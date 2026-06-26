#!/usr/bin/env python3
"""
Production-grade real-data Japan typhoon forecast pipeline.

The operational script accepts a target initialization timestamp, retrieves real
operational forecast grids from independent NWP centers, causally clips a real
ENSO index, applies a pre-trained real-observation-corrected ML model, and writes
one compressed NetCDF file plus a machine-readable provenance manifest.

No mocked, simulated, or analytically manufactured forecast/truth fields are used.
If a required real feed cannot be retrieved or decoded, the script fails closed.
"""
from __future__ import annotations

import argparse
import bz2
import calendar
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BBOX = {"lat_min": 24.0, "lat_max": 46.0, "lon_min": 122.0, "lon_max": 146.0}
GRID_DEG = 0.1
NLAT = 221
NLON = 241
NHOURS = 49
DEFAULT_OUTPUT = "typhoon_forecast_output.nc"
DEFAULT_MANIFEST = "provenance.json"
DEFAULT_MODEL = "models/typhoon_corrector.json"
NOAA_CPC_NINO34 = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
USER_AGENT = "typhoon-real-data-pipeline/1.0"

FEATURE_NAMES = [
    "intercept",
    "gfs",
    "ecmwf",
    "icon",
    "ensemble_mean",
    "ensemble_spread",
    "ensemble_min",
    "ensemble_max",
    "latitude_norm",
    "longitude_norm",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "nino34_anom",
    "baiu_front_proxy",
    "typhoon_wind_proxy",
    "orographic_rain_proxy",
]


class PipelineError(RuntimeError):
    pass


@dataclass
class DownloadRecord:
    source_provider: str
    url_or_archive_key: str
    model_cycle_init_time: str
    variables: List[str]
    lead_times: List[int]
    retrieval_timestamp: str
    local_path: str
    size_bytes: int
    sha256: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "source_provider": self.source_provider,
            "url_or_archive_key": self.url_or_archive_key,
            "model_cycle_init_time": self.model_cycle_init_time,
            "variables": self.variables,
            "lead_times": self.lead_times,
            "retrieval_timestamp": self.retrieval_timestamp,
            "local_path": self.local_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        out.update(self.extra)
        return out


@dataclass
class SourceRun:
    name: str
    provider: str
    cycle_time: datetime
    wind_speed: Any  # numpy array [time, lat, lon]
    precipitation: Any  # numpy array [time, lat, lon]
    records: List[DownloadRecord]
    notes: List[str] = field(default_factory=list)


def require_runtime_dependencies() -> Tuple[Any, Any, Any, Any]:
    missing: List[str] = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        missing.append("numpy")
        np = None
    try:
        import xarray as xr  # type: ignore
    except Exception:
        missing.append("xarray")
        xr = None
    try:
        import cfgrib  # type: ignore
    except Exception:
        missing.append("cfgrib")
        cfgrib = None
    try:
        import scipy.interpolate as spi  # type: ignore
    except Exception:
        missing.append("scipy")
        spi = None
    if missing:
        raise PipelineError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Create an environment inside this directory and install requirements.txt."
        )
    return np, xr, cfgrib, spi


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_init_timestamp(text: str) -> datetime:
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise PipelineError("Initialization timestamp must include timezone, e.g. 2026-06-25T00:00:00Z")
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_inside_root(path: Path) -> Path:
    root = ROOT.resolve()
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise PipelineError(f"Refusing to read or write outside working directory: {path}")
    return resolved


def local_path(relative_or_path: str | Path) -> Path:
    p = Path(relative_or_path)
    if not p.is_absolute():
        p = ROOT / p
    return ensure_inside_root(p)


def make_grid(np: Any) -> Tuple[Any, Any]:
    latitudes = np.round(np.linspace(BBOX["lat_min"], BBOX["lat_max"], NLAT), 1)
    longitudes = np.round(np.linspace(BBOX["lon_min"], BBOX["lon_max"], NLON), 1)
    if latitudes.shape[0] != NLAT or longitudes.shape[0] != NLON:
        raise PipelineError("Target grid dimensions are inconsistent")
    return latitudes, longitudes


def latest_cycle_at_or_before(init: datetime, cycle_hours: Sequence[int]) -> datetime:
    init = init.astimezone(timezone.utc)
    candidates: List[datetime] = []
    for delta_days in (0, -1, -2):
        base = (init + timedelta(days=delta_days)).date()
        for hour in cycle_hours:
            c = datetime(base.year, base.month, base.day, hour, tzinfo=timezone.utc)
            if c <= init:
                candidates.append(c)
    if not candidates:
        raise PipelineError(f"No cycle at or before {iso_z(init)} for hours {cycle_hours}")
    return max(candidates)


def download_url(url: str, dest: Path, source_provider: str, cycle: datetime, variables: List[str], leads: List[int], timeout: int = 180) -> DownloadRecord:
    dest = ensure_inside_root(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Optional cache reuse: when TYPHOON_REUSE_CACHE=1 and a previously retrieved
    # real file already exists on disk, reuse it instead of re-downloading.  The
    # bytes are still real provider data (their checksum is recomputed here); this
    # only avoids re-hitting the providers on a re-run and never synthesizes data.
    if os.environ.get("TYPHOON_REUSE_CACHE") == "1" and dest.exists() and dest.stat().st_size > 0:
        with dest.open("rb") as fh:
            start = fh.read(64).lstrip().lower()
        if not (start.startswith(b"<html") or start.startswith(b"<!doctype")):
            size, digest = file_sha256(dest)
            return DownloadRecord(
                source_provider=source_provider,
                url_or_archive_key=url,
                model_cycle_init_time=iso_z(cycle),
                variables=variables,
                lead_times=leads,
                retrieval_timestamp=utc_now_iso(),
                local_path=str(dest.relative_to(ROOT)),
                size_bytes=size,
                sha256=digest,
                extra={"reused_from_cache": True},
            )
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    retrieved_at = utc_now_iso()
    h = hashlib.sha256()
    size = 0
    try:
        with urlopen(req, timeout=timeout) as response, tmp.open("wb") as fh:
            status = getattr(response, "status", None)
            if status and int(status) >= 400:
                raise PipelineError(f"HTTP {status} retrieving {url}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
                fh.write(chunk)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    # Guard against provider error pages saved as data.
    with tmp.open("rb") as fh:
        start = fh.read(64).lstrip().lower()
    if start.startswith(b"<html") or start.startswith(b"<!doctype") or size == 0:
        tmp.unlink(missing_ok=True)
        raise PipelineError(f"Retrieved response from {url} is not a data file")
    tmp.replace(dest)
    return DownloadRecord(
        source_provider=source_provider,
        url_or_archive_key=url,
        model_cycle_init_time=iso_z(cycle),
        variables=variables,
        lead_times=leads,
        retrieval_timestamp=retrieved_at,
        local_path=str(dest.relative_to(ROOT)),
        size_bytes=size,
        sha256=h.hexdigest(),
    )


def file_sha256(path: Path) -> Tuple[int, str]:
    path = ensure_inside_root(path)
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return size, h.hexdigest()


def decompress_bz2(src: Path, dest: Path) -> None:
    src = ensure_inside_root(src)
    dest = ensure_inside_root(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with bz2.open(src, "rb") as inp, tmp.open("wb") as out:
        shutil.copyfileobj(inp, out)
    tmp.replace(dest)


def cfgrib_open_datasets(cfgrib: Any, path: Path) -> List[Any]:
    path = ensure_inside_root(path)
    try:
        return list(cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""}))
    except Exception as exc:
        raise PipelineError(f"Failed to decode GRIB with cfgrib: {path}: {exc}") from exc


def array_identity(da: Any) -> str:
    attrs = getattr(da, "attrs", {}) or {}
    parts = [str(getattr(da, "name", ""))]
    for key in ["GRIB_shortName", "GRIB_cfName", "GRIB_name", "long_name", "standard_name", "GRIB_units", "units"]:
        if key in attrs:
            parts.append(str(attrs[key]))
    return " ".join(parts).lower()


def find_data_array(datasets: Sequence[Any], candidates: Sequence[str]) -> Any:
    lowered = [c.lower() for c in candidates]
    for ds in datasets:
        for name, da in ds.data_vars.items():
            hay = array_identity(da)
            for candidate in lowered:
                if candidate in hay:
                    return da
    available = []
    for ds in datasets:
        for _name, da in ds.data_vars.items():
            available.append(array_identity(da))
    raise PipelineError(f"None of candidates {candidates} found in GRIB. Available variables: {available[:20]}")


def to_numpy(da: Any, np: Any) -> Any:
    return np.asarray(da.values, dtype="float64")


def coord_values(da: Any, np: Any, coord_names: Sequence[str]) -> Optional[Any]:
    for name in coord_names:
        if name in da.coords:
            return np.asarray(da.coords[name].values, dtype="float64")
        if name in da.dims and name in da.indexes:
            return np.asarray(da.indexes[name].values, dtype="float64")
    return None


def normalise_longitudes(lon: Any, np: Any) -> Any:
    lon = np.asarray(lon, dtype="float64")
    lon = np.where(lon < 0.0, lon + 360.0, lon)
    return lon


def build_unstructured_regridder(np: Any, native_lat: Any, native_lon: Any, target_lats: Any, target_lons: Any) -> Dict[str, Any]:
    """Precompute linear barycentric weights from an unstructured native grid."""
    from scipy.spatial import Delaunay, cKDTree  # type: ignore

    lat_flat = np.asarray(native_lat, dtype="float64").reshape(-1)
    lon_flat = normalise_longitudes(native_lon, np).reshape(-1)
    mask = (
        np.isfinite(lat_flat)
        & np.isfinite(lon_flat)
        & (lat_flat >= BBOX["lat_min"] - 1.0)
        & (lat_flat <= BBOX["lat_max"] + 1.0)
        & (lon_flat >= BBOX["lon_min"] - 1.0)
        & (lon_flat <= BBOX["lon_max"] + 1.0)
    )
    if int(mask.sum()) < 100:
        raise PipelineError("Too few native grid points inside Japan interpolation stencil")
    points = np.column_stack([lon_flat[mask], lat_flat[mask]])
    target_lon2d, target_lat2d = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])
    tri = Delaunay(points)
    simplex = tri.find_simplex(target_points)
    inside = simplex >= 0
    vertices = np.full((target_points.shape[0], 3), -1, dtype="int64")
    weights = np.zeros((target_points.shape[0], 3), dtype="float64")
    if inside.any():
        transform = tri.transform[simplex[inside], :2]
        delta = target_points[inside] - tri.transform[simplex[inside], 2]
        bary = np.einsum("ijk,ik->ij", transform, delta)
        weights[inside, :2] = bary
        weights[inside, 2] = 1.0 - bary.sum(axis=1)
        vertices[inside] = tri.simplices[simplex[inside]]
    tree = cKDTree(points)
    _dist, nearest = tree.query(target_points, k=1)
    return {
        "mask": mask,
        "vertices": vertices,
        "weights": weights,
        "inside": inside,
        "nearest": nearest,
        "shape": (NLAT, NLON),
    }


def apply_unstructured_regridder(np: Any, values: Any, regridder: Mapping[str, Any]) -> Any:
    values_flat = np.asarray(values, dtype="float64").reshape(-1)
    source = values_flat[regridder["mask"]]
    out = np.empty(regridder["inside"].shape[0], dtype="float64")
    inside = regridder["inside"]
    if inside.any():
        verts = regridder["vertices"][inside]
        w = regridder["weights"][inside]
        out[inside] = np.sum(source[verts] * w, axis=1)
    if (~inside).any():
        out[~inside] = source[regridder["nearest"][~inside]]
    return out.reshape(regridder["shape"])


def regrid_data_array(da: Any, np: Any, xr: Any, spi: Any, target_lats: Any, target_lons: Any) -> Any:
    values = to_numpy(da, np)
    while values.ndim > 2:
        values = values[0]
    lat = coord_values(da, np, ["latitude", "lat"])
    lon = coord_values(da, np, ["longitude", "lon"])
    if lat is None or lon is None:
        raise PipelineError(f"Data array {getattr(da, 'name', '')} has no latitude/longitude coordinates")
    lon = normalise_longitudes(lon, np)

    # Regular 1-D latitude/longitude grid.
    if lat.ndim == 1 and lon.ndim == 1 and values.ndim == 2:
        dims = list(da.dims)[-2:]
        work = xr.DataArray(values, dims=dims, coords={dims[0]: lat, dims[1]: lon})
        if lat[0] > lat[-1]:
            work = work.sortby(dims[0])
        if lon[0] > lon[-1]:
            work = work.sortby(dims[1])
        interp = work.interp({dims[0]: target_lats, dims[1]: target_lons}, method="linear")
        arr = np.asarray(interp.values, dtype="float64")
        if np.isnan(arr).any():
            nearest = work.interp({dims[0]: target_lats, dims[1]: target_lons}, method="nearest")
            arr = np.where(np.isnan(arr), np.asarray(nearest.values, dtype="float64"), arr)
        return arr

    # Curvilinear / unstructured grid (DWD ICON icosahedral).  Use real native
    # coordinates and interpolate; no values are invented except interpolation of
    # retrieved real fields.
    lat_flat = np.asarray(lat).reshape(-1)
    lon_flat = np.asarray(lon).reshape(-1)
    val_flat = np.asarray(values).reshape(-1)
    if lat_flat.shape[0] != val_flat.shape[0] or lon_flat.shape[0] != val_flat.shape[0]:
        raise PipelineError("Curvilinear coordinate/value shapes do not align")
    mask = (
        np.isfinite(lat_flat)
        & np.isfinite(lon_flat)
        & np.isfinite(val_flat)
        & (lat_flat >= BBOX["lat_min"] - 1.0)
        & (lat_flat <= BBOX["lat_max"] + 1.0)
        & (lon_flat >= BBOX["lon_min"] - 1.0)
        & (lon_flat <= BBOX["lon_max"] + 1.0)
    )
    if int(mask.sum()) < 100:
        raise PipelineError("Too few native grid points inside Japan interpolation stencil")
    points = np.column_stack([lon_flat[mask], lat_flat[mask]])
    target_lon2d, target_lat2d = np.meshgrid(target_lons, target_lats)
    target_points = np.column_stack([target_lon2d.ravel(), target_lat2d.ravel()])
    linear = spi.griddata(points, val_flat[mask], target_points, method="linear")
    if np.isnan(linear).any():
        nearest = spi.griddata(points, val_flat[mask], target_points, method="nearest")
        linear = np.where(np.isnan(linear), nearest, linear)
    return np.asarray(linear.reshape((NLAT, NLON)), dtype="float64")


def precip_to_mm(da: Any, arr: Any, np: Any) -> Any:
    attrs = getattr(da, "attrs", {}) or {}
    units = str(attrs.get("GRIB_units", attrs.get("units", ""))).lower()
    out = np.asarray(arr, dtype="float64")
    if units in {"m", "m of water equivalent"} or "metre" in units:
        out = out * 1000.0
    # kg m**-2 is numerically mm of liquid water.
    return np.maximum(out, 0.0)


def wind_speed_from_uv(u_da: Any, v_da: Any, np: Any, xr: Any, spi: Any, target_lats: Any, target_lons: Any) -> Any:
    u = regrid_data_array(u_da, np, xr, spi, target_lats, target_lons)
    v = regrid_data_array(v_da, np, xr, spi, target_lats, target_lons)
    return np.sqrt(u * u + v * v)


def interpolate_time_series(native_leads: Sequence[int], native_values: Sequence[Any], desired_leads: Sequence[float], np: Any) -> Any:
    leads = np.asarray(native_leads, dtype="float64")
    order = np.argsort(leads)
    leads = leads[order]
    stack = np.stack([native_values[int(i)] for i in order], axis=0)
    desired = np.asarray(desired_leads, dtype="float64")
    if desired.min() < leads.min() or desired.max() > leads.max():
        raise PipelineError(f"Native leads {leads.min()}..{leads.max()} do not bracket desired {desired.min()}..{desired.max()}")
    flat = stack.reshape((stack.shape[0], -1))
    out = np.empty((desired.shape[0], flat.shape[1]), dtype="float64")
    # Vectorized linear interpolation over lead time for each grid cell.
    for k, lead in enumerate(desired):
        hi = int(np.searchsorted(leads, lead, side="left"))
        if hi < len(leads) and leads[hi] == lead:
            out[k, :] = flat[hi, :]
        else:
            lo = hi - 1
            if lo < 0 or hi >= len(leads):
                raise PipelineError("Time interpolation bracket failure")
            weight = (lead - leads[lo]) / (leads[hi] - leads[lo])
            out[k, :] = flat[lo, :] * (1.0 - weight) + flat[hi, :] * weight
    return out.reshape((desired.shape[0],) + stack.shape[1:])


def hourly_precip_from_cumulative(native_leads: Sequence[int], cumulative_values: Sequence[Any], offset_hours: int, np: Any) -> Any:
    desired_cumulative_leads = [float(offset_hours + h) for h in range(NHOURS)]
    cumulative = interpolate_time_series(native_leads, cumulative_values, desired_cumulative_leads, np)
    hourly = np.empty_like(cumulative)
    hourly[0, :, :] = 0.0
    hourly[1:, :, :] = cumulative[1:, :, :] - cumulative[:-1, :, :]
    return np.maximum(hourly, 0.0)


def cumulative_from_hourly(native_leads: Sequence[int], hourly_values_by_lead: Mapping[int, Any], np: Any) -> Tuple[List[int], List[Any]]:
    ordered = sorted(native_leads)
    cumul: List[Any] = []
    running = None
    last_lead = -1
    for lead in ordered:
        if lead == 0:
            sample = hourly_values_by_lead.get(lead)
            if sample is None:
                sample = next(iter(hourly_values_by_lead.values()))
            running = np.zeros_like(sample, dtype="float64")
            cumul.append(running.copy())
            last_lead = 0
            continue
        if running is None:
            sample = hourly_values_by_lead[lead]
            running = np.zeros_like(sample, dtype="float64")
        # NOMADS GFS is requested at hourly leads; if a previous lead is absent,
        # fail rather than fabricate it.
        for missing in range(last_lead + 1, lead + 1):
            if missing not in hourly_values_by_lead:
                raise PipelineError(f"Missing hourly precipitation interval for GFS lead {missing}")
            running = running + np.maximum(hourly_values_by_lead[missing], 0.0)
        cumul.append(running.copy())
        last_lead = lead
    return ordered, cumul


def gfs_apcp_bucket_to_hourly(native_leads: Sequence[int], bucket_by_lead: Mapping[int, Any], np: Any, bucket_hours: int = 6) -> Dict[int, Any]:
    """Convert NOAA GFS APCP into true per-hour increments.

    NOAA/NCEP GFS surface APCP on the 0.25 deg pgrb2 product is *not* a forecast
    accumulation measured from initialization.  It accumulates within rolling
    ``bucket_hours`` (6 h) windows and resets at every window boundary, so the
    domain mean climbs f001..f006 and then drops at f007, again at f013, etc.
    The genuine incremental precipitation over the hour ending at lead ``h`` is::

        hourly[h] = acc[h]                if (h-1) % bucket_hours == 0   # first hour of a new bucket
                  = acc[h] - acc[h-1]     otherwise                      # same bucket, de-accumulate

    Only real decoded APCP fields are used; nothing is synthesized.  The returned
    true-hourly series is then re-accumulated from init and de-accumulated onto the
    target hourly grid by the shared accumulation helpers, exactly like the other
    centers whose precipitation already accumulates from initialization.
    """
    if bucket_hours < 1:
        raise PipelineError("bucket_hours must be a positive integer")
    forecast_leads = sorted(lead for lead in native_leads if lead >= 1)
    hourly: Dict[int, Any] = {}
    for lead in forecast_leads:
        acc = bucket_by_lead.get(lead)
        if acc is None:
            raise PipelineError(f"GFS APCP missing for lead {lead}; refusing to fabricate precipitation")
        if (lead - 1) % bucket_hours == 0:
            hourly[lead] = np.maximum(np.asarray(acc, dtype="float64"), 0.0)
        else:
            prev = bucket_by_lead.get(lead - 1)
            if prev is None:
                raise PipelineError(f"GFS APCP missing predecessor lead {lead - 1} needed for de-accumulation")
            hourly[lead] = np.maximum(np.asarray(acc, dtype="float64") - np.asarray(prev, dtype="float64"), 0.0)
    if 0 in native_leads:
        sample = next(iter(hourly.values())) if hourly else None
        hourly[0] = np.zeros_like(sample) if sample is not None else np.zeros((NLAT, NLON), dtype="float64")
    return hourly


def decode_grib_fields(path: Path, np: Any, xr: Any, cfgrib: Any, spi: Any, target_lats: Any, target_lons: Any) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    datasets = cfgrib_open_datasets(cfgrib, path)
    wind = None
    precip = None
    cumul_precip_da = None
    try:
        u_da = find_data_array(datasets, ["10u", "u10", "u component of wind", "u-component", "ugrd", "u_10m", "U_10M"])
        v_da = find_data_array(datasets, ["10v", "v10", "v component of wind", "v-component", "vgrd", "v_10m", "V_10M"])
        wind = wind_speed_from_uv(u_da, v_da, np, xr, spi, target_lats, target_lons)
    except PipelineError:
        wind = None
    try:
        p_da = find_data_array(datasets, ["tp", "total precipitation", "apcp", "precipitation", "tot_prec", "TOT_PREC"])
        raw = regrid_data_array(p_da, np, xr, spi, target_lats, target_lons)
        precip = precip_to_mm(p_da, raw, np)
        cumul_precip_da = p_da
    except PipelineError:
        precip = None
    return wind, precip, cumul_precip_da


class NOAAProvider:
    name = "gfs"
    provider = "NOAA/NCEP GFS via NOMADS filter_gfs_0p25"

    def retrieve(self, init: datetime, workdir: Path, np: Any, xr: Any, cfgrib: Any, spi: Any, target_lats: Any, target_lons: Any) -> SourceRun:
        cycle = latest_cycle_at_or_before(init, [0, 6, 12, 18])
        offset = int((init - cycle).total_seconds() // 3600)
        max_lead = offset + 48
        native_leads = list(range(0, max_lead + 1))
        records: List[DownloadRecord] = []
        wind_by_lead: Dict[int, Any] = {}
        precip_hour_by_lead: Dict[int, Any] = {}
        base_dir = ensure_inside_root(workdir / "noaa_gfs" / cycle.strftime("%Y%m%d%H"))
        for lead in native_leads:
            params = {
                "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
                "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}",
                "lev_10_m_above_ground": "on",
                "lev_surface": "on",
                "var_UGRD": "on",
                "var_VGRD": "on",
                "var_APCP": "on",
                "subregion": "",
                "toplat": f"{BBOX['lat_max']:.1f}",
                "leftlon": f"{BBOX['lon_min']:.1f}",
                "rightlon": f"{BBOX['lon_max']:.1f}",
                "bottomlat": f"{BBOX['lat_min']:.1f}",
            }
            url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?" + urlencode(params)
            dest = base_dir / f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}.japan.grib2"
            record = download_url(url, dest, self.provider, cycle, ["UGRD_10m", "VGRD_10m", "APCP_surface"], [lead])
            records.append(record)
            wind, precip, _p_da = decode_grib_fields(dest, np, xr, cfgrib, spi, target_lats, target_lons)
            if wind is not None:
                wind_by_lead[lead] = wind
            elif lead in (0,):
                raise PipelineError(f"GFS wind missing at required lead {lead}")
            if precip is not None:
                precip_hour_by_lead[lead] = precip
            elif lead == 0:
                # Lead-zero accumulated precipitation is conventionally zero if
                # the center does not encode APCP at f000.  This is not a synthetic
                # field used as model input; it is the definition of accumulation at init.
                precip_hour_by_lead[lead] = np.zeros((NLAT, NLON), dtype="float64")
        desired = [float(offset + h) for h in range(NHOURS)]
        missing_wind = [lead for lead in range(offset, offset + 49) if lead not in wind_by_lead]
        if missing_wind:
            raise PipelineError(f"GFS missing wind leads {missing_wind[:10]}")
        wind_values = [wind_by_lead[lead] for lead in sorted(wind_by_lead)]
        wind = interpolate_time_series(sorted(wind_by_lead), wind_values, desired, np)
        # GFS APCP accumulates in rolling 6-hour buckets and resets at each
        # boundary; convert it to genuine per-hour increments before the shared
        # accumulate/de-accumulate path maps it onto the hourly target grid.
        true_hourly_by_lead = gfs_apcp_bucket_to_hourly(native_leads, precip_hour_by_lead, np, bucket_hours=6)
        cumul_leads, cumul_values = cumulative_from_hourly(native_leads, true_hourly_by_lead, np)
        precip = hourly_precip_from_cumulative(cumul_leads, cumul_values, offset, np)
        notes = [
            "NOAA GFS surface APCP accumulates in rolling 6-hour buckets that reset at each boundary; "
            "it is de-accumulated to genuine per-hour increments before regridding to the hourly target.",
        ]
        return SourceRun(self.name, self.provider, cycle, wind, precip, records, notes)


class ECMWFProvider:
    name = "ecmwf"
    provider = "ECMWF IFS Open Data"

    def retrieve(self, init: datetime, workdir: Path, np: Any, xr: Any, cfgrib: Any, spi: Any, target_lats: Any, target_lons: Any) -> SourceRun:
        try:
            from ecmwf.opendata import Client  # type: ignore
        except Exception as exc:
            raise PipelineError("Missing dependency ecmwf-opendata needed for parameter-filtered ECMWF retrieval") from exc
        cycle = latest_cycle_at_or_before(init, [0, 12])
        offset = int((init - cycle).total_seconds() // 3600)
        max_lead = int(math.ceil((offset + 48) / 3.0) * 3)
        native_leads = list(range(0, max_lead + 1, 3))
        # AWS mirror of the same ECMWF open-data (the ecmwf.int portal throttles/resets
        # connections under load); identical real IFS data, more reliable CDN.
        client = Client(source="aws")
        records: List[DownloadRecord] = []
        wind_by_lead: Dict[int, Any] = {}
        cumul_precip_by_lead: Dict[int, Any] = {}
        base_dir = ensure_inside_root(workdir / "ecmwf_ifs" / cycle.strftime("%Y%m%d%H"))
        for lead in native_leads:
            # Use the official ecmwf-opendata client rather than downloading the
            # full per-step GRIB.  It byte-range retrieves only the real requested
            # parameters from ECMWF's open-data files, preserving provenance while
            # avoiding multi-GB operational runs.
            params = ["10u", "10v"] if lead == 0 else ["10u", "10v", "tp"]
            archive_key = (
                f"ecmwf-opendata://date={cycle:%Y%m%d}/time={cycle:%H}/stream=oper/"
                f"type=fc/step={lead}/param={','.join(params)}"
            )
            dest = base_dir / f"ecmwf_ifs_{cycle:%Y%m%d%H}_f{lead:03d}.grib2"
            dest = ensure_inside_root(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            retrieved_at = utc_now_iso()
            reused = False
            if os.environ.get("TYPHOON_REUSE_CACHE") == "1" and dest.exists() and dest.stat().st_size > 0:
                with dest.open("rb") as fh:
                    head = fh.read(4)
                if head == b"GRIB":
                    reused = True
            if not reused:
                client.retrieve(
                    date=f"{cycle:%Y%m%d}",
                    time=f"{cycle:%H}",
                    step=[lead],
                    stream="oper",
                    type="fc",
                    param=params,
                    target=str(dest),
                )
            size, digest = file_sha256(dest)
            record = DownloadRecord(
                source_provider=self.provider,
                url_or_archive_key=archive_key,
                model_cycle_init_time=iso_z(cycle),
                variables=params,
                lead_times=[lead],
                retrieval_timestamp=retrieved_at,
                local_path=str(dest.relative_to(ROOT)),
                size_bytes=size,
                sha256=digest,
                extra={"retrieval_tool": "ecmwf-opendata", "reused_from_cache": reused},
            )
            records.append(record)
            wind, precip, _p_da = decode_grib_fields(dest, np, xr, cfgrib, spi, target_lats, target_lons)
            if wind is not None:
                wind_by_lead[lead] = wind
            if precip is not None:
                cumul_precip_by_lead[lead] = precip
            elif lead == 0:
                cumul_precip_by_lead[lead] = np.zeros((NLAT, NLON), dtype="float64")
        if not all(lead in wind_by_lead for lead in native_leads):
            raise PipelineError("ECMWF wind was not decoded for every native lead")
        desired = [float(offset + h) for h in range(NHOURS)]
        wind = interpolate_time_series(native_leads, [wind_by_lead[lead] for lead in native_leads], desired, np)
        if not all(lead in cumul_precip_by_lead for lead in native_leads):
            missing = [lead for lead in native_leads if lead not in cumul_precip_by_lead]
            raise PipelineError(f"ECMWF cumulative precipitation missing leads {missing}")
        precip = hourly_precip_from_cumulative(native_leads, [cumul_precip_by_lead[lead] for lead in native_leads], offset, np)
        notes = ["ECMWF open-data IFS is available at 3-hour forecast steps; wind and cumulative precipitation are linearly interpolated/de-accumulated to the required hourly target grid."]
        return SourceRun(self.name, self.provider, cycle, wind, precip, records, notes)


class DWDICONProvider:
    name = "icon"
    provider = "DWD ICON Global Open Data (documented independent substitute for direct JMA GSM GRIB feed)"
    var_specs = {
        "u": ("u_10m", "U_10M"),
        "v": ("v_10m", "V_10M"),
        "p": ("tot_prec", "TOT_PREC"),
    }

    coord_specs = {
        "lat": ("clat", "CLAT"),
        "lon": ("clon", "CLON"),
    }

    def retrieve(self, init: datetime, workdir: Path, np: Any, xr: Any, cfgrib: Any, spi: Any, target_lats: Any, target_lons: Any) -> SourceRun:
        cycle = latest_cycle_at_or_before(init, [0, 6, 12, 18])
        offset = int((init - cycle).total_seconds() // 3600)
        max_lead = offset + 48
        native_leads = list(range(0, max_lead + 1))
        records: List[DownloadRecord] = []
        wind_by_lead: Dict[int, Any] = {}
        cumul_precip_by_lead: Dict[int, Any] = {}
        base_dir = ensure_inside_root(workdir / "dwd_icon" / cycle.strftime("%Y%m%d%H"))
        coord_paths: Dict[str, Path] = {}
        for key, (subdir, grib_name) in self.coord_specs.items():
            filename = f"icon_global_icosahedral_time-invariant_{cycle:%Y%m%d%H}_{grib_name}.grib2.bz2"
            url = f"https://opendata.dwd.de/weather/nwp/icon/grib/{cycle:%H}/{subdir}/{filename}"
            compressed = base_dir / filename
            record = download_url(url, compressed, self.provider, cycle, [grib_name], [0], timeout=300)
            record.extra["coordinate_file"] = True
            records.append(record)
            decoded = base_dir / filename[:-4]
            decompress_bz2(compressed, decoded)
            coord_paths[key] = decoded
        lat_ds = cfgrib_open_datasets(cfgrib, coord_paths["lat"])
        lon_ds = cfgrib_open_datasets(cfgrib, coord_paths["lon"])
        lat_da = find_data_array(lat_ds, ["tlat", "CLAT", "Latitude on T grid"])
        lon_da = find_data_array(lon_ds, ["tlon", "CLON", "Longitude on T grid"])
        icon_regridder = build_unstructured_regridder(np, lat_da.values, lon_da.values, target_lats, target_lons)
        for lead in native_leads:
            decoded_paths: Dict[str, Path] = {}
            for key, (subdir, grib_name) in self.var_specs.items():
                filename = f"icon_global_icosahedral_single-level_{cycle:%Y%m%d%H}_{lead:03d}_{grib_name}.grib2.bz2"
                url = f"https://opendata.dwd.de/weather/nwp/icon/grib/{cycle:%H}/{subdir}/{filename}"
                compressed = base_dir / filename
                record = download_url(url, compressed, self.provider, cycle, [grib_name], [lead], timeout=300)
                records.append(record)
                decoded = base_dir / filename[:-4]  # remove .bz2
                decompress_bz2(compressed, decoded)
                decoded_paths[key] = decoded
            u_ds = cfgrib_open_datasets(cfgrib, decoded_paths["u"])
            v_ds = cfgrib_open_datasets(cfgrib, decoded_paths["v"])
            p_ds = cfgrib_open_datasets(cfgrib, decoded_paths["p"])
            u_da = find_data_array(u_ds, ["u_10m", "U_10M", "10u", "u component of wind"])
            v_da = find_data_array(v_ds, ["v_10m", "V_10M", "10v", "v component of wind"])
            p_da = find_data_array(p_ds, ["tot_prec", "TOT_PREC", "total precipitation"])
            u_grid = apply_unstructured_regridder(np, u_da.values, icon_regridder)
            v_grid = apply_unstructured_regridder(np, v_da.values, icon_regridder)
            wind_by_lead[lead] = np.sqrt(u_grid * u_grid + v_grid * v_grid)
            raw_precip = apply_unstructured_regridder(np, p_da.values, icon_regridder)
            cumul_precip_by_lead[lead] = precip_to_mm(p_da, raw_precip, np)
        desired = [float(offset + h) for h in range(NHOURS)]
        wind = interpolate_time_series(native_leads, [wind_by_lead[lead] for lead in native_leads], desired, np)
        precip = hourly_precip_from_cumulative(native_leads, [cumul_precip_by_lead[lead] for lead in native_leads], offset, np)
        notes = [
            "DWD ICON Global is used as the real independent substitute because a stable no-auth direct JMA GSM GRIB2 endpoint is not exposed in this environment; no synthetic substitute is used.",
            "ICON native icosahedral fields are decoded on their native coordinates and interpolated to the target regular grid.",
        ]
        return SourceRun(self.name, self.provider, cycle, wind, precip, records, notes)


def fetch_nino34(init: datetime) -> Tuple[float, DownloadRecord, Dict[str, Any]]:
    dest = local_path("data/raw/noaa_cpc/sstoi.indices")
    record = download_url(NOAA_CPC_NINO34, dest, "NOAA CPC", init, ["NINO3.4_ANOM"], [0])
    text = dest.read_text(encoding="utf-8", errors="replace")
    values: Dict[Tuple[int, int], float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0].isdigit():
            values[(int(parts[0]), int(parts[1]))] = float(parts[9])
    best_key: Optional[Tuple[int, int]] = None
    best_publication: Optional[datetime] = None
    for year_month in values:
        y, m = year_month
        last_day = calendar.monthrange(y, m)[1]
        publication = datetime(y, m, last_day, tzinfo=timezone.utc) + timedelta(days=15)
        if publication <= init and (best_publication is None or publication > best_publication):
            best_key = year_month
            best_publication = publication
    if best_key is None or best_publication is None:
        raise PipelineError(f"No causally published NINO3.4 anomaly available at {iso_z(init)}")
    value = values[best_key]
    meta = {
        "enso_index": "NINO3.4 SST anomaly",
        "source": "NOAA CPC sstoi.indices",
        "used_year": best_key[0],
        "used_month": best_key[1],
        "assumed_publication_lag_days": 15,
        "publication_time_used_for_causal_fence": iso_z(best_publication),
        "value": value,
    }
    record.extra.update(meta)
    return value, record, meta


def load_model(model_path: Path, init: Optional[datetime] = None) -> Dict[str, Any]:
    model_path = ensure_inside_root(model_path)
    if not model_path.exists():
        raise PipelineError(
            f"Saved model artifact not found: {model_path.relative_to(ROOT)}. Run train_corrector.py first; the operational pipeline never trains."
        )
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("artifact_type") != "real_data_ridge_pointwise_corrector":
        raise PipelineError("Unsupported model artifact type")
    feature_names = model.get("feature_names")
    if feature_names != FEATURE_NAMES:
        raise PipelineError("Model feature schema does not match pipeline feature schema")
    cutoff = str(model.get("training_cutoff", ""))
    # Causal fence: the model must only have been trained on data that predates the
    # operational target init.  A model whose training cutoff is at or before the
    # target init introduces no leakage, regardless of absolute calendar date.
    if cutoff and init is not None and cutoff > iso_z(init):
        raise PipelineError(
            f"Model training cutoff {cutoff} is after the target init {iso_z(init)} (leakage)"
        )
    for target in ["wind_speed", "precipitation"]:
        coeffs = model.get("model_targets", {}).get(target, {}).get("coefficients")
        if not isinstance(coeffs, list) or len(coeffs) != len(FEATURE_NAMES):
            raise PipelineError(f"Model coefficients missing/malformed for {target}")
    return model


def build_features(np: Any, values: Sequence[Any], target_lats: Any, target_lons: Any, valid_times: Sequence[datetime], nino34: float) -> Any:
    gfs, ecmwf, icon = values
    stack = np.stack([gfs, ecmwf, icon], axis=0)
    ens_mean = np.mean(stack, axis=0)
    ens_spread = np.std(stack, axis=0)
    ens_min = np.min(stack, axis=0)
    ens_max = np.max(stack, axis=0)
    lat2d, lon2d = np.meshgrid(target_lats, target_lons, indexing="ij")
    lat_norm_2d = (lat2d - 35.0) / 11.0
    lon_norm_2d = (lon2d - 134.0) / 12.0
    features = np.empty((len(FEATURE_NAMES), NHOURS, NLAT, NLON), dtype="float64")
    for t_idx, vt in enumerate(valid_times):
        hour_angle = 2.0 * math.pi * (vt.hour / 24.0)
        month_angle = 2.0 * math.pi * ((vt.month - 1) / 12.0)
        baiu_month = 1.0 if vt.month in (6, 7) else 0.0
        baiu_lat = np.maximum(0.0, 1.0 - np.abs(lat2d - 33.0) / 7.0)
        south_west = np.maximum(0.0, (36.0 - lat2d) / 12.0) * np.maximum(0.0, (140.0 - lon2d) / 18.0)
        orographic_band = np.maximum(0.0, 1.0 - np.abs(lat2d - 34.5) / 5.0) * np.maximum(0.0, 1.0 - np.abs(lon2d - 136.5) / 8.0)
        cols = [
            np.ones((NLAT, NLON), dtype="float64"),
            gfs[t_idx],
            ecmwf[t_idx],
            icon[t_idx],
            ens_mean[t_idx],
            ens_spread[t_idx],
            ens_min[t_idx],
            ens_max[t_idx],
            lat_norm_2d,
            lon_norm_2d,
            np.full((NLAT, NLON), math.sin(hour_angle), dtype="float64"),
            np.full((NLAT, NLON), math.cos(hour_angle), dtype="float64"),
            np.full((NLAT, NLON), math.sin(month_angle), dtype="float64"),
            np.full((NLAT, NLON), math.cos(month_angle), dtype="float64"),
            np.full((NLAT, NLON), nino34, dtype="float64"),
            baiu_month * baiu_lat * np.maximum(ens_mean[t_idx], 0.0),
            south_west * ens_max[t_idx],
            orographic_band * np.maximum(ens_mean[t_idx], 0.0),
        ]
        for f_idx, col in enumerate(cols):
            features[f_idx, t_idx] = col
    return features


def apply_corrector(np: Any, model: Mapping[str, Any], runs: Mapping[str, SourceRun], target_lats: Any, target_lons: Any, valid_times: Sequence[datetime], nino34: float) -> Tuple[Any, Any]:
    wind_features = build_features(
        np,
        [runs["gfs"].wind_speed, runs["ecmwf"].wind_speed, runs["icon"].wind_speed],
        target_lats,
        target_lons,
        valid_times,
        nino34,
    )
    precip_features = build_features(
        np,
        [runs["gfs"].precipitation, runs["ecmwf"].precipitation, runs["icon"].precipitation],
        target_lats,
        target_lons,
        valid_times,
        nino34,
    )
    wind_coeffs = np.asarray(model["model_targets"]["wind_speed"]["coefficients"], dtype="float64")
    precip_coeffs = np.asarray(model["model_targets"]["precipitation"]["coefficients"], dtype="float64")
    wind = np.tensordot(wind_coeffs, wind_features, axes=(0, 0))
    precip = np.tensordot(precip_coeffs, precip_features, axes=(0, 0))
    precip = np.maximum(precip, 0.0)
    # T+0 is the initialization instant, so the preceding-hour accumulation
    # within the forecast horizon is exactly zero even after ML correction.
    precip[0, :, :] = 0.0
    return np.maximum(wind, 0.0), precip


def write_netcdf(output_path: Path, xr: Any, np: Any, latitudes: Any, longitudes: Any, valid_times: Sequence[datetime], wind: Any, precip: Any, attrs: Mapping[str, Any]) -> None:
    output_path = ensure_inside_root(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times64 = np.array([np.datetime64(vt.replace(tzinfo=None)) for vt in valid_times])
    ds = xr.Dataset(
        data_vars={
            "wind_speed": (("time", "latitude", "longitude"), wind.astype("float32"), {"units": "m s-1", "long_name": "ML-corrected 10 m sustained wind speed"}),
            "precipitation": (("time", "latitude", "longitude"), precip.astype("float32"), {"units": "mm", "long_name": "ML-corrected hourly accumulated precipitation"}),
        },
        coords={
            "time": ("time", times64),
            "latitude": ("latitude", latitudes.astype("float32"), {"units": "degrees_north"}),
            "longitude": ("longitude", longitudes.astype("float32"), {"units": "degrees_east"}),
        },
        attrs={k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in attrs.items()},
    )
    if ds.sizes.get("time") != NHOURS or ds.sizes.get("latitude") != NLAT or ds.sizes.get("longitude") != NLON:
        raise PipelineError(f"Output schema dimensions wrong: {dict(ds.sizes)}")
    encoding = {
        "wind_speed": {"zlib": True, "complevel": 9, "shuffle": True, "dtype": "float32"},
        "precipitation": {"zlib": True, "complevel": 9, "shuffle": True, "dtype": "float32"},
    }
    tmp = output_path.with_suffix(output_path.suffix + ".part")
    try:
        ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=encoding)
    except Exception as exc:
        raise PipelineError(f"Failed to write NetCDF4 output. Ensure netCDF4 is installed: {exc}") from exc
    tmp.replace(output_path)


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path = ensure_inside_root(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the real-data Japan typhoon forecast pipeline.")
    parser.add_argument("--init", required=True, help="Target initialization timestamp, e.g. 2026-06-25T00:00:00Z")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="NetCDF output path inside this directory")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Provenance JSON output path inside this directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Saved corrector JSON artifact inside this directory")
    parser.add_argument("--workdir", default="data/raw", help="Raw-data working directory inside this directory")
    args = parser.parse_args(argv)

    init = parse_init_timestamp(args.init)
    output_path = local_path(args.output)
    manifest_path = local_path(args.manifest)
    model_path = local_path(args.model)
    workdir = local_path(args.workdir)
    if output_path.name != "typhoon_forecast_output.nc":
        raise PipelineError("The required primary output filename is typhoon_forecast_output.nc")

    np, xr, cfgrib, spi = require_runtime_dependencies()
    latitudes, longitudes = make_grid(np)
    valid_times = [init + timedelta(hours=h) for h in range(NHOURS)]
    model = load_model(model_path, init)
    model_size, model_hash = file_sha256(model_path)
    nino34_value, nino_record, nino_meta = fetch_nino34(init)

    providers = [NOAAProvider(), ECMWFProvider(), DWDICONProvider()]
    runs: Dict[str, SourceRun] = {}
    for provider in providers:
        print(f"Retrieving {provider.provider} for target {iso_z(init)}", flush=True)
        run = provider.retrieve(init, workdir, np, xr, cfgrib, spi, latitudes, longitudes)
        if run.cycle_time > init:
            raise PipelineError(f"Causality violation: {run.provider} cycle {iso_z(run.cycle_time)} > init {iso_z(init)}")
        runs[run.name] = run

    wind, precip = apply_corrector(np, model, runs, latitudes, longitudes, valid_times, nino34_value)
    source_runs = {
        name: {
            "provider": run.provider,
            "cycle_time": iso_z(run.cycle_time),
            "notes": run.notes,
        }
        for name, run in runs.items()
    }
    attrs = {
        "title": "Real-data ML-corrected 48-hour gridded typhoon forecast over Japan",
        "initialization_timestamp": iso_z(init),
        "ml_model_weights": {"path": str(model_path.relative_to(ROOT)), "sha256": model_hash, "size_bytes": model_size, "artifact_type": model.get("artifact_type")},
        "source_model_runs_processed": source_runs,
        "real_data_only": "All NWP fields and ENSO data were retrieved from named external sources; no mocked or generated forecast fields are used.",
        "grid": {"latitude_min": BBOX["lat_min"], "latitude_max": BBOX["lat_max"], "longitude_min": BBOX["lon_min"], "longitude_max": BBOX["lon_max"], "resolution_degrees": GRID_DEG},
    }
    write_netcdf(output_path, xr, np, latitudes, longitudes, valid_times, wind, precip, attrs)
    output_size, output_hash = file_sha256(output_path)

    all_records: List[Dict[str, Any]] = [nino_record.as_dict()]
    for run in runs.values():
        all_records.extend(record.as_dict() for record in run.records)
    manifest = {
        "created_at": utc_now_iso(),
        "initialization_timestamp": iso_z(init),
        "causality": {
            "rule": "Only model cycles initialized before or on target init are used; ENSO is clipped by publication lag.",
            "target_init": iso_z(init),
            "source_model_runs": source_runs,
            "enso": nino_meta,
        },
        "output": {"path": str(output_path.relative_to(ROOT)), "size_bytes": output_size, "sha256": output_hash, "schema": {"time": NHOURS, "latitude": NLAT, "longitude": NLON, "variables": ["wind_speed", "precipitation"]}},
        "model_artifact": {"path": str(model_path.relative_to(ROOT)), "size_bytes": model_size, "sha256": model_hash, "verification": model.get("verification")},
        "ingested_real_files": all_records,
        "source_substitution": {
            "requested_center": "JMA GSM",
            "used_substitute": "DWD ICON Global Open Data",
            "reason": "No stable no-auth direct JMA GSM GRIB2 endpoint was exposed in this environment; the prompt permits documented real independent substitutes and forbids synthesis.",
        },
    }
    write_manifest(manifest_path, manifest)
    print(f"Wrote {output_path.relative_to(ROOT)} and {manifest_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
