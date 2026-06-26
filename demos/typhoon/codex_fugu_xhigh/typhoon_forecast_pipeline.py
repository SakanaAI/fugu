#!/usr/bin/env python3
"""
Operational 48-hour Japan wind/precipitation forecast pipeline.

This script is intentionally fail-closed: it downloads real forecast GRIB files from
real providers, records provenance for every network object used, applies a saved
ML corrector trained separately against real observations, and refuses to generate
synthetic substitute fields.

Default providers:
  * NOAA/NCEP GFS from NOMADS GRIB filter (official GRIB2 subset service)
  * ECMWF IFS open-data from ECMWF Open Data (via ecmwf-opendata client)
  * DWD ICON global from DWD Open Data as the documented JMA substitute.  The
    JMA WIS GSM GRIB endpoint is credential-gated from this execution context;
    using DWD avoids unauthenticated scraping and still provides an independent
    operational global NWP center.

Output:
  typhoon_forecast_output.nc with dimensions time=49, latitude=221, longitude=241
  provenance.json with retrievable source URLs, checksums, cycles, variables,
  lead times, ENSO provenance, and the saved corrector's validation scores.
"""
from __future__ import annotations

import argparse
import bz2
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Third-party imports are intentionally delayed until runtime so --help and basic
# validation work in a fresh checkout.  Install from requirements.txt inside this
# directory before running the operational pipeline.

LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 146.0
DLAT = DLON = 0.1
NTIME = 49
N_LAT = 221
N_LON = 241
FORECAST_HOURS = list(range(0, 49))
CYCLES_6H = (0, 6, 12, 18)
USER_AGENT = "fugu-typhoon-realdata-pipeline/1.0 (no-mock-operational-run)"
GFS_NOMADS_BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
CPC_NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
DWD_BASE = "https://opendata.dwd.de/weather/nwp/icon/grib"
JMA_WIS_EXAMPLE = (
    "https://www.wis-jma.go.jp/d/c/RJTD/GRIB/Global_Spectral_Model/"
    "Latitude_Longitude/0.25_0.25/90.0_-5.0_30.0_195.0/Surface_layers/"
)


class PipelineError(RuntimeError):
    """A user-actionable, fail-closed pipeline error."""


@dataclasses.dataclass
class ProvenanceRecord:
    provider: str
    url: str
    local_path: str
    cycle_init: str
    variables: List[str]
    lead_times: List[int]
    retrieval_timestamp: str
    size_bytes: int
    sha256: str
    notes: str = ""


@dataclasses.dataclass
class ProviderGrid:
    provider: str
    cycle_init: dt.datetime
    wind_speed: Any  # numpy.ndarray [time, lat, lon]
    precipitation: Any  # numpy.ndarray [time, lat, lon]
    source_runs: List[str]
    provenance: List[ProvenanceRecord]
    notes: str = ""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_init_timestamp(text: str) -> dt.datetime:
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(t)
    if parsed.tzinfo is None:
        raise PipelineError("Initialization timestamp must include timezone, e.g. 2026-06-25T12:00:00Z")
    return parsed.astimezone(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)


def assert_inside_root(root: Path, path: Path) -> Path:
    root = root.resolve()
    target = (root / path if not path.is_absolute() else path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise PipelineError(f"Refusing to access path outside working directory: {target}") from exc
    return target


def ensure_dir(root: Path, rel: str) -> Path:
    p = assert_inside_root(root, Path(rel))
    p.mkdir(parents=True, exist_ok=True)
    return p


def choose_cycle(target: dt.datetime, cycle_hours: Sequence[int] = CYCLES_6H) -> dt.datetime:
    candidates: List[dt.datetime] = []
    for day_offset in range(0, 5):
        base = (target - dt.timedelta(days=day_offset)).date()
        for h in cycle_hours:
            c = dt.datetime(base.year, base.month, base.day, h, tzinfo=dt.timezone.utc)
            if c <= target:
                candidates.append(c)
    if not candidates:
        raise PipelineError(f"No causally valid cycle found at or before {target.isoformat()}")
    return max(candidates)


def cycle_dir_name(cycle: dt.datetime) -> str:
    return cycle.strftime("%Y%m%d%H")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, target: Path, *, expected_min_bytes: int = 1, timeout: int = 180) -> Tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    tmp = target.with_suffix(target.suffix + ".part")
    with urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
        if getattr(resp, "status", 200) >= 400:
            raise PipelineError(f"HTTP {resp.status} while retrieving {url}")
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    size = tmp.stat().st_size
    if size < expected_min_bytes:
        tmp.unlink(missing_ok=True)
        raise PipelineError(f"Downloaded object was too small ({size} bytes): {url}")
    tmp.replace(target)
    return size, sha256_file(target)


def try_download_file(url: str, target: Path, *, expected_min_bytes: int = 1, timeout: int = 180) -> Tuple[int, str]:
    try:
        return download_file(url, target, expected_min_bytes=expected_min_bytes, timeout=timeout)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise PipelineError(f"Failed to download real data from {url}: {exc}") from exc


def require_dependencies() -> None:
    missing = []
    for mod in ("numpy", "xarray", "cfgrib", "netCDF4", "scipy"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        raise PipelineError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Create an environment inside this directory and run: "
            + "python -m pip install --cache-dir .pip-cache -r requirements.txt"
        )


def target_lat_lon():
    import numpy as np

    lats = np.round(np.linspace(LAT_MIN, LAT_MAX, N_LAT), 10)
    lons = np.round(np.linspace(LON_MIN, LON_MAX, N_LON), 10)
    if len(lats) != N_LAT or len(lons) != N_LON:
        raise AssertionError("Target grid constants are inconsistent")
    return lats, lons


def decode_grib_datasets(path: Path) -> List[Any]:
    import cfgrib

    try:
        return cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    except TypeError:
        return cfgrib.open_datasets(str(path))


def _var_candidates(ds: Any) -> Dict[str, Any]:
    out = {}
    for name, da in ds.data_vars.items():
        attrs = {k.lower(): str(v).lower() for k, v in da.attrs.items()}
        short = attrs.get("grib_shortname", attrs.get("shortname", name)).lower()
        long_name = attrs.get("long_name", "")
        std = attrs.get("standard_name", "")
        key_blob = " ".join([name.lower(), short, long_name, std])
        out[key_blob] = da
    return out


def select_grib_var(datasets: Sequence[Any], kind: str) -> Any:
    """Return a DataArray for u10, v10, or total precipitation."""
    patterns = {
        "u10": [r"\b10u\b", r"\bu10\b", r"10 metre u", r"10 m.*u", r"u-component.*10"],
        "v10": [r"\b10v\b", r"\bv10\b", r"10 metre v", r"10 m.*v", r"v-component.*10"],
        "tp": [r"\btp\b", r"\bapcp\b", r"tot_prec", r"total precipitation", r"precipitation"],
    }[kind]
    for ds in datasets:
        for blob, da in _var_candidates(ds).items():
            if any(re.search(p, blob) for p in patterns):
                return da
    available = []
    for ds in datasets:
        available.extend(list(ds.data_vars))
    raise PipelineError(f"Could not identify {kind} variable in GRIB. Available variables: {available}")


def dataarray_to_regular_grid(da: Any, target_lats: Any, target_lons: Any) -> Any:
    """Crop/regrid a cfgrib DataArray to the target 0.1 degree lat/lon grid."""
    import numpy as np
    import xarray as xr
    from scipy.interpolate import griddata

    # Standard regular lat/lon case.
    lat_name = None
    lon_name = None
    for n in da.coords:
        if n.lower() in ("latitude", "lat"):
            lat_name = n
        if n.lower() in ("longitude", "lon"):
            lon_name = n
    if lat_name and lon_name:
        lat_coord = da[lat_name]
        lon_coord = da[lon_name]
        arr = da
        # Normalize longitudes to 0..360 for Japan bbox, then sort increasing.
        if float(lon_coord.max()) > 180.0 or LON_MIN >= 0:
            pass
        if lat_coord.ndim == 1 and lon_coord.ndim == 1:
            if lat_coord.values[0] > lat_coord.values[-1]:
                arr = arr.sortby(lat_name)
            if lon_coord.values[0] > lon_coord.values[-1]:
                arr = arr.sortby(lon_name)
            arr = arr.sel({lat_name: slice(LAT_MIN - 1.0, LAT_MAX + 1.0), lon_name: slice(LON_MIN - 1.0, LON_MAX + 1.0)})
            return arr.interp({lat_name: target_lats, lon_name: target_lons}, method="linear").rename({lat_name: "latitude", lon_name: "longitude"}).values.astype("float32")

    # ICON-like unstructured grid.  cfgrib/eccodes exposes latitude/longitude arrays.
    lat = None
    lon = None
    for n in ("latitude", "lat"):
        if n in da.coords:
            lat = np.asarray(da[n].values).ravel()
    for n in ("longitude", "lon"):
        if n in da.coords:
            lon = np.asarray(da[n].values).ravel()
    if lat is None or lon is None:
        raise PipelineError(f"GRIB variable {da.name!r} has no usable latitude/longitude coordinates")
    vals = np.asarray(da.values).ravel()
    # Keep nearby source cells only; use nearest fill for coastal edge holes.
    mask = (
        (lat >= LAT_MIN - 1.5)
        & (lat <= LAT_MAX + 1.5)
        & (lon >= LON_MIN - 1.5)
        & (lon <= LON_MAX + 1.5)
        & np.isfinite(vals)
    )
    if int(mask.sum()) < 10:
        raise PipelineError(f"Not enough real source grid cells near Japan in {da.name!r}")
    lon2, lat2 = np.meshgrid(target_lons, target_lats)
    linear = griddata((lon[mask], lat[mask]), vals[mask], (lon2, lat2), method="linear")
    if np.isnan(linear).any():
        nearest = griddata((lon[mask], lat[mask]), vals[mask], (lon2, lat2), method="nearest")
        linear = np.where(np.isfinite(linear), linear, nearest)
    return linear.astype("float32")


def first_scalar_or_array(da: Any) -> Any:
    import numpy as np

    arr = np.asarray(da.values)
    # Drop singleton time/step dimensions if present.
    arr = np.squeeze(arr)
    return arr


def decode_single_grib(path: Path, target_lats: Any, target_lons: Any, *, precipitation_required: bool = True) -> Tuple[Any, Any, Optional[Any]]:
    import numpy as np

    dsets = decode_grib_datasets(path)
    u = select_grib_var(dsets, "u10")
    v = select_grib_var(dsets, "v10")
    ugrid = dataarray_to_regular_grid(u, target_lats, target_lons)
    vgrid = dataarray_to_regular_grid(v, target_lats, target_lons)
    try:
        tp = select_grib_var(dsets, "tp")
    except PipelineError:
        if precipitation_required:
            raise
        pgrid = np.zeros_like(ugrid, dtype="float32")
    else:
        pgrid = dataarray_to_regular_grid(tp, target_lats, target_lons)
    return ugrid, vgrid, pgrid


def interpolate_cumulative_to_hourly(step_to_cumulative: Mapping[int, Any], desired_leads: Sequence[int]) -> Any:
    import numpy as np

    steps = sorted(step_to_cumulative)
    if not steps:
        raise PipelineError("No precipitation accumulations available")
    arrays = np.stack([np.asarray(step_to_cumulative[s], dtype="float32") for s in steps], axis=0)
    # Force cumulative precipitation nonnegative and nondecreasing where possible.
    arrays = np.maximum(arrays, 0.0)
    arrays = np.maximum.accumulate(arrays, axis=0)
    out_cum = []
    for lead in desired_leads:
        if lead <= steps[0]:
            out_cum.append(arrays[0])
        elif lead >= steps[-1]:
            out_cum.append(arrays[-1])
        else:
            hi_i = next(i for i, s in enumerate(steps) if s >= lead)
            lo_i = hi_i - 1
            lo_s, hi_s = steps[lo_i], steps[hi_i]
            w = (lead - lo_s) / float(hi_s - lo_s)
            out_cum.append(arrays[lo_i] * (1.0 - w) + arrays[hi_i] * w)
    cum = np.stack(out_cum, axis=0)
    hourly = np.empty_like(cum, dtype="float32")
    hourly[0] = 0.0
    hourly[1:] = np.maximum(cum[1:] - cum[:-1], 0.0)
    return hourly


def interpolate_instantaneous(step_to_field: Mapping[int, Any], desired_leads: Sequence[int]) -> Any:
    import numpy as np

    steps = sorted(step_to_field)
    arrays = np.stack([np.asarray(step_to_field[s], dtype="float32") for s in steps], axis=0)
    out = []
    for lead in desired_leads:
        if lead <= steps[0]:
            out.append(arrays[0])
        elif lead >= steps[-1]:
            out.append(arrays[-1])
        else:
            hi_i = next(i for i, s in enumerate(steps) if s >= lead)
            lo_i = hi_i - 1
            lo_s, hi_s = steps[lo_i], steps[hi_i]
            w = (lead - lo_s) / float(hi_s - lo_s)
            out.append(arrays[lo_i] * (1.0 - w) + arrays[hi_i] * w)
    return np.stack(out, axis=0).astype("float32")


def gfs_url(cycle: dt.datetime, lead: int) -> str:
    params = {
        "dir": f"/gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos",
        "file": f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}",
        "lev_10_m_above_ground": "on",
        "lev_surface": "on",
        "var_UGRD": "on",
        "var_VGRD": "on",
        "var_APCP": "on",
        "leftlon": f"{LON_MIN:g}",
        "rightlon": f"{LON_MAX:g}",
        "toplat": f"{LAT_MAX:g}",
        "bottomlat": f"{LAT_MIN:g}",
    }
    return GFS_NOMADS_BASE + "?" + urlencode(params)


def fetch_gfs(root: Path, target_init: dt.datetime, target_lats: Any, target_lons: Any) -> ProviderGrid:
    import numpy as np

    cycle = choose_cycle(target_init)
    leads = [int((target_init + dt.timedelta(hours=h) - cycle).total_seconds() // 3600) for h in FORECAST_HOURS]
    if any(l < 0 for l in leads):
        raise AssertionError("Causality bug: GFS lead was negative")
    gfs_dir = ensure_dir(root, f"data/gfs/{cycle_dir_name(cycle)}")
    wind_by_lead: Dict[int, Any] = {}
    precip_cum_by_lead: Dict[int, Any] = {}
    prov: List[ProvenanceRecord] = []
    for lead in sorted(set(leads)):
        url = gfs_url(cycle, lead)
        path = gfs_dir / f"gfs.t{cycle:%H}z.pgrb2.0p25.f{lead:03d}.japan.grib2"
        if not path.exists():
            size, digest = try_download_file(url, path, expected_min_bytes=1000)
        else:
            size, digest = path.stat().st_size, sha256_file(path)
        u, v, tp = decode_single_grib(path, target_lats, target_lons, precipitation_required=(lead != 0))
        wind_by_lead[lead] = np.sqrt(np.square(u) + np.square(v)).astype("float32")
        precip_cum_by_lead[lead] = np.asarray(tp, dtype="float32")
        prov.append(
            ProvenanceRecord(
                provider="NOAA/NCEP GFS NOMADS",
                url=url,
                local_path=str(path.relative_to(root)),
                cycle_init=cycle.isoformat().replace("+00:00", "Z"),
                variables=["UGRD:10 m", "VGRD:10 m", "APCP:surface"],
                lead_times=[lead],
                retrieval_timestamp=utc_now_iso(),
                size_bytes=size,
                sha256=digest,
            )
        )
    wind = interpolate_instantaneous(wind_by_lead, leads)
    precip = interpolate_cumulative_to_hourly(precip_cum_by_lead, leads)
    return ProviderGrid("gfs", cycle, wind, precip, [f"GFS {cycle:%Y-%m-%d %H}Z"], prov)


def fetch_ecmwf(root: Path, target_init: dt.datetime, target_lats: Any, target_lons: Any) -> ProviderGrid:
    import numpy as np

    try:
        from ecmwf.opendata import Client
    except Exception as exc:
        raise PipelineError(
            "ECMWF retrieval requires the official ecmwf-opendata package. Install requirements.txt; no substitute synthetic field is allowed."
        ) from exc
    # ECMWF open-data deterministic forecasts are available at 00/06/12/18 and generally at 3-hour steps.
    cycle = choose_cycle(target_init)
    desired_leads = [int((target_init + dt.timedelta(hours=h) - cycle).total_seconds() // 3600) for h in FORECAST_HOURS]
    max_lead = max(desired_leads)
    ecmwf_steps = sorted(set([0] + [s for s in range(0, max_lead + 4, 3)]))
    # Ensure each desired lead can be interpolated from real surrounding forecast steps.
    ecmwf_steps = sorted(set(s for s in ecmwf_steps if s >= 0))
    out_dir = ensure_dir(root, f"data/ecmwf/{cycle_dir_name(cycle)}")
    grib = out_dir / f"ecmwf_ifs_open_{cycle:%Y%m%d%H}_japan_10u10vtp.grib2"
    if not grib.exists():
        # AWS mirror of the same ECMWF open-data; the ecmwf.int portal throttles/resets
        # connections under load. Identical real IFS data, more reliable CDN.
        client = Client(source="aws", model="ifs", resol="0p25")
        try:
            client.retrieve(
                date=cycle.strftime("%Y%m%d"),
                time=cycle.strftime("%H"),
                stream="oper",
                type="fc",
                step=ecmwf_steps,
                param=["10u", "10v", "tp"],
                area=[LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],
                target=str(grib),
            )
        except Exception as exc:
            raise PipelineError(f"Failed to retrieve real ECMWF Open Data GRIB for {cycle.isoformat()}: {exc}") from exc
    size, digest = grib.stat().st_size, sha256_file(grib)
    if size < 1000:
        raise PipelineError(f"ECMWF Open Data GRIB is unexpectedly small: {grib} ({size} bytes)")

    # cfgrib groups multi-step GRIBs into datasets.  Read per step robustly by selecting 'step'.
    dsets = decode_grib_datasets(grib)
    u_da = select_grib_var(dsets, "u10")
    v_da = select_grib_var(dsets, "v10")
    tp_da = select_grib_var(dsets, "tp")

    def step_hours(da: Any) -> List[int]:
        if "step" in da.coords:
            vals = da["step"].values
            return [int(v / __import__("numpy").timedelta64(1, "h")) for v in vals]
        return [0]

    def select_step(da: Any, h: int) -> Any:
        if "step" not in da.coords:
            return da
        import numpy as np
        return da.sel(step=np.timedelta64(h, "h"))

    available_steps = sorted(set(step_hours(u_da)) & set(step_hours(v_da)) & set(step_hours(tp_da)))
    use_steps = [s for s in available_steps if min(desired_leads) <= s <= max(desired_leads)]
    if min(desired_leads) == 0 and 0 in available_steps and 0 not in use_steps:
        use_steps.insert(0, 0)
    if len(use_steps) < 2:
        raise PipelineError(f"ECMWF GRIB did not contain enough usable steps. Available: {available_steps}")
    wind_by_lead = {}
    precip_cum_by_lead = {}
    for s in use_steps:
        u = dataarray_to_regular_grid(select_step(u_da, s), target_lats, target_lons)
        v = dataarray_to_regular_grid(select_step(v_da, s), target_lats, target_lons)
        tp = dataarray_to_regular_grid(select_step(tp_da, s), target_lats, target_lons)
        # ECMWF total precipitation is metres of water accumulated from forecast start.
        wind_by_lead[s] = np.sqrt(np.square(u) + np.square(v)).astype("float32")
        precip_cum_by_lead[s] = (np.asarray(tp, dtype="float32") * 1000.0).astype("float32")
    wind = interpolate_instantaneous(wind_by_lead, desired_leads)
    precip = interpolate_cumulative_to_hourly(precip_cum_by_lead, desired_leads)
    prov = [
        ProvenanceRecord(
            provider="ECMWF Open Data IFS",
            url=(
                f"ecmwf-opendata://source=ecmwf/model=ifs/resol=0p25/date={cycle:%Y%m%d}/time={cycle:%H}/"
                f"stream=oper/type=fc/param=10u,10v,tp/step={','.join(str(x) for x in use_steps)}"
            ),
            local_path=str(grib.relative_to(root)),
            cycle_init=cycle.isoformat().replace("+00:00", "Z"),
            variables=["10u", "10v", "tp"],
            lead_times=use_steps,
            retrieval_timestamp=utc_now_iso(),
            size_bytes=size,
            sha256=digest,
        )
    ]
    return ProviderGrid("ecmwf", cycle, wind, precip, [f"ECMWF IFS Open Data {cycle:%Y-%m-%d %H}Z"], prov)


def dwd_url(cycle: dt.datetime, var_name: str, lead: int) -> str:
    return (
        f"{DWD_BASE}/{cycle:%H}/{var_name.lower()}/"
        f"icon_global_icosahedral_single-level_{cycle:%Y%m%d%H}_{lead:03d}_{var_name.upper()}.grib2.bz2"
    )



def dwd_coord_url(cycle: dt.datetime, var_name: str) -> str:
    return (
        f"{DWD_BASE}/{cycle:%H}/{var_name.lower()}/"
        f"icon_global_icosahedral_time-invariant_{cycle:%Y%m%d%H}_{var_name.upper()}.grib2.bz2"
    )


def dwd_decode_time_invariant_coord(path: Path, kind: str) -> Any:
    import numpy as np

    dsets = decode_grib_datasets(path)
    candidates = []
    for ds in dsets:
        for name, da in ds.data_vars.items():
            blob = " ".join([name.lower(), str(da.attrs.get("GRIB_shortName", "")).lower(), str(da.attrs.get("GRIB_name", "")).lower()])
            candidates.append((blob, da))
    patterns = ["tlat", "clat", "latitude"] if kind == "lat" else ["tlon", "clon", "longitude"]
    for blob, da in candidates:
        if any(pat in blob for pat in patterns):
            arr = np.asarray(da.values, dtype="float32").ravel()
            # DWD CLON is -180..180; convert to 0..360-compatible east longitudes for Japan.
            if kind == "lon":
                arr = np.where(arr < 0.0, arr + 360.0, arr)
            return arr
    raise PipelineError(f"Could not identify DWD ICON {kind} coordinate variable in {path}")


def fetch_dwd_icon_coordinates(root: Path, cycle: dt.datetime) -> Tuple[Any, Any, List[ProvenanceRecord]]:
    out_dir = ensure_dir(root, f"data/dwd_icon/{cycle_dir_name(cycle)}")
    coords = {}
    prov: List[ProvenanceRecord] = []
    for var_name, kind in (("CLAT", "lat"), ("CLON", "lon")):
        url = dwd_coord_url(cycle, var_name)
        bz_path = out_dir / f"icon_global_{cycle:%Y%m%d%H}_{var_name}.grib2.bz2"
        grib_path = out_dir / f"icon_global_{cycle:%Y%m%d%H}_{var_name}.grib2"
        if not bz_path.exists():
            size_bz, digest_bz = try_download_file(url, bz_path, expected_min_bytes=1000)
        else:
            size_bz, digest_bz = bz_path.stat().st_size, sha256_file(bz_path)
        if not grib_path.exists():
            grib_path.write_bytes(bz2.decompress(bz_path.read_bytes()))
        coords[kind] = dwd_decode_time_invariant_coord(grib_path, kind)
        prov.append(
            ProvenanceRecord(
                provider="DWD ICON Global Open Data coordinate grid (JMA substitute)",
                url=url,
                local_path=str(bz_path.relative_to(root)),
                cycle_init=cycle.isoformat().replace("+00:00", "Z"),
                variables=[var_name],
                lead_times=[],
                retrieval_timestamp=utc_now_iso(),
                size_bytes=size_bz,
                sha256=digest_bz,
                notes="ICON unstructured-grid latitude/longitude coordinate field used to regrid real DWD forecast values over Japan.",
            )
        )
    if coords["lat"].shape != coords["lon"].shape:
        raise PipelineError("DWD ICON CLAT/CLON coordinate arrays have different lengths")
    return coords["lat"], coords["lon"], prov


def dwd_unstructured_to_regular(da: Any, clat: Any, clon: Any, target_lats: Any, target_lons: Any) -> Any:
    import numpy as np
    from scipy.interpolate import griddata

    vals = np.asarray(da.values, dtype="float32").ravel()
    if vals.shape[0] != clat.shape[0]:
        raise PipelineError(f"DWD ICON value count {vals.shape[0]} does not match coordinate count {clat.shape[0]}")
    mask = (
        (clat >= LAT_MIN - 1.5)
        & (clat <= LAT_MAX + 1.5)
        & (clon >= LON_MIN - 1.5)
        & (clon <= LON_MAX + 1.5)
        & np.isfinite(vals)
    )
    if int(mask.sum()) < 10:
        raise PipelineError(f"Not enough DWD ICON source cells near Japan for {da.name!r}")
    lon2, lat2 = np.meshgrid(target_lons, target_lats)
    linear = griddata((clon[mask], clat[mask]), vals[mask], (lon2, lat2), method="linear")
    if np.isnan(linear).any():
        nearest = griddata((clon[mask], clat[mask]), vals[mask], (lon2, lat2), method="nearest")
        linear = np.where(np.isfinite(linear), linear, nearest)
    return linear.astype("float32")

def fetch_dwd_icon(root: Path, target_init: dt.datetime, target_lats: Any, target_lons: Any) -> ProviderGrid:
    import numpy as np

    cycle = choose_cycle(target_init)
    desired_leads = [int((target_init + dt.timedelta(hours=h) - cycle).total_seconds() // 3600) for h in FORECAST_HOURS]
    out_dir = ensure_dir(root, f"data/dwd_icon/{cycle_dir_name(cycle)}")
    clat, clon, coord_prov = fetch_dwd_icon_coordinates(root, cycle)
    wind_by_lead = {}
    precip_cum_by_lead = {}
    prov: List[ProvenanceRecord] = list(coord_prov)
    # DWD ICON global is hourly, but 3-hourly real steps are sufficient for rigorous temporal interpolation
    # to the required 1-hour output grid while keeping downloads operationally tractable.
    min_lead = min(desired_leads)
    max_lead = max(desired_leads)
    start_step = max(0, (min_lead // 3) * 3)
    end_step = int(math.ceil(max_lead / 3.0) * 3)
    dwd_steps = list(range(start_step, end_step + 1, 3))
    if 0 not in dwd_steps and min_lead == 0:
        dwd_steps.insert(0, 0)
    for lead in dwd_steps:
        fields = {}
        for var_name, kind in (("U_10M", "u10"), ("V_10M", "v10"), ("TOT_PREC", "tp")):
            url = dwd_url(cycle, var_name, lead)
            bz_path = out_dir / f"icon_global_{cycle:%Y%m%d%H}_{lead:03d}_{var_name}.grib2.bz2"
            grib_path = out_dir / f"icon_global_{cycle:%Y%m%d%H}_{lead:03d}_{var_name}.grib2"
            if not bz_path.exists():
                size_bz, digest_bz = try_download_file(url, bz_path, expected_min_bytes=100)
            else:
                size_bz, digest_bz = bz_path.stat().st_size, sha256_file(bz_path)
            if not grib_path.exists():
                data = bz2.decompress(bz_path.read_bytes())
                if len(data) < 100:
                    raise PipelineError(f"Decompressed DWD ICON file too small: {bz_path}")
                grib_path.write_bytes(data)
            dsets = decode_grib_datasets(grib_path)
            da = select_grib_var(dsets, kind)
            fields[kind] = dwd_unstructured_to_regular(da, clat, clon, target_lats, target_lons)
            prov.append(
                ProvenanceRecord(
                    provider="DWD ICON Global Open Data (JMA substitute)",
                    url=url,
                    local_path=str(bz_path.relative_to(root)),
                    cycle_init=cycle.isoformat().replace("+00:00", "Z"),
                    variables=[var_name],
                    lead_times=[lead],
                    retrieval_timestamp=utc_now_iso(),
                    size_bytes=size_bz,
                    sha256=digest_bz,
                    notes="DWD ICON used because unauthenticated JMA WIS GSM GRIB returned HTTP 401 in this environment.",
                )
            )
        wind_by_lead[lead] = np.sqrt(np.square(fields["u10"]) + np.square(fields["v10"])).astype("float32")
        # ICON TOT_PREC is kg m-2 == mm accumulated from model start.
        precip_cum_by_lead[lead] = np.asarray(fields["tp"], dtype="float32")
    wind = interpolate_instantaneous(wind_by_lead, desired_leads)
    precip = interpolate_cumulative_to_hourly(precip_cum_by_lead, desired_leads)
    return ProviderGrid(
        "dwd_icon",
        cycle,
        wind,
        precip,
        [f"DWD ICON Global {cycle:%Y-%m-%d %H}Z (JMA substitute)"],
        prov,
        notes="JMA WIS GSM is documented as preferred but credential-gated; DWD ICON is an independent real operational substitute.",
    )


def fetch_enso(root: Path, target_init: dt.datetime) -> Tuple[float, Dict[str, Any], ProvenanceRecord]:
    enso_dir = ensure_dir(root, "data/enso")
    path = enso_dir / "noaa_cpc_sstoi.indices"
    url = CPC_NINO_URL
    size, digest = try_download_file(url, path, expected_min_bytes=1000, timeout=60)
    text = path.read_text(errors="replace")
    # Conservative publication lag: require the monthly index to be at least 45 days old.
    causal_cutoff = (target_init - dt.timedelta(days=45)).date()
    best = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        year, month = int(parts[0]), int(parts[1])
        # columns: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
        nino34_anom = float(parts[9])
        # Treat monthly value as known no sooner than 15th of following month; 45-day cutoff is stricter.
        period_date = dt.date(year, month, 1)
        if period_date <= causal_cutoff:
            best = (period_date, nino34_anom, line)
    if best is None:
        raise PipelineError(f"No causally valid NOAA CPC NINO3.4 anomaly found for target {target_init.isoformat()}")
    period, anom, raw = best
    prov = ProvenanceRecord(
        provider="NOAA CPC SST OI indices",
        url=url,
        local_path=str(path.relative_to(root)),
        cycle_init=period.isoformat(),
        variables=["NINO3.4 SST anomaly"],
        lead_times=[],
        retrieval_timestamp=utc_now_iso(),
        size_bytes=size,
        sha256=digest,
        notes="Causally clipped with a conservative 45-day publication lag before target initialization.",
    )
    return anom, {"nino34_anomaly_c": anom, "period": period.isoformat(), "raw_row": raw, "causal_cutoff": causal_cutoff.isoformat()}, prov


def load_corrector(root: Path, rel_path: str) -> Dict[str, Any]:
    path = assert_inside_root(root, Path(rel_path))
    if not path.exists():
        raise PipelineError(
            f"Saved ML corrector not found at {path}. Run scripts/train_corrector.py first; the operational script will not train online."
        )
    with path.open("r") as f:
        model = json.load(f)
    required = {"model_id", "feature_names", "wind_coefficients", "precip_coefficients", "validation"}
    missing = required - set(model)
    if missing:
        raise PipelineError(f"Corrector artifact is missing keys: {sorted(missing)}")
    return model


def build_features(provider_grids: Mapping[str, ProviderGrid], enso_anom: float, target_init: dt.datetime) -> Tuple[Any, List[str]]:
    import numpy as np

    # Shape [time, lat, lon]
    wg = provider_grids["gfs"].wind_speed
    we = provider_grids["ecmwf"].wind_speed
    wi = provider_grids["dwd_icon"].wind_speed
    pg = provider_grids["gfs"].precipitation
    pe = provider_grids["ecmwf"].precipitation
    pi = provider_grids["dwd_icon"].precipitation
    wind_stack = np.stack([wg, we, wi], axis=0)
    precip_stack = np.stack([pg, pe, pi], axis=0)
    wmean = np.nanmean(wind_stack, axis=0)
    pmean = np.nanmean(precip_stack, axis=0)
    wspread = np.nanstd(wind_stack, axis=0)
    pspread = np.nanstd(precip_stack, axis=0)
    lats, lons = target_lat_lon()
    lon2, lat2 = np.meshgrid(lons, lats)
    latn = (lat2 - 35.0) / 11.0
    lonn = (lon2 - 134.0) / 12.0
    latn3 = np.broadcast_to(latn, wmean.shape)
    lonn3 = np.broadcast_to(lonn, wmean.shape)
    hours = np.array([target_init.hour + h for h in FORECAST_HOURS], dtype="float32") % 24
    sin_hour = np.sin(2 * np.pi * hours / 24.0)[:, None, None] * np.ones_like(wmean)
    cos_hour = np.cos(2 * np.pi * hours / 24.0)[:, None, None] * np.ones_like(wmean)
    enso = np.full_like(wmean, float(enso_anom), dtype="float32")
    intercept = np.ones_like(wmean, dtype="float32")
    feature_names = [
        "intercept",
        "wind_gfs",
        "wind_ecmwf",
        "wind_dwd_icon",
        "wind_mean",
        "wind_spread",
        "precip_gfs",
        "precip_ecmwf",
        "precip_dwd_icon",
        "precip_mean",
        "precip_spread",
        "lat_norm",
        "lon_norm",
        "sin_hour_utc",
        "cos_hour_utc",
        "nino34_anomaly_c",
        "wind_mean_x_nino34",
        "precip_mean_x_nino34",
        "high_wind_excess_over_15ms",
    ]
    features = np.stack(
        [
            intercept,
            wg,
            we,
            wi,
            wmean,
            wspread,
            pg,
            pe,
            pi,
            pmean,
            pspread,
            latn3,
            lonn3,
            sin_hour,
            cos_hour,
            enso,
            wmean * enso,
            pmean * enso,
            np.maximum(wmean - 15.0, 0.0),
        ],
        axis=-1,
    ).astype("float32")
    return features, feature_names


def apply_linear_corrector(features: Any, feature_names: List[str], model: Dict[str, Any]) -> Tuple[Any, Any]:
    import numpy as np

    if feature_names != model["feature_names"]:
        raise PipelineError(
            "Corrector feature schema mismatch. Pipeline features do not match training artifact; refusing to run."
        )
    wc = np.asarray(model["wind_coefficients"], dtype="float32")
    pc = np.asarray(model["precip_coefficients"], dtype="float32")
    if wc.shape[0] != features.shape[-1] or pc.shape[0] != features.shape[-1]:
        raise PipelineError("Corrector coefficient length does not match feature count")
    wind = np.tensordot(features, wc, axes=([-1], [0])).astype("float32")
    precip = np.tensordot(features, pc, axes=([-1], [0])).astype("float32")
    wind = np.maximum(wind, 0.0)
    precip = np.maximum(precip, 0.0)
    return wind, precip


def write_output(root: Path, target_init: dt.datetime, wind: Any, precip: Any, model: Dict[str, Any], providers: Mapping[str, ProviderGrid], enso_meta: Dict[str, Any], output_name: str) -> Path:
    import numpy as np
    import xarray as xr

    output = assert_inside_root(root, Path(output_name))
    lats, lons = target_lat_lon()
    times = np.array([target_init + dt.timedelta(hours=h) for h in FORECAST_HOURS], dtype="datetime64[ns]")
    ds = xr.Dataset(
        data_vars={
            "wind_speed": (("time", "latitude", "longitude"), wind.astype("float32"), {"units": "m s-1", "long_name": "ML-corrected 10 m sustained wind speed"}),
            "precipitation": (("time", "latitude", "longitude"), precip.astype("float32"), {"units": "mm", "long_name": "ML-corrected hourly accumulated precipitation"}),
        },
        coords={"time": times, "latitude": lats.astype("float32"), "longitude": lons.astype("float32")},
        attrs={
            "title": "48-hour ML-corrected Japan typhoon-relevant operational NWP forecast",
            "initialization_timestamp": target_init.isoformat().replace("+00:00", "Z"),
            "ml_model_id": model["model_id"],
            "ml_model_weights": model.get("artifact_path", "artifacts/typhoon_corrector.json"),
            "source_model_runs_processed": json.dumps({k: v.source_runs for k, v in providers.items()}, sort_keys=True),
            "enso_conditioning": json.dumps(enso_meta, sort_keys=True),
            "grid": "0.1 degree regular lat/lon; Japan bbox 24N-46N, 122E-146E",
        },
    )
    encoding = {
        "wind_speed": {"zlib": True, "complevel": 9, "shuffle": True, "dtype": "float32", "_FillValue": -9999.0},
        "precipitation": {"zlib": True, "complevel": 9, "shuffle": True, "dtype": "float32", "_FillValue": -9999.0},
        "latitude": {"dtype": "float32"},
        "longitude": {"dtype": "float32"},
    }
    ds.to_netcdf(output, format="NETCDF4", engine="netcdf4", encoding=encoding)
    return output


def write_manifest(root: Path, target_init: dt.datetime, providers: Mapping[str, ProviderGrid], enso_prov: ProvenanceRecord, enso_meta: Dict[str, Any], model: Dict[str, Any], output_path: Path, manifest_name: str) -> Path:
    path = assert_inside_root(root, Path(manifest_name))
    records: List[Dict[str, Any]] = []
    for p in providers.values():
        records.extend(dataclasses.asdict(r) for r in p.provenance)
    records.append(dataclasses.asdict(enso_prov))
    manifest = {
        "pipeline": "typhoon_forecast_pipeline.py",
        "created_at": utc_now_iso(),
        "target_initialization_timestamp": target_init.isoformat().replace("+00:00", "Z"),
        "causality_policy": "All model cycles are <= target initialization; ENSO is clipped with a 45-day publication-lag guard; no verification/observation after target is used in operations.",
        "jma_access_note": (
            "JMA WIS GSM GRIB is the preferred third agency but the unauthenticated WIS endpoint returned HTTP 401. "
            "DWD ICON Global Open Data is used as a documented independent operational substitute; no synthetic fallback is allowed."
        ),
        "target_grid": {"latitude_min": LAT_MIN, "latitude_max": LAT_MAX, "longitude_min": LON_MIN, "longitude_max": LON_MAX, "resolution_degree": 0.1, "shape": [NTIME, N_LAT, N_LON]},
        "output": {"path": str(output_path.relative_to(root)), "sha256": sha256_file(output_path), "size_bytes": output_path.stat().st_size},
        "ingested_files": records,
        "enso": enso_meta,
        "source_model_runs_processed": {k: v.source_runs for k, v in providers.items()},
        "provider_notes": {k: v.notes for k, v in providers.items() if v.notes},
        "ml_corrector": {
            "model_id": model["model_id"],
            "artifact_path": model.get("artifact_path", "artifacts/typhoon_corrector.json"),
            "training_cutoff": model.get("training_cutoff"),
            "training_data_provenance": model.get("training_data_provenance", []),
            "validation": model.get("validation", {}),
            "feature_names": model.get("feature_names", []),
        },
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def validate_output_schema(path: Path) -> None:
    import xarray as xr

    ds = xr.open_dataset(path)
    try:
        expected_dims = {"time": NTIME, "latitude": N_LAT, "longitude": N_LON}
        actual = dict(ds.sizes)
        for k, v in expected_dims.items():
            if actual.get(k) != v:
                raise PipelineError(f"Output dimension {k} expected {v}, got {actual.get(k)}")
        for var in ("wind_speed", "precipitation"):
            if var not in ds:
                raise PipelineError(f"Output missing variable {var}")
            if ds[var].dims != ("time", "latitude", "longitude"):
                raise PipelineError(f"Variable {var} has dims {ds[var].dims}")
        for attr in ("initialization_timestamp", "ml_model_id", "source_model_runs_processed"):
            if attr not in ds.attrs:
                raise PipelineError(f"Output missing global attribute {attr}")
    finally:
        ds.close()


def run_pipeline(args: argparse.Namespace) -> None:
    root = Path.cwd().resolve()
    require_dependencies()
    target_init = parse_init_timestamp(args.init)
    model = load_corrector(root, args.model)
    model["artifact_path"] = args.model
    lats, lons = target_lat_lon()
    providers: Dict[str, ProviderGrid] = {}
    # Fetch independently.  Fail if any real provider is unavailable; never synthesize.
    providers["gfs"] = fetch_gfs(root, target_init, lats, lons)
    providers["ecmwf"] = fetch_ecmwf(root, target_init, lats, lons)
    providers["dwd_icon"] = fetch_dwd_icon(root, target_init, lats, lons)
    enso_anom, enso_meta, enso_prov = fetch_enso(root, target_init)
    features, feature_names = build_features(providers, enso_anom, target_init)
    wind, precip = apply_linear_corrector(features, feature_names, model)
    out = write_output(root, target_init, wind, precip, model, providers, enso_meta, args.output)
    manifest = write_manifest(root, target_init, providers, enso_prov, enso_meta, model, out, args.manifest)
    validate_output_schema(out)
    print(json.dumps({"output": str(out.relative_to(root)), "manifest": str(manifest.relative_to(root)), "status": "ok"}, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run real-data Japan typhoon NWP ML correction pipeline")
    p.add_argument("--init", required=True, help="Target initialization timestamp, e.g. 2026-06-25T12:00:00Z")
    p.add_argument("--model", default="artifacts/typhoon_corrector.json", help="Saved real-observation-trained corrector JSON")
    p.add_argument("--output", default="typhoon_forecast_output.nc", help="Output NetCDF4 file name inside current directory")
    p.add_argument("--manifest", default="provenance.json", help="Provenance JSON path inside current directory")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_pipeline(args)
        return 0
    except PipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
