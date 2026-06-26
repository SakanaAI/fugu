"""
NOAA GFS (0.25 deg) operational fetcher.

Source: NOAA Open Data on AWS S3 (no auth):
    s3://noaa-gfs-bdp-pds/gfs.<YYYYMMDD>/<HH>/atmos/gfs.t<HH>z.pgrb2.0p25.f<FFF>
We use the sidecar .idx inventory for byte-range subsetting of just the
10m wind components and surface APCP.

* 10m wind speed = sqrt(u10^2 + v10^2)  [m/s, instantaneous]
* precip: GFS APCP resets every 6 h; we de-accumulate to hourly then form a
  cumulative-since-init series (mm) for uniform temporal handling downstream.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import xarray as xr

import config as cfg
import gribio

GFS_BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# padded native subset box for clean interpolation onto the target grid
PAD = 1.5
BLAT = (cfg.LAT_MIN - PAD, cfg.LAT_MAX + PAD)
BLON = (cfg.LON_MIN - PAD, cfg.LON_MAX + PAD)


def _cycle_url(day: dt.date, cyc: int, lead: int) -> str:
    return (f"{GFS_BUCKET}/gfs.{day:%Y%m%d}/{cyc:02d}/atmos/"
            f"gfs.t{cyc:02d}z.pgrb2.0p25.f{lead:03d}")


def pick_cycle(target_init: dt.datetime, need_lead: int) -> tuple[dt.date, int]:
    """Latest GFS cycle initialized <= target_init whose f{need_lead} exists."""
    # candidate cycles every 6h going back up to 2 days
    t = target_init.replace(minute=0, second=0, microsecond=0)
    # round target down to a 6-hourly cycle
    cyc_hour = (target_init.hour // 6) * 6
    cand = target_init.replace(hour=cyc_hour, minute=0, second=0, microsecond=0)
    for _ in range(9):  # search back up to ~2 days
        if cand <= target_init:
            url = _cycle_url(cand.date(), cand.hour, need_lead) + ".idx"
            if gribio.head_ok(url):
                return cand.date(), cand.hour
        cand -= dt.timedelta(hours=6)
    raise RuntimeError("no available GFS cycle found <= target")


def _subset(ds: xr.Dataset) -> xr.Dataset:
    # GFS latitude descending (90..-90), longitude ascending 0..359.75
    return ds.sel(latitude=slice(BLAT[1], BLAT[0]), longitude=slice(BLON[0], BLON[1]))


def _clean(da: xr.DataArray) -> xr.DataArray:
    """Drop cfgrib scalar coords (step/time/heightAboveGround/...) that vary per file."""
    keep = {"latitude", "longitude"}
    drop = [c for c in da.coords if c not in keep]
    return da.drop_vars(drop, errors="ignore")


def fetch(target_init: dt.datetime, horizon: int = 48, max_lead_override: int | None = None):
    # leads must cover [target_init, target_init+horizon]; pick a cycle that can
    # reach the far edge (a cycle <= target is at most 6 h before it).
    provis_far = 6 + horizon + 1
    day, cyc = pick_cycle(target_init, provis_far)
    init = dt.datetime(day.year, day.month, day.day, cyc, tzinfo=dt.timezone.utc)

    offset = int((target_init - init).total_seconds() // 3600)
    L1 = offset + horizon
    if max_lead_override is not None:
        L1 = min(L1, max_lead_override)
    leads = list(range(0, L1 + 1))

    prov = []
    winds = {}      # lead -> wind speed DataArray (lat,lon)
    apcp = {}       # lead -> APCP field (mm) as archived (bucket accumulation)

    def _one(lead: int):
        base = _cycle_url(day, cyc, lead)
        idx = gribio.fetch_idx(base + ".idx")
        pats = [r":UGRD:10 m above ground:", r":VGRD:10 m above ground:"]
        if lead >= 1:
            pats.append(r":APCP:surface:")
        sels = gribio.select_ranges(idx, pats)
        dest = cfg.FORECAST_DIR / f"gfs_{init:%Y%m%d%H}_f{lead:03d}.grib2"
        if dest.exists() and dest.stat().st_size > 256 and open(dest, "rb").read(4) == b"GRIB":
            return lead, dest, dest.stat().st_size, base
        nbytes = gribio.download_ranges(base, sels, dest)
        return lead, dest, nbytes, base

    results = gribio.thread_map(_one, leads, max_workers=8)

    for lead, dest, nbytes, base in sorted(results):
        ds = _subset(gribio.open_grib(dest))
        spd = _clean(np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2))
        winds[lead] = spd
        if lead >= 1 and "tp" in ds:
            apcp[lead] = _clean(ds["tp"])  # kg m-2 == mm, bucket accumulation
        prov.append({
            "center": "NOAA GFS 0.25deg",
            "provider": "NOAA Open Data / AWS S3 (noaa-gfs-bdp-pds)",
            "url": base,
            "idx_url": base + ".idx",
            "init": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lead_hours": lead,
            "valid_time": (init + dt.timedelta(hours=lead)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "variables": ["UGRD/VGRD 10m", "APCP surface" if lead >= 1 else "(no precip at f000)"],
            "bytes": int(nbytes),
            "sha256": gribio.sha256_file(dest),
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    # assemble wind cube
    lead_arr = np.array(sorted(winds))
    lat = winds[lead_arr[0]].latitude.values
    lon = winds[lead_arr[0]].longitude.values
    wind_cube = xr.concat([winds[l] for l in lead_arr], dim="lead")
    wind_cube = wind_cube.assign_coords(lead=("lead", lead_arr)).rename("wind")
    valid = np.array([init + dt.timedelta(hours=int(l)) for l in lead_arr])
    wind_cube = wind_cube.assign_coords(valid_time=("lead", valid))

    # de-bucket APCP -> hourly -> cumulative since init
    hourly = {0: xr.zeros_like(winds[lead_arr[0]])}
    for L in lead_arr:
        if L == 0:
            continue
        bs = 6 * ((L - 1) // 6)
        if (L - 1) > bs and (L - 1) in apcp:
            h = apcp[L] - apcp[L - 1]
        else:
            h = apcp[L]
        hourly[int(L)] = h.clip(min=0)
    cum = {}
    run = xr.zeros_like(winds[lead_arr[0]])
    for L in lead_arr:
        run = run + hourly[int(L)]
        cum[int(L)] = run.copy()
    tp_cube = xr.concat([cum[int(l)] for l in lead_arr], dim="lead")
    tp_cube = tp_cube.assign_coords(lead=("lead", lead_arr), valid_time=("lead", valid)).rename("tp_cum")

    return {
        "center": "gfs",
        "center_label": "NOAA GFS 0.25deg",
        "init": init,
        "wind": wind_cube,
        "tp_cum": tp_cube,
        "provenance": prov,
    }


if __name__ == "__main__":
    out = fetch(cfg.get_target_init(), horizon=48, max_lead_override=9)
    print("init", out["init"], "leads", out["wind"].lead.values)
    print("wind dims", out["wind"].sizes, "mean", float(out["wind"].mean()))
    print("tp_cum last mean", float(out["tp_cum"].isel(lead=-1).mean()))
