"""
CMC GDPS (Global Deterministic Prediction System, 0.15 deg) operational fetcher.

Source: Meteorological Service of Canada (MSC) Open Data / Datamart (no auth):
    https://dd.weather.gc.ca/today/model_gdps/15km/<HH>/<hhh>/
    <YYYYMMDD>T<HH>Z_MSC_GDPS_<VAR>_LatLon0.15_PT<hhh>H.grib2

GDPS is used as the documented independent substitute for JMA (whose GSM/MSM
GRIB is not available from a free, no-auth archive on this machine).

* WindSpeed_AGL-10m : 10m sustained wind speed [m/s], instantaneous (native).
* Precip-Accum_Sfc  : total precipitation accumulated since init [mm] (cumulative).

Files are single-variable single-level, so each is downloaded whole and decoded
with cfgrib. We fetch only the lead range that brackets the target valid window.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import xarray as xr

import config as cfg
import gribio

GDPS_BASE = "https://dd.weather.gc.ca/today/model_gdps/15km"
PAD = 1.5
BLAT = (cfg.LAT_MIN - PAD, cfg.LAT_MAX + PAD)
BLON = (cfg.LON_MIN - PAD, cfg.LON_MAX + PAD)


def _url(day: dt.date, cyc: int, lead: int, var: str) -> str:
    return (f"{GDPS_BASE}/{cyc:02d}/{lead:03d}/"
            f"{day:%Y%m%d}T{cyc:02d}Z_MSC_GDPS_{var}_LatLon0.15_PT{lead:03d}H.grib2")


def pick_cycle(target_init: dt.datetime, need_lead: int) -> tuple[dt.date, int]:
    """Latest GDPS cycle (00/12) <= target_init whose far-lead wind file exists."""
    # candidate cycle hours 12, 0 of target day, then previous day 12, 0 ...
    cands = []
    base = target_init.replace(minute=0, second=0, microsecond=0)
    for back in range(0, 3):
        d = (base - dt.timedelta(days=back)).date()
        for ch in (12, 0):
            c = dt.datetime(d.year, d.month, d.day, ch, tzinfo=dt.timezone.utc)
            if c <= target_init:
                cands.append(c)
    cands.sort(reverse=True)
    for c in cands:
        url = _url(c.date(), c.hour, need_lead, "WindSpeed_AGL-10m")
        if gribio.head_ok(url):
            return c.date(), c.hour
    raise RuntimeError("no available GDPS cycle found <= target")


def _subset(ds: xr.Dataset) -> xr.Dataset:
    lat = ds.latitude
    # GDPS latitude may be ascending; build slice in the right direction
    if float(lat[0]) < float(lat[-1]):
        sl_lat = slice(BLAT[0], BLAT[1])
    else:
        sl_lat = slice(BLAT[1], BLAT[0])
    return ds.sel(latitude=sl_lat, longitude=slice(BLON[0], BLON[1]))


def _clean(da: xr.DataArray) -> xr.DataArray:
    keep = {"latitude", "longitude"}
    return da.drop_vars([c for c in da.coords if c not in keep], errors="ignore")


def fetch(target_init: dt.datetime, horizon: int = 48):
    far = 60  # provisional; refined after cycle pick
    # find a cycle; need lead up to offset+horizon, offset<=12 so 60+ is safe up front
    day, cyc = pick_cycle(target_init, far)
    init = dt.datetime(day.year, day.month, day.day, cyc, tzinfo=dt.timezone.utc)
    offset = int((target_init - init).total_seconds() // 3600)
    leads = list(range(offset, offset + horizon + 1))  # cover V_0..V_horizon exactly

    prov, winds, tpc = [], {}, {}

    def _one(lead: int):
        recs = {}
        for var, dest_tag in (("WindSpeed_AGL-10m", "ws"), ("Precip-Accum_Sfc", "pa")):
            url = _url(day, cyc, lead, var)
            dest = cfg.FORECAST_DIR / f"gdps_{init:%Y%m%d%H}_f{lead:03d}_{dest_tag}.grib2"
            try:
                nb = gribio.ensure_file(url, dest)
            except Exception:  # noqa
                nb = None
                dest = None
            recs[var] = (url, dest, nb)
        return lead, recs

    results = gribio.thread_map(_one, leads, max_workers=4)

    for lead, recs in sorted(results):
        valid = init + dt.timedelta(hours=lead)
        _, wdest, _ = recs["WindSpeed_AGL-10m"]
        _, pdest, _ = recs["Precip-Accum_Sfc"]
        if wdest is None or pdest is None:
            continue  # skip a missing lead; temporal interpolation fills the gap
        wds = gribio.open_grib(wdest)
        wv = "si10" if "si10" in wds.data_vars else list(wds.data_vars)[0]
        winds[lead] = _subset(_clean(wds[wv]).to_dataset(name="wind"))["wind"]
        pds = gribio.open_grib(pdest)
        pv = list(pds.data_vars)[0]
        tpc[lead] = _subset(_clean(pds[pv]).to_dataset(name="tp"))["tp"]
        for var, (url, dest, nb) in recs.items():
            prov.append({
                "center": "CMC GDPS 0.15deg",
                "provider": "MSC Open Data / Datamart (dd.weather.gc.ca)",
                "url": url,
                "init": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "lead_hours": lead,
                "valid_time": valid.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "variables": [var],
                "bytes": int(nb) if nb else None,
                "sha256": gribio.sha256_file(dest) if dest else None,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            })

    lead_arr = np.array(sorted(winds))
    valid = np.array([init + dt.timedelta(hours=int(l)) for l in lead_arr])
    wind_cube = xr.concat([winds[int(l)] for l in lead_arr], dim="lead").assign_coords(
        lead=("lead", lead_arr), valid_time=("lead", valid)).rename("wind")
    tp_cube = xr.concat([tpc[int(l)] for l in lead_arr], dim="lead").assign_coords(
        lead=("lead", lead_arr), valid_time=("lead", valid)).rename("tp_cum")

    return {
        "center": "gdps",
        "center_label": "CMC GDPS 0.15deg (substitute for JMA)",
        "init": init,
        "wind": wind_cube,
        "tp_cum": tp_cube,
        "provenance": prov,
    }


if __name__ == "__main__":
    out = fetch(cfg.get_target_init(), horizon=6)
    print("init", out["init"], "leads", out["wind"].lead.values)
    print("wind", out["wind"].sizes, "mean", float(out["wind"].mean()))
    print("tp_cum first/last mean", float(out["tp_cum"].isel(lead=0).mean()),
          float(out["tp_cum"].isel(lead=-1).mean()))
    print("lat", float(out["wind"].latitude.min()), float(out["wind"].latitude.max()))
