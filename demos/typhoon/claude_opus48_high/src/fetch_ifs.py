"""
ECMWF IFS open-data (0.25 deg) operational fetcher via the `ecmwf-opendata` client.

Source: ECMWF open-data (CC BY 4.0), replicated on AWS/Azure/GCP, no auth.
Native lead-time stepping is 3-hourly (handled downstream by temporal
interpolation onto the 1-hourly target axis).

* 10u/10v -> 10m wind speed [m/s, instantaneous]
* tp      -> total precipitation accumulated since init [m] -> mm (cumulative)
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import xarray as xr
from ecmwf.opendata import Client

import config as cfg
import gribio

PAD = 1.5
BLAT = (cfg.LAT_MIN - PAD, cfg.LAT_MAX + PAD)
BLON = (cfg.LON_MIN - PAD, cfg.LON_MAX + PAD)


def pick_cycle(target_init: dt.datetime, client: Client) -> dt.datetime:
    """Most recent IFS oper cycle that is both published and <= target_init."""
    latest = client.latest(type="fc", stream="oper", param="10u")
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=dt.timezone.utc)
    ch = (target_init.hour // 6) * 6
    cand = target_init.replace(hour=ch, minute=0, second=0, microsecond=0)
    init = min(cand, latest)
    return init


def _subset(ds: xr.Dataset) -> xr.Dataset:
    return ds.sel(latitude=slice(BLAT[1], BLAT[0]), longitude=slice(BLON[0], BLON[1]))


def fetch(target_init: dt.datetime, horizon: int = 48):
    # AWS mirror of the same ECMWF open-data; the ecmwf.int portal throttles/resets
    # connections under load. Identical real IFS data, more reliable CDN.
    client = Client(source="aws")
    init = pick_cycle(target_init, client)
    offset = int((target_init - init).total_seconds() // 3600)
    s0 = max(0, ((offset - 3) // 3) * 3)
    s1 = offset + horizon + 3
    steps = list(range(s0, s1 + 1, 3))

    dest = cfg.FORECAST_DIR / f"ifs_{init:%Y%m%d%H}.grib2"
    result = None
    if not (dest.exists() and dest.stat().st_size > 1024 and open(dest, "rb").read(4) == b"GRIB"):
        result = client.retrieve(type="fc", stream="oper",
                                 date=init.strftime("%Y-%m-%d"), time=init.hour,
                                 step=steps, param=["10u", "10v", "tp"], target=str(dest))

    ds = _subset(gribio.open_grib(dest))
    # ds has dims (step, lat, lon); coords time, step, valid_time
    spd = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
    tp_mm = ds["tp"] * 1000.0  # m -> mm, cumulative since init

    # rebuild along 'lead' with valid_time
    steps_td = ds["step"].values
    lead_hours = (steps_td / np.timedelta64(1, "h")).astype(int)
    valid = np.array([init + dt.timedelta(hours=int(h)) for h in lead_hours])

    def _relabel(da, name):
        da = da.rename({"step": "lead"})
        da = da.assign_coords(lead=("lead", lead_hours), valid_time=("lead", valid))
        keep = {"latitude", "longitude", "lead", "valid_time"}
        return da.drop_vars([c for c in da.coords if c not in keep], errors="ignore").rename(name)

    wind_cube = _relabel(spd, "wind")
    tp_cube = _relabel(tp_mm, "tp_cum")

    # canonical, resolvable ECMWF open-data URLs (one GRIB per step)
    base = f"https://data.ecmwf.int/forecasts/{init:%Y%m%d}/{init:%H}z/ifs/0p25/oper"
    step_urls = [f"{base}/{init:%Y%m%d%H}0000-{int(h)}h-oper-fc.grib2" for h in lead_hours]

    prov = [{
        "center": "ECMWF IFS open-data 0.25deg",
        "provider": "ECMWF open-data (CC BY 4.0; AWS/Azure/GCP mirrors)",
        "url": base + "/",
        "step_files": step_urls,
        "retrieval_client": "ecmwf-opendata (subset to 10u/10v/tp)",
        "init": init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lead_hours": [int(h) for h in lead_hours],
        "valid_time": [v.strftime("%Y-%m-%dT%H:%M:%SZ") for v in valid],
        "variables": ["10u", "10v", "tp"],
        "step_hours": "3-hourly (interpolated to 1-hourly target)",
        "bytes": int(dest.stat().st_size),
        "sha256": gribio.sha256_file(dest),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }]

    return {
        "center": "ifs",
        "center_label": "ECMWF IFS open-data 0.25deg",
        "init": init,
        "wind": wind_cube,
        "tp_cum": tp_cube,
        "provenance": prov,
    }


if __name__ == "__main__":
    out = fetch(cfg.get_target_init(), horizon=12)
    print("init", out["init"], "leads", out["wind"].lead.values)
    print("wind", out["wind"].sizes, "mean", float(out["wind"].mean()))
    print("tp_cum last mean", float(out["tp_cum"].isel(lead=-1).mean()))
