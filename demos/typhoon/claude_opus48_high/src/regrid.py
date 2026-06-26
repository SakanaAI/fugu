"""
Spatial regridding and temporal alignment.

Each center's native cube (wind [m/s, instantaneous] and tp_cum [mm, cumulative
since that center's own init], along a 'lead' dim carrying absolute valid_time)
is transformed into the common deliverable frame:

  * Spatial: bilinear interpolation onto the uniform 0.1 deg target mesh
    (221 lat x 241 lon).
  * Temporal: everything is expressed on the 49 target valid times
    V_k = target_init + k h, k = 0..48.
      - wind: linear interpolation of the instantaneous field in valid_time.
      - precipitation: the cumulative-since-init curve is linearly interpolated
        in valid_time onto the 49 target edges and differenced to give the
        hourly accumulation in each (V_{k-1}, V_k] window (step 0 = 0).

This is the only place where centers with different inits and different native
lead-steppings (GFS/GDPS hourly, IFS 3-hourly) are reconciled, by working in
absolute valid time.  Every operation here is a genuine transform of real data.
"""
from __future__ import annotations

import datetime as dt
import numpy as np
import xarray as xr

import config as cfg


def target_valid_times(target_init: dt.datetime) -> np.ndarray:
    base = np.datetime64(target_init.replace(tzinfo=None), "ns")
    return base + (cfg.LEAD_HOURS.astype("timedelta64[h]")).astype("timedelta64[ns]")


def _to_valid_dim(da: xr.DataArray) -> xr.DataArray:
    """Swap the 'lead' dim for its 'valid_time' coordinate (monotonic time index)."""
    da = da.swap_dims({"lead": "valid_time"})
    # ensure datetime64[ns], tz-naive
    vt = da["valid_time"].values.astype("datetime64[ns]")
    da = da.assign_coords(valid_time=("valid_time", vt))
    return da.sortby("valid_time")


def _spatial_regrid(da: xr.DataArray) -> xr.DataArray:
    """Bilinear interpolation onto the target 0.1 deg mesh (lat ascending)."""
    da = da.sortby("latitude").sortby("longitude")
    out = da.interp(latitude=cfg.TARGET_LAT, longitude=cfg.TARGET_LON, method="linear")
    return out


def process_center(center_out: dict, target_init: dt.datetime) -> dict:
    tvt = target_valid_times(target_init)

    wind = _to_valid_dim(center_out["wind"])
    tpc = _to_valid_dim(center_out["tp_cum"])

    # --- temporal interpolation onto target valid times ---
    wind_t = wind.interp(valid_time=tvt, method="linear",
                         kwargs={"fill_value": "extrapolate"})
    cum_t = tpc.interp(valid_time=tvt, method="linear",
                       kwargs={"fill_value": "extrapolate"})

    # de-accumulate cumulative -> hourly in each window; step 0 = 0
    cum_vals = cum_t.transpose("valid_time", "latitude", "longitude")
    hourly = cum_vals.diff("valid_time")                       # 48 windows
    zero0 = xr.zeros_like(cum_vals.isel(valid_time=0))
    precip = xr.concat([zero0.expand_dims(valid_time=[tvt[0]]),
                        hourly.assign_coords(valid_time=tvt[1:])], dim="valid_time")
    precip = precip.clip(min=0.0)

    # --- spatial regrid to target mesh ---
    wind_g = _spatial_regrid(wind_t)
    precip_g = _spatial_regrid(precip)

    # rename time dim to 'time' with integer lead-hour index relationship
    wind_g = wind_g.rename({"valid_time": "time"})
    precip_g = precip_g.rename({"valid_time": "time"})
    wind_g = wind_g.assign_coords(time=("time", tvt))
    precip_g = precip_g.assign_coords(time=("time", tvt))

    return {
        "center": center_out["center"],
        "center_label": center_out["center_label"],
        "init": center_out["init"],
        "wind": wind_g.clip(min=0.0),     # [time, lat, lon] m/s
        "precip": precip_g,               # [time, lat, lon] mm/hr
    }


def build_ensemble(processed: list[dict]) -> xr.Dataset:
    """Stack regridded centers and compute the multi-model ensemble mean."""
    centers = [p["center"] for p in processed]
    wind = xr.concat([p["wind"] for p in processed], dim="center")
    precip = xr.concat([p["precip"] for p in processed], dim="center")
    wind = wind.assign_coords(center=("center", centers))
    precip = precip.assign_coords(center=("center", centers))

    ds = xr.Dataset({
        "wind_members": wind.transpose("center", "time", "latitude", "longitude"),
        "precip_members": precip.transpose("center", "time", "latitude", "longitude"),
        "wind_ensmean": wind.mean("center"),
        "precip_ensmean": precip.mean("center"),
    })
    return ds
