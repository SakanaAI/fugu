"""
DELIVERABLE operational pipeline.

Given a causal target initialization timestamp it:
  1. ingests real operational GRIB2 forecasts from three independent centers
     (NOAA GFS, ECMWF IFS, CMC GDPS) for runs initialized <= the target,
  2. ingests the real, causally-lagged CPC NINO3.4 anomaly,
  3. regrids every field onto the uniform 0.1 deg / 49-step target frame,
  4. LOADS the pre-trained XGBoost correctors (never trains) and applies the
     ML error-correction layer cell-by-cell,
  5. exports a single compressed NetCDF4 `typhoon_forecast_output.nc` plus a
     machine-readable `provenance.json`.

No plots are produced.  No field is synthesized.
Run:  python src/run_pipeline.py [--init 2026-06-25T12:00:00Z]
"""
from __future__ import annotations

import argparse
import json
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb

import config as cfg
import enso
import fetch_gfs
import fetch_ifs
import fetch_gdps
import regrid
from build_training import FEATURES


def build_feature_matrix(ens: xr.Dataset, nino34: float, tvt: np.ndarray) -> np.ndarray:
    """Assemble the per-cell feature matrix in the exact training FEATURES order."""
    nt, nlat, nlon = cfg.N_TIME, cfg.N_LAT, cfg.N_LON

    def member(var, center):
        return ens[var].sel(center=center).transpose("time", "latitude", "longitude").values

    wsp = {c: member("wind_members", c) for c in ("gfs", "ifs", "gdps")}
    pr = {c: member("precip_members", c) for c in ("gfs", "ifs", "gdps")}
    wsp_ens = ens["wind_ensmean"].transpose("time", "latitude", "longitude").values
    pr_ens = ens["precip_ensmean"].transpose("time", "latitude", "longitude").values
    wsp_stack = np.stack([wsp["gfs"], wsp["ifs"], wsp["gdps"]], axis=0)
    pr_stack = np.stack([pr["gfs"], pr["ifs"], pr["gdps"]], axis=0)
    wsp_spread = wsp_stack.std(axis=0)
    pr_spread = pr_stack.std(axis=0)

    lat2d = np.broadcast_to(cfg.TARGET_LAT[None, :, None], (nt, nlat, nlon))
    lon2d = np.broadcast_to(cfg.TARGET_LON[None, None, :], (nt, nlat, nlon))
    hours = np.array([pd.Timestamp(t).hour for t in tvt])
    doy = np.array([pd.Timestamp(t).dayofyear for t in tvt])
    hour3d = np.broadcast_to(hours[:, None, None], (nt, nlat, nlon))
    doy_sin3d = np.broadcast_to(np.sin(2 * np.pi * doy / 365.25)[:, None, None], (nt, nlat, nlon))
    doy_cos3d = np.broadcast_to(np.cos(2 * np.pi * doy / 365.25)[:, None, None], (nt, nlat, nlon))
    nino3d = np.full((nt, nlat, nlon), float(nino34), dtype=np.float32)

    cols = {
        "wsp_gfs": wsp["gfs"], "wsp_ifs": wsp["ifs"], "wsp_gem": wsp["gdps"],
        "pr_gfs": pr["gfs"], "pr_ifs": pr["ifs"], "pr_gem": pr["gdps"],
        "wsp_ens": wsp_ens, "pr_ens": pr_ens,
        "wsp_spread": wsp_spread, "pr_spread": pr_spread,
        "lat": lat2d, "lon": lon2d, "hour": hour3d,
        "doy_sin": doy_sin3d, "doy_cos": doy_cos3d, "nino34": nino3d,
    }
    X = np.stack([np.asarray(cols[f], dtype=np.float32).reshape(-1) for f in FEATURES], axis=1)
    return X, wsp_ens, pr_ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=cfg.DEFAULT_TARGET_INIT)
    ap.add_argument("--horizon", type=int, default=48)
    args = ap.parse_args()

    target_init = cfg.parse_init(args.init)
    print(f"target init (causal fence): {target_init.isoformat()}")
    tvt = regrid.target_valid_times(target_init)

    # --- ENSO conditioning (causally fenced) ---
    enso_res = enso.get_nino34(target_init)
    print(f"NINO3.4 anomaly {enso_res.nino34_anom} (month {enso_res.valid_month})")

    # --- ingest the three centers ---
    print("ingesting NOAA GFS ...")
    gfs = fetch_gfs.fetch(target_init, args.horizon)
    print("ingesting ECMWF IFS ...")
    ifs = fetch_ifs.fetch(target_init, args.horizon)
    print("ingesting CMC GDPS ...")
    gdps = fetch_gdps.fetch(target_init, args.horizon)

    raw_outs = [gfs, ifs, gdps]
    processed = [regrid.process_center(o, target_init) for o in raw_outs]
    ens = regrid.build_ensemble(processed)
    print("ensemble grid:", dict(ens.sizes))

    # --- ML correction (load only; never train) ---
    wind_model = xgb.XGBRegressor()
    wind_model.load_model(cfg.MODELS_DIR / "wind_corrector.json")
    precip_model = xgb.XGBRegressor()
    precip_model.load_model(cfg.MODELS_DIR / "precip_corrector.json")
    model_metrics = json.loads((cfg.MODELS_DIR / "model_metrics.json").read_text())

    X, wsp_ens, pr_ens = build_feature_matrix(ens, enso_res.nino34_anom, tvt)
    shape = (cfg.N_TIME, cfg.N_LAT, cfg.N_LON)

    wind_resid = wind_model.predict(X).reshape(shape)
    wind_corr = np.clip(wsp_ens + wind_resid, 0, None).astype(np.float32)
    precip_corr = np.clip(np.expm1(precip_model.predict(X)).reshape(shape), 0, None).astype(np.float32)
    # T+0 precipitation is identically zero (no accumulation window)
    precip_corr[0, :, :] = 0.0

    # --- assemble output dataset (rigid schema) ---
    out = xr.Dataset(
        {
            "wind_speed": (("time", "latitude", "longitude"), wind_corr),
            "precipitation": (("time", "latitude", "longitude"), precip_corr),
        },
        coords={
            "time": cfg.LEAD_HOURS.astype("int32"),
            "latitude": cfg.TARGET_LAT.astype("float32"),
            "longitude": cfg.TARGET_LON.astype("float32"),
        },
    )
    out = out.assign_coords(valid_time=("time", tvt.astype("datetime64[ns]")))
    out["valid_time"].attrs.update(long_name="absolute valid time (UTC)")
    out["time"].attrs.update(units="hours", long_name="forecast lead time (hours since initialization_timestamp)",
                             standard_name="forecast_period")
    out["latitude"].attrs.update(units="degrees_north", standard_name="latitude")
    out["longitude"].attrs.update(units="degrees_east", standard_name="longitude")
    out["wind_speed"].attrs.update(units="m s-1", long_name="10m sustained wind speed (ML-corrected)")
    out["precipitation"].attrs.update(units="mm", long_name="hourly accumulated precipitation (ML-corrected)")

    source_runs = []
    for o in raw_outs:
        source_runs.append(f"{o['center_label']} init {o['init'].strftime('%Y-%m-%dT%H:%M:%SZ')}")
    out.attrs.update(
        title="ML-ensembled 48h typhoon wind & precipitation forecast over Japan",
        initialization_timestamp=target_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        causal_fence="no model run, observation or ENSO value dated after the initialization_timestamp was used",
        ml_model_weights="wind_corrector.json, precip_corrector.json (XGBoost)",
        ml_label_source="ERA5 reanalysis (Open-Meteo archive)",
        source_model_runs="; ".join(source_runs),
        nino34_anomaly=float(enso_res.nino34_anom),
        nino34_valid_month=enso_res.valid_month,
        grid="0.1deg uniform, 24-46N x 122-146E",
        oos_wind_rmse_corrected=model_metrics["wind"]["ml_corrected"]["rmse"],
        oos_wind_rmse_baseline=model_metrics["wind"]["baseline_ensmean"]["rmse"],
        oos_precip_rmse_corrected=model_metrics["precip"]["ml_corrected"]["rmse"],
        oos_precip_rmse_baseline=model_metrics["precip"]["baseline_ensmean"]["rmse"],
        created_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        Conventions="CF-1.8 (subset)",
    )

    enc = {v: {"zlib": True, "complevel": 6, "dtype": "float32"} for v in ("wind_speed", "precipitation")}
    out_path = cfg.OUTPUT_DIR / "typhoon_forecast_output.nc"
    out.to_netcdf(out_path, format="NETCDF4", encoding=enc)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")

    # --- provenance manifest ---
    provenance = {
        "initialization_timestamp": target_init.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_grid": {"lat": [cfg.LAT_MIN, cfg.LAT_MAX], "lon": [cfg.LON_MIN, cfg.LON_MAX],
                        "resolution_deg": cfg.DLAT, "n_time": cfg.N_TIME,
                        "n_lat": cfg.N_LAT, "n_lon": cfg.N_LON},
        "enso": {
            "nino34_anomaly": enso_res.nino34_anom, "valid_month": enso_res.valid_month,
            "source": enso_res.source, "url": enso_res.url,
            "sha256": enso_res.sha256, "retrieved_utc": enso_res.retrieved_utc,
            "causal_note": "monthly value strictly precedes init month (respects publication lag)",
        },
        "centers": {
            o["center"]: {
                "label": o["center_label"], "init": o["init"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "n_files": len(o["provenance"]), "files": o["provenance"],
            } for o in raw_outs
        },
        "jma_substitution_note": (
            "JMA GSM/MSM GRIB is not available from a free, no-auth archive on this "
            "machine; per the task's allowance we substitute CMC GDPS, a fully "
            "independent operational center on a regular lat-lon grid."),
        "ml_model": {
            "wind_weights": "models/wind_corrector.json",
            "precip_weights": "models/precip_corrector.json",
            "features": FEATURES,
            "out_of_sample_verification": {"wind": model_metrics["wind"],
                                           "precip": model_metrics["precip"],
                                           "oos_split": model_metrics["oos_split"],
                                           "n_test": model_metrics["n_test"]},
            "label_source": model_metrics["label_source"],
            "feature_source": model_metrics["feature_source"],
            "train_window": "2024-06-01 .. 2026-06-20 (<= 2026-06-25 op-init fence)",
        },
        "output_file": "output/typhoon_forecast_output.nc",
    }
    prov_path = cfg.OUTPUT_DIR / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2))
    print(f"wrote {prov_path}")

    print("\n=== summary ===")
    print(f"wind_speed  mean {float(np.nanmean(wind_corr)):.2f} m/s  max {float(np.nanmax(wind_corr)):.2f}")
    print(f"precip      mean {float(np.nanmean(precip_corr)):.3f} mm/h  max {float(np.nanmax(precip_corr)):.2f}")


if __name__ == "__main__":
    main()
