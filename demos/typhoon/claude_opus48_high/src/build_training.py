"""
Build a REAL training dataset for the ML error-correction layer.

Inputs (features)  -- real operational model *historical forecasts* of the SAME
three centers used operationally, retrieved from the Open-Meteo historical
forecast archive (free, no-auth):
    gfs_seamless    <-> NOAA GFS
    ecmwf_ifs025    <-> ECMWF IFS
    gem_seamless    <-> CMC GEM/GDPS
(Past-cycle native GRIB for these centers is not retained on the free servers,
so the archive of their issued forecasts is the real, traceable source.)

Labels (truth) -- real ERA5 reanalysis of 10m wind and precipitation from the
Open-Meteo ERA5 archive API.  Labels are NEVER self-generated.

Plus NINO3.4 climate-state conditioning (CPC, causally lagged per month).

Temporal fence: window ends 2026-06-20 (<= 2026-06-25 operational init), so the
table now includes the spring + early-summer 2026 typhoon season (Tropical Storm
Jangmi 2026-05-26..06-03 and the early spin-up of Typhoon Mekkhala from 06-18).
This predates the operational late-June 2026 inits, so there is no leakage.
ERA5 labels via Open-Meteo currently extend only to ~2026-06-20; rows with
missing/null labels are dropped in add_features (never fabricated).  Verification
is done out-of-sample by holding out a typhoon-free slice of dates (see train.py).
"""
from __future__ import annotations

import json
import time
import datetime as dt
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config as cfg
import enso

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"

MODELS = ["gfs_seamless", "ecmwf_ifs025", "gem_seamless"]
MODEL_SHORT = {"gfs_seamless": "gfs", "ecmwf_ifs025": "ifs", "gem_seamless": "gem"}

START = "2024-06-01"
END = "2026-06-20"     # <= 2026-06-25 operational-init fence; ERA5 labels end ~2026-06-20

# Sample points across the Japan box (mainland + immediate marine approaches).
def sample_points() -> list[tuple[float, float]]:
    lats = np.arange(26.0, 45.0 + 0.1, 3.0)
    lons = np.arange(126.0, 144.0 + 0.1, 3.0)
    pts = [(round(float(la), 2), round(float(lo), 2)) for la in lats for lo in lons]
    return pts


def _get(url: str, params: dict, max_tries: int = 8) -> dict:
    for k in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=90)
            j = r.json()
        except Exception:
            time.sleep(10); continue
        if isinstance(j, dict) and j.get("error"):
            # rate limited or other; back off
            time.sleep(15)
            continue
        return j
    raise RuntimeError(f"failed to fetch {url} after {max_tries} tries")


def fetch_point(lat: float, lon: float) -> tuple[pd.DataFrame, list[dict]]:
    prov = []
    # --- model historical forecasts (all 3 models in one call) ---
    fp = dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
              hourly="wind_speed_10m,precipitation", models=",".join(MODELS),
              wind_speed_unit="ms", timezone="GMT")
    fj = _get(HIST_URL, fp)
    h = fj["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for m in MODELS:
        s = MODEL_SHORT[m]
        df[f"wsp_{s}"] = h.get(f"wind_speed_10m_{m}")
        df[f"pr_{s}"] = h.get(f"precipitation_{m}")
    prov.append({"role": "features", "provider": "Open-Meteo Historical Forecast API",
                 "url": requests.Request("GET", HIST_URL, params=fp).prepare().url,
                 "models": MODELS, "point": [lat, lon]})

    # --- ERA5 truth ---
    ep = dict(latitude=lat, longitude=lon, start_date=START, end_date=END,
              hourly="wind_speed_10m,precipitation", wind_speed_unit="ms", timezone="GMT")
    ej = _get(ERA5_URL, ep)
    he = ej["hourly"]
    truth = pd.DataFrame({"time": pd.to_datetime(he["time"]),
                          "wsp_obs": he["wind_speed_10m"],
                          "pr_obs": he["precipitation"]})
    prov.append({"role": "labels", "provider": "Open-Meteo ERA5 archive API",
                 "url": requests.Request("GET", ERA5_URL, params=ep).prepare().url,
                 "variable": "ERA5 10m wind & precipitation", "point": [lat, lon]})

    df = df.merge(truth, on="time", how="inner")
    df["lat"] = lat
    df["lon"] = lon
    return df, prov


def add_features(df: pd.DataFrame, nino_series) -> pd.DataFrame:
    # drop rows missing any model feature or truth
    cols = ["wsp_gfs", "wsp_ifs", "wsp_gem", "pr_gfs", "pr_ifs", "pr_gem",
            "wsp_obs", "pr_obs"]
    df = df.dropna(subset=cols).copy()
    if df.empty:
        return df
    # ensemble baselines + spread
    df["wsp_ens"] = df[["wsp_gfs", "wsp_ifs", "wsp_gem"]].mean(axis=1)
    df["pr_ens"] = df[["pr_gfs", "pr_ifs", "pr_gem"]].mean(axis=1)
    df["wsp_spread"] = df[["wsp_gfs", "wsp_ifs", "wsp_gem"]].std(axis=1)
    df["pr_spread"] = df[["pr_gfs", "pr_ifs", "pr_gem"]].std(axis=1)
    # temporal / seasonal encodings (Baiu front, diurnal)
    doy = df["time"].dt.dayofyear.values
    df["hour"] = df["time"].dt.hour
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # NINO3.4 conditioning, causally lagged per month
    def nino_lookup(t):
        return enso.nino34_for_date(nino_series, t.date())
    df["nino34"] = df["time"].apply(nino_lookup)
    df = df.dropna(subset=["nino34"]).copy()
    return df


FEATURES = ["wsp_gfs", "wsp_ifs", "wsp_gem", "pr_gfs", "pr_ifs", "pr_gem",
            "wsp_ens", "pr_ens", "wsp_spread", "pr_spread",
            "lat", "lon", "hour", "doy_sin", "doy_cos", "nino34"]


def main():
    pts = sample_points()
    nino_series = enso.load_series_cached()
    all_df = []
    prov_all = []
    raw_dir = cfg.TRAIN_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    for i, (lat, lon) in enumerate(pts):
        cache = raw_dir / f"pt_{lat:+06.2f}_{lon:+07.2f}.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            # provenance still recorded
            prov_all.append({"role": "cached", "point": [lat, lon], "file": cache.name})
        else:
            try:
                df, prov = fetch_point(lat, lon)
            except Exception as e:  # noqa - resilient to API quota exhaustion
                print(f"[{i+1}/{len(pts)}] ({lat},{lon}) SKIPPED: {e}", flush=True)
                continue
            df.to_parquet(cache)
            prov_all.extend(prov)
            time.sleep(1.0)  # be gentle on the API
        all_df.append(df)
        print(f"[{i+1}/{len(pts)}] ({lat},{lon}) rows={len(df)}", flush=True)

    raw = pd.concat(all_df, ignore_index=True)
    feat = add_features(raw, nino_series)
    out = cfg.TRAIN_DIR / "training_table.parquet"
    feat.to_parquet(out)
    print("TOTAL rows", len(feat), "->", out)

    prov_file = cfg.TRAIN_DIR / "training_provenance.json"
    prov_file.write_text(json.dumps({
        "window": {"start": START, "end": END, "fence": "<= 2026-06-25 (op init); ERA5 labels ~<= 2026-06-20"},
        "models": MODELS,
        "n_points": len(pts),
        "n_rows": int(len(feat)),
        "feature_columns": FEATURES,
        "records": prov_all,
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }, indent=2))
    print("provenance ->", prov_file)


if __name__ == "__main__":
    main()
