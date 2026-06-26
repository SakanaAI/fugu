"""
Assemble the training table from cached per-point parquet files (no network).
Used when the full point sweep is throttled by the Open-Meteo quota; the cached
points already provide hundreds of thousands of real rows.
"""
from __future__ import annotations

import json
import datetime as dt
import re

import pandas as pd

import config as cfg
import enso
from build_training import add_features, FEATURES, MODELS, START, END


def main():
    raw_dir = cfg.TRAIN_DIR / "raw"
    files = sorted(raw_dir.glob("pt_*.parquet"))
    nino_series = enso.load_series_cached()
    frames, points = [], []
    for f in files:
        df = pd.read_parquet(f)
        frames.append(df)
        m = re.match(r"pt_([+-][0-9.]+)_([+-][0-9.]+)\.parquet", f.name)
        if m:
            points.append([float(m.group(1)), float(m.group(2))])
    raw = pd.concat(frames, ignore_index=True)
    feat = add_features(raw, nino_series)
    out = cfg.TRAIN_DIR / "training_table.parquet"
    feat.to_parquet(out)
    print("assembled", len(files), "points ->", len(feat), "rows ->", out)

    prov = {
        "window": {"start": START, "end": END, "fence": "<= 2026-05-01"},
        "models": MODELS,
        "n_points": len(files),
        "points": points,
        "n_rows": int(len(feat)),
        "feature_columns": FEATURES,
        "feature_source": "Open-Meteo Historical Forecast API (gfs_seamless, ecmwf_ifs025, gem_seamless)",
        "label_source": "Open-Meteo ERA5 archive API (10m wind & precipitation)",
        "enso_source": "NOAA CPC ERSSTv5 monthly NINO3.4 (causally lagged per month)",
        "note": ("Per-point cached real downloads under data/train/raw/. Point sweep was "
                 "throttled by the free Open-Meteo quota; assembled from the cached real points."),
        "built_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (cfg.TRAIN_DIR / "training_provenance.json").write_text(json.dumps(prov, indent=2))
    print("provenance written")


if __name__ == "__main__":
    main()
