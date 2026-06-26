"""
Train and verify the ML error-correction layer (run OFFLINE; the deliverable
pipeline only loads the saved models).

Two XGBoost regressors:
  * wind  : predicts the residual (ERA5 obs - ensemble-mean) -> additive correction.
  * precip: predicts log1p(ERA5 obs) -> expm1 -> non-negative corrected precip.

Honest out-of-sample verification: hold out a typhoon-FREE winter/early-spring
slice (2026-01-01 .. 2026-03-31) as a test set the model never sees, and report
RMSE/MAE of the corrector vs the raw multi-model ensemble-mean baseline.  The
2026 in-season typhoon data (Tropical Storm Jangmi late-May/early-June and the
early spin-up of Typhoon Mekkhala from 06-18) is kept in TRAINING along with the
full 2024-2025 history, so the corrector learns from those cases.  This is not
leakage: all training data predates the operational late-June 2026 inits.
"""
from __future__ import annotations

import json
import datetime as dt

import numpy as np
import pandas as pd
import xgboost as xgb

import config as cfg
from build_training import FEATURES

# Held-out OOS test window: a typhoon-FREE winter/early-spring 2026 slice.
# Everything outside this window (2024-2025 history PLUS the 2026 in-season
# typhoon data from 2026-04-01 onward) is used for TRAINING.
OOS_TEST_START = pd.Timestamp("2026-01-01")   # inclusive
OOS_TEST_END = pd.Timestamp("2026-04-01")     # exclusive (covers Jan-Mar 2026)

WIND_PARAMS = dict(n_estimators=600, max_depth=7, learning_rate=0.03,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                   objective="reg:squarederror", n_jobs=-1, random_state=0)
PRECIP_PARAMS = dict(n_estimators=700, max_depth=8, learning_rate=0.03,
                     subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                     objective="reg:squarederror", n_jobs=-1, random_state=0)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    df = pd.read_parquet(cfg.TRAIN_DIR / "training_table.parquet")
    df = df.dropna(subset=FEATURES + ["wsp_obs", "pr_obs"]).reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])

    in_test = (df["time"] >= OOS_TEST_START) & (df["time"] < OOS_TEST_END)
    train = df[~in_test]
    test = df[in_test]
    print(f"train rows {len(train)}  test rows {len(test)}")
    print(f"train date span {train['time'].min()} .. {train['time'].max()}")
    print(f"test  date span {test['time'].min()} .. {test['time'].max()}")

    Xtr, Xte = train[FEATURES], test[FEATURES]

    # ---------------- WIND (residual correction) ----------------
    ytr_w = (train["wsp_obs"] - train["wsp_ens"]).values
    wind_model = xgb.XGBRegressor(**WIND_PARAMS)
    wind_model.fit(Xtr, ytr_w)
    wind_corr = test["wsp_ens"].values + wind_model.predict(Xte)
    wind_corr = np.clip(wind_corr, 0, None)

    wind_metrics = {
        "baseline_ensmean": {"rmse": rmse(test["wsp_ens"], test["wsp_obs"]),
                             "mae": mae(test["wsp_ens"], test["wsp_obs"])},
        "ml_corrected": {"rmse": rmse(wind_corr, test["wsp_obs"]),
                         "mae": mae(wind_corr, test["wsp_obs"])},
        "per_member_rmse": {m: rmse(test[f"wsp_{m}"], test["wsp_obs"])
                            for m in ("gfs", "ifs", "gem")},
    }

    # ---------------- PRECIP (log1p target) ----------------
    ytr_p = np.log1p(train["pr_obs"].clip(lower=0).values)
    precip_model = xgb.XGBRegressor(**PRECIP_PARAMS)
    precip_model.fit(Xtr, ytr_p)
    precip_corr = np.clip(np.expm1(precip_model.predict(Xte)), 0, None)

    precip_metrics = {
        "baseline_ensmean": {"rmse": rmse(test["pr_ens"], test["pr_obs"]),
                             "mae": mae(test["pr_ens"], test["pr_obs"])},
        "ml_corrected": {"rmse": rmse(precip_corr, test["pr_obs"]),
                         "mae": mae(precip_corr, test["pr_obs"])},
        "per_member_rmse": {m: rmse(test[f"pr_{m}"], test["pr_obs"])
                            for m in ("gfs", "ifs", "gem")},
    }

    def improvement(metrics, key):
        b = metrics["baseline_ensmean"][key]
        c = metrics["ml_corrected"][key]
        return round(100 * (b - c) / b, 2)

    wind_metrics["rmse_improvement_pct"] = improvement(wind_metrics, "rmse")
    wind_metrics["mae_improvement_pct"] = improvement(wind_metrics, "mae")
    precip_metrics["rmse_improvement_pct"] = improvement(precip_metrics, "rmse")
    precip_metrics["mae_improvement_pct"] = improvement(precip_metrics, "mae")

    # save models
    wind_model.save_model(cfg.MODELS_DIR / "wind_corrector.json")
    precip_model.save_model(cfg.MODELS_DIR / "precip_corrector.json")

    summary = {
        "trained_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "features": FEATURES,
        "oos_split": f"{OOS_TEST_START.date()} .. {(OOS_TEST_END - pd.Timedelta(days=1)).date()} (typhoon-free held-out test window)",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "wind": wind_metrics,
        "precip": precip_metrics,
        "wind_params": WIND_PARAMS,
        "precip_params": PRECIP_PARAMS,
        "framing": {
            "wind": "residual: obs - ensemble_mean, additive",
            "precip": "log1p(obs) target, expm1 inverse",
        },
        "label_source": "ERA5 reanalysis (Open-Meteo archive API)",
        "feature_source": "Open-Meteo historical forecasts of GFS/IFS(ECMWF)/GEM(CMC)",
    }
    (cfg.MODELS_DIR / "model_metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== OUT-OF-SAMPLE SKILL (typhoon-free held-out 2026-01-01 .. 2026-03-31) ===")
    print("WIND   baseline RMSE %.3f -> corrected RMSE %.3f  (%.1f%% better)" % (
        wind_metrics["baseline_ensmean"]["rmse"], wind_metrics["ml_corrected"]["rmse"],
        wind_metrics["rmse_improvement_pct"]))
    print("PRECIP baseline RMSE %.3f -> corrected RMSE %.3f  (%.1f%% better)" % (
        precip_metrics["baseline_ensmean"]["rmse"], precip_metrics["ml_corrected"]["rmse"],
        precip_metrics["rmse_improvement_pct"]))
    print("metrics ->", cfg.MODELS_DIR / "model_metrics.json")


if __name__ == "__main__":
    main()
