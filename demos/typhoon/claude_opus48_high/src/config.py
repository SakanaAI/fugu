"""
Shared configuration for the Japan typhoon NWP ensembling / error-correction pipeline.

Everything in this project is confined to the directory that contains prompt.md.
This module is the single source of truth for the target grid, the spatial box,
the causal target initialization timestamp and all on-disk paths.
"""
from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Filesystem sandbox -- everything lives under the directory holding prompt.md.
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
assert (ROOT / "prompt.md").exists(), f"sandbox root {ROOT} must contain prompt.md"

DATA_DIR = ROOT / "data"
FORECAST_DIR = DATA_DIR / "forecast"      # raw operational GRIB downloads
TRAIN_DIR = DATA_DIR / "train"            # training dataset (point samples)
ENSO_DIR = DATA_DIR / "enso"              # NINO3.4 text products
MODELS_DIR = ROOT / "models"             # saved XGBoost correctors + metrics
OUTPUT_DIR = ROOT / "output"             # typhoon_forecast_output.nc, provenance.json
LOG_DIR = ROOT / "logs"

for _d in (DATA_DIR, FORECAST_DIR, TRAIN_DIR, ENSO_DIR, MODELS_DIR, OUTPUT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Target space-time grid (rigid schema required by the deliverable).
# ----------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 24.0, 46.0
LON_MIN, LON_MAX = 122.0, 146.0
DLAT = DLON = 0.1

# 221 latitudes (24.0 .. 46.0 inclusive), 241 longitudes (122.0 .. 146.0 inclusive)
TARGET_LAT = np.round(np.arange(LAT_MIN, LAT_MAX + DLAT / 2, DLAT), 4)
TARGET_LON = np.round(np.arange(LON_MIN, LON_MAX + DLON / 2, DLON), 4)
assert TARGET_LAT.size == 221, TARGET_LAT.size
assert TARGET_LON.size == 241, TARGET_LON.size

N_LAT = TARGET_LAT.size          # 221
N_LON = TARGET_LON.size          # 241

# 48-hour horizon at 1-hour steps -> 49 lead times T+0 .. T+48
LEAD_HOURS = np.arange(0, 49, dtype=int)
N_TIME = LEAD_HOURS.size          # 49

# ----------------------------------------------------------------------------
# Causal target initialization timestamp.
# The pipeline may only ingest model runs initialized BEFORE OR ON this time,
# and may use no observation / ENSO value dated after it.
# ----------------------------------------------------------------------------
DEFAULT_TARGET_INIT = "2026-06-25T12:00:00Z"


def parse_init(ts: str) -> dt.datetime:
    """Parse an ISO-8601 'Z' timestamp into a tz-aware UTC datetime."""
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    d = dt.datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def get_target_init() -> dt.datetime:
    return parse_init(os.environ.get("TARGET_INIT", DEFAULT_TARGET_INIT))


# Operational centers used for the deliverable run.  JMA's GSM/MSM GRIB is not
# available from a free, no-auth archive on this machine, so per the prompt we
# substitute DWD ICON -- a fully independent operational center -- and document
# the substitution in the provenance manifest.
CENTERS = ("gfs", "ifs", "icon")

# Variable canonical names
V_WIND = "wind_speed"
V_PRECIP = "precipitation"

# Training temporal fence: training data must be dated <= this day.
TRAIN_MAX_DATE = dt.date(2026, 6, 20)
