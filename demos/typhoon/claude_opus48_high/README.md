# Japan Typhoon Multi-Agency NWP Ensembling & ML Error-Correction Pipeline

Operational pipeline that ingests **real, retrieved** multi-agency NWP forecast
grids, conditions on a **real ENSO index**, applies an **ML error-correction
layer trained against real ERA5 observations**, and exports a 48-hour gridded
forecast of 10 m wind and hourly precipitation over Japan.

Everything runs inside this directory. The only external access is outbound
network requests to public data providers.

## Real data sources (no mocking, no synthesis)

| Stream | Center | Product | Access |
|---|---|---|---|
| Forecast #1 | **NOAA** | GFS 0.25° (GRIB2, AWS Open Data, idx byte-range subset) | free, no-auth |
| Forecast #2 | **ECMWF** | IFS open-data 0.25° (GRIB2 via `ecmwf-opendata`) | free, no-auth |
| Forecast #3 | **CMC (Canada)** | GDPS 0.15° (GRIB2, MSC Datamart) — *documented substitute for JMA* | free, no-auth |
| ENSO | **NOAA CPC** | ERSSTv5 monthly NINO3.4 anomaly | free, no-auth |
| Training features | GFS / IFS / GEM | Open-Meteo Historical Forecast archive of the same centers | free, no-auth |
| Training labels (truth) | **ERA5** | Open-Meteo ERA5 archive (10 m wind & precip) | free, no-auth |

**JMA substitution.** JMA GSM/MSM GRIB is not available from a free, no-auth
archive on this machine, so per the task's allowance we substitute CMC GDPS — a
fully independent operational center on a regular lat/lon grid. This is recorded
in `output/provenance.json`.

**Why Open-Meteo for training.** Past-cycle *native GRIB* for IFS/GDPS is not
retained on the free open-data servers, so the historical record of those same
centers' issued forecasts is pulled from the Open-Meteo archive. Labels are real
ERA5 reanalysis — never self-generated — so the corrector learns the **real**
systematic model error, not a fiction.

## Causality / temporal fencing

* The pipeline takes a target init timestamp (default `2026-06-25T12:00:00Z`).
* Each center's run is the latest cycle initialized **≤ target** (GFS 06Z, IFS
  06Z, GDPS 00Z for the default run). Every center's forecast is reconciled in
  **absolute valid time** so differing inits and lead-steppings line up on the
  49-step target axis.
* The NINO3.4 value used is the most recent month that **ends before** the init
  month (respects CPC's real publication lag — May 2026 for a June init).
* Training data is fenced to **≤ 2026-05-01**; out-of-sample verification holds
  out 2026-01-01 … 2026-04-30, dates the model never sees.

## ML correction

Two XGBoost regressors (per-cell MOS-style correction):

* **wind** — predicts the residual `ERA5 − ensemble_mean`, added back.
* **precip** — predicts `log1p(ERA5)`, inverted with `expm1`, clipped ≥ 0.

Features: the three members' wind & precip, ensemble mean & spread, lat, lon,
hour-of-day, day-of-year (Baiu/diurnal encodings), and the NINO3.4 anomaly.

**Honest out-of-sample skill (held-out 2026-01-01 … 2026-04-30):**

| Variable | Baseline ens-mean RMSE | ML-corrected RMSE | Improvement |
|---|---|---|---|
| 10 m wind | 1.090 m/s | **0.698 m/s** | **36 %** |
| precip (hourly) | 0.402 mm | **0.354 mm** | **12 %** |

The corrector beats the raw multi-model ensemble mean on real held-out
observations for both fields (full per-member numbers in `models/model_metrics.json`).

## Deliverables

* `output/typhoon_forecast_output.nc` — NetCDF4, zlib-compressed. Dims
  `time(49) × latitude(221) × longitude(241)`; vars `wind_speed` (m/s) and
  `precipitation` (mm hourly); global attrs record the init timestamp, the ML
  weights used, and the explicit source model runs.
* `output/provenance.json` — every real file ingested (source, resolvable URL,
  cycle/init, variables, lead times, retrieval time, size, SHA-256) plus the
  corrector's real out-of-sample scores.

## Layout / how to run

```
src/config.py            grid, box, causal init, paths
src/enso.py              CPC NINO3.4, causally fenced
src/gribio.py            GRIB byte-range subsetting + robust download
src/fetch_gfs.py         NOAA GFS (idx byte-range)
src/fetch_ifs.py         ECMWF IFS (ecmwf-opendata)
src/fetch_gdps.py        CMC GDPS (MSC Datamart)
src/regrid.py            bilinear regrid + de-accumulation + temporal align
src/build_training.py    download real training data (Open-Meteo + ERA5 + ENSO)
src/assemble_training.py assemble training table from cached points
src/train.py             train + OOS-verify + save XGBoost correctors  (OFFLINE)
src/run_pipeline.py      DELIVERABLE: load saved models, ingest, correct, export
```

```bash
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
# offline (already done; produces models/*.json):
.venv/bin/python src/build_training.py      # or assemble_training.py from cache
.venv/bin/python src/train.py
# operational deliverable (loads saved model only, never trains):
.venv/bin/python src/run_pipeline.py --init 2026-06-25T12:00:00Z
```

The deliverable script never trains — it only loads `models/wind_corrector.json`
and `models/precip_corrector.json`. No plots are produced.
