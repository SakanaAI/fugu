# Japan typhoon real-data forecast pipeline

This directory implements the instructions in `prompt.md` as a self-contained Python architecture.

## Files

- `typhoon_pipeline.py` — operational pipeline. It **only loads** the saved ML artifact and fails closed if real NWP/ENSO data cannot be retrieved or decoded.
- `train_corrector.py` — offline training utility that retrieves real archived forecasts, real ERA5 labels, and real NOAA CPC NINO3.4 indices, then writes `models/typhoon_corrector.json`.
- `models/typhoon_corrector.json` — saved ML corrector artifact after running training.
- `requirements.txt` — runtime packages for decoding GRIB2, interpolation, and compressed NetCDF4 export.

## Real data sources

Operational pipeline:

1. NOAA/NCEP GFS 0.25° operational GRIB2 subsets via NOMADS `filter_gfs_0p25.pl`.
2. ECMWF IFS Open Data GRIB2 forecast files.
3. DWD ICON Global Open Data GRIB2 files. This is the documented independent substitute for direct JMA GSM GRIB2 because a stable no-auth JMA GSM native GRIB2 endpoint was not available in this environment. The pipeline never falls back to synthetic fields.
4. NOAA CPC `sstoi.indices` for monthly NINO3.4 anomaly, clipped with a 15-day publication lag.

Training utility:

- Open-Meteo Historical Forecast API for archived operational GFS, ECMWF IFS, and ICON forecasts.
- Open-Meteo Archive API / ERA5 for real observed/reanalysis labels.
- NOAA CPC NINO3.4 anomalies.

## Setup

All commands should be run from this directory.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Train / refresh the saved corrector

`train_corrector.py` uses only dates up to 2026-05-01 and reports out-of-sample scores honestly. It does not make plots.

```bash
python train_corrector.py
```

## Run the operational pipeline

The script accepts a target initialization timestamp and enforces that every model cycle is initialized before or at that timestamp. Forecast valid times after the initialization are allowed because they are forecast lead times from causal model runs.

```bash
python typhoon_pipeline.py --init 2026-06-25T00:00:00Z
```

To re-run reproducibly from already-retrieved real files without re-hitting the
providers, set `TYPHOON_REUSE_CACHE=1`. The bytes are still the real provider
files previously downloaded into `data/raw/`; their checksums are recomputed and
nothing is synthesized:

```bash
TYPHOON_REUSE_CACHE=1 python typhoon_pipeline.py --init 2026-06-25T00:00:00Z
```

Outputs:

- `typhoon_forecast_output.nc` — compressed NetCDF4 with dimensions `time=49`, `latitude=221`, `longitude=241`, variables `wind_speed` and `precipitation`.
- `provenance.json` — machine-readable manifest with every retrieved real file/API payload, checksums, source cycles, leads, URLs/archive keys, ENSO causal clipping, and ML validation metrics.

## Precipitation de-accumulation (per center)

Precipitation lead-time stepping and accumulation conventions differ by center,
and the pipeline handles each on its real native convention:

- **NOAA GFS** surface `APCP` accumulates in **rolling 6-hour buckets that reset**
  at each boundary (the domain mean climbs f001→f006 then drops at f007). It is
  de-accumulated to genuine per-hour increments (`acc[h]` at a bucket start,
  `acc[h]-acc[h-1]` within a bucket) before being mapped to the hourly target.
- **ECMWF IFS** `tp` and **DWD ICON** `tot_prec` accumulate monotonically from
  initialization; they are de-accumulated by simple consecutive differencing of
  the cumulative series interpolated onto the hourly grid.

## Failure mode

If an operational source is unavailable, a dependency cannot decode GRIB2, or the saved model artifact is absent, the operational script exits with an error. It does **not** synthesize or mock missing fields.
