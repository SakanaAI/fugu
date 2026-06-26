# Japan typhoon operational NWP ML pipeline

This directory implements the instructions in `prompt.md` with a fail-closed real-data pipeline. It never fabricates forecast, ENSO, or training-label fields.

## Files

- `typhoon_forecast_pipeline.py` — operational 48-hour pipeline. It only loads a saved corrector and writes `typhoon_forecast_output.nc` plus `provenance.json`.
- `scripts/train_corrector.py` — offline trainer that creates `artifacts/typhoon_corrector.json` from real Open-Meteo historical forecasts, ERA5 reanalysis targets, and NOAA CPC ENSO.
- `requirements.txt` — Python dependencies for GRIB decoding, interpolation, ECMWF open-data retrieval, and NetCDF4 writing.

## Data sources

Operational forecast grids:

1. NOAA/NCEP GFS via NOMADS GRIB2 filter, Japan subset, 10 m U/V and surface APCP.
2. ECMWF IFS via the official ECMWF Open Data client, 10u/10v/tp GRIB.
3. DWD ICON Global via DWD Open Data GRIB2.bz2 as a documented substitute for JMA GSM. JMA WIS GSM GRIB paths are credential-gated from this environment (HTTP 401 without registration), so the code uses DWD ICON rather than any mock or generated field.

Climate conditioning:

- NOAA CPC `sstoi.indices`; the NINO3.4 anomaly is causally clipped with a conservative 45-day publication-lag guard.

Offline ML training:

- Open-Meteo historical forecast API for `gfs_global`, `ecmwf_ifs025`, `icon_global` point histories around Japan.
- Open-Meteo archive API ERA5 hourly wind/precipitation as real verification targets.
- NOAA CPC NINO3.4 anomaly as climate-state feature.

## Run

Create/install an environment inside this directory, keeping pip cache inside this directory:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --cache-dir .pip-cache -r requirements.txt
```

Train the saved corrector offline (one-time, real data only):

```bash
python scripts/train_corrector.py
```

Run the operational pipeline for a causally fenced target initialization timestamp:

```bash
python typhoon_forecast_pipeline.py --init 2026-06-25T12:00:00Z
```

Expected outputs:

- `typhoon_forecast_output.nc`: NetCDF4, compressed, dimensions `time=49`, `latitude=221`, `longitude=241`, variables `wind_speed` and `precipitation`.
- `provenance.json`: manifest listing every downloaded real file/API object with provider, URL/archive key, cycle/init, variables, lead times, retrieval time, size/checksum, and ML validation scores.

If any real provider is unavailable, the pipeline exits with an error. It does not fall back to mock, climatological, analytic, or self-generated forecast fields.
