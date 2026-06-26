#!/usr/bin/env python3
"""
Offline training utility for the Japan typhoon forecast corrector.

This script deliberately uses only real retrieved data:
  * Historical operational forecasts from Open-Meteo's historical forecast API
    for GFS, ECMWF IFS, and DWD ICON global.
  * ERA5 reanalysis fields from Open-Meteo's archive API as the observed / truth
    target.
  * NOAA CPC monthly NINO3.4 SST anomalies, causally lagged.

It writes a compact JSON ridge-regression model consumed by typhoon_pipeline.py.
The operational pipeline never trains; it only loads the saved JSON artifact.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / "data" / "training"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "typhoon_corrector.json"
OPEN_METEO_HISTORICAL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NOAA_CPC_NINO34 = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
TRAINING_CUTOFF = "2026-06-21T23:59:59Z"
MODELS = ["gfs_global", "ecmwf_ifs025", "icon_global"]
TARGETS = ["wind_speed", "precipitation"]
FEATURE_NAMES = [
    "intercept",
    "gfs",
    "ecmwf",
    "icon",
    "ensemble_mean",
    "ensemble_spread",
    "ensemble_min",
    "ensemble_max",
    "latitude_norm",
    "longitude_norm",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "nino34_anom",
    "baiu_front_proxy",
    "typhoon_wind_proxy",
    "orographic_rain_proxy",
]

# Small but physically diverse Japan-domain sample.  All coordinates are inside
# the required target box and include marine approaches where typhoon wind/rain
# errors matter most.
SAMPLE_POINTS = [
    (26.2, 127.7, "okinawa_ryukyu_marine"),
    (30.0, 140.0, "izu_bonin_marine"),
    (31.6, 130.6, "kyushu_kagoshima"),
    (34.7, 135.5, "kansai_osaka"),
    (35.7, 139.8, "kanto_tokyo_bay"),
    (38.3, 142.0, "sanriku_offshore"),
    (43.0, 141.4, "hokkaido_sapporo"),
]

TRAIN_WINDOWS = [
    ("2025-06-15", "2025-07-20", "baiu_front_heavy_rain_season"),
    ("2025-08-20", "2025-09-25", "western_north_pacific_typhoon_season"),
    ("2026-01-15", "2026-02-15", "cool_season_control"),
    ("2026-05-18", "2026-06-03", "ts_jangmi_2026_japan_landfall"),
    ("2026-06-10", "2026-06-20", "typhoon_mekkhala_2026_spinup"),
]
VALIDATION_WINDOWS = [
    ("2026-04-01", "2026-04-30", "out_of_sample_spring_2026_before_cutoff"),
]


class DataError(RuntimeError):
    pass


def ensure_inside_root(path: Path) -> Path:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise DataError(f"Refusing to access path outside working directory: {path}")
    return resolved


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def fetch_url(url: str, timeout: int = 60) -> Tuple[bytes, Dict[str, object]]:
    req = Request(url, headers={"User-Agent": "typhoon-real-data-pipeline/1.0"})
    retrieved_at = utc_now_iso()
    with urlopen(req, timeout=timeout) as response:
        body = response.read()
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type")
    digest = hashlib.sha256(body).hexdigest()
    record = {
        "url": url,
        "retrieval_timestamp": retrieved_at,
        "http_status": status,
        "content_type": content_type,
        "size_bytes": len(body),
        "sha256": digest,
    }
    if status and int(status) >= 400:
        raise DataError(f"HTTP {status} for {url}")
    return body, record


def query_json(base_url: str, params: Dict[str, str]) -> Tuple[Dict[str, object], Dict[str, object]]:
    url = base_url + "?" + urlencode(params)
    body, record = fetch_url(url)
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise DataError(f"Non-JSON response from {url}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise DataError(f"Provider returned error for {url}: {payload}")
    return payload, record


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def parse_hour(s: str) -> datetime:
    # Open-Meteo uses YYYY-MM-DDTHH:MM in UTC when timezone=UTC.
    return datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def month_publication_time(year: int, month: int, lag_days: int = 15) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    return datetime(year, month, last_day, tzinfo=timezone.utc) + timedelta(days=lag_days)


def load_nino34(provenance: List[Dict[str, object]]) -> Dict[Tuple[int, int], float]:
    body, record = fetch_url(NOAA_CPC_NINO34)
    record.update({
        "source_provider": "NOAA CPC",
        "dataset": "Monthly NINO3.4 SST anomaly (sstoi.indices)",
        "variables": ["NINO3.4_ANOM"],
    })
    provenance.append(record)
    text = body.decode("utf-8", errors="replace")
    values: Dict[Tuple[int, int], float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 10 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        month = int(parts[1])
        # Header: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
        anom = float(parts[9])
        values[(year, month)] = anom
    if not values:
        raise DataError("No NINO3.4 values parsed from NOAA CPC file")
    return values


def causally_available_nino34(valid_time: datetime, monthly: Dict[Tuple[int, int], float]) -> float:
    # For feature engineering, use the most recent monthly NINO3.4 anomaly whose
    # publication time is not after the valid forecast/observation time.  This
    # avoids peeking at unpublished climate indices.
    best_key: Optional[Tuple[int, int]] = None
    best_pub: Optional[datetime] = None
    for (year, month), _value in monthly.items():
        pub = month_publication_time(year, month)
        if pub <= valid_time and (best_pub is None or pub > best_pub):
            best_key = (year, month)
            best_pub = pub
    if best_key is None:
        raise DataError(f"No causally published NINO3.4 anomaly for {valid_time.isoformat()}")
    return monthly[best_key]


def safe_number(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def fetch_point_window(lat: float, lon: float, start: str, end: str) -> Tuple[Dict[str, object], Dict[str, object], List[Dict[str, object]]]:
    model_string = ",".join(MODELS)
    forecast_params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,precipitation",
        "models": model_string,
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    obs_params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start,
        "end_date": end,
        "hourly": "wind_speed_10m,precipitation",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": "era5",
    }
    forecast, f_record = query_json(OPEN_METEO_HISTORICAL, forecast_params)
    obs, o_record = query_json(OPEN_METEO_ARCHIVE, obs_params)
    f_record.update({
        "source_provider": "Open-Meteo Historical Forecast API",
        "dataset": "Archived operational NWP point forecasts",
        "model_cycle_or_archive_key": model_string,
        "variables": ["wind_speed_10m", "precipitation"],
        "latitude": lat,
        "longitude": lon,
        "date_start": start,
        "date_end": end,
    })
    o_record.update({
        "source_provider": "Open-Meteo Archive API / ERA5",
        "dataset": "ERA5 reanalysis point fields",
        "variables": ["wind_speed_10m", "precipitation"],
        "latitude": lat,
        "longitude": lon,
        "date_start": start,
        "date_end": end,
    })
    return forecast, obs, [f_record, o_record]


def feature_vector(values: Sequence[float], lat: float, lon: float, valid_time: datetime, nino: float) -> List[float]:
    gfs, ecmwf, icon = values
    ens_mean = sum(values) / 3.0
    ens_min = min(values)
    ens_max = max(values)
    variance = sum((v - ens_mean) ** 2 for v in values) / 3.0
    spread = math.sqrt(max(variance, 0.0))
    lat_norm = (lat - 35.0) / 11.0
    lon_norm = (lon - 134.0) / 12.0
    hour_angle = 2.0 * math.pi * (valid_time.hour / 24.0)
    month_angle = 2.0 * math.pi * ((valid_time.month - 1) / 12.0)
    # June-July stationary Baiu-front proxy strongest around 30-36N.
    baiu_month = 1.0 if valid_time.month in (6, 7) else 0.0
    baiu_lat = max(0.0, 1.0 - abs(lat - 33.0) / 7.0)
    baiu_front_proxy = baiu_month * baiu_lat * max(ens_mean, 0.0)
    # Typhoon risk proxy: high winds in the southern/western Japan approaches.
    south_west = max(0.0, (36.0 - lat) / 12.0) * max(0.0, (140.0 - lon) / 18.0)
    typhoon_wind_proxy = south_west * ens_max
    # Very coarse orographic rain proxy for Honshu/Shikoku/Kyushu windward bands.
    orographic_band = max(0.0, 1.0 - abs(lat - 34.5) / 5.0) * max(0.0, 1.0 - abs(lon - 136.5) / 8.0)
    orographic_rain_proxy = orographic_band * max(ens_mean, 0.0)
    return [
        1.0,
        gfs,
        ecmwf,
        icon,
        ens_mean,
        spread,
        ens_min,
        ens_max,
        lat_norm,
        lon_norm,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(month_angle),
        math.cos(month_angle),
        nino,
        baiu_front_proxy,
        typhoon_wind_proxy,
        orographic_rain_proxy,
    ]


@dataclass
class Example:
    target: str
    x: List[float]
    y: float
    baseline: float
    valid_time: str
    point: str


def build_examples(
    point_name: str,
    lat: float,
    lon: float,
    forecast: Dict[str, object],
    obs: Dict[str, object],
    nino34: Dict[Tuple[int, int], float],
) -> List[Example]:
    f_hourly = forecast.get("hourly")
    o_hourly = obs.get("hourly")
    if not isinstance(f_hourly, dict) or not isinstance(o_hourly, dict):
        raise DataError("Missing hourly data in provider response")
    f_times = [parse_hour(t) for t in f_hourly.get("time", [])]
    o_times = [parse_hour(t) for t in o_hourly.get("time", [])]
    obs_index = {t: i for i, t in enumerate(o_times)}
    examples: List[Example] = []
    for idx, valid_time in enumerate(f_times):
        if valid_time not in obs_index:
            continue
        oi = obs_index[valid_time]
        nino = causally_available_nino34(valid_time, nino34)
        for target in TARGETS:
            if target == "wind_speed":
                base_name = "wind_speed_10m"
                obs_key = "wind_speed_10m"
            else:
                base_name = "precipitation"
                obs_key = "precipitation"
            raw_values: List[float] = []
            missing = False
            for model in MODELS:
                # Open-Meteo uses suffixed names when multiple models are requested.
                value = safe_number(f_hourly.get(f"{base_name}_{model}", [None] * len(f_times))[idx])
                if value is None:
                    missing = True
                    break
                raw_values.append(max(value, 0.0) if target == "precipitation" else value)
            y = safe_number(o_hourly.get(obs_key, [None] * len(o_times))[oi])
            if missing or y is None:
                continue
            if target == "precipitation":
                y = max(y, 0.0)
            x = feature_vector(raw_values, lat, lon, valid_time, nino)
            examples.append(
                Example(
                    target=target,
                    x=x,
                    y=y,
                    baseline=sum(raw_values) / 3.0,
                    valid_time=valid_time.isoformat().replace("+00:00", "Z"),
                    point=point_name,
                )
            )
    return examples


def transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(col) for col in zip(*matrix)]


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise DataError("Singular design matrix during ridge solve")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def fit_ridge(examples: Sequence[Example], ridge_lambda: float = 0.25) -> List[float]:
    if not examples:
        raise DataError("No examples supplied to fit_ridge")
    p = len(examples[0].x)
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for ex in examples:
        for i in range(p):
            xty[i] += ex.x[i] * ex.y
            for j in range(p):
                xtx[i][j] += ex.x[i] * ex.x[j]
    for i in range(1, p):  # do not regularize intercept
        xtx[i][i] += ridge_lambda
    return solve_linear_system(xtx, xty)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def metrics(examples: Sequence[Example], coeffs: Sequence[float]) -> Dict[str, float]:
    if not examples:
        raise DataError("No examples supplied for metrics")
    err_model = []
    err_base = []
    for ex in examples:
        pred = dot(coeffs, ex.x)
        pred = max(pred, 0.0)
        err_model.append(pred - ex.y)
        err_base.append(max(ex.baseline, 0.0) - ex.y)
    n = float(len(examples))
    rmse_model = math.sqrt(sum(e * e for e in err_model) / n)
    rmse_base = math.sqrt(sum(e * e for e in err_base) / n)
    mae_model = sum(abs(e) for e in err_model) / n
    mae_base = sum(abs(e) for e in err_base) / n
    return {
        "n": int(n),
        "corrected_rmse": rmse_model,
        "baseline_rmse": rmse_base,
        "rmse_improvement": rmse_base - rmse_model,
        "corrected_mae": mae_model,
        "baseline_mae": mae_base,
        "mae_improvement": mae_base - mae_model,
        "beats_baseline_rmse": rmse_model < rmse_base,
        "beats_baseline_mae": mae_model < mae_base,
    }


def collect_examples() -> Tuple[List[Example], List[Example], List[Dict[str, object]]]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    provenance: List[Dict[str, object]] = []
    nino34 = load_nino34(provenance)
    train: List[Example] = []
    valid: List[Example] = []
    for split_name, windows, bucket in [
        ("train", TRAIN_WINDOWS, train),
        ("validation", VALIDATION_WINDOWS, valid),
    ]:
        for start, end, label in windows:
            if parse_date(end) > parse_date("2026-06-21"):
                raise DataError(f"{split_name} window extends beyond training cutoff: {start}..{end}")
            for lat, lon, point_name in SAMPLE_POINTS:
                forecast, obs, records = fetch_point_window(lat, lon, start, end)
                for record in records:
                    record.update({"split": split_name, "window_label": label, "point_name": point_name})
                    provenance.append(record)
                examples = build_examples(point_name, lat, lon, forecast, obs, nino34)
                bucket.extend(examples)
                print(f"{split_name:10s} {label:42s} {point_name:24s} -> {len(examples):5d} examples")
    return train, valid, provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the saved real-data typhoon corrector artifact.")
    parser.add_argument("--output", default=str(MODEL_PATH), help="Model JSON path inside this working directory")
    parser.add_argument("--ridge", type=float, default=0.25, help="Ridge regularization strength")
    args = parser.parse_args()
    output = ensure_inside_root(Path(args.output))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, valid, provenance = collect_examples()
    model_targets: Dict[str, object] = {}
    verification: Dict[str, object] = {}
    for target in TARGETS:
        train_target = [ex for ex in train if ex.target == target]
        valid_target = [ex for ex in valid if ex.target == target]
        coeffs = fit_ridge(train_target, ridge_lambda=args.ridge)
        model_targets[target] = {
            "coefficients": coeffs,
            "coefficient_units_note": "Linear correction in target units from real operational forecast features and causal NINO3.4.",
        }
        verification[target] = {
            "train": metrics(train_target, coeffs),
            "out_of_sample_validation": metrics(valid_target, coeffs),
        }
    artifact = {
        "artifact_type": "real_data_ridge_pointwise_corrector",
        "created_at": utc_now_iso(),
        "training_cutoff": TRAINING_CUTOFF,
        "training_description": (
            "Ridge linear multi-model error-correction trained on real archived GFS, ECMWF IFS, "
            "and DWD ICON forecasts against real ERA5 reanalysis point labels over Japan and marine approaches."
        ),
        "operational_source_alignment": {
            "gfs": "NOAA GFS operational grids",
            "ecmwf": "ECMWF IFS open-data operational grids",
            "icon": "DWD ICON global operational grids used as the documented independent substitute for direct JMA GSM GRIB ingestion",
        },
        "feature_names": FEATURE_NAMES,
        "model_targets": model_targets,
        "verification": verification,
        "training_windows": TRAIN_WINDOWS,
        "validation_windows": VALIDATION_WINDOWS,
        "sample_points": [{"latitude": lat, "longitude": lon, "name": name} for lat, lon, name in SAMPLE_POINTS],
        "provenance": provenance,
        "real_data_only_statement": (
            "No synthetic forecast or truth fields were generated. All fitted examples came from retrieved "
            "Open-Meteo historical operational forecasts, Open-Meteo ERA5 archive observations, and NOAA CPC NINO3.4."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = ensure_inside_root(TRAINING_DIR / "training_manifest.json")
    manifest_path.write_text(json.dumps({"created_at": utc_now_iso(), "provenance": provenance}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote model artifact: {output}")
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
