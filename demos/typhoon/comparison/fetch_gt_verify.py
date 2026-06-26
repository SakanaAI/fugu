"""Verify-aware ground-truth fetch: real ERA5/analysis (Open-Meteo archive best_match)
for a 48h window, WITHOUT fabricating unobserved hours.

vs fetch_gt.py: (1) caps end_date at today UTC so a not-yet-complete window doesn't
trigger the empty-range bug; (2) leaves fully-unobserved lead steps as all-NaN (the
verification render skips them); observed steps get spatial NaNs filled with that step's
mean so the field is clean.

Usage: fetch_gt_verify.py <ISO_INIT_Z> <out_npz>
"""
import sys, time
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from scipy.interpolate import RegularGridInterpolator

INIT = datetime.fromisoformat(sys.argv[1].replace("Z", "+00:00")).astimezone(timezone.utc)
OUT = sys.argv[2]

TLAT = np.round(np.arange(24.0, 46.0 + 1e-6, 0.1), 1)
TLON = np.round(np.arange(122.0, 146.0 + 1e-6, 0.1), 1)
CLAT = np.round(np.arange(24.0, 46.0 + 1e-6, 0.5), 2)
CLON = np.round(np.arange(122.0, 146.0 + 1e-6, 0.5), 2)

TIMES = [INIT + timedelta(hours=h) for h in range(49)]
TIME_KEYS = [t.strftime("%Y-%m-%dT%H:%M") for t in TIMES]
today = datetime.now(timezone.utc).date()
start_date = (INIT - timedelta(hours=1)).strftime("%Y-%m-%d")
# cap end_date at today (the archive returns empty for a wholly-future end_date)
end_dt = min(TIMES[-1].date(), today)
end_date = end_dt.strftime("%Y-%m-%d")
print(f"init={INIT.isoformat()} window->{TIMES[-1].isoformat()}  fetch {start_date}..{end_date} (today={today})")

pts = [(la, lo) for la in CLAT for lo in CLON]
URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_batch(batch):
    params = {"latitude": ",".join(f"{la:.2f}" for la, lo in batch),
              "longitude": ",".join(f"{lo:.2f}" for la, lo in batch),
              "start_date": start_date, "end_date": end_date,
              "hourly": "wind_speed_10m,precipitation", "wind_speed_unit": "ms", "timezone": "UTC"}
    for attempt in range(8):
        r = requests.get(URL, params=params, timeout=90)
        if r.status_code == 200:
            return r.json()
        print(f"  status {r.status_code} retry {attempt}: {r.text[:120]}")
        time.sleep(8 * (attempt + 1))
    raise RuntimeError("batch failed")

wind_c = np.full((49, CLAT.size, CLON.size), np.nan, np.float32)
prec_c = np.full((49, CLAT.size, CLON.size), np.nan, np.float32)
results = []
for i in range(0, len(pts), 100):
    js = fetch_batch(pts[i:i+100])
    if isinstance(js, dict):
        js = [js]
    results.extend(js)
    print(f"  fetched {min(i+100,len(pts))}/{len(pts)}")
    time.sleep(1.2)

for loc in results:
    ila = int(np.argmin(np.abs(CLAT - float(loc["latitude"]))))
    ilo = int(np.argmin(np.abs(CLON - float(loc["longitude"]))))
    h = loc["hourly"]; tindex = {t: k for k, t in enumerate(h["time"])}
    ws = h["wind_speed_10m"]; pr = h["precipitation"]
    for k, tkey in enumerate(TIME_KEYS):
        j = tindex.get(tkey)
        if j is None:
            continue
        if ws[j] is not None: wind_c[k, ila, ilo] = ws[j]
        if pr[j] is not None: prec_c[k, ila, ilo] = pr[j]

# Per-step: if the step is observed at all, fill its spatial NaNs with the step mean;
# if entirely unobserved, leave all-NaN (render skips it).
def consolidate(arr):
    for k in range(49):
        if np.isfinite(arr[k]).any():
            m = np.nanmean(arr[k])
            arr[k] = np.where(np.isfinite(arr[k]), arr[k], m)
    return arr
wind_c = consolidate(wind_c); prec_c = consolidate(prec_c)

def interp(coarse):
    out = np.full((49, TLAT.size, TLON.size), np.nan, np.float32)
    LO, LA = np.meshgrid(TLON, TLAT)
    tgt = np.column_stack([LA.ravel(), LO.ravel()])
    for k in range(49):
        if not np.isfinite(coarse[k]).any():
            continue
        f = RegularGridInterpolator((CLAT, CLON), coarse[k], method="linear",
                                    bounds_error=False, fill_value=None)
        out[k] = f(tgt).reshape(TLAT.size, TLON.size)
    return out
wind = interp(wind_c)
prec = np.clip(interp(prec_c), 0.0, None)

# Only past hours are real observations. The best_match archive backfills "today's"
# future hours with IFS forecast — null any step whose valid time is after now so we
# never verify forecast-against-forecast.
now = datetime.now(timezone.utc)
for k, t in enumerate(TIMES):
    if t > now:
        wind[k, :, :] = np.nan
        prec[k, :, :] = np.nan

n_valid = int(sum(np.isfinite(wind[k]).any() for k in range(49)))
print(f"observed lead steps: {n_valid}/49  wind max {np.nanmax(wind):.1f}  precip max {np.nanmax(prec):.1f}")
np.savez_compressed(OUT, wind_speed=wind, precipitation=prec,
                    latitude=TLAT, longitude=TLON,
                    times=np.array([t.isoformat() for t in TIMES]))
print("saved", OUT)
