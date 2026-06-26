"""MODE-style object comparison for WIND over the 24h prediction.

The object is the storm's strong-wind swath = the 24h peak-wind field thresholded at 15 m/s
(smoothed, small specks removed). For each agent we match that swath to the analysis swath and
report the standard MODE attributes:
  location   centroid displacement (km)        -> did you place the strong winds correctly
  area ratio forecast swath / analysis swath   -> 1 = right coverage; >1 too much, <1 too little
  intensity  95th-pct peak wind, forecast - analysis (m/s)
"""
import numpy as np, xarray as xr
from scipy import ndimage

z = np.load("comparison/round2/data/ground_truth_now_partial.npz", allow_pickle=True)
tw = z["wind_speed"]
valid = [k for k in range(49) if np.isfinite(tw[k]).any()]
steps = [k for k in valid if k <= 24]

def swath(field_max, thr=15.0, sigma=2.0, minarea=20):
    lab, n = ndimage.label(ndimage.gaussian_filter(field_max, sigma) >= thr)
    keep = np.zeros(field_max.shape, bool)
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= minarea:
            keep |= m
    if keep.sum() == 0:
        return None
    ys, xs = np.where(keep); w = field_max[keep]
    return dict(area=int(keep.sum()),
                cy=np.average(ys, weights=w), cx=np.average(xs, weights=w),
                peak=float(np.percentile(field_max[keep], 95)))

obs = swath(np.nanmax(tw[steps], 0))
files = {"claude": "claude_now.nc", "fugu": "fugu_now.nc", "fugu-ultra": "ultra_now.nc"}
print(f"24h peak-wind swath (>=15 m/s).  analysis swath area = {obs['area']} cells, "
      f"95th-pct peak = {obs['peak']:.1f} m/s\n")
print(f"  {'model':11s}   location(km)   area ratio   peak bias (m/s)")
for m, fn in files.items():
    ds = xr.open_dataset(f"comparison/round2/data/{fn}"); w = ds.wind_speed.values; ds.close()
    s = swath(np.nanmax(w[steps], 0))
    if s is None:
        print(f"  {m:11s}   (no swath)"); continue
    dy = (s["cy"] - obs["cy"]) * 0.1 * 111.0
    dx = (s["cx"] - obs["cx"]) * 0.1 * 111.0 * np.cos(np.deg2rad(35))
    dist = np.hypot(dy, dx)
    print(f"  {m:11s}   {dist:5.0f}          {s['area']/obs['area']:.2f}         {s['peak']-obs['peak']:+.1f}")
