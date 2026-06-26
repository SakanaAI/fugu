"""Shared helpers for the typhoon-pipeline comparison animations."""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, TwoSlopeNorm, Normalize

DATA = "comparison/data"
MODELS = ["claude", "fugu", "ultra"]
MODEL_LABEL = {"claude": "Model: claude", "fugu": "Model: fugu", "ultra": "Model: ultra"}
MODEL_COLOR = {"claude": "#e6550d", "fugu": "#3182bd", "ultra": "#31a354"}

TLAT = np.round(np.arange(24.0, 46.0 + 1e-6, 0.1), 1)   # 221
TLON = np.round(np.arange(122.0, 146.0 + 1e-6, 0.1), 1)  # 241
ASPECT = 1.0 / np.cos(np.deg2rad(35.0))                  # degree aspect for ~35N

# pcolormesh cell edges
def _edges(c):
    d = (c[1] - c[0]) / 2.0
    return np.concatenate([[c[0] - d], c + d])
LON_E = _edges(TLON)
LAT_E = _edges(TLAT)

WIND_CMAP = "viridis"
PRECIP_CMAP = "turbo"
ERR_CMAP = "RdBu_r"


def load_forecast(model, date):
    ds = xr.open_dataset(f"{DATA}/{model}_{date}.nc")
    w = ds.wind_speed.values.astype(np.float32)
    p = ds.precipitation.values.astype(np.float32)
    times = ds.time.values
    ds.close()
    return w, p, times


def load_ground_truth(date):
    z = np.load(f"{DATA}/ground_truth_{date}.npz", allow_pickle=True)
    return z["wind_speed"].astype(np.float32), z["precipitation"].astype(np.float32)


def load_coastline():
    z = np.load(f"{DATA}/coastline.npz", allow_pickle=True)
    return list(z["segs"])


def draw_coast(ax, segs, lw=0.6, color="k"):
    for s in segs:
        ax.plot(s[:, 0], s[:, 1], color=color, lw=lw, alpha=0.85, zorder=5)


def style_map(ax):
    ax.set_xlim(TLON[0], TLON[-1])
    ax.set_ylim(TLAT[0], TLAT[-1])
    ax.set_aspect(ASPECT)
    ax.set_xticks([]); ax.set_yticks([])


def robust_vmax(arrs, q=99.0):
    return float(np.percentile(np.concatenate([a.ravel() for a in arrs]), q))


def rmse_series(pred, truth):
    """RMSE over space at each lead time -> (49,) array."""
    diff = pred - truth
    return np.sqrt(np.nanmean(diff * diff, axis=(1, 2)))
