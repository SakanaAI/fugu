"""Per-variable verification animation: three forecasters vs ERA5 analysis, for ONE variable
(wind OR precip), as a 2x4 grid.

  Row 0: claude | fugu | fugu-ultra | ANALYSIS        (field maps)
  Row 1: claude | fugu | fugu-ultra | rollout         (error maps + a rollout panel)

Rollout depends on variable:
  - wind  -> non-physical-structure index (straight grid-aligned edge area; analysis ~0 ref)
  - precip-> spatial RMSE vs lead

Models are aligned by lead index (T+0..T+48). Truth may be partial; the map panels continue
as forecast-only past the observed window.
"""
import argparse, os, sys
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viz_common as V

DISP = {"ultra": "fugu-ultra"}
DOMAIN = 221 * 241


def load_var(path, var):
    ds = xr.open_dataset(path)
    a = (ds.wind_speed if var == "wind" else ds.precipitation).values.astype(np.float32)
    ds.close()
    assert a.shape == (49, 221, 241), f"{path}: {a.shape}"
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--variable", required=True, choices=["wind", "precip"])
    ap.add_argument("--data", default="comparison/round2/data")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="predictions")
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    var = args.variable
    is_wind = var == "wind"
    init = datetime.fromisoformat(args.init.replace("Z", "+00:00"))
    segs = V.load_coastline()

    fc = {m: load_var(f"{args.data}/{m}_{args.label}.nc", var) for m in V.MODELS}
    if args.placeholder:
        t = np.mean([fc[m] for m in V.MODELS], axis=0).astype(np.float32)
    else:
        z = np.load(args.gt, allow_pickle=True)
        t = (z["wind_speed"] if is_wind else z["precipitation"]).astype(np.float32)

    valid = [k for k in range(49) if np.isfinite(t[k]).any()]
    if not valid:
        raise SystemExit("no valid truth steps")
    last = valid[-1]
    print(f"[{var}] valid through T+{last}h ({len(valid)} steps)")

    units = "m/s" if is_wind else "mm/h"
    fld_cmap = V.WIND_CMAP if is_wind else V.PRECIP_CMAP
    fld_vmax = V.robust_vmax([fc[m] for m in V.MODELS] + [t[valid]], 99.0)
    fld_norm = Normalize(0, fld_vmax) if is_wind else PowerNorm(0.6, 0, fld_vmax)
    err_vmax = V.robust_vmax([np.abs(fc[m][valid] - t[valid]) for m in V.MODELS], 98.0) or 1.0
    err_norm = Normalize(-err_vmax, err_vmax)

    lead = np.arange(49)
    if is_wind:
        # non-physical-structure index: % of domain on long, straight, grid-aligned edges.
        def roll_series(field, gthr=2.0, L=10):
            out = np.full(49, np.nan)
            for k in range(49):
                f = field[k]
                if not np.isfinite(f).any():
                    continue
                gx = np.zeros_like(f); gx[:, 1:] = np.abs(np.diff(f, axis=1))
                gy = np.zeros_like(f); gy[1:, :] = np.abs(np.diff(f, axis=0))
                vv = ndimage.binary_opening(gx >= gthr, structure=np.ones((L, 1)))
                hh = ndimage.binary_opening(gy >= gthr, structure=np.ones((1, L)))
                out[k] = 100.0 * (vv | hh).sum() / DOMAIN
            return out
        roll = {m: roll_series(fc[m]) for m in V.MODELS}
        for m in V.MODELS:
            roll[m][last + 1:] = np.nan  # stop at the analysis frontier; no extrapolation past obs
        roll_title, roll_ylabel = "non-physical structure vs lead", "straight-edge area (% dom)"
    else:
        def roll_series(field):
            out = np.full(49, np.nan)
            for k in valid:
                d = field[k] - t[k]
                out[k] = np.sqrt(np.nanmean(d * d))
            return out
        roll = {m: roll_series(fc[m]) for m in V.MODELS}
        roll_ref = None
        roll_title, roll_ylabel = "spatial RMSE vs lead", "RMSE (mm/h)"

    truthlbl = "PLACEHOLDER TRUTH" if args.placeholder else "ANALYSIS (ERA5)"
    var_row = "10 m wind speed" if is_wind else "hourly precipitation"
    vlabel = "10 m wind" if is_wind else "hourly rain"

    fig = plt.figure(figsize=(16, 7.4))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.05],
                  hspace=0.10, wspace=0.06, left=0.035, right=0.93, top=0.85, bottom=0.08)
    col_titles = [DISP.get(m, m) for m in V.MODELS] + [truthlbl]

    axes = {}
    def add_map(r, c, norm, cmap, title=None):
        ax = fig.add_subplot(gs[r, c]); axes[(r, c)] = ax
        mesh = ax.pcolormesh(V.LON_E, V.LAT_E, np.zeros((221, 241)), norm=norm, cmap=cmap, shading="flat")
        V.draw_coast(ax, segs); V.style_map(ax)
        if title:
            ax.set_title(title, fontsize=10)
        return mesh

    meshes = {}
    for c, m in enumerate(V.MODELS):
        meshes[("f", m)] = add_map(0, c, fld_norm, fld_cmap, col_titles[c])
    meshes[("f", "gt")] = add_map(0, 3, fld_norm, fld_cmap, col_titles[3])
    for c, m in enumerate(V.MODELS):
        meshes[("e", m)] = add_map(1, c, err_norm, V.ERR_CMAP)

    cax0 = fig.add_subplot(gs[0, 4])
    fig.colorbar(plt.cm.ScalarMappable(norm=fld_norm, cmap=fld_cmap), cax=cax0).set_label(units, fontsize=8)
    cax1 = fig.add_subplot(gs[1, 4])
    fig.colorbar(plt.cm.ScalarMappable(norm=err_norm, cmap=V.ERR_CMAP), cax=cax1).set_label(units, fontsize=8)

    axr = fig.add_subplot(gs[1, 3])
    axr.set_ylim(0, max(np.nanmax(v) for v in roll.values()) * 1.25 + 1e-3)
    for m in V.MODELS:
        axr.plot(lead, roll[m], color=V.MODEL_COLOR[m], lw=1.8, label=DISP.get(m, m))
    cur = axr.axvline(valid[0], color="k", lw=1.0, ls=":")
    axr.set_xlim(0, 48)
    axr.set_xlabel("lead hour", fontsize=8); axr.set_ylabel(roll_ylabel, fontsize=8)
    axr.tick_params(labelsize=7); axr.legend(fontsize=7, loc="upper left", framealpha=0.6)
    axr.grid(alpha=0.3); axr.set_title(roll_title, fontsize=9)

    fig.text(0.007, 0.63, var_row, rotation=90, va="center", ha="left", fontsize=10, weight="bold")
    fig.text(0.007, 0.27, "error: forecast − truth", rotation=90, va="center", ha="left",
             fontsize=10, weight="bold")
    wt_txt = axes[(0, 3)].text(0.5, 0.5, "", transform=axes[(0, 3)].transAxes, ha="center",
                               va="center", fontsize=12, color="#444", style="italic", zorder=6)
    suptitle = fig.suptitle("", fontsize=15, weight="bold")

    NAN = np.full(221 * 241, np.nan, np.float32)
    def update(k):
        has = bool(np.isfinite(t[k]).any())
        for m in V.MODELS:
            meshes[("f", m)].set_array(fc[m][k].ravel())
            meshes[("e", m)].set_array((fc[m][k] - t[k]).ravel() if has else NAN)
        meshes[("f", "gt")].set_array(t[k].ravel() if has else NAN)
        wt_txt.set_text("" if has else "no analysis yet")
        cur.set_xdata([k, k])
        valid_dt = (init + timedelta(hours=k)).strftime("%Y-%m-%d %H:%MZ")
        suptitle.set_text(f"{args.name} — {vlabel} vs {truthlbl}  |  init {args.init}   "
                          f"T+{k:02d}h   valid {valid_dt}")
        return []

    anim = FuncAnimation(fig, update, frames=49, blit=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=4000), dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
