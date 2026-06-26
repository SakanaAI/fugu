"""SAL (Structure / Amplitude / Location; Wernli et al. 2008) for the 24h prediction.

Each component is 0 when perfect:
  A in [-2,2]  amplitude bias   (+ = forecast too wet/strong, - = too dry/weak)
  L in [0,2]   location error   (0 = storm placed correctly)
  S in [-2,2]  structure        (+ = objects too large/flat/widespread, - = too peaked/small)

Precip uses the 24h-accumulated field (SAL's natural use); wind uses the 24h-mean field.
Objects = contiguous regions >= (1/15)*field max, per the standard SAL threshold.
"""
import numpy as np, xarray as xr
from scipy import ndimage

z = np.load("comparison/round2/data/ground_truth_now_partial.npz", allow_pickle=True)
tw, tp = z["wind_speed"], z["precipitation"]
valid = [k for k in range(49) if np.isfinite(tw[k]).any()]
steps = [k for k in valid if k <= 24]          # the 24h window
ny, nx = 221, 241
d = float(np.hypot(ny - 1, nx - 1))            # max distance in the domain (cells)

def com(field):
    tot = field.sum()
    if tot <= 0:
        return None
    ys, xs = np.indices(field.shape)
    return np.array([(ys * field).sum() / tot, (xs * field).sum() / tot])

def objects(field, f=1.0 / 15.0):
    lab, n = ndimage.label(field >= f * field.max())
    return [field * (lab == i) for i in range(1, n + 1)]

def sal(fc, ob):
    Df, Do = fc.mean(), ob.mean()
    A = (Df - Do) / (0.5 * (Df + Do)) if (Df + Do) > 0 else np.nan
    of, oo = objects(fc), objects(ob)
    def Vscaled(objs):
        num = den = 0.0
        for o in objs:
            Rn, mx = o.sum(), o.max()
            if Rn <= 0 or mx <= 0:
                continue
            num += Rn * (Rn / mx); den += Rn
        return num / den if den > 0 else 0.0
    Vf, Vo = Vscaled(of), Vscaled(oo)
    S = (Vf - Vo) / (0.5 * (Vf + Vo)) if (Vf + Vo) > 0 else np.nan
    cf, co = com(fc), com(ob)
    if cf is None or co is None:
        return S, A, np.nan
    L1 = np.hypot(*(cf - co)) / d
    def r_avg(objs, c):
        num = den = 0.0
        for o in objs:
            Rn = o.sum(); oc = com(o)
            if Rn <= 0 or oc is None:
                continue
            num += Rn * np.hypot(*(oc - c)); den += Rn
        return num / den if den > 0 else 0.0
    L2 = 2.0 * abs(r_avg(of, cf) - r_avg(oo, co)) / d
    return S, A, L1 + L2

files = {"claude": "claude_now.nc", "fugu": "fugu_now.nc", "fugu-ultra": "ultra_now.nc"}
print(f"24h window: T+{steps[0]}..T+{steps[-1]}h\n")
print("PRECIP (24h accumulation) — SAL")
print(f"  {'model':11s}   S(struct)  A(amplitude)  L(location)")
for m, fn in files.items():
    ds = xr.open_dataset(f"comparison/round2/data/{fn}"); p = ds.precipitation.values; ds.close()
    S, A, L = sal(np.nansum(p[steps], 0), np.nansum(tp[steps], 0))
    print(f"  {m:11s}   {S:+.2f}      {A:+.2f}        {L:.2f}")
print("\nWIND (24h mean) — SAL  [note: SAL assumes a mostly-zero field; wind has no zero background]")
print(f"  {'model':11s}   S(struct)  A(amplitude)  L(location)")
for m, fn in files.items():
    ds = xr.open_dataset(f"comparison/round2/data/{fn}"); w = ds.wind_speed.values; ds.close()
    S, A, L = sal(np.nanmean(w[steps], 0), np.nanmean(tw[steps], 0))
    print(f"  {m:11s}   {S:+.2f}      {A:+.2f}        {L:.2f}")
