# Typhoon Forecast

Three coding agents were each given the same prompt (`prompt.md`) to build a
production-grade, real-data numerical weather prediction (NWP) and
machine-learning pipeline that forecasts 48 hours of 10 m wind and hourly
precipitation over Japan. We compare what they built on a real event, the
convergence of tropical storms Mekkhala and Higos in late June 2026.

## Agents

| Folder | Agent | Model / settings |
|--------|-------|------------------|
| `claude_opus48_high/` | Claude | Opus 4.8, high reasoning effort |
| `codex_fugu_xhigh/` | fugu (Codex wrapper) | xhigh reasoning effort |
| `codex_fugu-ultra_xhigh/` | fugu-ultra (Codex wrapper) | xhigh reasoning effort |

## The task

The task was to create a forecasting model specific to typhoons in Japan, based
on real weather forecasts. Each agent had to ingest real operational forecast
grids from three independent centers (JMA / ECMWF / NOAA-GFS, or documented
substitutes), regrid onto a uniform 0.1° mesh over Japan (24–46°N, 122–146°E;
49 hourly steps T+0…T+48), condition on a real NINO3.4 ENSO index, train an ML
error-correction layer against real ERA5 observations with a causal/rolling
training cutoff, and export a NetCDF tensor plus a provenance manifest. The
full spec is in `prompt.md`. The prompt forbids mocking: every forecast field,
the ENSO index, and the ML training labels must be real, retrieved data,
causally fenced (nothing dated after the forecast initialization). 

## What each agent built

| Agent | 3rd center (JMA substitute) | ML corrector | Forecast feeds |
|-------|------------------------------|--------------|----------------|
| Claude | CMC GDPS | XGBoost (per-cell; wind residual + log-precip) | GFS (AWS), ECMWF IFS open-data |
| fugu | DWD ICON | ridge regression (multi-task linear) | GFS (NOMADS), ECMWF IFS open-data |
| fugu-ultra | DWD ICON | ridge regression (multi-task linear) | GFS (NOMADS), ECMWF IFS open-data |

All three train their corrector on real ERA5 labels and verify out-of-sample.

## Results & observations

We score predictions after 24 hours (init 2026-06-25 06:00Z) compared to ERA5
with object-based metrics that separate the kinds of error. 

**Wind — MODE** on the 24 h peak-wind swath (≥ 15 m/s):

| Agent | location (km) | area ratio | peak bias (m/s) |
|-------|--------------:|-----------:|----------------:|
| Claude | 126 | 0.91 | +0.4 |
| fugu | **87** | 1.28 | +5.4 |
| fugu-ultra | 199 | 2.92 | +11.7 |

*area ratio 1 = right coverage; peak bias = 95th-pct wind vs analysis.*
**fugu** places the swath best (87 km) but has a stronger peak bias; Claude
gets coverage and intensity nearly right, but not location; fugu-ultra
over-intensifies with ~3× too much strong-wind area and +11.7 m/s too strong.
Overall, **fugu** has the best prediction of wind at 24 hours.

**Realism (separate from skill):** Claude's wind field also contains a large
rectangular block of anomalous wind next to the Korean coast, a CMC GDPS
tile/regridding artifact. This is a physicality problem, not a skill one, and
difficult to represent in error metrics like MODE/RMSE, but is obvious in video
forecast. The non-physical-structure index, a metric based on the shape of the
prediction, is ~10× the others while the ERA5 analysis sits at ~0.

**Precipitation — SAL** (Structure / Amplitude / Location; each 0 = perfect):

| Agent | S (structure) | A (amplitude) | L (location) |
|-------|--------------:|--------------:|-------------:|
| Claude | +0.09 | −0.21 | 0.14 |
| fugu | +0.09 | −0.05 | 0.16 |
| fugu-ultra | +0.46 | +0.05 | 0.18 |

*A > 0 too wet, S > 0 rain too spread/flat, L = displacement.* **fugu** is the most balanced
(near-zero amplitude bias); Claude under-predicts total rain (A −0.21); fugu-ultra spreads
the rain too broadly (S +0.46). Rain placement is comparable for all three.

## Videos

Two animations are provided separately:

- **wind** — each agent's 10 m wind vs the ERA5 analysis, error maps, and a non-physical-structure
  index vs lead time.
- **rain** — each agent's hourly precipitation vs the analysis, error maps, and spatial RMSE vs lead.

Regenerate with `comparison/animate_verify.py` after producing each agent's forecast and fetching
ERA5 truth with `comparison/fetch_gt_verify.py`. The SAL and MODE tables above are produced by
`comparison/sal.py` (precip) and `comparison/mode.py` (wind).

## Folder layout

```
typhoon/
├── prompt.md                  shared task given to all three agents
├── claude_opus48_high/        Claude's pipeline (XGBoost; CMC GDPS)
├── codex_fugu_xhigh/          fugu's pipeline (ridge; DWD ICON)
├── codex_fugu-ultra_xhigh/    fugu-ultra's pipeline (ridge; DWD ICON)
└── comparison/                verification + animation scripts
```

## Reproduce

To create a forecaster, instruct agents to follow the prompt in `prompt.md`. To
use the existing forecast scripts, follow the `README.md` provided in each
agent folder. Trained weights and downloaded data are not committed; the
training scripts regenerate them.

