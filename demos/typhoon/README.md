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
Claude and fugu fetch filtered prediction data (ECMWF at 10u/10v/tp only),
which can result in glitches that intermittently drop fields; occasionally this
can lose an entire prediction. fugu-ultra's pipeline fetches a full ECMWF
open-data file and filters afterwards, so it does not exhibit this problem.
While fugu-ultra's prediction did not perform as well due to a difference in
calibration parameters, fugu-ultra built the most reliable forecasting code.

## Results & observations

We verify the Jun 25 06:00Z forecast against ERA5 at two horizons: 24 and 48
hours. 48 hours is a very long horizon without re-forecasting, especially for a
dynamic storm system, but it is included to compare the models built by each
agent on long-horizon chaotic systems forecasting. All metrics are
object-/shape-based:

Wind columns — **loc** (MODE storm-center distance, km), **area** (strong-wind coverage ratio,
1 = right), **peak** (95th-pct wind bias, m/s), **corr** (spatial pattern correlation, ↑),
**spread** (strong-wind effective radius vs analysis; closer = better), **SSIM** (↑).
Rain — **SAL** (Structure / Amplitude / Location, each 0 = perfect; A > 0 too wet, S > 0 rain too
spread/flat, L = displacement).

### 24 hours

**Wind** (analysis strong-wind radius ≈ 169 km)

| Agent | loc km | area | peak | corr | spread | SSIM |
|-------|-------:|-----:|-----:|-----:|-------:|-----:|
| Claude | 129 | 0.81 | **+0.3** | 0.933 | 227 (1.35×) | **0.778** |
| fugu | **91** | 1.15 | +5.4 | **0.944** | **199 (1.18×)** | 0.746 |
| fugu-ultra | 202 | 2.61 | +11.6 | 0.912 | 259 (1.53×) | 0.727 |

**Rain — SAL**

| Agent | S | A | L |
|-------|--:|--:|--:|
| Claude | +0.08 | −0.23 | 0.13 |
| fugu | +0.08 | **−0.06** | 0.15 |
| fugu-ultra | +0.45 | +0.04 | 0.18 |

On **wind**, fugu is the most trustworthy: best center location (91 km) and
best coherence (corr 0.944, tightest spread 1.18×). Fugu predicts the correct
storm center, just too intense (+5.4). Claude predicts storm intensity (+0.3)
well and scores highest on SSIM, but places the center worse and spreads it
wider (1.35×); fugu-ultra badly over-intensifies (+11.6, 2.6× coverage) and is
the most smeared. The scalar metrics don't fully agree (SSIM favours Claude;
correlation/spread favour fugu), but animation of the predictions shows
**fugu** to have a reliable forecast. On **rain**, fugu is the most balanced
(amplitude −0.06); Claude predicted too little rain (-0.23) but is the most
accurately placed (0.13); fugu-ultra's rain is good on amount and placement,
but is too spread (S +0.45).

### 48 hours

**Wind** (analysis strong-wind radius ≈ 253 km)

| Agent | loc km | area | peak | corr | spread | SSIM |
|-------|-------:|-----:|-----:|-----:|-------:|-----:|
| Claude | 96 | 0.83 | −1.0 | 0.914 | 259 (1.02×) | 0.751 |
| fugu | 145 | 1.23 | +2.2 | 0.926 | 240 (0.95×) | 0.717 |
| fugu-ultra | 432 | 1.54 | +9.3 | 0.891 | 290 (1.14×) | 0.699 |

**Rain — SAL**

| Agent | S | A | L |
|-------|--:|--:|--:|
| Claude | −0.41 | −0.23 | 0.09 |
| fugu | +0.22 | −0.08 | 0.07 |
| fugu-ultra | −0.30 | +0.01 | 0.10 |

Two days out, the complex Mekkhala–Higos interaction makes all three
predictions diverge further from observation. fugu-ultra's storm center is
measured at 432 km off, as it spreads wind along the coast. Claude's rain
structure collapses, although it predicts the closest storm center in wind.
fugu predicts a much more intense storm than what was observed, but preserves
the storm shape the best. As noted above, prediction out to this long horizon
without integration of more recent data is very difficult and unlikely
operationally, but it does illustrate the predictive capabilities of these
coding agents.

Finally, through the prediction, Claude's wind field contains a large
rectangular block of anomalous wind next to the Korean coast. This is an
artifact from the data loading section of the script that Claude wrote, an
error in CMC GDPS tile/regridding. This makes the result look physically
implausible, but it is difficult to represent in the error metrics. This issue
is obvious in the video, where its non-physical-structure index runs ~10× the
others while the ERA5 analysis sits at ~0.

## Videos

Two animations are provided separately:

- **wind** — each agent's 10 m wind vs the ERA5 analysis, error maps, and a non-physical-structure
  index vs lead time.
- **rain** — each agent's hourly precipitation vs the analysis, error maps, and spatial RMSE vs lead.

Regenerate the videos with `comparison/animate_verify.py` after producing each agent's forecast and
fetching ERA5 truth with `comparison/fetch_gt_verify.py`. The score tables come from
`comparison/mode.py` (wind MODE) and `comparison/sal.py` (rain SAL); the wind shape metrics
(pattern correlation, spread, SSIM) are computed alongside.

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

