# GPU mode — evolutionary GPU-kernel search

Artifacts from running **Fugu in "GPU mode"**: an evolutionary search that
optimizes a GPU kernel against a live [GPU MODE](https://www.gpumode.com/)
leaderboard. This folder ships the orchestration prompts, the evaluation harness
scripts, the shared seed kernel, and the head-to-head figure. It does **not**
include the per-candidate generation trees or either agent's evolved kernels —
both trajectories are shown via the figure and the results table below.

This demo compares two agents on the **identical** task and seed:

- **Fugu Ultra** (xhigh)
- **GPT-5.5** (xhigh)

In this comparison run, each model generated 5 generations of 10 candidates each, both at xhigh reasoning effort under the same Codex harness. As the figure below shows, Fugu Ultra discovers fast kernels 4.6× more effectively than GPT-5.5 for only 0.6× more cost.

<p align="center">
  <img src="qr_fugu_vs_oai.png" alt="Fugu Ultra (xhigh) vs GPT-5.5 (xhigh) on QR-kernel optimization" width="900">
</p>

Live leaderboard (QR decomposition, `qr_v2`, B200):
**<https://www.gpumode.com/leaderboard/774?tab=rankings>**

## Background

A **meta-agent** runs an evolutionary loop. It does **not** optimize the kernel
itself; each generation it launches a batch of independent Codex coding agents
("individuals"), each of which tries to make the kernel faster, evaluates
locally on a B200 GPU pool, and submits the best candidate to the remote
leaderboard. The accepted best is propagated forward as the seed for the next
generation. The full procedure is in [`meta_local.md`](meta_local.md) (Fugu
Ultra) and [`meta_local_oai.md`](meta_local_oai.md) (GPT-5.5); the only
difference between them is the agent launch line:

| Prompt | Agent | Launch |
|--------|-------|--------|
| [`meta_local.md`](meta_local.md) | Fugu Ultra | `codex -p fugu …` |
| [`meta_local_oai.md`](meta_local_oai.md) | GPT-5.5 | `codex -m gpt-5.5 -c model_reasoning_effort="xhigh" …` |

Both individuals are handed the same per-individual task prompt,
[`PROMPT_qr_v2_triton.md`](PROMPT_qr_v2_triton.md).

### The task

Optimize a **batched QR decomposition** kernel (the
[`qr_v2`](https://www.gpumode.com/leaderboard/774?tab=rankings) leaderboard)
written in **Triton**, run on an **NVIDIA B200**. Score is the **geomean**
wall-clock time (ms) across the leaderboard's benchmark shapes — **lower is
better**. Both runs start from the **byte-identical seed**
[`init.py`](init.py) (a naive unblocked Householder QR, **61.17 ms**), so it is
an apples-to-apples comparison.

## Results

Each run is **5 generations of 10 individuals**, starting from the same 61.17 ms
seed. The figure tracks the **best geomean per generation**, plus the dollar
cost reconstructed from Codex session token logs.

| Generation | Fugu Ultra (xhigh) | GPT-5.5 (xhigh) |
|-----------:|-------------------:|----------------:|
| 0 (seed)   | 61.17 ms | 61.17 ms |
| 1          | **13.43** | 42.44 |
| 2          | **6.43**  | 37.25 |
| 3          | **5.76**  | 27.00 |
| 4          | **5.23**  | 26.86 |
| 5          | **4.72**  | 26.62 |
| **5-gen speedup** | **12.9×** | **2.3×** |
| Cost (gens 1–5) | $163 | $99 |

Fugu Ultra reaches **5.6× faster** kernels by generation 5 (4.72 ms vs 26.62 ms)
because it switches to a structurally better algorithm — a WY block-reflector
panel with a cuBLAS batched trailing update — by generation 2, whereas GPT-5.5
stays on the seed's unblocked per-column Householder kernel and only tunes
tiles/warp counts, plateauing near 26.6 ms.

## Folder layout

```
gpu-mode/
├── README.md
├── qr_fugu_vs_oai.png        # the head-to-head figure above
├── init.py                   # shared seed kernel (61.17 ms) — both runs start here
├── meta_local.md             # meta-agent loop (Fugu Ultra variant)
├── meta_local_oai.md         # meta-agent loop (GPT-5.5 variant)
├── PROMPT_qr_v2_triton.md    # per-individual kernel-optimization prompt (shared)
├── evaluation.py             # local correctness + timing evaluator
├── local_eval.sh             # spreads evals across the B200 GPU pool
└── submission.py             # submits a kernel to the GPU MODE leaderboard
```

## Artifact files

### `init.py`

The shared seed both agents start from — a naive unblocked Householder QR in
Triton (geomean **61.17 ms**). Both runs are seeded from this identical file;
its header records the seed's speed and trick.

### Harness scripts

`evaluation.py`, `local_eval.sh`, and `submission.py` are the evaluation harness
the meta-agent drives (see [How to reproduce](#how-to-reproduce)).

## How to reproduce

The harness scripts are bundled here, but a run still needs the reference
problem, hardware, a venv, and (for submission/search) leaderboard access and
Codex. Nothing here hardcodes a path you must edit — every script takes its
paths as arguments or env vars.

### Requirements at a glance

| Need | Why | Used by |
|------|-----|---------|
| `gpu-mode/reference-kernels` checkout | provides the `qr_v2` problem (`task.yml`, `eval.py`, `reference.py`, `task.py`, `utils.py`) — **not** vendored here | every eval |
| NVIDIA GPU + CUDA toolkit (`nvcc` on `PATH`) | Triton/cuSOLVER kernels compile and run on-device; a **B200** matches the leaderboard | every eval |
| venv with `torch`, `triton`, `numpy`, `pyyaml` | the eval interpreter | every eval |
| `popcorn-cli` + registered account | remote leaderboard submit | `submission.py` |
| Codex (`codex -p fugu` / `-m gpt-5.5`) | drives the evolutionary loop | full search |

A **single local eval** (Quick start below) needs only the first three rows.
Submission and the evolutionary loop are optional on top of that.

### Quick start — one local eval on the seed kernel

The minimal runnable path: benchmark the bundled seed [`init.py`](init.py)
against the `qr_v2` problem, entirely offline.

```bash
cd demos/gpu-mode

# 1. get the problem definition (NOT vendored here)
git clone https://github.com/gpu-mode/reference-kernels.git

# 2. create the eval venv. Any torch+triton+CUDA build runs locally; use a
#    cu130 stack (torch 2.12 / CUDA 13.0 / triton 3.7) to track the leaderboard
#    — see the parity note in meta_local.md.
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch triton numpy pyyaml

# 3a. run it straight through evaluation.py (no GPU pool, no locks)
.venv/bin/python evaluation.py init.py \
    --problem-dir reference-kernels/problems/linalg/qr_v2 \
    --mode benchmark

# 3b. OR via the multi-GPU wrapper: auto-picks a free GPU, enforces the
#     leaderboard's 300 s budget, and tees a log to ./logs/
./local_eval.sh init.py \
    --problem-dir reference-kernels/problems/linalg/qr_v2 \
    --mode benchmark
```

- `evaluation.py` uses whatever interpreter you launch it with.
- `local_eval.sh` defaults to `./.venv/bin/python`; override with
  `FUGU_PYTHON=/path/to/other/bin/python ./local_eval.sh …` (e.g. to point at a
  separate cu130 venv). It also auto-discovers all GPUs `nvidia-smi` reports and
  gates each with its own `flock`, so many evals can run concurrently.
- **Modes:** `test` (correctness only, fast), `benchmark` (timing, default),
  `leaderboard` (timing + per-run correctness re-checks).

### Submit to the live leaderboard

```bash
# one-time: install popcorn-cli and register (writes ~/.popcorn.yaml)
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord

# submit the seed (qr_v2 on B200, ranked); the remote board is the
# authoritative acceptance gate
.venv/bin/python submission.py init.py
```

### Full reproduction — the evolutionary search

The 5-generation search additionally needs **Codex** and a task workspace.

1. Set up a task workspace with the seed [`init.py`](init.py) and the shared
   prompt [`PROMPT_qr_v2_triton.md`](PROMPT_qr_v2_triton.md).
2. Point the meta-agent at [`meta_local.md`](meta_local.md) (Fugu Ultra) or
   [`meta_local_oai.md`](meta_local_oai.md) (GPT-5.5) and answer its Step-0
   questions (task, generations, individuals per generation, GPU pool).
3. Per generation the meta-agent: copies the parent kernel into each individual
   folder, launches the Codex individuals in parallel, evaluates each via
   `local_eval.sh` / `evaluation.py`, submits the best-ranked candidate with
   `submission.py`, propagates the accepted kernel forward, and appends a row to
   `meta_history.csv`.

> **Heads up — absolute paths in the meta prompts.** `meta_local*.md` reference
> absolute paths from the original Fugu workspace
> (`/home/lfsm/code/fugu_on_gpu_mode/…`). Adjust those to wherever you place the
> harness, the cloned `reference-kernels`, and the task workspace. The harness
> scripts themselves need no editing — they take all paths as arguments.

See the prompts themselves for the exact per-generation procedure, GPU-pool
gating, leaderboard-stack parity, and submission/retry rules.
