# Task — optimize `qr` (batched Householder QR) on B200 — **Triton variant**

Optimize a single GPU kernel: **batched square compact-Householder QR
factorization**. For each matrix in a batch, factor `A = Q R` and return the
result in **exactly `torch.geqrf(A)`'s compact convention**.
Target: **NVIDIA B200 (sm_100)**. Ranking is geomean of runtime over the 12
benchmark shapes, among submissions that pass correctness **and finish the full
evaluation within the leaderboard's hard 300 s server budget** — a real gate that
silently rejects otherwise-fast kernels (see "⚠️ Submission budget" below).

Language: **Triton**, via `@triton.jit` kernels launched from Python.
`torch` is allowed for glue (allocation, reshaping, dense `matmul`/BLAS building
blocks, `torch.linalg`). You may also write helper `@triton.jit` kernels and
compose them. (This is the Triton sibling of `PROMPT_qr_v2_cuda.md`; the task,
gate, baseline, and eval flow are identical — only the kernel language differs.)

> Policy — shape-dispatch hybrid is allowed (and encouraged): `torch.geqrf`
> (cuSOLVER) is a **legal per-shape fallback**. You may branch on `(batch, n)`
> and dispatch to your own custom Triton kernel on some shapes and to
> `torch.geqrf` on others. What is **not** allowed is the trivial no-op of
> returning `torch.geqrf(A)` for *every* shape — that just reproduces the
> baseline and scores 0 improvement.
>
> The win comes from your custom kernel on the shapes where cuSOLVER is weak —
> the batched small/medium cases, notably `(640,512)` (>1 s baseline), `(60,1024)`,
> and `(40,352)`. cuSOLVER is genuinely strong on the
> **few-large** shapes (`(2,4096)`, `(8,2048)`); falling back to `torch.geqrf`
> there is fine and usually optimal. Net effect: geomean can only improve over
> the 131.16 ms baseline, because you only replace cuSOLVER where you are faster.
> `torch.linalg.qr` / `householder_product` / `ormqr` / `orgqr` are also
> available if useful, but `torch.geqrf` already returns the exact `(H, tau)`
> contract for the fallback path.

## ⚠️ Submission budget — the 300 s wall-clock gate (read this; it decides acceptance)

The leaderboard server kills any submission whose **full evaluation** (compile +
22 tests + 12 benchmarks) exceeds a **hard 300 s function budget** with a
`FunctionTimeoutError`. A kernel that times out is **not ranked at all** — a great
geomean is worthless if the eval doesn't finish in time.

**The budget is driven by your kernel's timing *variance*, not by its geomean.**
The benchmark harness re-runs each shape until the mean is stable (relative
std-error < 0.1 %), so the repeat count scales with run-to-run jitter as
**≈ (CoV / 0.1 %)²**. A *steady* kernel needs ~3 repeats per shape; a *noisy* one
needs ~250 — even when each call is faster. Measured on the CUDA sibling of this
task (the variance mechanism is harness-level and identical in Triton):

| kernel | per-call speed | n=512 jitter | benchmark repeats | full-eval wall-clock | result |
|---|---|---|---|---|---|
| steady (modest `num_warps`) | 101 ms | 0.1 % CoV | 3 | 58 s | ✅ accepted |
| fast-but-noisy (max `num_warps`, deep `num_stages`) | 73 ms (faster!) | 1.7 % CoV | ~250 | 214–368 s | ❌ FunctionTimeoutError |

The **faster** kernel was rejected. **Optimize for low runtime AND low timing
variance.** What creates the killer jitter, and how to avoid it:

- **Too many warps / deep pipelines per program.** One program per matrix running
  a long sequential column loop with a large `num_warps` and deep `num_stages` has
  jittery warp scheduling → high CoV. Prefer **modest `num_warps` (4–8)** and
  **fewer, larger steps** (blocked/panel Householder: factor a panel, then one
  tiled trailing update, instead of touching the whole trailing matrix per column).
- **Python-side launch loops.** Launching one Triton kernel per column from Python
  (~2000 launches/call) is slow on the server *and* jittery. Keep each matrix's
  factorization inside **one or a few kernel launches** — a runtime column loop
  *inside* the kernel, not a Python `for` loop of launches.
- **Wave-tail effects.** A batch grid that doesn't tile evenly over the SMs leaves
  a partial last wave whose timing wobbles; grid-stride the batch so waves are full.
- **Autotune jitter.** Re-autotuning on the timed path adds variance; warm up once
  so the chosen config is cached before the measured run.
- **Goal:** keep n=512 run-to-run **CoV under ~0.3 %** (repeats stay < ~10).

**You MUST verify this before finalizing** with `--mode leaderboard` (see "How to
evaluate"). `local_eval.sh` now enforces the 300 s budget by default and prints
`[eval-walltime]`; a `[budget WARNING]` or `[SERVER TIMEOUT]` means the kernel
will be rejected on submission no matter how good its geomean. **A steady kernel
that finishes well under budget beats a faster one that rides the budget.**

> ⚠️ **Triton reality check.** Householder QR splits into a *sequential* panel
> factorization (per-column reflector → norm → apply, with a cross-column
> dependency) and a *parallel* trailing GEMM update. Triton is excellent at the
> GEMM half — but `torch.matmul` already hits cuBLAS there, so Triton adds little.
> The lever in this benchmark (batched-medium `(640,512)`) lives in the sequential
> half, which is exactly where Triton's per-program / fixed-tile model is most
> awkward (no exposed block-barrier across a long sequential loop; a whole 512×512
> matrix does not fit in one program's SRAM). Expect a Triton kernel to be
> **correct and to beat cuSOLVER, but to trail a hand-tuned CUDA kernel.** Lean
> into Triton's strengths: tiled trailing updates, `tl.dot` for panel/WY GEMMs,
> autotuned block sizes.

## Inputs / outputs (per `reference.py` + `task.py`)

- `input` (`data`): `A`, a `float32` tensor of shape `(batch, n, n)`, contiguous,
  on `cuda`. **Treat it as read-only** (see the in-place gotcha below).
- `output`: a tuple `(H, tau)`:
  - `H`: `float32`, shape `(batch, n, n)` — `R` in the **upper triangle**,
    Householder reflector vectors **below** the diagonal (geqrf packing).
  - `tau`: `float32`, shape `(batch, n)` — reflector coefficients.

The checker materializes `Q = torch.linalg.householder_product(H, tau)` and
`R = triu(H)`. **You must return the compact reflector form**, not `Q`/`R`
directly — computing `Q,R` by Gram–Schmidt and returning them will fail unless
you convert to reflectors (which costs as much as just doing Householder). This
strongly favors an actual Householder-based algorithm.

Shapes evaluated:
- test → 22 cases, `n ∈ {32,176,352,512,1024,2048,4096}` with stress structures
  (including `mixed` heterogeneous-conditioning batches).
- benchmark → 12 timing shapes (same `n` values, at production batch sizes).
  Ranking is geomean over them.
- The batched medium shapes `(640,512)` and `(60,1024)` are the heaviest terms in
  the cuSOLVER baseline, so they are the biggest levers. The benchmark includes
  structured variants of both (`mixed`/`rankdef`/`clustered`/`nearrank`), so your
  custom path must stay correct on ill-conditioned / rank-deficient batches, not
  just plain dense ones.

> Note: passing the orthogonality gate alone does **not** mean your factorization
> is correct. `QᵀQ ≈ I` only proves each `(v, tau)` is a valid reflector; the
> *reconstruction* gate `triu(H) ≈ Qᵀ@A` is what catches a buggy trailing update.
> Check both when debugging.

## Correctness (hard gate — relative tolerance, measured in FP64)

The returned factors must satisfy, per matrix (L1 matrix norms):
- **Factor residual:** `‖triu(H) − Qᵀ@A‖₁ ≤ 20·n·eps32 · ‖A‖₁`
- **Orthogonality:**   `‖QᵀQ − I‖₁ ≤ 100·n·eps32 · ‖I‖₁`

`eps32 ≈ 1.19e-7`. There is **no `atol`** — tolerance is purely relative, so you
need genuine FP32 accuracy. Low-bit (FP16/FP8/NVFP4/TF32) is allowed **only as an
internal strategy**; returned `H`/`tau` must be FP32 and meet these gates.

> ⚠️ Triton's `tl.dot` defaults to TF32 inputs on recent hardware
> (`allow_tf32=True` / `input_precision="tf32"`). The QR gate is tight and has no
> `atol`; TF32 reductions can fail orthogonality / reconstruction on
> ill-conditioned cases. Use `tl.dot(..., allow_tf32=False)` (or
> `input_precision="ieee"`), or accumulate dot products in fp32 with explicit
> error correction. Validate on every stress case.

Stress cases your kernel must survive (all `n=512` or larger):
`dense` (wide dynamic range via `cond`), `rankdef`, `nearrank`, `band`,
`rowscale`, `nearcollinear`, `clustered`, `upper`, and `mixed`. **These break
naive Cholesky-QR / classical Gram–Schmidt** (loss of orthogonality on
ill-conditioned / rank-deficient input). If you use a Cholesky-QR-style fast
path, you must make it robust (e.g. shifted CholeskyQR2, or fall back to
Householder) or you will FAIL the gate. Householder QR is unconditionally stable
and is the safe default.

> The `mixed` case is a **heterogeneous batch**: each matrix gets an independent
> conditioning profile (dense/rankdef/nearrank/clustered/band/rowscale/
> nearcollinear) at a random position in the batch. It specifically defeats the
> trick of sampling a few matrices, deciding the whole batch is well-conditioned,
> and routing it all to a fast path that is only valid for well-conditioned
> inputs. You must handle **each matrix on its own merits** — a per-matrix
> robustness decision, not a per-batch one. `mixed` shapes appear in both the
> test set and the ranked benchmark, so getting them wrong costs both correctness
> and score.

## API + template (start here)

```python
# Required shim: the harness imports this module inside a multiprocessing worker
# where sys.stdout/sys.stderr are None. Restore them before importing torch so
# any library that flushes during import does not crash on None.flush().
import sys, io
if sys.stdout is None: sys.stdout = io.StringIO()
if sys.stderr is None: sys.stderr = io.StringIO()

from task import input_t, output_t
import torch
import triton
import triton.language as tl

@triton.jit
def qr_kernel(H_ptr, Tau_ptr, n, stride_b, stride_i, stride_j, stride_tb,
              BLOCK_M: tl.constexpr, BLOCK_J: tl.constexpr):
    # ... your batched Householder QR in Triton here ...
    # Tip: one program per matrix (grid=(batch,)) with a runtime `for k in range(n)`
    # column loop keeps the loop a real loop (not unrolled). Keep `n` a runtime arg,
    # NOT a tl.constexpr, or a constexpr `range(n)` will fully unroll and blow up
    # compile time for n=512/1024.
    pass

def custom_kernel(data: input_t) -> output_t:
    A = data                      # (batch, n, n) float32 cuda — DO NOT mutate
    b, n, _ = A.shape
    # Shape dispatch: custom Triton kernel where it beats cuSOLVER, geqrf else.
    if n in (32, 176, 352, 512):  # tune this predicate from the benchmark numbers
        H = A.clone()             # factor a copy; never overwrite the input
        Tau = torch.empty((b, n), dtype=A.dtype, device=A.device)
        BLOCK_M = 1 << ((n) .bit_length())   # power of two strictly > n (guard row)
        qr_kernel[(b,)](H, Tau, n,
                        H.stride(0), H.stride(1), H.stride(2), Tau.stride(0),
                        BLOCK_M=BLOCK_M, BLOCK_J=16, num_warps=8)
        return H, Tau
    return torch.geqrf(A)         # cuSOLVER is strong on few-large matrices
```

Rules:
- Single Python file. Triton kernels JIT-compile **lazily on first call** (and
  autotune, if you use `@triton.autotune`), so the first timed run pays a
  one-time cold-compile cost — warm up once if it skews your local reading.
- Keep the stdout/stderr shim at the very top — it is **required for local
  `local_eval.sh`** (its worker has `sys.stdout = None`). You do **not** remove
  it: the meta-agent strips it automatically before leaderboard submission (see
  "Leaderboard submission rules").
- **Give each iteration a fresh kernel identity** so a stale compile/autotune
  cache is never silently reused: rename the `@triton.jit` function per iteration
  (e.g. `qr_kernel_v0003`) — this is the Triton analogue of the CUDA `name=` tag.
  If you use `@triton.autotune`, vary its `key=`/configs deliberately rather than
  copying a previous iteration's cache.
- **FP32 accuracy:** disable TF32 in `tl.dot` (`allow_tf32=False` /
  `input_precision="ieee"`) unless all 22 test cases still PASS with it on.
- ⚠️ **Guard-row gotcha (verified on this box):** an exact-fit row tile
  (`BLOCK_M == n`, hit at n=32 and n=512) miscompiles the axis-0 reduction in the
  trailing update and silently corrupts the factorization (orthogonality still
  passes, reconstruction fails). Pad `BLOCK_M` to the next power of two **strictly
  greater than n** so there is always ≥1 masked guard row, and mask all
  loads/stores with `offs < n`.
- ⚠️ **Do not mutate the input `A` in place.** The benchmark loop reuses the same
  input tensor across all timed repeats *without re-cloning*; an in-place
  destroy corrupts repeats 2..N and fails the leaderboard `recheck`. Factor a
  `A.clone()` (or a fresh scratch buffer), not `A` itself.

### ⚠️ Leaderboard submission rules (or the kernel is rejected before it runs)

The leaderboard runs a **static source scan** and rejects a submission in ~0.5 s,
*before compiling or running it*, with the error **"your code contains work on
another stream"** if the file's *source text* contains certain patterns:

1. **Explicit CUDA stream usage (a GPU stream).** Literal
   `at::cuda::getCurrentCUDAStream()`, `cudaStream_t`, or a 4-argument launch
   `kernel<<<grid, block, 0, stream>>>`. **Triton does not emit any of these
   strings** — a `@triton.jit` kernel launched as `kernel[grid](...)` runs on the
   current CUDA stream internally but the *source* is clean, so it passes the
   scan. Do **not** hand-manage streams (`torch.cuda.Stream`, `stream=` kwargs);
   just launch `kernel[grid](...)` and let it use the default stream.
2. **The stdout/stderr shim (a std stream — unrelated to GPU streams).**
   `sys.stdout = io.StringIO()`. It is needed for local eval but trips this filter
   when a custom kernel is present. **Keep it in your working file** (local eval
   needs it); the **meta-agent strips it before remote submission**, so leave it in.

So: write plain Triton (`kernel[grid](...)`, no explicit streams) and keep the
shim as-is. `torch.geqrf` and other `torch`/cuSOLVER calls are **not** flagged and
remain a legal per-shape fallback. (The Triton import / `@triton.jit` decorator
itself is not flagged.)

### Possible algorithmic directions (feel free to use your own if you have a better idea)
- **Shape dispatch (do this first)**: branch on `(batch, n)` — route the batched
  small/medium shapes (esp. `(640,512)`) to your custom kernel and the few-large
  shapes (`(2,4096)`, `(8,2048)`) to `torch.geqrf`. Locks in geomean ≤ baseline
  immediately, then improve the custom path iteration by iteration.
- **Tiled trailing update**: the seed re-streams the whole trailing matrix from
  HBM every column. Block the column loop (panel of `b` columns), accumulate the
  block reflector `(I − V T Vᵀ)`, and apply the trailing update as a `tl.dot`
  GEMM over tiles — this is where Triton is strong. Standard LAPACK `geqrf` shape.
- **Add the n=1024 path**: the seed falls back to `torch.geqrf` for n=1024 (three
  `(60,1024)` cases, ~242 ms each — the second-biggest lever). A correct Triton
  n=1024 path is an obvious early win.
- **Mixed precision internally**: BF16/TF32 trailing GEMMs with FP32 accumulation
  and a correction pass — but keep the final factors at the FP32 gate. Validate on
  the ill-conditioned stress cases.
- **Batched-aware scheduling**: for the batched medium shapes, tune
  `num_warps`/`num_stages`/block sizes so each matrix's program saturates an SM
  and many run concurrently.

## Where to write kernels

Write the kernel directly into your current workspace (the directory you were
started in). Do not create a `kernels_b200/qr/` subfolder. Do not write to `/tmp`
or anywhere else.

### ⛔ Workspace confinement — no external solutions (hard rule)

Your solution must be derived **only** from `init.py` plus your own iterations in
this workspace. You may read **only**: files in your current workspace (`init.py`,
`PROMPT`, `history.csv`, your own `v00*.py`) and the read-only reference problem
files needed to evaluate (`task.py`, `reference.py`, and the `reference-kernels/`
problem dir). **Everything else is off-limits.**

In particular you may **not** read, open, `cat`, `grep`, `find`, list, or copy
from — directly or via any tool/subprocess — any of:

- **other task workspaces** under `kernels_b200/` (e.g. any sibling
  `kernels_b200/qr_*` directory other than your own), or any other run's
  generations / individuals;
- any `best.py`, `final.py`, or `v00*.py` belonging to a different workspace or a
  different agent;
- any pre-existing optimized QR solution anywhere on disk or in git history.

Do not search the filesystem for faster QR implementations to port in. A diff
shows when an iteration matches an outside solution; **copying or "porting" an
existing solution is cheating and disqualifies the run.** If your `v0001` jumps to
a fully blocked/WY/hybrid kernel in one step without an incremental edit from
`init.py`, that is the signature of a copy and will be rejected. Build up the
optimization yourself, one verifiable iteration at a time.

### Start from `init.py`

Your workspace contains **`init.py`** — a verified passing Triton baseline (PASS,
geomean 61.17 ms on B200, qr_v2 12-shape benchmark). It is
a **naive unblocked Householder QR in Triton**: one program per matrix, a runtime
column loop, the trailing matrix re-streamed from HBM each column. It routes
`n ∈ {32,176,352,512}` to the Triton kernel and `n ∈ {1024,2048,4096}` to
`torch.geqrf`. **Do not modify `init.py`.** Your iteration 1 (`v0001_*.py`) copies
`init.py`'s shim + Triton scaffold, renames the `@triton.jit` function, and
applies your first optimization (e.g. a tiled/blocked trailing update, or adding
the n=1024 path).

Treat `init.py`'s numbers as the floor you must beat — if `v0001` doesn't improve
geomean, the change isn't paying for itself.

Every iteration file starts with a 3-line header (canonical record is in
`history.csv` — keep this terse):

```python
# v0003_20260616_111245.py  |  parent: v0002_20260616_104812.py
# status: PASS  |  geomean: 41.2 ms
# trick: blocked Householder, panel=32, WY trailing update via tl.dot (allow_tf32=False)
```

## History (mandatory)

Maintain `history.csv` in your current workspace with this exact header (create
it on iteration 1 if missing, and seed it with the `init.py` baseline row):

```csv
iteration,timestamp_utc,filename,status,geomean_ms,parent,trick
0000,2026-06-16T14:20:00Z,init.py,PASS,61.1692,none,naive unblocked Householder QR in Triton (one program per matrix); custom n in {32,176,352,512}, torch.geqrf fallback otherwise
```

Append **one row per iteration** immediately after eval completes. Empty strings
for timing fields on FAIL rows. Quote the `trick` cell if it contains commas.
The CSV is your source of truth for "what worked" — consult it before each
iteration so you don't re-try a failed idea.

## How to evaluate (local B200)

```bash
LOCAL_EVAL=/home/lfsm/code/fugu_on_gpu_mode/local_eval.sh
PROBLEM=/home/lfsm/code/fugu_on_gpu_mode/reference-kernels/problems/linalg/qr_v2
KERNEL="$PWD/v0003_20260616_111245.py"   # kernel file in your current workspace

$LOCAL_EVAL "$KERNEL" --problem-dir "$PROBLEM" --mode test        --time 00:09:00  # 22 correctness cases
$LOCAL_EVAL "$KERNEL" --problem-dir "$PROBLEM" --mode benchmark   --time 00:09:00  # 12 timing cases + geomean
$LOCAL_EVAL "$KERNEL" --problem-dir "$PROBLEM" --mode leaderboard --time 00:09:00  # FULL server replica — must finish under the 300 s budget
```

These commands assume `FUGU_PYTHON` points at the cu130 venv (the meta-agent
exports it so eval matches the board's torch 2.12 / CUDA 13.0 / triton 3.7
stack). Don't override it — unset falls back to the cu128 `.venv`, which
mis-ranks candidates the board scores on cu130.

`local_eval.sh` enforces the **300 s leaderboard budget by default**
(`--server-timeout 300`). The `--mode leaderboard` run is the one that reproduces
the server (compile + tests + benchmarks together); a kernel that WARNs or TIMES
OUT there will be rejected on submission. Run it on any candidate you intend to
keep — `--mode benchmark` alone (a subset) does **not** prove you fit the budget.

`local_eval.sh` runs synchronously, streams output, and `flock`-serializes the
single B200 (so concurrent runs queue rather than collide). Gate benchmark on a
PASS from test mode. Exit 0 = PASS.

### Bash-tool timeout
A fresh iteration JIT-compiles Triton then runs. Set the coding-agent **Bash
tool `timeout = 600_000` ms (10 min)** for both `--mode test` and `--mode
benchmark`. The large `(640,512)`/`n=1024` cases are slow under the naive kernel;
the default 2-minute Bash timeout is far too short and you'll lose iterations.
Keep `local_eval.sh`'s `--time` **below** the Bash ceiling (the examples use
`00:09:00`) so a hung kernel triggers a clean `TIMED OUT` from `local_eval.sh`
before the Bash tool hard-kills the call. If you genuinely need longer than
~10 min, run the eval in the background instead.

### Reading the output
- `Overall: PASS  (22/22 passed)` and `[exit] 0` → test success.
- `Overall: PASS  |  geomean: X.XXXX ms` → the benchmark line you record. Pull
  `geomean` into `history.csv`.
- `[eval-walltime] Xs` (printed every run) → wall-clock of this eval. In
  `--mode leaderboard` this is what the server's 300 s budget caps.
  - `[budget WARNING] ... % of the 300 s server budget` → **high risk** of a
    remote `FunctionTimeoutError` (this box is faster than the server). Reduce
    timing variance/runtime before trusting this kernel.
  - `[SERVER TIMEOUT] ... exceeded the 300 s leaderboard budget` (`[exit] 124`) →
    **would be rejected** on submission; not submittable as-is.
- `Overall: FAIL` → read the per-case lines:
  - `'NoneType' object has no attribute 'flush'` → you dropped the stdout shim.
  - `R - Q.T @ A is too large` → reconstruction failed: your trailing update is
    wrong (often a tiling/mask bug — see the guard-row gotcha) or you lost
    accuracy on an ill-conditioned stress case.
  - `Q is not orthogonal enough` → numerical: your reflectors drifted; check
    TF32 in `tl.dot`.
  - shape/dtype errors → `H` must be `(batch,n,n)` fp32, `tau` `(batch,n)` fp32.
  - Triton compile error → fix the kernel (check constexpr vs runtime args,
    tile-size powers of two, mask shapes).

**Do not inspect the evaluation harness internals.** Treat `local_eval.sh`,
`evaluation.py`, `task.py`, `reference.py`, `eval.py`, and the problem directory
as a black box — the description above is sufficient. Once test+benchmark run on
the baseline, stop investigating the harness and spend your effort on the kernel.

## Loop

0. **Once at start:** read `init.py` (verified PASS, geomean 61.17 ms). Create
   `history.csv` and seed it with the `0000 … init.py … PASS` baseline row so the
   floor is on record. Do not re-run eval on `init.py` — its numbers are known.
1. Pick next `NNNN` (starting at `0001`), get UTC `TS`, save to `v{NNNN}_{TS}.py`
   with header `status: pending`. For `v0001`, copy `init.py`'s shim/scaffold,
   rename the `@triton.jit` function to `qr_kernel_v0001`, and apply your first
   real optimization.
2. `local_eval.sh … --mode test`. On FAIL → header `status: FAIL`, append a FAIL
   row to `history.csv`, write the next iteration.
3. `local_eval.sh … --mode benchmark`. On PASS → update header (`status: PASS`,
   fill geomean), append the PASS row.
4. One hypothesis → one change → next iteration. Consult `history.csv` first;
   only keep an iteration if its geomean improves on the best PASS so far,
   otherwise note it and keep the prior best as parent.
5. **If you get stuck** (consecutive iterations fail to improve), use `WebSearch`
   for concrete techniques (e.g. "blocked Householder QR Triton", "WY
   representation tensor core", "Triton tl.dot fp32 accuracy"), or worklogs like
   Colfax (https://research.colfax-intl.com/?cst) and Cudaforfun
   (https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).
   Bring back one technique you haven't tried and apply it next. Don't loop in
   place repeating the same class of change.

## Termination — five attempts, then finalize

You get **exactly 5 iteration attempts**: `v0001`–`v0005`. An "attempt" is any
iteration written to disk and evaluated, PASS or FAIL. `0000` (`init.py`) does
not count.

As soon as the 5th attempt's CSV row is written, **finalize immediately**:
1. Read `history.csv`. Filter to `status == PASS` (includes the `0000` init row).
   Consider candidates from best (smallest) `geomean_ms` upward and pick the
   **fastest one that also fits the 300 s budget**: run
   `local_eval.sh "$cand" --problem-dir "$PROBLEM" --mode leaderboard` and accept
   it only if it ends `[exit] 0` with **no `[SERVER TIMEOUT]`** (and ideally no
   `[budget WARNING]`). If your best-geomean kernel TIMES OUT or rides the budget,
   **skip it and take the next-fastest budget-passing kernel** — a steady kernel
   that submits beats a faster one that gets rejected. `init.py` is budget-safe by
   definition, so it is always a valid fallback.
2. Copy the chosen kernel to `final.py` (overwrite if present).
3. Prepend a single line:
   `# final: copied from <filename>; best of 5 attempts (geomean <X> ms)`
4. Report: best filename, geomean, the one-line trick, and the PASS/FAIL ratio
   across the 5 attempts.

If all 5 attempts FAIL, fall back to `init.py` (PASS by definition). Do not run
more than 5 attempts.

## Don't

- Don't fall back to `torch.geqrf` on *every* shape — that's the baseline no-op.
  Custom Triton kernels must win the batched small/medium shapes (esp.
  `(640,512)`); cuSOLVER fallback is only for shapes where it genuinely beats your
  kernel.
- Don't mutate the input tensor `A` in place.
- Don't finalize a kernel that exceeds or rides close to the **300 s leaderboard
  budget** (a `[budget WARNING]` / `[SERVER TIMEOUT]` in `--mode leaderboard`) —
  it is rejected with `FunctionTimeoutError` regardless of geomean. Low timing
  variance matters as much as low runtime: steady beats fast-but-noisy.
- Don't try to hack, game, or short-circuit the benchmark or eval pipeline.
- Don't run `python evaluation.py` directly or call `popcorn-cli`/`submission.py`
  (the meta-agent handles remote submission).
- Don't modify `reference.py`/`task.py`/`eval.py`/`task.yml`/`utils.py`; only
  write files in your current workspace.
- Don't modify `init.py` — it's the immutable baseline. Copy its scaffold into
  `v0001_*.py` and edit that.
- Don't hand-manage CUDA streams (`torch.cuda.Stream`, `stream=` kwargs, or any
  explicit-stream construct). Launch Triton kernels plainly as
  `kernel[grid](...)` on the default stream — explicit-stream code is statically
  rejected by the leaderboard with "work on another stream" (see "Leaderboard
  submission rules").
- Don't leave TF32 on in `tl.dot` unless all 22 test cases still PASS — the QR
  gate has no `atol` and TF32 will silently fail orthogonality on hard cases.
