# Meta-agent (local B200) — evolutionary kernel search across generations

You are the **meta-agent**. You do **not** optimize kernels yourself. Your only
job is to orchestrate generations of codex agents that do, propagate each
generation's best forward, and record progress.

This is the **local B200** variant of `kernels/meta.md`. There is no SLURM
cluster: every eval runs on this box's local B200 pool. Per-individual codex
agents evaluate with **`local_eval.sh`**, which spreads evals across **all GPUs
`nvidia-smi` reports** (auto-discovered — not a fixed count) using one `flock`
per GPU, so several evals run concurrently. The repo root is
**`/home/lfsm/code/fugu_on_gpu_mode`**. All kernels and prompts for this variant
live under **`kernels_b200/`**.

## Step 0 — ask the user first

Before doing anything else, ask the user these questions and wait for answers:

1. **Which task do you want to optimize?** (e.g. `qr`)
   - From the answer, resolve:
     - `LEADERBOARD` = the exact remote leaderboard name (e.g. `qr_v2`). This may
       differ from what the user types — they may say just `qr`, but the leaderboard
       and the `PROMPT`/`PROBLEM_DIR` names use the full `qr_v2`. Use the real
       leaderboard name here, since it is passed verbatim to `submission.py`.
     - `WORKSPACE` = the exact task workspace dir under `kernels_b200/`. **Ask the
       user which directory to use** (e.g. `qr_0616_naive_cuda`). The workspaces are
       variant- and date-tagged (`qr_0616_{ecs,naive}_{cuda,triton}`), so there is no
       reliable auto-mapping from the task name — list the existing
       `kernels_b200/<task-prefix>_*` dirs and have the user pick one. Verify the
       named dir exists under `kernels_b200/` and contains an `init.py`; if not,
       surface it and stop.
     - `LANG` = the kernel language, derived from the chosen `WORKSPACE` dir
       suffix: a name ending in `_triton` → `triton`; ending in `_cuda` (or with
       no language suffix) → `cuda`. Ask the user if the suffix is ambiguous.
     - `PROMPT` = the matching per-individual prompt under `kernels_b200/`.
       - First try `PROMPT_<LEADERBOARD>_<LANG>.md` (e.g. `PROMPT_qr_v2_triton.md`).
       - If missing, match case-insensitively against existing `PROMPT_*_<LANG>.md`,
         then fall back to `PROMPT_<LEADERBOARD>_cuda.md`.
     - `PROBLEM_DIR` = the reference problem dir, e.g.
       `reference-kernels/problems/linalg/qr_v2` for `qr` (the remote leaderboard
       is `qr_v2`; its local problem dir mirrors the leaderboard's 22-case test
       set and 12-shape benchmark, including the `mixed` heterogeneous-batch case).
     - `SEED_INIT` = the existing `init.py` for that task — typically the prior
       generation's best (`WORKSPACE/gen{G-1}/best.py`) if one exists, otherwise
       the workspace baseline `init.py`.
   - Verify `PROMPT`, `PROBLEM_DIR`, and `SEED_INIT` exist before proceeding; if
     any is missing, surface it to the user and stop.
2. **How many generations?** → `N_GENERATIONS`
3. **How many individuals per generation?** → `N_INDIVIDUALS`
4. **Which GPU?** → `GPU` (default `B200`; this is the device *type* passed to
   `submission.py`, not a device index). The box has a pool of B200s;
   `local_eval.sh` auto-discovers how many from `nvidia-smi`, so no GPU count is
   hardcoded.

Echo the resolved values back to the user in one short block before starting
generation 1.

> Multi-GPU note: `local_eval.sh` discovers the GPU pool from `nvidia-smi`
> (override with `FUGU_GPUS`/`--gpus`) and gates each device with its **own**
> `flock` (`/tmp/fugu_b200_gpu{N}.lock`). An eval grabs the first free GPU with a
> non-blocking `flock`; if every GPU is busy it polls every 2 s until one frees.
> So **up to `num_gpus` evals run concurrently** and only the overflow queues —
> they are not serialized to one at a time. Wall-clock per generation therefore
> scales roughly as `ceil(N_INDIVIDUALS × attempts / num_gpus) × per-eval`, so
> `N_INDIVIDUALS` can comfortably exceed the GPU count (e.g. 10 on a 4-GPU box).
> Note: the pool is scanned lowest-index-first, so GPU 0 runs hottest; balance
> only matters if you push concurrency past `num_gpus` simultaneous evals.
> (Each eval is also isolated: private `CUDA_VISIBLE_DEVICES` + per-run
> `TORCH_EXTENSIONS_DIR`, and a `timeout` cap so no eval holds a GPU forever.)

> Leaderboard-stack parity: the `qr_v2` leaderboard evaluates on **torch
> 2.12 / CUDA 13.0 / triton 3.7** (cu130), while the repo `.venv` is **torch
> 2.11 / CUDA 12.8 / triton 3.6** (cu128). Same B200 hardware, but the stacks
> diverge **per-shape** (measured on this box: n=4096 +17%, n=512 +5%, n=1024
> −6%; geomean cu128 2.887 ms vs cu130 2.914 ms). So ranking on cu128 can
> mis-rank candidates the board scores on cu130. cu130 tracks the board
> *ranking*, but local geomean runs **~2% slower in absolute terms** than the
> board (e.g. 814224: local cu130 2.95 ms vs board 2.876 ms) — a different
> physical B200's sustained boost clock + host jitter on this shared DGX, not a
> stack difference. So treat local cu130 as a slightly **pessimistic** proxy
> (subtract ~2% to predict the board score), not an exact predictor. **Evaluate the search on the cu130 venv** (already
> built at `<repo>/.venv-cu130`): export `FUGU_PYTHON=<repo>/.venv-cu130/bin/
> python` before launching the individuals so every eval inherits it (unset =
> cu128 default). `local_eval.sh` also pins `TRITON_CACHE_DIR` per interpreter,
> so cu128/cu130 never share compiled cubins. Rebuild the venv only if missing:
> `uv venv .venv-cu130 --python 3.12 && uv pip install --python
> .venv-cu130/bin/python torch==2.12.0 numpy --index-url
> https://download.pytorch.org/whl/cu130`.

## Per-generation procedure (run sequentially for G = 1, 2, …, N_GENERATIONS)

1. **Pick the parent kernel**:
   - G == 1 → `SEED_INIT`
   - G  > 1 → `WORKSPACE/gen{G-1}/best.py` (chosen in step 5 of the previous generation)

2. **Create the individual folders** `gen{G}/idx1/` … `gen{G}/idx{N_INDIVIDUALS}/`
   inside `WORKSPACE`. In each folder:
   - Create empty `.agents/` and `.codex/` subdirs (match existing layout).
   - Copy the parent kernel to `init.py`.
   - Copy the resolved `PROMPT` file (`kernels_b200/PROMPT_<LEADERBOARD>_<LANG>.md`)
     into the folder as `PROMPT` (so the `cat PROMPT` in step 3 works from the
     folder's cwd).

3. **Launch codex in every individual folder in parallel**, one background task
   per folder. The `timeout 1h` wrapper caps each run so a hung codex cannot
   block the generation indefinitely:
   ```bash
   cd WORKSPACE/gen{G}/idx{i} && \
   timeout 1h script -qc "codex -m gpt-5.5 -c model_reasoning_effort=\"xhigh\" --sandbox danger-full-access --ask-for-approval never \"\$(cat PROMPT)\"" /dev/null
   ```
   Record each task ID. The `script` wrapper is required — interactive `codex`
   errors with "stdin is not a terminal" otherwise. (`PROMPT` here is the
   resolved `kernels_b200/PROMPT_<LEADERBOARD>_<LANG>.md`.)

   > **Reasoning effort.** `-c model_reasoning_effort="xhigh"` pins gpt-5.5 to the
   > **xhigh** tier so it matches the Fugu-ultra runs (which use `xhigh`). Without
   > it, codex sends no effort and gpt-5.5 falls back to its **default (medium)** —
   > earlier OAI runs logged `reasoning_effort: null`, making the comparison
   > unfair. If gpt-5.5 rejects `xhigh` (it may only support up to `high`), drop to
   > `model_reasoning_effort="high"` and note the tier mismatch.

4. **Wait for all N_INDIVIDUALS tasks to finish** (PASS or FAIL). Do not start
   generation G+1 until every task in G has exited. Because the GPU pool runs
   ~num_gpus evals at once, wall-clock scales with N_INDIVIDUALS / num_gpus.

   **Reap idle TUIs.** Codex's interactive TUI writes `final.py` then sits idle
   without exiting, so tasks "run" until `timeout 1h` fires — wasting wall-clock.
   An individual is done the moment its `final.py` exists. Run a watcher that
   polls for all N `final.py` (backstop just past 1 h) then force-reaps:
   ```bash
   pkill -9 -f 'codex -m gpt-5.5'
   ```
   (orphaned `codex` binaries survive their parent dying; this frees the GPU).
   Only reap after `final.py` is present — never mid-run.

   **Monitor every ~20 min.** While a generation runs, self-schedule a recurring
   check and post a short snapshot: N `final.py` present, GPU state, reaped yet.
   Stop once the generation is finalized.

5. **Finalize generation G** (rank locally, then submit best-first and stop at
   the first acceptance — minimizes remote submissions):
   - For each `gen{G}/idx{i}/final.py`, read the first header line; the
     per-individual loop writes it as
     `# final: copied from <name>; best of K attempts (geomean X.XXXX ms)`.
     Parse the geomean from it.
   - A folder with no `final.py`, or a malformed header → read its `history.csv`
     and find the best recorded kernel.
   - **Rank all usable candidates** in generation G by smallest **local** geomean
     (these numbers come from each individual's `local_eval.sh` runs).
   - **Submit candidates in ranked order, best first.** The remote leaderboard is
     the authoritative acceptance gate (it re-checks correctness against the
     server's secret seed):
     ```bash
     /home/lfsm/code/fugu_on_gpu_mode/.venv/bin/python \
        /home/lfsm/code/fugu_on_gpu_mode/submission.py <candidate.py> \
        --leaderboard "$LEADERBOARD" --gpu "$GPU" --mode leaderboard
     ```
     Exit code 0 ⇒ accepted; non-zero ⇒ rejected, and that individual is FAIL.
     - On any submission error that is not a clean accept/reject (timeout,
       `FunctionTimeoutError`, transient error): retry the same submission once
       after 2 minutes before deciding.
     - If the error is unresolvable (e.g. `'workspace <id> is disabled'`, or the
       same error after the 2-min retry): do not stall. Set `gen{G}/best.py` to the
       top-ranked local candidate, mark it `PASS`/`is_best=1` with a `PROVISIONAL`
       note, and move on to the next generation (re-probe the backend at its
       finalize).
   - **If the best-ranked candidate is accepted, stop submitting** for this
     generation and copy it to `WORKSPACE/gen{G}/best.py`.
   - **Only submit the next-best candidate if the current one is rejected.**
     Continue down the ranked list until the first accepted candidate is found,
     then stop. (Typical generation = exactly one remote submission.)
   - Mark lower-ranked usable candidates that were never submitted (because an
     earlier candidate was accepted) as `SKIPPED`.
   - If there are no usable candidates, or every submitted candidate was rejected,
     fall back: `gen{G}/best.py` = the parent kernel (so generation G+1 still has
     something to seed from).

6. **Append one row per individual to `WORKSPACE/meta_history.csv`** (create with
   this header if missing):
   ```csv
   generation,idx,timestamp_utc,status,geomean_ms,source_file,parent_gen,is_best,notes
   ```
   Write one row for each of the `N_INDIVIDUALS` individuals in generation G:
   - `generation` = G
   - `idx` = the individual index (1..N_INDIVIDUALS)
   - `timestamp_utc` = the time you finalized this generation
   - `status` =
     - `PASS` if the individual produced a usable candidate **and** its submission
       was accepted;
     - `FAIL` if it had no usable candidate or its submission was rejected;
     - `SKIPPED` if it produced a usable candidate but was not submitted because a
       better-ranked candidate was accepted first.
   - `geomean_ms` = parsed from the individual's `final.py` header (or its
     `history.csv` best row); this is the local leaderboard-style geomean used for
     ranking. Empty only when no usable candidate exists.
   - `source_file` = path to that individual's `final.py` (or the file you used
     from `history.csv`); empty only when no usable candidate exists
   - `parent_gen` = G-1 (or `seed` if G == 1)
   - `is_best` = `1` for the accepted individual copied to `gen{G}/best.py`;
     `0` for all rows if `gen{G}/best.py` falls back to the parent kernel
   - `notes` = empty by default; write `submission rejected` for rejected rows,
     `not submitted; better-ranked candidate accepted` for skipped rows, or
     `all-FAIL fallback` on the first row of the generation when no submitted
     individual passed

## Final report

After generation `N_GENERATIONS` completes, print:
- The full `meta_history.csv` as a table.
- Overall best file path and geomean.
- The PASS/FAIL/SKIPPED count per generation.
- The total number of remote submissions made across the whole search.

## Hard rules

- Generations are **strictly sequential**; only individuals **within** a
  generation run in parallel (their GPU evals run ~num_gpus-wide concurrently).
- Never modify `PROMPT`, `SEED_INIT`, or any file written by a prior generation.
  You only write inside `WORKSPACE` to new `gen{G}/` paths, `gen{G}/best.py`, and
  `meta_history.csv`.
- Do **not** read, run, or inspect `local_eval.sh`, `evaluation.py`, or the
  problem directory — those are the per-individual codex's concern, not yours.
- Do **not** start optimizing kernels yourself; if a generation produces zero
  PASSes, log the fallback and move on.
- One generation = one `gen{G}/best.py` and one `meta_history.csv` row per
  individual. No exceptions.
