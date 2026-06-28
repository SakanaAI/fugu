#!/usr/bin/env bash
# Local multi-GPU wrapper for GPU MODE evaluation.py.
#
# Replaces slurm_eval.sh. The old script fanned eval jobs out across a
# multi-node H100 SLURM cluster; this box has a pool of local B200s. Jobs run
# CONCURRENTLY — one per GPU — and only queue when every GPU is busy. This
# runner executes a job on a free GPU with full isolation:
#
#   1. PER-GPU flocks (/tmp/fugu_b200_gpu{N}.lock) gate each device. A job grabs
#      the first FREE GPU with a non-blocking flock; if all are busy it polls
#      until one frees. So two evals never share a GPU or corrupt each other's
#      CUDA build cache. Launch as many as you like — they spread, then queue.
#   2. Each run gets a private $TORCH_EXTENSIONS_DIR on node-local /tmp, so
#      concurrent nvcc builds can't clobber each other's .so. (evaluation.py
#      already gives each run a private tempfile workdir.)
#   3. Each run is wrapped in `timeout` so a hung kernel can't hold a GPU
#      lock forever.
#
# Usage:
#   ./local_eval.sh <kernel_file> [--mode MODE] [--problem-dir DIR]
#                                 [--time HH:MM:SS] [--gpu N] [--gpus "0 1 2"]
#                                 [--lock-prefix PATH] [--no-lock] [--keep]
#                                 [--server-timeout SECS] [--no-server-timeout]
#
# --gpu N pins a device; without it a free GPU is auto-picked from the pool. The
# pool defaults to every GPU nvidia-smi reports; override via --gpus "0 2 5" or
# the FUGU_GPUS env var.
#
# Defaults: --mode benchmark, --problem-dir = parent dir of <kernel_file>,
#           --time 00:10:00, --gpus = all detected GPUs, --server-timeout 300.
#
# --server-timeout replicates the leaderboard server's hard function budget: the
# qr_v2 (and other) leaderboards kill any submission whose full eval (compile +
# tests + benchmarks) exceeds 300s with a FunctionTimeoutError. This wrapper now
# enforces the same budget locally, so a kernel that would time out on
# submission also FAILS here (exit 124) instead of passing. Because this local
# B200 is FASTER than the Modal server, an eval whose wall-clock is anywhere
# near the budget almost certainly times out remotely — a >60%-of-budget warning
# is printed. Pass --no-server-timeout (or --server-timeout 0) to disable.
#
# The run is always synchronous: stdout/stderr stream to your terminal AND
# are tee'd to logs/. (The old --wait flag is accepted but a no-op now.)
#
# Examples:
#   ./local_eval.sh init.py --mode test \
#       --problem-dir reference-kernels/problems/linalg/qr_v2
#
#   # fire several at once from a loop — they spread across the GPU pool and
#   # the per-GPU flocks queue any overflow:
#   for f in kernels_b200/qr/gen1/idx*/final.py; do
#     ./local_eval.sh "$f" --problem-dir reference-kernels/problems/linalg/qr_v2 &
#   done; wait

set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# FUGU_PYTHON overrides the eval interpreter (e.g. .venv-cu130 to match the leaderboard stack).
PY="${FUGU_PYTHON:-${ROOT}/.venv/bin/python}"
# Per-interpreter Triton cache so different torch/triton stacks never share cubins.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$(basename "$(dirname "$(dirname "$PY")")")}"

# Default pool = every GPU nvidia-smi reports; falls back to GPU 0.
detect_gpus() {
    local idx
    idx=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | xargs)
    echo "${idx:-0}"
}

TIME_LIMIT="00:10:00"
GPU=""                                  # empty = auto-select a free GPU from the pool
GPU_POOL="${FUGU_GPUS:-$(detect_gpus)}" # devices to schedule across
# Namespace the lock by SLURM job (each job's GPU is remapped to local index 0,
# so a bare gpu0.lock collides across jobs on different physical GPUs). Falls back
# to the shared name when not under SLURM.
_slurm_tag="${SLURM_JOB_NAME:-${SLURM_JOB_ID:-}}"
_slurm_tag="${_slurm_tag//[^A-Za-z0-9_.-]/_}"
LOCK_PREFIX="/tmp/fugu_b200${_slurm_tag:+_${_slurm_tag}}_gpu"  # per-GPU lock = ${LOCK_PREFIX}${N}.lock
POLL_INTERVAL=2                         # secs between retries when all GPUs busy
USE_LOCK=1
SERVER_TIMEOUT=300   # leaderboard FunctionTimeoutError budget (s); 0 disables

KERNEL_FILE=""
MODE="benchmark"
PROBLEM_DIR=""
KEEP=""
LOCK_FILE=""         # resolved to the acquired GPU's lock at runtime
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)         MODE="$2"; shift 2 ;;
        --problem-dir)  PROBLEM_DIR="$2"; shift 2 ;;
        --time)         TIME_LIMIT="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --gpus)         GPU_POOL="$2"; shift 2 ;;
        --lock-prefix)  LOCK_PREFIX="$2"; shift 2 ;;
        --lock-file)    LOCK_PREFIX="${2%.lock}"; shift 2 ;;  # back-compat alias
        --no-lock)      USE_LOCK=0; shift ;;
        --server-timeout)    SERVER_TIMEOUT="$2"; shift 2 ;;
        --no-server-timeout) SERVER_TIMEOUT=0; shift ;;
        --keep)         KEEP="--keep"; shift ;;
        --wait)         shift ;;   # accepted for slurm_eval.sh compatibility; no-op
        -h|--help)      sed -n '2,52p' "$0"; exit 0 ;;
        *)              KERNEL_FILE="$1"; shift ;;
    esac
done

[[ -n "$KERNEL_FILE" ]] || { echo "[error] missing kernel file argument" >&2; exit 2; }
[[ -f "$KERNEL_FILE" ]] || { echo "[error] kernel file not found: $KERNEL_FILE" >&2; exit 2; }
[[ -x "$PY" ]]          || { echo "[error] venv python not found at $PY" >&2; exit 2; }
KERNEL_FILE=$(realpath "$KERNEL_FILE")

PROBLEM_ARG=()
if [[ -n "$PROBLEM_DIR" ]]; then
    PROBLEM_DIR=$(realpath "$PROBLEM_DIR")
    PROBLEM_ARG=(--problem-dir "$PROBLEM_DIR")
fi

# HH:MM:SS (or MM:SS, or plain seconds) -> seconds, for `timeout`.
hms_to_secs() {
    local t="$1"
    IFS=: read -ra p <<< "$t"
    case "${#p[@]}" in
        3) echo $(( 10#${p[0]}*3600 + 10#${p[1]}*60 + 10#${p[2]} )) ;;
        2) echo $(( 10#${p[0]}*60 + 10#${p[1]} )) ;;
        1) echo $(( 10#${p[0]} )) ;;
        *) echo 600 ;;
    esac
}
TIMEOUT_S=$(hms_to_secs "$TIME_LIMIT")

# torch's JIT cpp_extension needs CUDA_HOME + nvcc on PATH to build kernels.
# Respect an already-exported CUDA_HOME; otherwise pick the system toolkit that
# matches torch's CUDA build (e.g. cu128 -> /usr/local/cuda-12.8).
if [[ -z "${CUDA_HOME:-}" ]]; then
    torch_cu=$("$PY" -c 'import torch;v=torch.version.cuda or "";print(v.replace(".",""))' 2>/dev/null)
    for cand in "/usr/local/cuda-${torch_cu:0:2}.${torch_cu:2}" /usr/local/cuda /usr/local/cuda-12.8; do
        if [[ -x "$cand/bin/nvcc" ]]; then CUDA_HOME="$cand"; break; fi
    done
fi
if [[ -z "${CUDA_HOME:-}" || ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "[error] no usable CUDA toolkit (nvcc) found. Set CUDA_HOME to a CUDA install root." >&2
    echo "        torch wants CUDA ${torch_cu:-?}; checked /usr/local/cuda*" >&2
    exit 2
fi
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"

TS=$(date -u +%Y%m%d_%H%M%S)
JOB_NAME="local_eval_${MODE}_${TS}_$$"
mkdir -p "${ROOT}/logs"
OUT="${ROOT}/logs/${JOB_NAME}.out"

# This is the actual evaluation, run under the GPU lock (if enabled).
run_eval() {
    local te_dir="/tmp/torch_extensions_${JOB_NAME}"
    mkdir -p "$te_dir"
    trap 'rm -rf "$te_dir"' RETURN

    {
        echo "[node]    $(hostname)"
        echo "[date]    $(date -u +%FT%TZ)"
        echo "[gpu]     CUDA_VISIBLE_DEVICES=${GPU}  $(CUDA_VISIBLE_DEVICES=${GPU} nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
        echo "[kernel]  ${KERNEL_FILE}"
        echo "[cuda]    CUDA_HOME=${CUDA_HOME}"
        echo "[mode]    ${MODE}"
        echo "[cache]   ${te_dir}"
        echo "[lock]    $([[ $USE_LOCK -eq 1 ]] && echo "${LOCK_FILE}" || echo "disabled")"
        echo "[timeout] ${TIMEOUT_S}s (--time ${TIME_LIMIT})"
        echo "[budget]  $([[ $SERVER_TIMEOUT -gt 0 ]] && echo "${SERVER_TIMEOUT}s (leaderboard FunctionTimeoutError budget)" || echo "disabled")"
        echo "================================================================"
    } | tee "$OUT"

    # Effective wall-clock cap = the smaller of the hung-kernel safety timeout
    # (--time) and the leaderboard server budget (--server-timeout, when > 0).
    # Hitting the server budget is reported as a leaderboard FunctionTimeoutError
    # so a kernel that would time out on submission also fails here.
    local wrap_timeout=$TIMEOUT_S
    local server_budget_active=0
    # Use -le so that when the server budget equals the safety timeout, hitting
    # it is still reported as a SERVER TIMEOUT (the more informative cause) and
    # not as a generic hung-kernel timeout.
    if [[ $SERVER_TIMEOUT -gt 0 && $SERVER_TIMEOUT -le $TIMEOUT_S ]]; then
        wrap_timeout=$SERVER_TIMEOUT
        server_budget_active=1
    fi

    local t_start=$(date +%s)
    CUDA_VISIBLE_DEVICES="${GPU}" \
    TORCH_EXTENSIONS_DIR="$te_dir" \
    PYTHONUNBUFFERED=1 \
        timeout --signal=TERM --kill-after=15 "${wrap_timeout}" \
        "$PY" -u "${ROOT}/evaluation.py" "${KERNEL_FILE}" --mode "${MODE}" ${KEEP} "${PROBLEM_ARG[@]}" \
        2>&1 | tee -a "$OUT"
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - t_start ))

    {
        echo "================================================================"
        echo "[eval-walltime] ${elapsed}s"
        if [[ $rc -eq 124 && $server_budget_active -eq 1 ]]; then
            echo "[SERVER TIMEOUT] eval exceeded the ${SERVER_TIMEOUT}s leaderboard budget"
            echo "    -> would FAIL on the leaderboard with FunctionTimeoutError."
            echo "    -> this local B200 is FASTER than the Modal server, so anything near"
            echo "       ${SERVER_TIMEOUT}s locally is a near-certain remote timeout."
            echo "[exit] 124 (SERVER BUDGET EXCEEDED, ${SERVER_TIMEOUT}s)"
        elif [[ $rc -eq 124 ]]; then
            echo "[exit] $rc (TIMED OUT after ${wrap_timeout}s)"
        else
            if [[ $SERVER_TIMEOUT -gt 0 && $elapsed -gt $(( SERVER_TIMEOUT * 60 / 100 )) ]]; then
                echo "[budget WARNING] eval used ${elapsed}s = $(( elapsed * 100 / SERVER_TIMEOUT ))% of the ${SERVER_TIMEOUT}s server budget."
                echo "    -> the Modal server is slower than this box; HIGH RISK of remote"
                echo "       FunctionTimeoutError. Cut eval wall-clock (lower benchmark timing"
                echo "       variance and/or runtime) before submitting."
            fi
            echo "[exit] $rc"
        fi
    } | tee -a "$OUT"
    return $rc
}

echo "[start]   ${JOB_NAME}"
echo "[log]     ${OUT}"

# Acquire a GPU on fd 9 (sets GPU + LOCK_FILE). With --gpu N, block on that one;
# otherwise take the first free GPU in the pool, polling when all are busy.
acquire_gpu() {
    local announced=0 g
    while true; do
        for g in ${GPU:-$GPU_POOL}; do
            LOCK_FILE="${LOCK_PREFIX}${g}.lock"
            exec 9>"$LOCK_FILE" || { echo "[error] cannot open $LOCK_FILE" >&2; exit 2; }
            { [[ -n "$GPU" ]] && flock 9 || flock -n 9; } && { GPU="$g"; return 0; }
        done
        [[ $announced -eq 0 ]] && { echo "[wait]    all GPUs [${GPU_POOL}] busy; waiting..."; announced=1; }
        sleep "$POLL_INTERVAL"
    done
}

if [[ $USE_LOCK -eq 1 ]]; then
    acquire_gpu
    echo "[gpu]     acquired GPU ${GPU} (lock ${LOCK_FILE})"
    run_eval
    rc=$?
    flock -u 9
else
    GPU="${GPU:-0}"   # no lock: honor --gpu, else default to GPU 0
    run_eval
    rc=$?
fi

exit $rc
