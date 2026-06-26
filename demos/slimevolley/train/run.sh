#!/usr/bin/env bash
# Train a Slime Volley policy. Usage: ./run.sh --model <claude|gemini|fugu> [--iters N]
# Then pick the champion with:  uv run python assess.py --model <model>
set -euo pipefail
cd "$(dirname "$0")"
[ -d ./.codex ] && export CODEX_HOME="$PWD/.codex"   # use local (gitignored) codex profiles if present, else ~/.codex
uv sync
exec uv run python train.py "$@"
