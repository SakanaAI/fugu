"""Train a Slime Volley policy with one model: run its CLI agent up to a target number of iterations,
carrying a one-file note between them. Re-entrant: re-running resumes from the last finished iteration,
so raising --iters just adds more. Scoring and champion selection are assess.py.

    uv run python train.py --model gemini [--iters N] [--run-dir PATH]
"""

from __future__ import annotations

import argparse
import os
import shutil

from common import HERE, MODELS, Model, run_agent, run_directory

DEFAULT_ITERATIONS = 10
SHARED_INTO_RUN = ("env", "assess.py", "common.py")  # symlinked so the agent can self-score from its cwd


def train(profile: str, model: Model, run_dir: str, target: int) -> None:
    """Run `model` up to `target` iterations into `run_dir`/iter_NN/, resuming from the last finished one."""
    os.makedirs(run_dir, exist_ok=True)
    start, note = _resume_point(run_dir)
    if start >= target:
        print(f"[train] {profile}: already at {start}/{target} iterations, nothing to do")
        return
    print(
        f"[train] {profile}: {'resuming at' if start else 'starting'} iteration {start}, target {target} -> {run_dir}"
    )
    for iteration in range(start, target):
        iter_dir = os.path.join(run_dir, f"iter_{iteration:02d}")
        if os.path.isdir(iter_dir):  # an interrupted attempt: clear it for a clean redo
            shutil.rmtree(iter_dir)
        os.makedirs(os.path.join(iter_dir, "kit", "policies"))
        _link_shared_files(iter_dir)
        print(f"[train] iteration {iteration}: running {profile} ...")
        run_agent(model, build_prompt(note, iteration, target), cwd=iter_dir)
        note = _read(os.path.join(iter_dir, "note.md"), default=note)
    print(f"[train] done. pick the champion with:  uv run python assess.py --model {profile}")


def build_prompt(note: str, iteration: int, target: int) -> str:
    """The agent's prompt: the standing instructions, the note it left last time, and this turn's ask."""
    instructions = _read(os.path.join(HERE, "INSTRUCTIONS.md"))
    last_note = note.strip() or "(empty: this is the first iteration)"
    this_turn = (
        f"Iteration {iteration + 1} of {target}. Improve on your best prior policy with ONE change, and keep "
        f"it only if the win-rate does not regress. Leave kit/policies/champion.py (make_policy(seat=0) -> "
        f"act(obs) -> [forward, back, jump]) and note.md in this directory, then stop. No internet."
    )
    return (
        f"===== INSTRUCTIONS =====\n{instructions}\n\n"
        f"===== YOUR NOTE FROM THE LAST ITERATION =====\n{last_note}\n\n"
        f"===== THIS ITERATION =====\n{this_turn}"
    )


def _resume_point(run_dir: str) -> tuple[int, str]:
    """Walk iter_00, iter_01, ... while each finished (wrote note.md), carrying its note. The first
    iteration with no note.md is the one to (re)run; return (that index, the last note)."""
    note, start = "", 0
    while os.path.exists(os.path.join(run_dir, f"iter_{start:02d}", "note.md")):
        note = _read(os.path.join(run_dir, f"iter_{start:02d}", "note.md"))
        start += 1
    return start, note


def _link_shared_files(iter_dir: str) -> None:
    """Symlink the env + scorer into the agent's working dir so it can self-evaluate."""
    for name in SHARED_INTO_RUN:
        link = os.path.join(iter_dir, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(HERE, name), link)


def _read(path: str, default: str = "") -> str:
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Slime Volley policy with one model.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS), help="which model to run")
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERATIONS, help="target total iterations (default 10)"
    )
    parser.add_argument("--run-dir", help="where the run lives (default ./run/<model>)")
    args = parser.parse_args()
    train(args.model, MODELS[args.model], run_directory(args.model, args.run_dir), args.iters)


if __name__ == "__main__":
    main()
