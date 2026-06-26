"""Preflight: confirm each model profile is runnable (a quick one-turn spawn). Run it before training,
so a missing key or codex profile shows up here instead of five iterations into a run.

    uv run python preflight.py
"""

from __future__ import annotations

import os
import tempfile

from common import MODELS, Model, run_agent


def main() -> None:
    print("== model profiles (each needs its own key / config) ==")
    for name, model in MODELS.items():
        print(f"  {name:7} {_probe(model)}")


def _probe(model: Model) -> str:
    """Spawn one trivial turn; report whether the model's CLI, auth, and config are in place."""
    with tempfile.TemporaryDirectory() as work_dir:
        try:
            run_agent(model, "Reply with exactly: OK", cwd=work_dir, timeout=300)
        except Exception as error:  # a missing CLI or bad config should report, not crash the preflight
            return f"FAILED: {error!r}"
        with open(os.path.join(work_dir, "transcript.txt"), encoding="utf-8") as handle:
            return "runnable" if handle.read().strip() else "no output (check key / config)"


if __name__ == "__main__":
    main()
