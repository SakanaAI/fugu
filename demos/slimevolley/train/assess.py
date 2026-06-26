"""Score Slime Volley policies by playing them against the built-in baseline in the real solid-net
env, 500 games spread across worker processes. The single source of truth for whether a policy
improved: it drives the env through the public make_policy ABI and trusts no self-report.

    uv run python assess.py path/to/policy.py        # score one file
    uv run python assess.py <profile> [run_dir]      # rank a run's policies and pick the champion
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import multiprocessing
import os
import shutil
import sys
import warnings
from dataclasses import dataclass

import numpy as np

from common import run_directory

# The vendored env predates numpy 2; restore the names it imports, then quiet its deprecation notice.
for _name, _value in {"bool8": np.bool_, "float_": np.float64}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)
warnings.filterwarnings("ignore", message="Gym has been unmaintained")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "env"))
from slimevolleygym.slimevolley import SlimeVolleyEnv  # noqa: E402  (needs the path + shims above)

GAMES = 500
MAX_STEPS = 3000  # bounds one rally; a tie at the cap counts as a draw
BASE_SEED = 42


@dataclass(frozen=True)
class Score:
    """The outcome of playing a policy for `games` games against the baseline."""

    value: float  # (wins + 0.5 * draws) / games, in [0, 1]
    wins: int
    draws: int
    losses: int
    games: int

    def __str__(self) -> str:
        return f"score={self.value:.4f}  W{self.wins}/D{self.draws}/L{self.losses}  ({self.games} games)"


def score_policy(path: str, games: int = GAMES, timeout: float | None = None) -> Score:
    """Score `path` over `games` games vs the baseline, split across worker processes.

    Each game has a fixed seed, so the result is reproducible regardless of how many workers run it.
    `timeout` (seconds, optional) guards against a pathological policy (an infinite loop, or a file
    caught mid-write): if the games do not finish in time the workers are killed and TimeoutError is
    raised for the caller to handle.
    """
    seeds = [BASE_SEED + i for i in range(games)]
    worker_count = min(os.cpu_count() or 1, games)
    shards = [(path, seeds[i::worker_count]) for i in range(worker_count)]
    with multiprocessing.Pool(worker_count) as pool:
        pending = pool.starmap_async(_play_games, shards)
        try:
            results = pending.get(timeout=timeout)
        except multiprocessing.TimeoutError:
            pool.terminate()
            raise TimeoutError(f"scoring did not finish within {timeout}s") from None
    wins = sum(result[0] for result in results)
    draws = sum(result[1] for result in results)
    losses = sum(result[2] for result in results)
    return Score((wins + 0.5 * draws) / games, wins, draws, losses, games)


def find_champion(run_dir: str, per_candidate_timeout: float = 120.0) -> tuple[str, Score]:
    """Score every candidate policy in `run_dir`, print the ranking, copy the best to champion.py.

    Robust to a partial / in-flight run: a candidate that fails to import, errors, or hangs (a file
    being written, or a broken policy) is skipped with a note instead of sinking the ranking.
    """
    candidates = sorted(set(glob.glob(os.path.join(run_dir, "iter_*", "kit", "policies", "*.py"))))
    if not candidates:
        sys.exit(f"no candidate policies under {run_dir}/iter_*/kit/policies/*.py")
    ranked: list[tuple[Score, str]] = []
    for path in candidates:
        try:
            score = score_policy(path, timeout=per_candidate_timeout)
        except Exception as error:  # broken, hung, or mid-write candidate: skip it, keep the run alive
            print(f"  skipped {os.path.relpath(path, run_dir)}: {error!r}")
            continue
        ranked.append((score, path))
        print(f"  {score.value:.4f}  {os.path.relpath(path, run_dir)}")
    if not ranked:
        sys.exit("no scorable candidate yet")
    ranked.sort(key=lambda item: item[0].value, reverse=True)
    best_score, best_path = ranked[0]
    champion = os.path.join(run_dir, "champion.py")
    shutil.copy(best_path, champion)
    _record_in_inventory(run_dir, champion)
    print(f"champion: {os.path.relpath(best_path, run_dir)}  {best_score}  ->  {champion}")
    return best_path, best_score


def _record_in_inventory(run_dir: str, champion: str) -> None:
    """Record this run's champion in run/inventory.json, a git-ignored map of {run name: champion
    path}. Paths are relative to that file's own dir (run/), so the demo can load it over HTTP."""
    run_root = os.path.join(_HERE, "run")
    if not os.path.abspath(champion).startswith(os.path.abspath(run_root) + os.sep):
        return  # a custom run dir outside run/: nothing the demo could resolve, so skip it
    inventory_path = os.path.join(run_root, "inventory.json")
    inventory: dict[str, str] = {}
    if os.path.exists(inventory_path):
        with open(inventory_path, encoding="utf-8") as handle:
            inventory = json.load(handle)
    inventory[os.path.basename(os.path.normpath(run_dir))] = os.path.relpath(champion, run_root)
    with open(inventory_path, "w", encoding="utf-8") as handle:
        json.dump(inventory, handle, indent=2)


def _play_games(path: str, seeds: list[int]) -> tuple[int, int, int]:
    """One worker: play `path` for the given per-game seeds; return (wins, draws, losses)."""
    act = _load_policy(path)
    env = SlimeVolleyEnv()
    wins = draws = losses = 0
    for seed in seeds:
        env.seed(seed)
        observation = env.reset()
        total_reward = 0.0
        for _ in range(MAX_STEPS):
            action = act(observation)
            action = np.zeros(3) if action is None else np.asarray(action).reshape(-1)[:3]
            observation, reward, done, _ = env.step(action)
            total_reward += float(reward)
            if done:
                break
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1
    return wins, draws, losses


def _load_policy(path: str, seat: int = 0):
    """Import a policy module and return its act(obs) callable via the make_policy ABI."""
    spec = importlib.util.spec_from_file_location("_policy_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    make_policy = getattr(module, "make_policy", None)
    policy = make_policy(seat=seat) if callable(make_policy) else getattr(module, "act", None)
    return getattr(policy, "act", policy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a Slime Volley policy, or pick a run's champion.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--file", help="score a single policy .py file")
    target.add_argument("--model", help="find + rank the champion of a run (a model profile name)")
    parser.add_argument("--run-dir", help="run dir for --model (default ./run/<model>)")
    args = parser.parse_args()
    if args.file:
        print(f"{score_policy(args.file)}  {args.file}")
    else:
        find_champion(run_directory(args.model, args.run_dir))


if __name__ == "__main__":
    main()
