"""
Re-run FROZEN model solvers on the CURRENT eval set, WITHOUT calling any API.

Each model already wrote its solver once; that code is saved at
results/<key>_solver.py. This driver runs those exact frozen solvers against
eval_cubes.json (the 300-cube set) via solver_runner.py, re-verifies every
solution with the trusted engine (cube.py), and writes the summary JSON named
by RUBIK_SUMMARY_OUT (default results/summary_100.json). Pure local CPU --
zero tokens, fully reproducible, standard library only.

Usage:
  RUBIK_CUBE_TIMEOUT=600 RUBIK_SUMMARY_OUT=summary_300.json python3 rerun_saved.py
  python3 rerun_saved.py opus48 gemini   # a subset
  python3 rerun_saved.py                 # all frozen solvers
"""
import os
import sys
import json
import time
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rubik_compare as rc   # top-level imports are stdlib + cube only (no API SDKs)

# Output file is overridable so parallel single-model runs don't race on one JSON.
OUT = HERE / "results" / os.environ.get("RUBIK_SUMMARY_OUT", "summary_100.json")


def main():
    keys = sys.argv[1:] or [m[0] for m in rc.MODELS]
    cubes = json.loads(rc.EVAL_CUBES.read_text(encoding="utf-8"))
    label = {k: l for k, l, _ in rc.MODELS}
    print(f"rerun_saved: {len(keys)} model(s) on {len(cubes)} cubes "
          f"(per-cube cap {rc.PER_CUBE_TIMEOUT:g}s) -- NO API calls\n", flush=True)

    merged = {}
    if OUT.exists():
        try:
            for s in json.loads(OUT.read_text(encoding="utf-8")):
                merged[s["key"]] = s
        except Exception:
            pass

    for key in keys:
        solver_file = HERE / "results" / f"{key}_solver.py"
        if not solver_file.exists():
            print(f"[{key}] no frozen solver file ({solver_file.name}), skip")
            continue
        code = solver_file.read_text(encoding="utf-8")
        print(f"[{key}] running frozen solver on {len(cubes)} cubes...", flush=True)
        t0 = time.time()
        results, err = rc.run_solver(key, code, cubes)
        dt = time.time() - t0
        if results is None:
            merged[key] = {"key": key, "label": label.get(key, key), "ok": False,
                           "stage": "solver", "detail": err, "n_cubes": len(cubes),
                           "n_solved": 0, "solve_rate": 0.0, "solve_sec": round(dt, 1)}
            print(f"[{key}] FAIL: {err}  ({dt:.0f}s)")
            _flush(merged)
            continue
        solved = [r for r in results if r["solved"]]
        turns = [r["turns"] for r in solved]
        hero = next((r for r in results if r["hero"]), None)
        merged[key] = {
            "key": key, "label": label.get(key, key), "ok": True,
            "solve_sec": round(dt, 1), "n_cubes": len(results),
            "n_solved": len(solved), "solve_rate": round(len(solved) / len(results), 3),
            "hero_solved": bool(hero and hero["solved"]),
            "hero_turns": hero["turns"] if hero and hero["solved"] else None,
            "mean_turns": round(statistics.mean(turns), 2) if turns else None,
            "median_turns": statistics.median(turns) if turns else None,
            "min_turns": min(turns) if turns else None,
            "max_turns": max(turns) if turns else None,
            "results": results,
        }
        print(f"[{key}] solved {len(solved)}/{len(results)}, "
              f"mean {merged[key]['mean_turns']}, hero {merged[key]['hero_turns']}  ({dt:.0f}s)")
        _flush(merged)

    print(f"\nwrote {OUT}")


def _flush(merged):
    order = [m[0] for m in rc.MODELS]
    OUT.write_text(json.dumps([merged[k] for k in order if k in merged], indent=2),
                   encoding="utf-8")


if __name__ == "__main__":
    main()
