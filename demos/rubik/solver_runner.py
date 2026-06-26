"""
Subprocess-side runner for ONE model's solver.

Imports the model-written `solve(facelet)` from a file path, runs it against
every cube in the eval set with a per-cube wall-clock cap (SIGALRM), and prints
the RAW solution string for each cube. It does NOT judge correctness -- the
parent re-verifies every solution with the trusted engine (cube.py), so a model
cannot fake a solved cube by lying in its output.

Usage (invoked by rubik_compare.py, not by hand):
  python solver_runner.py <solver_file.py> <eval_cubes.json> <per_cube_timeout_s>

Output: one line per cube, prefixed with a sentinel so the model's own prints
can't be mistaken for results:
  ##R## {"id": <int>, "solution": <str|null>, "error": <str|null>, "secs": <float>}
On import / contract failure (no usable solve function):
  ##CONTRACT_FAIL## <reason>
"""
import sys
import json
import time
import signal
import importlib.util


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _load_solve(path):
    spec = importlib.util.spec_from_file_location("model_solver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # may raise (e.g. forbidden import)
    fn = getattr(mod, "solve", None)
    if not callable(fn):
        raise AttributeError("module defines no callable solve(facelet)")
    return fn


def main():
    solver_path, cubes_path, per_cube = sys.argv[1], sys.argv[2], float(sys.argv[3])
    cubes = json.loads(open(cubes_path, encoding="utf-8").read())

    try:
        solve = _load_solve(solver_path)
    except Exception as exc:
        print(f"##CONTRACT_FAIL## {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        return

    signal.signal(signal.SIGALRM, _alarm_handler)
    for cube in cubes:
        rec = {"id": cube["id"], "solution": None, "error": None, "secs": 0.0}
        t0 = time.time()
        signal.setitimer(signal.ITIMER_REAL, per_cube)
        try:
            out = solve(cube["facelet"])
            rec["solution"] = out if isinstance(out, str) else str(out)
        except _Timeout:
            rec["error"] = f"timeout(>{per_cube:g}s)"
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        rec["secs"] = round(time.time() - t0, 2)
        print(f"##R## {json.dumps(rec)}", flush=True)


if __name__ == "__main__":
    main()
