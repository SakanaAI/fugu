"""
Checkpointed, resumable runner for ONE frozen solver over a cube set.

Streams solver_runner.py's per-cube output and appends each result to a JSONL
checkpoint the instant it arrives. If the process/machine dies partway, re-running
skips cubes already in the checkpoint and continues from where it stopped — a long
run never has to start over (lesson: feedback_long_jobs_must_checkpoint).

No API calls: runs the FROZEN results/<key>_solver.py on the given cubes, then
re-verifies every solution with the trusted engine (cube.py) at summarize time.

Usage:
  RUBIK_CUBE_TIMEOUT=300 python run_checkpointed.py <key> <cubes_json> <ckpt_jsonl> <summary_json>
"""
import os
import sys
import json
import time
import subprocess
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cube import verify_solution   # trusted engine

SOLVER_RUNNER = HERE / "solver_runner.py"
PER_CUBE = float(os.environ.get("RUBIK_CUBE_TIMEOUT", "300"))


def load_done(ckpt: Path):
    done = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done[rec["id"]] = rec
            except Exception:
                pass
    return done


def main():
    key, cubes_path, ckpt_path, summary_path = sys.argv[1:5]
    cubes = json.loads(Path(cubes_path).read_text(encoding="utf-8"))
    ckpt = Path(ckpt_path)
    solver_file = HERE / "results" / f"{key}_solver.py"
    if not solver_file.exists():
        sys.exit(f"[{key}] no frozen solver file: {solver_file}")

    done = load_done(ckpt)
    remaining = [c for c in cubes if c["id"] not in done]
    print(f"[{key}] {len(done)} done / {len(cubes)} total -> running {len(remaining)} "
          f"(per-cube cap {PER_CUBE:g}s)", flush=True)

    if remaining:
        # write remaining-cubes temp file; run the UNCHANGED solver_runner on it,
        # streaming each ##R## line into the checkpoint as it completes.
        tmp = HERE / f".tmp_remaining_{key}.json"
        tmp.write_text(json.dumps(remaining), encoding="utf-8")
        with ckpt.open("a", encoding="utf-8") as cf:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(SOLVER_RUNNER),
                 str(solver_file), str(tmp), str(PER_CUBE)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            contract_fail = None
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith("##R## "):
                    cf.write(line[len("##R## "):] + "\n")
                    cf.flush()
                elif line.startswith("##CONTRACT_FAIL## "):
                    contract_fail = line[len("##CONTRACT_FAIL## "):]
            proc.wait()
            if contract_fail:
                # solver failed to import -> record all remaining as contract fail
                for c in remaining:
                    cf.write(json.dumps({"id": c["id"], "solution": None,
                                         "error": f"contract_fail: {contract_fail}",
                                         "secs": 0.0}) + "\n")
                cf.flush()
        tmp.unlink(missing_ok=True)

    # --- summarize from the (now complete) checkpoint, re-verifying with cube.py ---
    done = load_done(ckpt)
    by_id = {c["id"]: c for c in cubes}
    results = []
    for cid in sorted(by_id):
        rec = done.get(cid, {"solution": None, "error": "missing", "secs": 0.0})
        cube = by_id[cid]
        solved, turns, err = False, None, rec.get("error")
        if rec.get("solution") is not None and not err:
            try:
                solved, turns = verify_solution(cube["facelet"], rec["solution"])
                if not solved:
                    err, turns = "wrong (did not reach solved)", None
            except Exception as exc:
                err = f"bad moves: {type(exc).__name__}: {str(exc)[:120]}"
        results.append({"id": cid, "hero": cube.get("hero", cid == 0), "solved": solved,
                        "turns": turns, "solution": rec.get("solution"),
                        "error": err, "secs": rec.get("secs", 0.0)})

    solved = [r for r in results if r["solved"]]
    turns = [r["turns"] for r in solved]
    hero = next((r for r in results if r["hero"]), None)
    summary = {
        "key": key, "ok": True, "n_cubes": len(results), "n_solved": len(solved),
        "solve_rate": round(len(solved) / len(results), 4) if results else 0.0,
        "hero_solved": bool(hero and hero["solved"]),
        "hero_turns": hero["turns"] if hero and hero["solved"] else None,
        "mean_turns": round(statistics.mean(turns), 2) if turns else None,
        "median_turns": statistics.median(turns) if turns else None,
        "min_turns": min(turns) if turns else None,
        "max_turns": max(turns) if turns else None,
        "n_timeout": sum(1 for r in results if r["error"] and "timeout" in str(r["error"])),
        "max_secs": round(max((r["secs"] for r in results), default=0.0), 1),
        "results": results,
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{key}] DONE: solved {len(solved)}/{len(results)}, mean {summary['mean_turns']}, "
          f"timeouts {summary['n_timeout']}, max {summary['max_secs']}s -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
