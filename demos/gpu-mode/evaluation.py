#!/usr/bin/env python3
"""
GPU MODE local evaluator.

Run the gpu-mode/reference-kernels eval.py harness locally against any
kernel file, without contacting the gpumode.com server. Reports the same
metrics (per-case timing + geomean) the server would report.

Usage:
    python evaluation.py <codefile> [--problem-dir DIR] [--mode MODE]

If --problem-dir is omitted, the directory containing <codefile> is used
(it must contain task.yml, task.py, reference.py).

<codefile> is the kernel solution. It will be copied in as submission.py
inside a temp dir, alongside the problem's reference.py / task.py / utils.py
/ eval.py, so the harness sees the exact layout it expects.

Modes:
    test         correctness only (fast)
    benchmark    timing on benchmark cases (default)
    leaderboard  timing + per-run correctness re-checks (most thorough)

Examples:
    # benchmark the qr_v2 seed kernel
    python evaluation.py init.py \
        --problem-dir reference-kernels/problems/linalg/qr_v2

    # correctness only
    python evaluation.py mykernel.py \
        --problem-dir reference-kernels/problems/linalg/qr_v2 \
        --mode test

    # full leaderboard-equivalent run
    python evaluation.py mykernel.py \
        --problem-dir reference-kernels/problems/linalg/qr_v2 \
        --mode leaderboard

Requires: torch + numpy in the active Python env (and PyYAML to parse task.yml;
without it only strict-JSON-per-line task.yml files are understood). Invoke with
that venv's interpreter, e.g.:
    .venv/bin/python evaluation.py init.py \
        --problem-dir reference-kernels/problems/linalg/qr_v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# ----- task.yml parsing ----------------------------------------------------

def _case_to_eval_line(case: dict) -> str:
    """Return eval.py's `key: val; ...` case format."""
    return "; ".join(f"{key}: {value}" for key, value in case.items())


def _yaml_available() -> bool:
    try:
        import yaml  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


def _load_task_yaml(path: Path) -> dict | None:
    """Load task.yml when PyYAML is present; keep the harness usable without it."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else None


def _extract_json_list_section(src: str, section: str) -> list[dict]:
    out = []
    in_section = False
    for line in src.splitlines():
        if re.match(rf"^{section}:\s*$", line):
            in_section = True
            continue
        if not in_section:
            continue
        if re.match(r"^\S", line):
            break
        if not line.strip() or not line.lstrip().startswith("-"):
            continue
        body = line.strip().lstrip("-").strip()
        try:
            out.append(json.loads(body))
        except json.JSONDecodeError:
            pass
    return out


def parse_task_yaml(path: Path) -> tuple[list[str], list[str]]:
    """Return (test_lines, benchmark_lines) in eval.py's `key: val; ...` format."""
    data = _load_task_yaml(path)
    if data:
        tests = data.get("tests") or []
        benches = data.get("benchmarks") or []
        return ([_case_to_eval_line(case) for case in tests],
                [_case_to_eval_line(case) for case in benches])

    src = path.read_text()
    return ([_case_to_eval_line(case) for case in _extract_json_list_section(src, "tests")],
            [_case_to_eval_line(case) for case in _extract_json_list_section(src, "benchmarks")])


def parse_task_files(path: Path) -> list[dict]:
    data = _load_task_yaml(path)
    if data and data.get("files"):
        return data["files"]
    return _extract_json_list_section(path.read_text(), "files")


# ----- problem-dir layout discovery ----------------------------------------

REQUIRED_IN_PROBLEM_DIR = ("task.yml",)
DEFAULT_FILES = (
    {"name": "submission.py", "source": "@SUBMISSION@"},
    {"name": "task.py", "source": "task.py"},
    {"name": "reference.py", "source": "reference.py"},
    {"name": "eval.py", "source": "eval.py"},
    {"name": "utils.py", "source": "utils.py"},
)


def find_shared_file(name: str, start: Path) -> Path:
    """Walk up from `start` until `name` is found, or raise."""
    d = start
    for _ in range(6):
        cand = d / name
        if cand.is_file():
            return cand
        if d == d.parent:
            break
        d = d.parent
    raise FileNotFoundError(f"could not find {name} starting from {start}")


def assemble_workdir(codefile: Path, problem_dir: Path, dest: Path) -> None:
    """Mirror the layout the server constructs from task.yml's `files:` list."""
    for fname in REQUIRED_IN_PROBLEM_DIR:
        src = problem_dir / fname
        if not src.is_file():
            sys.exit(f"[error] {problem_dir} missing required {fname}")
        shutil.copy(src, dest / fname)
    files = parse_task_files(problem_dir / "task.yml") or list(DEFAULT_FILES)
    for item in files:
        name = item.get("name")
        source = item.get("source")
        if not name or not source:
            continue
        dst = dest / name
        if source == "@SUBMISSION@":
            shutil.copy(codefile, dst)
            continue
        src = (problem_dir / source).resolve()
        if not src.is_file():
            try:
                src = find_shared_file(Path(source).name, problem_dir)
            except FileNotFoundError as e:
                sys.exit(f"[error] {e}")
        shutil.copy(src, dst)


# ----- log parsing & pretty print -----------------------------------------

def parse_log(text: str) -> dict:
    """Parse PopcornOutput's `key: value` lines into a nested dict."""
    out: dict = {}
    for line in text.splitlines():
        m = re.match(r"^([\w.\-]+):\s*(.*)$", line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
        cursor[parts[-1]] = val
    return out


def print_test_results(parsed: dict) -> int:
    n = parsed.get("test-count", 0)
    bucket = parsed.get("test", {})
    fails = 0
    print(f"\nCorrectness: {n} test cases\n" + "-" * 78)
    for i in range(n):
        rec = bucket.get(str(i), {})
        ok = rec.get("status") == "pass"
        mark = "PASS" if ok else "FAIL"
        spec = rec.get("spec", "?")
        msg = rec.get("message", rec.get("error", ""))
        print(f"  [{mark}] {spec}{(' — ' + str(msg)) if msg else ''}")
        if not ok:
            fails += 1
    overall = parsed.get("check", "?")
    print("-" * 78)
    print(f"Overall: {overall.upper()}  ({n - fails}/{n} passed)")
    return 0 if overall == "pass" else 1


def print_benchmark_results(parsed: dict) -> int:
    n = parsed.get("benchmark-count", 0)
    bucket = parsed.get("benchmark", {})
    overall = parsed.get("check", "?")
    print(f"\nBenchmarks: {n} cases (times in ms unless noted)\n" + "-" * 78)
    rows = []
    means_ms = []
    for i in range(n):
        rec = bucket.get(str(i), {})
        spec = rec.get("spec", "?")
        if rec.get("status") == "fail":
            err = rec.get("error", "")
            rows.append((spec, "FAIL", "", "", "", str(err)))
            continue
        mean_ns = rec.get("mean", 0)
        best_ns = rec.get("best", 0)
        worst_ns = rec.get("worst", 0)
        runs = rec.get("runs", 0)
        rows.append((spec, f"{mean_ns/1e6:.4f}", f"{best_ns/1e6:.4f}",
                     f"{worst_ns/1e6:.4f}", str(runs), ""))
        if mean_ns > 0:
            means_ms.append(mean_ns / 1e6)
    w_spec = max(len(r[0]) for r in rows) if rows else 0
    print(f"  {'spec'.ljust(w_spec)}  {'mean':>10}  {'best':>10}  {'worst':>10}  {'runs':>5}")
    for r in rows:
        if r[5]:
            print(f"  {r[0].ljust(w_spec)}  {'FAIL':>10}  {' ':>10}  {' ':>10}  {' ':>5}  {r[5]}")
        else:
            print(f"  {r[0].ljust(w_spec)}  {r[1]:>10}  {r[2]:>10}  {r[3]:>10}  {r[4]:>5}")
    print("-" * 78)
    print(f"Overall: {overall.upper()}", end="")
    if means_ms:
        geomean = math.exp(sum(math.log(m) for m in means_ms) / len(means_ms))
        total = sum(means_ms)
        print(f"  |  geomean: {geomean:.4f} ms  (sum-of-means: {total:.4f} ms)")
    else:
        print()
    return 0 if overall == "pass" else 1


# ----- main ----------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Run GPU MODE eval.py harness locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("codefile", help="Path to the kernel solution to evaluate.")
    p.add_argument("--problem-dir", default=None,
                   help="Problem directory containing task.yml/task.py/reference.py. "
                        "Defaults to the directory of <codefile>.")
    p.add_argument("--mode", default="benchmark",
                   choices=["test", "benchmark", "leaderboard"],
                   help="Evaluation mode (default: benchmark).")
    p.add_argument("--keep", action="store_true",
                   help="Don't delete the working directory (useful for debugging).")
    args = p.parse_args()

    codefile = Path(args.codefile).resolve()
    if not codefile.is_file():
        sys.exit(f"[error] code file not found: {codefile}")

    problem_dir = Path(args.problem_dir).resolve() if args.problem_dir else codefile.parent
    if not (problem_dir / "task.yml").is_file():
        sys.exit(f"[error] no task.yml in {problem_dir}. Pass --problem-dir.")

    tests, benches = parse_task_yaml(problem_dir / "task.yml")
    # Without PyYAML the fallback parser only handles strict-JSON list items at
    # column 0, so a normal task.yml silently yields nothing. Say so explicitly
    # rather than claiming the section is missing.
    hint = "" if _yaml_available() else (
        " — PyYAML is not installed and the fallback parser only understands "
        "strict-JSON list items at column 0; run `pip install pyyaml`"
    )
    if not tests and args.mode == "test":
        sys.exit(f"[error] no tests parsed from task.yml{hint}")
    if not benches and args.mode in ("benchmark", "leaderboard"):
        sys.exit(f"[error] no benchmarks parsed from task.yml{hint}")

    cases = tests if args.mode == "test" else benches
    work = Path(tempfile.mkdtemp(prefix="gpumode-eval-"))
    try:
        assemble_workdir(codefile, problem_dir, work)
        cases_path = work / ("tests.txt" if args.mode == "test" else "benchmarks.txt")
        cases_path.write_text("\n".join(cases) + "\n")

        print(f"problem:    {problem_dir}", file=sys.stderr)
        print(f"codefile:   {codefile}", file=sys.stderr)
        print(f"mode:       {args.mode}  ({len(cases)} cases)", file=sys.stderr)
        print(f"workdir:    {work}", file=sys.stderr)

        env = os.environ.copy()
        env["POPCORN_FD"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(
            [sys.executable, "eval.py", args.mode, cases_path.name],
            cwd=str(work), env=env, capture_output=True, text=True,
        )
        if proc.returncode not in (0, 112):
            print("[error] eval.py crashed:", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return proc.returncode

        parsed = parse_log(proc.stdout)
        if args.mode == "test":
            return print_test_results(parsed)
        return print_benchmark_results(parsed)
    finally:
        if args.keep:
            print(f"[kept workdir] {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
