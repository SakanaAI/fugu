#!/usr/bin/env python3
"""Run AxiDraw's official offline Plot Preview against a generated SVG.

This script shells out to axicli with preview enabled. It does not communicate
with USB/serial hardware and does not move an AxiDraw.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "outputs" / "offline_axidraw_demo.svg"


def resolve_axicli(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("AXICLI_PATH", "").strip()):
        if not candidate:
            continue
        explicit_path = Path(candidate)
        if explicit_path.exists():
            return str(explicit_path)
        found_explicit = shutil.which(candidate)
        if found_explicit:
            return found_explicit
        raise RuntimeError(f"axicli was not found at {candidate}.")
    found = shutil.which("axicli")
    if found:
        return found
    raise RuntimeError(
        "axicli was not found. Install AxiDraw API/CLI, set AXICLI_PATH, or pass --axicli /path/to/axicli."
    )


def parse_preview_output(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    patterns = {
        "estimated_print_time": r"Estimated print time:\s*(.+)",
        "path_to_draw_m": r"Length of path to draw:\s*([0-9.]+)\s*m",
        "pen_up_travel_m": r"Pen-up travel distance:\s*([0-9.]+)\s*m",
        "total_movement_m": r"Total movement distance:\s*([0-9.]+)\s*m",
        "estimate_runtime": r"This estimate took\s*(.+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(1).strip()
        metrics[key] = float(value) if value.replace(".", "", 1).isdigit() else value

    warnings: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        if block.strip().startswith(("Note", "Warning")):
            warnings.append(" ".join(block.split()))
    return {"metrics": metrics, "warnings": warnings, "raw_output": text}


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline-simulate an SVG with AxiDraw Plot Preview.")
    parser.add_argument("svg", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--axicli", help="Path to axicli. Defaults to local miniconda axicli or PATH.")
    parser.add_argument("--model", type=int, default=1, help="AxiDraw model code. 1 is V2/V3/SE-A4.")
    parser.add_argument("--rendering", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--out-svg", type=Path, help="Rendered preview SVG path.")
    parser.add_argument("--report-json", type=Path, help="Write parsed simulation report JSON.")
    parser.add_argument("--strict-warnings", action="store_true", help="Exit non-zero if axicli emits notes/warnings.")
    args = parser.parse_args()

    svg = args.svg.resolve()
    if not svg.exists():
        print(f"SVG not found: {svg}", file=sys.stderr)
        return 2

    out_svg = args.out_svg or svg.with_name(f"{svg.stem}.axidraw_preview.svg")
    report_json = args.report_json or svg.with_name(f"{svg.stem}.axidraw_preview.report.json")
    try:
        axicli = resolve_axicli(args.axicli)
    except RuntimeError as exc:
        out_svg.write_text(svg.read_text(encoding="utf-8"), encoding="utf-8")
        report = {
            "input_svg": str(svg),
            "preview_svg": str(out_svg),
            "command": [],
            "returncode": 0,
            "metrics": {},
            "warnings": [str(exc), "AxiDraw axicli preview is unavailable; showing generated SVG directly."],
            "raw_output": str(exc),
        }
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(str(exc), file=sys.stderr)
        print("AxiDraw preview fallback: showing generated SVG directly.")
        print(f"wrote preview SVG: {out_svg}")
        print(f"wrote report JSON: {report_json}")
        return 0

    cmd = [
        axicli,
        str(svg),
        "-vT",
        f"-g{args.rendering}",
        f"-L{args.model}",
        "-o",
        str(out_svg),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    combined = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    report = parse_preview_output(combined)
    report.update(
        {
            "input_svg": str(svg),
            "preview_svg": str(out_svg),
            "command": cmd,
            "returncode": proc.returncode,
        }
    )

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(combined.strip())
    print(f"wrote preview SVG: {out_svg}")
    print(f"wrote report JSON: {report_json}")

    if proc.returncode != 0:
        return proc.returncode
    if args.strict_warnings and report["warnings"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
