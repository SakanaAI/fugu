#!/usr/bin/env python3
"""Generate a Sakana Fugu prompt-based AxiDraw demo and immediately Plot Preview it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import generate_axidraw_svg as generator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
GENERATED_SUMMARY = OUTPUT_DIR / "generated_summary.json"


def python_path() -> str:
    return sys.executable


def load_generated_summary() -> list[dict[str, Any]]:
    if not GENERATED_SUMMARY.exists():
        return []
    data = json.loads(GENERATED_SUMMARY.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def write_generated_summary(item: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [entry for entry in load_generated_summary() if entry.get("demo") != item["demo"]]
    GENERATED_SUMMARY.write_text(
        json.dumps([item, *existing][:12], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one prompt-to-plotter demo with Sakana Fugu.")
    parser.add_argument("--model", choices=["fugu", "fugu-ultra"], default="fugu")
    parser.add_argument("--request", required=True)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    raw_plan = generator.call_fugu(args.model, args.request)
    plan = generator.sanitize_plan(raw_plan, source_model=args.model)
    base_title = generator._clean_title(plan.title)[:42].strip("_") or "prompt_plot"
    unique_title = f"fugu_{base_title}_{time.strftime('%Y%m%d_%H%M%S')}"
    plan = replace(plan, title=unique_title)
    json_path, svg_path = generator.write_outputs(plan, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {svg_path}")
    print(f"commands: {len(plan.commands)}")

    preview = subprocess.run(
        [python_path(), str(SCRIPT_DIR / "simulate_axidraw_preview.py"), str(svg_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in [preview.stdout, preview.stderr] if part)
    if combined.strip():
        print(combined.strip())
    report_path = svg_path.with_name(f"{svg_path.stem}.axidraw_preview.report.json")
    if preview.returncode == 0 and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        preview_svg_path = svg_path.with_name(f"{svg_path.stem}.axidraw_preview.svg")
        preview_svg_path.write_text(svg_path.read_text(encoding="utf-8"), encoding="utf-8")
        report = {
            "input_svg": str(svg_path),
            "preview_svg": str(preview_svg_path),
            "command": [],
            "returncode": preview.returncode,
            "metrics": {},
            "warnings": ["AxiDraw axicli preview is unavailable; showing generated SVG directly."],
            "raw_output": combined,
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("AxiDraw preview fallback: showing generated SVG directly.")
    item_warnings = [*plan.warnings, *report.get("warnings", [])]
    item = {
        "demo": plan.title,
        "description": plan.description or args.request[:180],
        "category": "Generated with Sakana Fugu",
        "model": args.model,
        "prompt": args.request,
        "source": "prompt",
        "svg": str(svg_path),
        "preview_svg": report["preview_svg"],
        "report_json": str(report_path),
        "metrics": report["metrics"],
        "warning_count": len(item_warnings),
        "warnings": item_warnings,
    }
    write_generated_summary(item)
    print(f"generated demo: {plan.title}")
    print(f"updated {GENERATED_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
