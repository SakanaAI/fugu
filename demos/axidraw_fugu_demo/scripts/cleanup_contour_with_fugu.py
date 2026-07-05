#!/usr/bin/env python3
"""Clean a fast webcam contour plan with Sakana Fugu and preview it for AxiDraw."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import generate_axidraw_svg as generator
import vectorize_camera_frame as vectorizer


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


def compact_polyline(points: list[tuple[float, float]], max_points: int) -> list[list[float]]:
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) >= 0.6:
            cleaned.append(point)
    if len(cleaned) <= max_points:
        sampled = cleaned
    else:
        sampled = []
        for index in range(max_points):
            source_index = round(index * (len(cleaned) - 1) / float(max_points - 1))
            sampled.append(cleaned[source_index])
    return [[round(x, 2), round(y, 2)] for x, y in sampled]


def polyline_bbox(points: list[list[float]] | list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def is_probable_long_artifact(points: list[list[float]] | list[tuple[float, float]]) -> bool:
    if len(points) < 2:
        return True
    left, top, right, bottom = polyline_bbox(points)
    width = right - left
    height = bottom - top
    if width > 95.0 and height < 18.0:
        return True
    segments = [
        math.hypot(float(x2) - float(x1), float(y2) - float(y1))
        for (x1, y1), (x2, y2) in zip(points, points[1:])
    ]
    return bool(segments and max(segments) > 72.0 and height < 24.0)


def drop_probable_artifacts(plan: generator.SanitizedPlan) -> generator.SanitizedPlan:
    commands: list[dict[str, Any]] = []
    for command in plan.commands:
        if command["type"] == "line":
            points = [[command["x1"], command["y1"]], [command["x2"], command["y2"]]]
            if is_probable_long_artifact(points):
                continue
        elif command["type"] == "polyline" and is_probable_long_artifact(command["points"]):
            continue
        commands.append(command)
    return replace(plan, commands=commands)


def build_source_contours(args: argparse.Namespace) -> tuple[dict[str, Any], list[list[tuple[float, float]]]]:
    image = vectorizer.read_image(args)
    crop, crop_info = vectorizer.detect_face_crop(image)
    working = vectorizer.resize_for_edges(crop)
    edges = vectorizer.preprocess_edges(working)
    polylines_px = vectorizer.contours_to_polylines(
        edges,
        args.simplify_eps,
        max(12, args.max_source_polylines),
    )
    if not polylines_px:
        raise RuntimeError("no usable contours were found in the webcam frame")

    polylines = vectorizer.fit_to_page(
        polylines_px,
        working.shape[1],
        working.shape[0],
        args.plot_width_mm,
        args.plot_height_mm,
    )
    polylines = vectorizer.sort_polylines_nearest(polylines)
    compact = [
        compact_polyline(line, args.max_points_per_source_line)
        for line in polylines[: args.max_source_polylines]
        if len(line) >= 2
    ]
    compact = [line for line in compact if len(line) >= 2 and not is_probable_long_artifact(line)]
    point_count = sum(len(line) for line in compact)
    source_plan = {
        "title": "webcam_contour_cleanup_source",
        "description": "Fast OpenCV contour geometry from the real webcam frame.",
        "commands": [{"type": "polyline", "points": line} for line in compact],
        "metrics": {
            "command_count": len(compact),
            "point_count": point_count,
            "source_image_px": [int(image.shape[1]), int(image.shape[0])],
            "working_image_px": [int(working.shape[1]), int(working.shape[0])],
            "plot_area_mm": [round(args.plot_width_mm, 2), round(args.plot_height_mm, 2)],
            **crop_info,
        },
        "warnings": [],
    }
    print(f"source payload chars: {len(json.dumps(source_plan, separators=(',', ':')))}")
    return source_plan, [[(float(x), float(y)) for x, y in line] for line in compact]


def run_preview(svg_path: Path) -> tuple[dict[str, Any], str]:
    proc = subprocess.run(
        [python_path(), str(SCRIPT_DIR / "simulate_axidraw_preview.py"), str(svg_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(part for part in [proc.stdout, proc.stderr] if part)
    report_path = svg_path.with_name(f"{svg_path.stem}.axidraw_preview.report.json")
    if proc.returncode == 0 and report_path.exists():
        return json.loads(report_path.read_text(encoding="utf-8")), combined

    preview_svg_path = svg_path.with_name(f"{svg_path.stem}.axidraw_preview.svg")
    preview_svg_path.write_text(svg_path.read_text(encoding="utf-8"), encoding="utf-8")
    report = {
        "input_svg": str(svg_path),
        "preview_svg": str(preview_svg_path),
        "command": [],
        "returncode": proc.returncode,
        "metrics": {},
        "warnings": ["AxiDraw axicli preview is unavailable; showing generated SVG directly."],
        "raw_output": combined,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report, combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean webcam contours with Sakana Fugu.")
    parser.add_argument("--model", choices=["fugu", "fugu-ultra"], default="fugu")
    parser.add_argument("--request", required=True)
    parser.add_argument("--image-data-file", type=Path, help="Text file containing a data:image/... URL from the webcam.")
    parser.add_argument("--image", type=Path, help="Read an image file instead of a data URL.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--simplify-eps", type=float, default=2.35)
    parser.add_argument("--max-source-polylines", type=int, default=14)
    parser.add_argument("--max-points-per-source-line", type=int, default=8)
    parser.add_argument("--plot-width-mm", type=float, default=170.0)
    parser.add_argument("--plot-height-mm", type=float, default=132.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.image_data_file and not args.image:
        raise ValueError("pass --image-data-file or --image")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    source_plan, source_polylines = build_source_contours(args)
    source_title = f"webcam_contour_cleanup_source_{timestamp}"
    source_plan["title"] = source_title
    source_plan_path = args.out_dir / f"{source_title}.plan.json"
    source_svg_path = args.out_dir / f"{source_title}.svg"
    source_plan_path.write_text(json.dumps(source_plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    vectorizer.write_svg(
        source_svg_path,
        source_title,
        "Compacted source contour plan before Fugu cleanup.",
        source_polylines,
    )
    print(f"wrote source {source_plan_path}")
    print(f"source contours: {source_plan['metrics']['command_count']}")
    print(f"source points: {source_plan['metrics']['point_count']}")
    print(f"face crop: {source_plan['metrics'].get('face_detected')}")

    raw_plan = generator.call_fugu_contour_cleanup(
        args.model,
        args.request,
        source_plan,
    )
    plan = generator.sanitize_plan(raw_plan, source_model=args.model)
    plan = drop_probable_artifacts(plan)
    base_title = generator._clean_title(plan.title)[:42].strip("_") or "webcam_contour_cleanup"
    unique_title = f"fugu_{base_title}_{timestamp}"
    plan = replace(plan, title=unique_title)
    json_path, svg_path = generator.write_outputs(plan, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {svg_path}")
    print(f"commands: {len(plan.commands)}")

    report, preview_output = run_preview(svg_path)
    if preview_output.strip():
        print(preview_output.strip())
    item_warnings = [*plan.warnings, *report.get("warnings", [])]
    item = {
        "demo": plan.title,
        "description": plan.description or "Fugu-cleaned webcam contour plot.",
        "category": "Generated from Webcam Contours",
        "model": args.model,
        "prompt": args.request,
        "source": "webcam_fugu_contour_cleanup",
        "source_contour_plan": str(source_plan_path),
        "source_contour_svg": str(source_svg_path),
        "svg": str(svg_path),
        "preview_svg": report["preview_svg"],
        "report_json": str(svg_path.with_name(f"{svg_path.stem}.axidraw_preview.report.json")),
        "metrics": report.get("metrics", {}),
        "warning_count": len(item_warnings),
        "warnings": item_warnings,
        "vector_metrics": source_plan["metrics"],
    }
    write_generated_summary(item)
    print(f"generated demo: {plan.title}")
    print(f"updated {GENERATED_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
