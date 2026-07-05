#!/usr/bin/env python3
"""Vectorize one webcam frame into plotter-friendly SVG contours."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
GENERATED_SUMMARY = OUTPUT_DIR / "generated_summary.json"

PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0
MARGIN_MM = 13.0
DEFAULT_PLOT_WIDTH_MM = 155.0
DEFAULT_PLOT_HEIGHT_MM = 120.0
STROKE_WIDTH_MM = 0.28
TARGET_LONG_SIDE_PX = 620
MIN_CONTOUR_LEN = 24
MAX_POLYLINES = 28
MAX_POINTS_PER_LINE = 220


def clean_title(raw: str) -> str:
    title = raw.strip().lower()
    title = re.sub(r"[^a-z0-9_ -]+", "", title)
    title = re.sub(r"[\s-]+", "_", title).strip("_")
    return title[:64] or "webcam_contour"


def decode_data_url(data_url: str) -> np.ndarray:
    match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url.strip(), flags=re.DOTALL)
    if not match:
        raise ValueError("expected a base64 data:image/... URL")
    raw = base64.b64decode(match.group(1), validate=True)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode the webcam image")
    return image


def read_image(args: argparse.Namespace) -> np.ndarray:
    if args.image_data_file:
        return decode_data_url(args.image_data_file.read_text(encoding="utf-8"))
    if args.image:
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV could not read image: {args.image}")
        return image
    raise ValueError("pass --image-data-file or --image")


def detect_face_crop(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    info: dict[str, Any] = {"face_detected": False, "crop": "center"}
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        detector = cv2.CascadeClassifier(str(cascade_path))
        faces = detector.detectMultiScale(
            cv2.equalizeHist(gray),
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(70, 70),
        )
        if len(faces):
            x, y, w, h = max(faces, key=lambda face: int(face[2]) * int(face[3]))
            cx = x + w / 2.0
            cy = y + h / 2.0
            crop_w = w * 2.65
            crop_h = h * 3.05
            left = int(max(0, cx - crop_w / 2.0))
            top = int(max(0, cy - crop_h * 0.42))
            right = int(min(image.shape[1], cx + crop_w / 2.0))
            bottom = int(min(image.shape[0], top + crop_h))
            info.update(
                {
                    "face_detected": True,
                    "crop": "face",
                    "face_box": [int(x), int(y), int(w), int(h)],
                    "crop_box": [left, top, right - left, bottom - top],
                }
            )
            return image[top:bottom, left:right], info

    h, w = image.shape[:2]
    crop_w = int(min(w, h * 1.18))
    crop_h = int(min(h, crop_w / 1.18))
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    info["crop_box"] = [left, top, crop_w, crop_h]
    return image[top : top + crop_h, left : left + crop_w], info


def resize_for_edges(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = TARGET_LONG_SIDE_PX / float(max(h, w))
    if scale >= 1.0:
        return image.copy()
    return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def preprocess_edges(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)

    median = float(np.median(gray))
    lo = int(max(0, 0.55 * median))
    hi = int(min(255, max(lo + 20, 1.28 * median)))
    canny = cv2.Canny(gray, lo, hi)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        9,
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    edges = cv2.bitwise_or(canny, adaptive)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)


def contours_to_polylines(edges: np.ndarray, simplify_eps: float, max_polylines: int) -> list[list[tuple[float, float]]]:
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates: list[tuple[float, list[tuple[float, float]]]] = []
    for contour in contours:
        if len(contour) < MIN_CONTOUR_LEN:
            continue
        arc_len = float(cv2.arcLength(contour, closed=False))
        if arc_len < 18.0:
            continue
        simplified = cv2.approxPolyDP(contour, simplify_eps, closed=False).reshape(-1, 2)
        if len(simplified) < 2:
            continue
        if len(simplified) > MAX_POINTS_PER_LINE:
            simplified = simplified[:: math.ceil(len(simplified) / MAX_POINTS_PER_LINE)]
        candidates.append((arc_len, [(float(x), float(y)) for x, y in simplified]))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in candidates[:max_polylines]]


def fit_to_page(
    polylines: list[list[tuple[float, float]]],
    width_px: int,
    height_px: int,
    plot_width_mm: float,
    plot_height_mm: float,
) -> list[list[tuple[float, float]]]:
    drawable_w = min(PAGE_WIDTH_MM - MARGIN_MM * 2.0, plot_width_mm)
    drawable_h = min(PAGE_HEIGHT_MM - MARGIN_MM * 2.0, plot_height_mm)
    scale = min(drawable_w / float(width_px), drawable_h / float(height_px))
    offset_x = (PAGE_WIDTH_MM - width_px * scale) / 2.0
    offset_y = (PAGE_HEIGHT_MM - height_px * scale) / 2.0
    return [
        [(round(offset_x + x * scale, 3), round(offset_y + y * scale, 3)) for x, y in line]
        for line in polylines
    ]


def estimate_distance(polylines: list[list[tuple[float, float]]]) -> float:
    total = 0.0
    for line in polylines:
        for (x1, y1), (x2, y2) in zip(line, line[1:]):
            total += math.hypot(x2 - x1, y2 - y1)
    return round(total, 1)


def sort_polylines_nearest(polylines: list[list[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    remaining = [list(line) for line in polylines if len(line) >= 2]
    ordered: list[list[tuple[float, float]]] = []
    cursor = (MARGIN_MM, MARGIN_MM)
    while remaining:
        best_index = 0
        best_reverse = False
        best_distance = float("inf")
        for index, line in enumerate(remaining):
            start_distance = math.hypot(line[0][0] - cursor[0], line[0][1] - cursor[1])
            end_distance = math.hypot(line[-1][0] - cursor[0], line[-1][1] - cursor[1])
            if start_distance < best_distance:
                best_index = index
                best_reverse = False
                best_distance = start_distance
            if end_distance < best_distance:
                best_index = index
                best_reverse = True
                best_distance = end_distance
        chosen = remaining.pop(best_index)
        if best_reverse:
            chosen.reverse()
        ordered.append(chosen)
        cursor = chosen[-1]
    return ordered


def write_svg(path: Path, title: str, description: str, polylines: list[list[tuple[float, float]]]) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH_MM}mm" '
            f'height="{PAGE_HEIGHT_MM}mm" viewBox="0 0 {PAGE_WIDTH_MM} {PAGE_HEIGHT_MM}">'
        ),
        f"  <title>{title}</title>",
        f"  <desc>{description}</desc>",
        (
            '  <g fill="none" stroke="black" '
            f'stroke-width="{STROKE_WIDTH_MM}" stroke-linecap="round" stroke-linejoin="round">'
        ),
    ]
    for line in polylines:
        if len(line) >= 2:
            points = " ".join(f"{x},{y}" for x, y in line)
            lines.append(f'    <polyline points="{points}" />')
    lines.extend(["  </g>", "</svg>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_vpype_optimize(raw_svg: Path, final_svg: Path) -> list[str]:
    vpype = shutil.which("vpype")
    if not vpype:
        shutil.copyfile(raw_svg, final_svg)
        return []

    cmd = [
        vpype,
        "read",
        str(raw_svg),
        "linemerge",
        "--tolerance",
        "0.2mm",
        "linesimplify",
        "--tolerance",
        "0.08mm",
        "reloop",
        "linesort",
        "write",
        str(final_svg),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False, timeout=45)
    if proc.returncode == 0:
        return []
    shutil.copyfile(raw_svg, final_svg)
    detail = " ".join(part.strip() for part in [proc.stdout, proc.stderr] if part.strip())
    return [f"vpype optimize failed; using raw contours: {detail[:240]}"]


def run_preview(svg_path: Path) -> tuple[dict[str, Any], str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "simulate_axidraw_preview.py"), str(svg_path)],
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


def load_generated_summary() -> list[dict[str, Any]]:
    if not GENERATED_SUMMARY.exists():
        return []
    data = json.loads(GENERATED_SUMMARY.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def write_generated_summary(item: dict[str, Any]) -> None:
    existing = [entry for entry in load_generated_summary() if entry.get("demo") != item["demo"]]
    GENERATED_SUMMARY.write_text(
        json.dumps([item, *existing][:12], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorize one webcam frame into an AxiDraw preview SVG.")
    parser.add_argument("--image-data-file", type=Path, help="Text file containing a data:image/... URL from the webcam.")
    parser.add_argument("--image", type=Path, help="Read an image file instead of a data URL.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--title", default="webcam_contour_portrait")
    parser.add_argument("--simplify-eps", type=float, default=2.35)
    parser.add_argument("--max-polylines", type=int, default=MAX_POLYLINES)
    parser.add_argument("--plot-width-mm", type=float, default=DEFAULT_PLOT_WIDTH_MM)
    parser.add_argument("--plot-height-mm", type=float, default=DEFAULT_PLOT_HEIGHT_MM)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(args)
    crop, crop_info = detect_face_crop(image)
    working = resize_for_edges(crop)
    edges = preprocess_edges(working)
    polylines_px = contours_to_polylines(edges, args.simplify_eps, max(20, args.max_polylines))
    if not polylines_px:
        raise RuntimeError("no usable contours were found in the webcam frame")
    polylines = fit_to_page(
        polylines_px,
        working.shape[1],
        working.shape[0],
        args.plot_width_mm,
        args.plot_height_mm,
    )
    polylines = sort_polylines_nearest(polylines)

    title = f"{clean_title(args.title)}_{time.strftime('%Y%m%d_%H%M%S')}"
    description = "OpenCV Canny and adaptive-threshold contours from the webcam frame."
    raw_svg = args.out_dir / f"{title}.raw.svg"
    svg_path = args.out_dir / f"{title}.svg"
    plan_path = args.out_dir / f"{title}.plan.json"

    write_svg(raw_svg, title, description, polylines)
    warnings = maybe_vpype_optimize(raw_svg, svg_path)
    point_count = sum(len(line) for line in polylines)
    vector_metrics = {
        "command_count": len(polylines),
        "point_count": point_count,
        "estimated_pen_distance_mm": estimate_distance(polylines),
        "plot_area_mm": [round(args.plot_width_mm, 2), round(args.plot_height_mm, 2)],
        "source_image_px": [int(image.shape[1]), int(image.shape[0])],
        "working_image_px": [int(working.shape[1]), int(working.shape[0])],
        **crop_info,
    }
    plan_path.write_text(
        json.dumps(
            {
                "title": title,
                "description": description,
                "source_model": "opencv-contours",
                "page": {"width_mm": PAGE_WIDTH_MM, "height_mm": PAGE_HEIGHT_MM},
                "commands": [{"type": "polyline", "points": line} for line in polylines],
                "metrics": vector_metrics,
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"wrote {plan_path}")
    print(f"wrote {svg_path}")
    print(f"contours: {len(polylines)}")
    print(f"points: {point_count}")
    print(f"face crop: {crop_info.get('face_detected')}")
    for warning in warnings:
        print(f"warning: {warning}")

    report, preview_output = run_preview(svg_path)
    if preview_output.strip():
        print(preview_output.strip())
    item_warnings = [*warnings, *report.get("warnings", [])]
    write_generated_summary(
        {
            "demo": title,
            "description": description,
            "category": "Generated from Webcam Contours",
            "model": "opencv-contours",
            "prompt": "webcam contour portrait",
            "source": "webcam_contour",
            "svg": str(svg_path),
            "preview_svg": report["preview_svg"],
            "report_json": str(svg_path.with_name(f"{svg_path.stem}.axidraw_preview.report.json")),
            "metrics": report.get("metrics", {}),
            "warning_count": len(item_warnings),
            "warnings": item_warnings,
            "vector_metrics": vector_metrics,
        }
    )
    print(f"generated demo: {title}")
    print(f"updated {GENERATED_SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
