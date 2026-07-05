#!/usr/bin/env python3
"""Generate a safe AxiDraw-oriented SVG from a constrained Sakana Fugu drawing plan.

Default mode is offline and hardware-free: it validates a built-in sample plan,
then writes JSON and SVG files under ./outputs. Use --online to request a fresh
plan from Sakana Fugu. This script never invokes axicli.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = (os.environ.get("SAKANA_BASE_URL") or "https://api.sakana.ai/v1").rstrip("/")
DEFAULT_MODEL = "fugu"
DISPLAY_MODEL_NAME = "Sakana Fugu"
API_TIMEOUT_SECONDS = int(os.environ.get("FUGU_API_TIMEOUT_SECONDS", "45"))
MAX_OUTPUT_TOKENS = int(os.environ.get("FUGU_MAX_OUTPUT_TOKENS", "1536"))
FUGU_INSTRUCTIONS = os.environ.get(
    "FUGU_INSTRUCTIONS",
    "You convert user drawing requests into safe AxiDraw JSON plans. "
    "Answer with valid JSON only and do not fabricate unsupported primitives.",
)
PAGE_WIDTH_MM = 297.0
PAGE_HEIGHT_MM = 210.0
MAX_COMMANDS = 80
MAX_POINTS_PER_POLYLINE = 80
ALLOWED_COMMANDS = {"line", "polyline", "circle", "text"}

GLYPHS: dict[str, list[str]] = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ">": ["10000", "01000", "00100", "00010", "00100", "01000", "10000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
}


SAMPLE_PLAN: dict[str, Any] = {
    "title": "offline_axidraw_demo",
    "description": "A deterministic sample plan that requires no API key.",
    "commands": [
        {"type": "line", "x1": 35, "y1": 40, "x2": 262, "y2": 40},
        {
            "type": "polyline",
            "points": [[40, 135], [75, 82], [112, 122], [150, 72], [190, 122], [225, 82], [257, 135]],
        },
        {"type": "circle", "cx": 148.5, "cy": 105, "r": 23},
        {"type": "line", "x1": 96, "y1": 165, "x2": 201, "y2": 165},
        {"type": "text", "x": 92, "y": 188, "value": "Sakana Fugu -> SVG -> AxiDraw", "size": 6},
    ],
}


PROMPT_TEMPLATE = """\
Create a compact AxiDraw v3 pen-plotter artwork plan as JSON only.

Canvas:
- Units are millimeters.
- Width: 297, height: 210.
- Keep all geometry inside the canvas with at least 10 mm margin.
- Use 4 to 6 commands total.
- Keep each polyline under 12 points.
- Keep the whole JSON response under 1000 characters.
- Think like a single black pen on white paper: clear silhouette, a few contour lines, ripples, hatching, and graceful curves.
- Prefer image-like line art over diagrams.
- Avoid UI boxes, flowcharts, arrows, legends, model names, worker nodes, and explanatory labels unless the user explicitly asks for a diagram.
- Avoid text unless explicitly requested.

Allowed command objects:
- {{"type":"line","x1":number,"y1":number,"x2":number,"y2":number}}
- {{"type":"polyline","points":[[x,y], ...]}}
- {{"type":"circle","cx":number,"cy":number,"r":number}}
- {{"type":"text","x":number,"y":number,"value":"short label","size":number}}

Return exactly this top-level JSON shape and no markdown:
{{"title":"snake_case_title","description":"one sentence","commands":[...]}}

User drawing request: {request}
"""


CONTOUR_CLEANUP_TEMPLATE = """\
Clean up this webcam contour plan for an AxiDraw v3 pen plotter. Return JSON only.

You are given machine-generated OpenCV contour polylines from the actual webcam frame.
Use those contours as the source of truth. Keep the drawing realistic and recognizable:
- Preserve the subject outline, face contour, hair silhouette, glasses/eyes/mouth, shoulders, and any clear room cues already present.
- Keep coordinates close to the source contour plan. Do not invent unrelated objects, decorative symbols, text, diagrams, or labels.
- Remove noisy speckles, duplicate fragments, tiny jagged strokes, and accidental long crossing lines.
- Merge nearby contour fragments into smoother polylines when doing so preserves the real shape.
- Optimize for AxiDraw: continuous contours, fewer pen lifts, no dense hatching, no fills, no shading.

Visual target:
- Use a ShadowDraw-like sketch style: loose freehand observational line drawing, not a geometric icon.
- Improve face spacing and proportions while preserving the subject's own look and pose.
- Prefer confident curved contour strokes that suggest a real portrait: head/hair outline, aligned glasses or eyes, centered nose, mouth, neck, and shoulders.
- Keep moderate portrait detail: several hair contour strokes, glasses with bridge/eyes, nose, mouth, jaw, neck, shoulders, and a few pose-defining interior contours.
- Do not reduce the face to a smiley icon. Preserve asymmetry, head tilt, curly/irregular hair, and real portrait proportions.
- Avoid low-poly triangles, floating symbols, perfect diagram geometry, or cartoon abstractions.
- Treat weak/noisy contour fragments like faint shadows: use only the strokes that agree with the portrait structure.
- If a cropped webcam reference image is attached, use it only to recover real details such as hair, glasses, face angle, and shoulders. The contour plan is still the geometry scaffold.

Canvas:
- Units are millimeters.
- Width: 297, height: 210.
- Keep all geometry inside the canvas with at least 10 mm margin.
- Use 12 to 22 commands total.
- Prefer polyline commands; use circles only for clear circular features such as glasses.
- Keep each polyline under 14 points.
- Keep the response under 2400 characters.

Allowed command objects:
- {{"type":"line","x1":number,"y1":number,"x2":number,"y2":number}}
- {{"type":"polyline","points":[[x,y], ...]}}
- {{"type":"circle","cx":number,"cy":number,"r":number}}

Return exactly this top-level JSON shape and no markdown:
{{"title":"webcam_contour_cleanup","description":"one sentence","commands":[...]}}

User goal: {request}

Source contour plan:
{source_plan}
"""


@dataclass(frozen=True)
class SanitizedPlan:
    title: str
    description: str
    commands: list[dict[str, Any]]
    warnings: list[str]
    source_model: str = "offline"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _number(raw: Any, default: float = 0.0) -> float:
    if isinstance(raw, bool):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _point(raw: Any, warnings: list[str], label: str) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        warnings.append(f"dropped malformed point for {label}")
        return None
    x = _clamp(_number(raw[0]), 0.0, PAGE_WIDTH_MM)
    y = _clamp(_number(raw[1]), 0.0, PAGE_HEIGHT_MM)
    return [round(x, 3), round(y, 3)]


def _clean_title(raw: Any) -> str:
    title = str(raw or "axidraw_demo").strip().lower()
    title = re.sub(r"[^a-z0-9_ -]+", "", title)
    title = re.sub(r"[\s-]+", "_", title).strip("_")
    return title[:64] or "axidraw_demo"


def sanitize_plan(raw_plan: dict[str, Any], source_model: str = "offline") -> SanitizedPlan:
    raw_warnings = raw_plan.get("warnings")
    warnings: list[str] = [str(warning)[:240] for warning in raw_warnings] if isinstance(raw_warnings, list) else []
    title = _clean_title(raw_plan.get("title"))
    description = str(raw_plan.get("description") or "").strip()[:240]
    raw_commands = raw_plan.get("commands")
    if not isinstance(raw_commands, list):
        raw_commands = []
        warnings.append("missing commands list; emitted an empty plan")

    commands: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_commands[:MAX_COMMANDS]):
        if not isinstance(raw, dict):
            warnings.append(f"dropped command {index}: not an object")
            continue
        command_type = str(raw.get("type") or "").lower()
        if command_type not in ALLOWED_COMMANDS:
            warnings.append(f"dropped command {index}: unsupported type {command_type!r}")
            continue

        if command_type == "line":
            commands.append(
                {
                    "type": "line",
                    "x1": round(_clamp(_number(raw.get("x1")), 0.0, PAGE_WIDTH_MM), 3),
                    "y1": round(_clamp(_number(raw.get("y1")), 0.0, PAGE_HEIGHT_MM), 3),
                    "x2": round(_clamp(_number(raw.get("x2")), 0.0, PAGE_WIDTH_MM), 3),
                    "y2": round(_clamp(_number(raw.get("y2")), 0.0, PAGE_HEIGHT_MM), 3),
                }
            )
        elif command_type == "polyline":
            raw_points = raw.get("points")
            if not isinstance(raw_points, list):
                warnings.append(f"dropped command {index}: polyline points must be a list")
                continue
            points = [_point(point, warnings, f"polyline {index}") for point in raw_points[:MAX_POINTS_PER_POLYLINE]]
            clean_points = [point for point in points if point is not None]
            if len(clean_points) < 2:
                warnings.append(f"dropped command {index}: polyline needs at least two valid points")
                continue
            commands.append({"type": "polyline", "points": clean_points})
        elif command_type == "circle":
            cx = _clamp(_number(raw.get("cx")), 0.0, PAGE_WIDTH_MM)
            cy = _clamp(_number(raw.get("cy")), 0.0, PAGE_HEIGHT_MM)
            max_radius = min(cx, cy, PAGE_WIDTH_MM - cx, PAGE_HEIGHT_MM - cy)
            radius = _clamp(_number(raw.get("r"), 1.0), 0.5, max(0.5, max_radius))
            commands.append({"type": "circle", "cx": round(cx, 3), "cy": round(cy, 3), "r": round(radius, 3)})
        elif command_type == "text":
            value = str(raw.get("value") or "")[:80]
            if not value:
                warnings.append(f"dropped command {index}: empty text")
                continue
            commands.append(
                {
                    "type": "text",
                    "x": round(_clamp(_number(raw.get("x")), 0.0, PAGE_WIDTH_MM), 3),
                    "y": round(_clamp(_number(raw.get("y")), 0.0, PAGE_HEIGHT_MM), 3),
                    "value": value,
                    "size": round(_clamp(_number(raw.get("size"), 6.0), 2.0, 16.0), 3),
                }
            )

    if len(raw_commands) > MAX_COMMANDS:
        warnings.append(f"truncated command list from {len(raw_commands)} to {MAX_COMMANDS}")

    return SanitizedPlan(
        title=title,
        description=description,
        commands=commands,
        warnings=warnings,
        source_model=source_model,
    )


def estimate_pen_distance_mm(commands: list[dict[str, Any]]) -> float:
    total = 0.0
    for command in commands:
        if command["type"] == "line":
            dx = command["x2"] - command["x1"]
            dy = command["y2"] - command["y1"]
            total += (dx * dx + dy * dy) ** 0.5
        elif command["type"] == "polyline":
            points = command["points"]
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                dx = x2 - x1
                dy = y2 - y1
                total += (dx * dx + dy * dy) ** 0.5
        elif command["type"] == "circle":
            total += 2 * 3.14159 * command["r"]
        elif command["type"] == "text":
            total += len(command["value"]) * command["size"] * 0.7
    return round(total, 1)


def text_command_to_paths(command: dict[str, Any]) -> list[str]:
    text = command["value"].upper()
    cell = command["size"] / 7.0
    spacing = cell
    char_width = 5 * cell + spacing
    origin_x = command["x"]
    origin_y = command["y"] - command["size"]
    paths: list[str] = []
    for char_index, char in enumerate(text):
        if char == " ":
            continue
        glyph = GLYPHS.get(char)
        if not glyph:
            continue
        char_x = origin_x + char_index * char_width
        for row, pattern in enumerate(glyph):
            for col, bit in enumerate(pattern):
                if bit != "1":
                    continue
                x = round(char_x + col * cell, 3)
                y = round(origin_y + row * cell, 3)
                s = round(cell * 0.72, 3)
                paths.append(f'M {x} {y} h {s} v {s} h {-s} Z')
    return paths


def plan_to_svg(plan: SanitizedPlan) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH_MM}mm" '
            f'height="{PAGE_HEIGHT_MM}mm" viewBox="0 0 {PAGE_WIDTH_MM} {PAGE_HEIGHT_MM}">'
        ),
        f"  <title>{html.escape(plan.title)}</title>",
        f"  <desc>{html.escape(plan.description)}</desc>",
        '  <g fill="none" stroke="black" stroke-width="0.35" stroke-linecap="round" stroke-linejoin="round">',
    ]
    for command in plan.commands:
        command_type = command["type"]
        if command_type == "line":
            lines.append(
                f'    <line x1="{command["x1"]}" y1="{command["y1"]}" '
                f'x2="{command["x2"]}" y2="{command["y2"]}" />'
            )
        elif command_type == "polyline":
            points = " ".join(f"{x},{y}" for x, y in command["points"])
            lines.append(f'    <polyline points="{points}" />')
        elif command_type == "circle":
            lines.append(f'    <circle cx="{command["cx"]}" cy="{command["cy"]}" r="{command["r"]}" />')
        elif command_type == "text":
            lines.append(f'    <g aria-label="{html.escape(command["value"])}">')
            for path_data in text_command_to_paths(command):
                lines.append(f'      <path d="{path_data}" />')
            lines.append("    </g>")
    lines.extend(["  </g>", "</svg>", ""])
    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Sakana Fugu response JSON must be an object")
    return parsed


def fallback_plan_from_request(request: str, reason: str = "live model could not produce a usable plan") -> dict[str, Any]:
    payload = json.loads(json.dumps(SAMPLE_PLAN))
    title = _clean_title(request)[:42].strip("_") or "fallback_plot"
    payload["title"] = title
    payload["description"] = f"Safe fallback plot generated because {reason}: {request[:130]}"
    payload["warnings"] = [f"safe fallback used because {reason}"]
    return payload


def fallback_plan_from_source(source_plan: dict[str, Any], reason: str) -> dict[str, Any]:
    payload = json.loads(json.dumps(source_plan))
    payload["title"] = _clean_title(payload.get("title") or "webcam_contour_cleanup")
    payload["description"] = f"Fast contour fallback shown because {reason}."
    raw_warnings = payload.get("warnings")
    warnings = raw_warnings if isinstance(raw_warnings, list) else []
    payload["warnings"] = [*warnings, f"Fugu cleanup fallback used because {reason}"]
    return payload


def _extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                parts.append(content["text"])
            elif isinstance(content.get("delta"), str):
                parts.append(content["delta"])
    return "".join(parts)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _responses_completion(body: dict[str, Any], api_key: str) -> str:
    req = urllib.request.Request(
        f"{BASE_URL}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    if not isinstance(payload, dict):
        raise ValueError("Responses API returned a non-object payload")
    return _extract_response_text(payload)


def _streaming_responses_completion(body: dict[str, Any], api_key: str) -> str:
    req = urllib.request.Request(
        f"{BASE_URL}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    text_parts: list[str] = []
    final_payloads: list[dict[str, Any]] = []
    final_text = ""
    with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                continue
            chunk = json.loads(payload)
            event_type = chunk.get("type")
            if event_type == "response.output_text.delta" and isinstance(chunk.get("delta"), str):
                text_parts.append(chunk["delta"])
            elif event_type == "response.output_text.done" and isinstance(chunk.get("text"), str):
                final_text = chunk["text"]
            if isinstance(chunk.get("response"), dict):
                final_payloads.append(chunk["response"])
    if text_parts:
        return "".join(text_parts)
    if final_text:
        return final_text
    for payload in reversed(final_payloads):
        text = _extract_response_text(payload)
        if text:
            return text
    return ""


def call_fugu(model: str, request: str) -> dict[str, Any]:
    api_key = os.environ.get("SAKANA_API_KEY")
    if not api_key:
        print(
            f"{DISPLAY_MODEL_NAME} API key is not loaded; using safe fallback plan.",
            file=sys.stderr,
        )
        return fallback_plan_from_request(request, "no API key was loaded")

    api_model = model
    body: dict[str, Any] = {
        "model": api_model,
        "instructions": FUGU_INSTRUCTIONS,
        "input": PROMPT_TEMPLATE.format(request=request),
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stream": _env_flag("FUGU_STREAM", False),
    }
    if os.environ.get("FUGU_REASONING_EFFORT", "").strip():
        body["reasoning"] = {"effort": os.environ["FUGU_REASONING_EFFORT"].strip()}
    if "FUGU_STORE" in os.environ:
        body["store"] = _env_flag("FUGU_STORE", False)
    if api_model not in {"fugu", "fugu-ultra"}:
        raise ValueError(f"unknown Sakana Fugu API model id: {model}")

    start = time.monotonic()
    try:
        if body["stream"]:
            content = _streaming_responses_completion(body, api_key)
        else:
            content = _responses_completion(body, api_key)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if len(detail) > 400:
            detail = f"{detail[:400]}..."
        print(
            f"{DISPLAY_MODEL_NAME} API request failed; using safe fallback plan "
            f"(HTTP {exc.code}: {detail or exc.reason}).",
            file=sys.stderr,
        )
        return fallback_plan_from_request(request, "the live model request failed")
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        print(
            f"{DISPLAY_MODEL_NAME} API request timed out or failed; using safe fallback plan ({exc}).",
            file=sys.stderr,
        )
        return fallback_plan_from_request(request, "the live model request timed out or failed")
    elapsed_ms = int((time.monotonic() - start) * 1000)
    print(f"{DISPLAY_MODEL_NAME} response: {len(content)} chars from {api_model} in {elapsed_ms} ms", file=sys.stderr)
    try:
        return extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            f"{DISPLAY_MODEL_NAME} response was not valid plan JSON; using safe fallback plan ({exc}).",
            file=sys.stderr,
        )
        return fallback_plan_from_request(request, "the live model returned invalid plan JSON")


def call_fugu_contour_cleanup(
    model: str,
    request: str,
    source_plan: dict[str, Any],
) -> dict[str, Any]:
    api_key = os.environ.get("SAKANA_API_KEY")
    if not api_key:
        print(
            f"{DISPLAY_MODEL_NAME} API key is not loaded; using fast contour fallback.",
            file=sys.stderr,
        )
        return fallback_plan_from_source(source_plan, "no API key was loaded")

    api_model = model
    if api_model not in {"fugu", "fugu-ultra"}:
        raise ValueError(f"unknown Sakana Fugu API model id: {model}")

    source_json = json.dumps(source_plan, separators=(",", ":"), ensure_ascii=False)
    prompt_text = CONTOUR_CLEANUP_TEMPLATE.format(request=request, source_plan=source_json)

    body: dict[str, Any] = {
        "model": api_model,
        "instructions": FUGU_INSTRUCTIONS,
        "input": prompt_text,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "stream": _env_flag("FUGU_STREAM", False),
    }
    if os.environ.get("FUGU_REASONING_EFFORT", "").strip():
        body["reasoning"] = {"effort": os.environ["FUGU_REASONING_EFFORT"].strip()}
    if "FUGU_STORE" in os.environ:
        body["store"] = _env_flag("FUGU_STORE", False)

    start = time.monotonic()
    try:
        if body["stream"]:
            content = _streaming_responses_completion(body, api_key)
        else:
            content = _responses_completion(body, api_key)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        if len(detail) > 400:
            detail = f"{detail[:400]}..."
        print(
            f"{DISPLAY_MODEL_NAME} contour cleanup failed; using fast contour fallback "
            f"(HTTP {exc.code}: {detail or exc.reason}).",
            file=sys.stderr,
        )
        return fallback_plan_from_source(source_plan, "the live contour cleanup request failed")
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        print(
            f"{DISPLAY_MODEL_NAME} contour cleanup timed out or failed; using fast contour fallback ({exc}).",
            file=sys.stderr,
        )
        return fallback_plan_from_source(source_plan, "the live contour cleanup request timed out or failed")
    elapsed_ms = int((time.monotonic() - start) * 1000)
    print(
        f"{DISPLAY_MODEL_NAME} contour cleanup response: {len(content)} chars from {api_model} in {elapsed_ms} ms",
        file=sys.stderr,
    )
    try:
        return extract_json_object(content)
    except (json.JSONDecodeError, ValueError) as exc:
        print(
            f"{DISPLAY_MODEL_NAME} contour cleanup was not valid plan JSON; using fast contour fallback ({exc}).",
            file=sys.stderr,
        )
        return fallback_plan_from_source(source_plan, "the live contour cleanup returned invalid plan JSON")


def write_outputs(plan: SanitizedPlan, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{plan.title}.plan.json"
    svg_path = output_dir / f"{plan.title}.svg"
    json_payload = {
        "title": plan.title,
        "description": plan.description,
        "source_model": plan.source_model,
        "page": {"width_mm": PAGE_WIDTH_MM, "height_mm": PAGE_HEIGHT_MM},
        "commands": plan.commands,
        "metrics": {
            "command_count": len(plan.commands),
            "estimated_pen_distance_mm": estimate_pen_distance_mm(plan.commands),
        },
        "warnings": plan.warnings,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    svg_path.write_text(plan_to_svg(plan), encoding="utf-8")
    return json_path, svg_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate validated JSON and SVG files for an AxiDraw demo.")
    parser.add_argument("--online", action="store_true", help="Call Sakana Fugu using SAKANA_API_KEY.")
    parser.add_argument("--model", choices=["fugu", "fugu-ultra"], default=DEFAULT_MODEL)
    parser.add_argument("--request", default="Draw a clean geometric wave with a small Sakana Fugu label.")
    parser.add_argument("--plan-json", type=Path, help="Read an existing plan JSON file instead of using the sample or --online.")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--print-axicli", action="store_true", help="Print a future axicli command, but do not execute it.")
    args = parser.parse_args()

    source_model = "offline"
    if args.plan_json:
        raw_plan = json.loads(args.plan_json.read_text(encoding="utf-8"))
        source_model = f"file:{args.plan_json}"
    elif args.online:
        raw_plan = call_fugu(args.model, args.request)
        source_model = args.model
    else:
        raw_plan = SAMPLE_PLAN

    if not isinstance(raw_plan, dict):
        print("plan JSON must be an object", file=sys.stderr)
        return 2

    plan = sanitize_plan(raw_plan, source_model=source_model)
    json_path, svg_path = write_outputs(plan, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {svg_path}")
    print(f"commands: {len(plan.commands)}")
    print(f"estimated pen distance: {estimate_pen_distance_mm(plan.commands)} mm")
    if plan.warnings:
        print("warnings:")
        for warning in plan.warnings:
            print(f"- {warning}")
    if args.print_axicli:
        print()
        print("future hardware command, not executed:")
        print(f"axicli {svg_path} --mode plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
