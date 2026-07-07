#!/usr/bin/env python3
"""Run a hardware-free AxiDraw prompt benchmark across model buckets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import generate_axidraw_svg as generator


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "benchmark_results"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ORDER = ("fugu_ultra", "fugu", "gpt", "gemini", "opus")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    provider: str
    default_api_model: str | None = None


@dataclass(frozen=True)
class DrawingPrompt:
    slug: str
    request: str


MODEL_SPECS = {
    "fugu_ultra": ModelSpec("fugu_ultra", "Fugu Ultra (xhigh)", "sakana_responses", "fugu-ultra"),
    "fugu": ModelSpec("fugu", "Fugu (xhigh)", "sakana_responses", "fugu"),
    "gpt": ModelSpec("gpt", "GPT-5.5", "openai_compatible"),
    "gemini": ModelSpec("gemini", "Gemini 3.1 Pro", "openai_compatible"),
    "opus": ModelSpec("opus", "Claude Opus 4.8 (max)", "openai_compatible"),
}

PROMPTS = (
    DrawingPrompt("simple_fish", "Draw a simple fish using clean pen-plotter line art."),
    DrawingPrompt("robot_face", "Draw a friendly robot face with simple geometric features."),
    DrawingPrompt("flower", "Draw a single flower with stem, leaves, and several petals."),
    DrawingPrompt("geometric_turtle", "Draw a geometric turtle with shell pattern and small legs."),
    DrawingPrompt("skyline", "Draw a compact city skyline with a moon and simple windows."),
)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def fixture_plan(model_key: str, prompt: DrawingPrompt) -> dict[str, Any]:
    model_index = MODEL_ORDER.index(model_key)
    dx = (model_index - 2) * 1.8
    dy = ((model_index % 3) - 1) * 1.2
    title = f"{prompt.slug}_{model_key}"
    description = f"Dry-run fixture for {MODEL_SPECS[model_key].display_name} on {prompt.slug}."

    if prompt.slug == "simple_fish":
        commands = [
            {"type": "polyline", "points": [[68 + dx, 104 + dy], [98 + dx, 82 + dy], [151 + dx, 79 + dy], [196 + dx, 105 + dy], [151 + dx, 131 + dy], [98 + dx, 128 + dy], [68 + dx, 104 + dy]]},
            {"type": "polyline", "points": [[196 + dx, 105 + dy], [236 + dx, 80 + dy], [229 + dx, 105 + dy], [236 + dx, 130 + dy], [196 + dx, 105 + dy]]},
            {"type": "circle", "cx": 104 + dx, "cy": 100 + dy, "r": 3.6 + model_index * 0.15},
            {"type": "line", "x1": 126 + dx, "y1": 88 + dy, "x2": 118 + dx, "y2": 120 + dy},
            {"type": "polyline", "points": [[74 + dx, 151 + dy], [118 + dx, 145 + dy], [164 + dx, 150 + dy], [210 + dx, 144 + dy]]},
        ]
    elif prompt.slug == "robot_face":
        commands = [
            {"type": "polyline", "points": [[85 + dx, 62 + dy], [212 + dx, 62 + dy], [222 + dx, 74 + dy], [222 + dx, 150 + dy], [211 + dx, 162 + dy], [86 + dx, 162 + dy], [75 + dx, 150 + dy], [75 + dx, 74 + dy], [85 + dx, 62 + dy]]},
            {"type": "circle", "cx": 117 + dx, "cy": 102 + dy, "r": 11 + model_index * 0.2},
            {"type": "circle", "cx": 181 + dx, "cy": 102 + dy, "r": 11 + model_index * 0.2},
            {"type": "polyline", "points": [[111 + dx, 134 + dy], [132 + dx, 144 + dy], [165 + dx, 144 + dy], [186 + dx, 134 + dy]]},
            {"type": "line", "x1": 148.5 + dx, "y1": 62 + dy, "x2": 148.5 + dx, "y2": 37 + dy},
            {"type": "circle", "cx": 148.5 + dx, "cy": 31 + dy, "r": 5.5},
        ]
    elif prompt.slug == "flower":
        commands = [
            {"type": "line", "x1": 148 + dx, "y1": 106 + dy, "x2": 148 + dx, "y2": 178 + dy},
            {"type": "circle", "cx": 148 + dx, "cy": 86 + dy, "r": 12 + model_index * 0.25},
            {"type": "circle", "cx": 124 + dx, "cy": 86 + dy, "r": 16},
            {"type": "circle", "cx": 172 + dx, "cy": 86 + dy, "r": 16},
            {"type": "circle", "cx": 148 + dx, "cy": 62 + dy, "r": 16},
            {"type": "polyline", "points": [[148 + dx, 142 + dy], [118 + dx, 128 + dy], [132 + dx, 155 + dy], [148 + dx, 142 + dy], [178 + dx, 128 + dy], [164 + dx, 155 + dy], [148 + dx, 142 + dy]]},
        ]
    elif prompt.slug == "geometric_turtle":
        commands = [
            {"type": "circle", "cx": 149 + dx, "cy": 107 + dy, "r": 42 + model_index * 0.25},
            {"type": "circle", "cx": 206 + dx, "cy": 104 + dy, "r": 13},
            {"type": "polyline", "points": [[115 + dx, 93 + dy], [149 + dx, 75 + dy], [183 + dx, 93 + dy], [169 + dx, 128 + dy], [129 + dx, 128 + dy], [115 + dx, 93 + dy]]},
            {"type": "line", "x1": 149 + dx, "y1": 75 + dy, "x2": 149 + dx, "y2": 149 + dy},
            {"type": "polyline", "points": [[113 + dx, 141 + dy], [91 + dx, 160 + dy], [124 + dx, 154 + dy], [184 + dx, 141 + dy], [207 + dx, 160 + dy], [174 + dx, 154 + dy]]},
        ]
    else:
        commands = [
            {"type": "line", "x1": 46 + dx, "y1": 164 + dy, "x2": 251 + dx, "y2": 164 + dy},
            {"type": "polyline", "points": [[56 + dx, 164 + dy], [56 + dx, 118 + dy], [82 + dx, 118 + dy], [82 + dx, 91 + dy], [111 + dx, 91 + dy], [111 + dx, 164 + dy], [139 + dx, 164 + dy], [139 + dx, 104 + dy], [171 + dx, 104 + dy], [171 + dx, 164 + dy], [205 + dx, 164 + dy], [205 + dx, 78 + dy], [234 + dx, 78 + dy], [234 + dx, 164 + dy]]},
            {"type": "circle", "cx": 67 + dx, "cy": 58 + dy, "r": 11},
            {"type": "polyline", "points": [[91 + dx, 131 + dy], [101 + dx, 131 + dy], [101 + dx, 142 + dy], [91 + dx, 142 + dy], [91 + dx, 131 + dy]]},
            {"type": "polyline", "points": [[185 + dx, 119 + dy], [194 + dx, 119 + dy], [194 + dx, 130 + dy], [185 + dx, 130 + dy], [185 + dx, 119 + dy]]},
        ]

    return {
        "title": title,
        "description": description,
        "commands": commands,
    }


def prompt_payload(request: str) -> str:
    return generator.PROMPT_TEMPLATE.format(request=request)


def extract_responses_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) or []:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "".join(parts)


def call_sakana_responses(model_id: str, request: str) -> tuple[str, int]:
    api_key = os.environ.get("SAKANA_API_KEY")
    if not api_key:
        raise RuntimeError("SAKANA_API_KEY is required for live Fugu benchmark runs")
    base_url = (os.environ.get("SAKANA_BASE_URL") or "https://api.sakana.ai/v1").rstrip("/")
    body = {
        "model": model_id,
        "instructions": generator.FUGU_INSTRUCTIONS,
        "input": prompt_payload(request),
        "max_output_tokens": int(os.environ.get("AXIDRAW_BENCHMARK_MAX_OUTPUT_TOKENS", str(generator.MAX_OUTPUT_TOKENS))),
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base_url}/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=int(os.environ.get("AXIDRAW_BENCHMARK_TIMEOUT_SECONDS", "60"))) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    return extract_responses_text(payload), int((time.monotonic() - start) * 1000)


def openai_model_id(model_key: str) -> str:
    env_name = f"AXIDRAW_BENCHMARK_{model_key.upper()}_MODEL"
    model_id = os.environ.get(env_name, "").strip()
    if not model_id:
        raise RuntimeError(f"{env_name} is required for live {MODEL_SPECS[model_key].display_name} runs")
    return model_id


def call_openai_compatible(model_key: str, request: str) -> tuple[str, int]:
    api_key = (
        os.environ.get(f"AXIDRAW_BENCHMARK_{model_key.upper()}_API_KEY")
        or os.environ.get("AXIDRAW_BENCHMARK_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY or AXIDRAW_BENCHMARK_API_KEY is required for live baseline runs")
    base_url = (
        os.environ.get(f"AXIDRAW_BENCHMARK_{model_key.upper()}_BASE_URL")
        or os.environ.get("AXIDRAW_BENCHMARK_BASE_URL")
        or DEFAULT_OPENROUTER_BASE_URL
    ).rstrip("/")
    body = {
        "model": openai_model_id(model_key),
        "messages": [
            {"role": "system", "content": generator.FUGU_INSTRUCTIONS},
            {"role": "user", "content": prompt_payload(request)},
        ],
        "temperature": 0.2,
        "max_tokens": int(os.environ.get("AXIDRAW_BENCHMARK_MAX_OUTPUT_TOKENS", str(generator.MAX_OUTPUT_TOKENS))),
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=int(os.environ.get("AXIDRAW_BENCHMARK_TIMEOUT_SECONDS", "60"))) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace") or "{}")
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return "", int((time.monotonic() - start) * 1000)
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else ""
    return str(content or ""), int((time.monotonic() - start) * 1000)


def call_model(model_key: str, request: str, dry_run: bool) -> tuple[str, int | None]:
    if dry_run:
        return json.dumps(fixture_plan(model_key, DrawingPrompt("_", request)), separators=(",", ":")), None
    spec = MODEL_SPECS[model_key]
    if spec.provider == "sakana_responses":
        if not spec.default_api_model:
            raise RuntimeError(f"missing API model for {model_key}")
        return call_sakana_responses(spec.default_api_model, request)
    return call_openai_compatible(model_key, request)


def count_page_bounds_violations(raw_plan: dict[str, Any]) -> int:
    commands = raw_plan.get("commands")
    if not isinstance(commands, list):
        return 1

    violations = 0

    def check_point(x: Any, y: Any) -> None:
        nonlocal violations
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            violations += 1
            return
        if xf < 0 or xf > generator.PAGE_WIDTH_MM or yf < 0 or yf > generator.PAGE_HEIGHT_MM:
            violations += 1

    for command in commands:
        if not isinstance(command, dict):
            violations += 1
            continue
        command_type = str(command.get("type") or "").lower()
        if command_type == "line":
            check_point(command.get("x1"), command.get("y1"))
            check_point(command.get("x2"), command.get("y2"))
        elif command_type == "polyline":
            points = command.get("points")
            if not isinstance(points, list):
                violations += 1
                continue
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    violations += 1
                    continue
                check_point(point[0], point[1])
        elif command_type == "circle":
            try:
                cx = float(command.get("cx"))
                cy = float(command.get("cy"))
                radius = float(command.get("r"))
            except (TypeError, ValueError):
                violations += 1
                continue
            if (
                cx - radius < 0
                or cy - radius < 0
                or cx + radius > generator.PAGE_WIDTH_MM
                or cy + radius > generator.PAGE_HEIGHT_MM
            ):
                violations += 1
        elif command_type == "text":
            check_point(command.get("x"), command.get("y"))
        else:
            violations += 1
    return violations


def normalize_preview_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in ("input_svg", "preview_svg"):
        value = report.get(key)
        if isinstance(value, str):
            path = Path(value)
            if path.is_absolute():
                report[key] = relpath(path)
    command = report.get("command")
    if isinstance(command, list):
        normalized_command = []
        for item in command:
            if isinstance(item, str):
                path = Path(item)
                if path.is_absolute():
                    normalized = relpath(path)
                    normalized_command.append(path.name if normalized == item else normalized)
                else:
                    normalized_command.append(item)
            else:
                normalized_command.append(item)
        report["command"] = normalized_command
    write_json(report_path, report)
    return report


def run_preview(svg_path: Path, preview_svg: Path, report_json: Path) -> tuple[bool, dict[str, Any], str | None]:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "simulate_axidraw_preview.py"),
            str(svg_path),
            "--out-svg",
            str(preview_svg),
            "--report-json",
            str(report_json),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    report = normalize_preview_report(report_json)
    ok = proc.returncode == 0 and preview_svg.exists() and report_json.exists()
    failure = None if ok else (proc.stderr.strip() or proc.stdout.strip() or f"preview exited {proc.returncode}")
    return ok, report, failure


def plan_payload(plan: generator.SanitizedPlan) -> dict[str, Any]:
    return {
        "title": plan.title,
        "description": plan.description,
        "source_model": plan.source_model,
        "page": {"width_mm": generator.PAGE_WIDTH_MM, "height_mm": generator.PAGE_HEIGHT_MM},
        "commands": plan.commands,
        "warnings": plan.warnings,
    }


def run_one(model_key: str, prompt: DrawingPrompt, model_dir: Path, dry_run: bool) -> dict[str, Any]:
    notes: list[str] = []
    failure_reason: str | None = None
    raw_text = ""
    latency_ms: int | None = None
    valid_json = False
    sanitized_success = False
    svg_generated = False
    preview_generated = False
    command_count = 0
    estimated_pen_distance_mm = 0.0
    page_bounds_violations = 0
    svg_rel = ""
    preview_rel = ""
    report_rel = ""
    sanitized_payload: dict[str, Any] | None = None

    try:
        if dry_run:
            raw_text = json.dumps(fixture_plan(model_key, prompt), indent=2, ensure_ascii=True)
        else:
            raw_text, latency_ms = call_model(model_key, prompt.request, dry_run=False)
        raw_plan = generator.extract_json_object(raw_text)
        valid_json = True
        page_bounds_violations = count_page_bounds_violations(raw_plan)
    except (json.JSONDecodeError, ValueError, RuntimeError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raw_plan = {}
        failure_reason = str(exc)

    if valid_json:
        plan = generator.sanitize_plan(raw_plan, source_model=model_key)
        plan = replace(plan, title=f"{prompt.slug}_{model_key}")
        sanitized_payload = plan_payload(plan)
        sanitized_success = len(plan.commands) > 0
        command_count = len(plan.commands)
        estimated_pen_distance_mm = generator.estimate_pen_distance_mm(plan.commands)
        notes.extend(plan.warnings)

        if sanitized_success:
            svg_path = model_dir / f"{prompt.slug}.svg"
            svg_path.write_text(generator.plan_to_svg(plan), encoding="utf-8")
            svg_generated = svg_path.exists()
            svg_rel = relpath(svg_path)
            preview_svg = model_dir / f"{prompt.slug}.preview.svg"
            report_json = model_dir / f"{prompt.slug}.preview.report.json"
            preview_generated, report, preview_failure = run_preview(svg_path, preview_svg, report_json)
            preview_rel = relpath(preview_svg) if preview_svg.exists() else ""
            report_rel = relpath(report_json) if report_json.exists() else ""
            notes.extend(str(warning) for warning in report.get("warnings", []) if warning)
            if preview_failure:
                failure_reason = preview_failure
        elif not failure_reason:
            failure_reason = "sanitizer produced no drawable commands"

    if dry_run:
        notes.append("dry-run fixture response")

    return {
        "prompt": prompt.slug,
        "request": prompt.request,
        "valid_json": valid_json,
        "sanitized_success": sanitized_success,
        "svg_generated": svg_generated,
        "preview_generated": preview_generated,
        "command_count": command_count,
        "estimated_pen_distance_mm": estimated_pen_distance_mm,
        "page_bounds_violations": page_bounds_violations,
        "latency_ms": latency_ms,
        "notes": sorted(set(notes)),
        "failure_reason": failure_reason,
        "artifacts": {
            "svg": svg_rel,
            "preview_svg": preview_rel,
            "preview_report": report_rel,
        },
        "sanitized_plan": sanitized_payload,
        "raw_model_response": raw_text,
    }


def aggregate_model(model_key: str, prompt_results: list[dict[str, Any]], dry_run: bool) -> dict[str, Any]:
    count = len(prompt_results)
    latencies = [item["latency_ms"] for item in prompt_results if isinstance(item.get("latency_ms"), int)]
    notes = sorted({note for item in prompt_results for note in item.get("notes", [])})
    return {
        "model": model_key,
        "display_name": MODEL_SPECS[model_key].display_name,
        "mode": "dry_run" if dry_run else "live",
        "prompt_count": count,
        "valid_json": sum(1 for item in prompt_results if item["valid_json"]),
        "sanitized_success": sum(1 for item in prompt_results if item["sanitized_success"]),
        "svg_generated": sum(1 for item in prompt_results if item["svg_generated"]),
        "preview_generated": sum(1 for item in prompt_results if item["preview_generated"]),
        "average_command_count": round(mean([item["command_count"] for item in prompt_results]) if prompt_results else 0, 1),
        "average_estimated_pen_distance_mm": round(
            mean([item["estimated_pen_distance_mm"] for item in prompt_results]) if prompt_results else 0,
            1,
        ),
        "page_bounds_violations": sum(item["page_bounds_violations"] for item in prompt_results),
        "average_latency_ms": round(mean(latencies), 1) if latencies else None,
        "failure_count": sum(1 for item in prompt_results if item["failure_reason"]),
        "notes": notes,
    }


def load_model_metrics(results_dir: Path, model_key: str) -> dict[str, Any] | None:
    path = results_dir / model_key / "metrics.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("aggregate", data) if isinstance(data, dict) else None


def fraction(value: int, total: int) -> str:
    return f"{value}/{total}"


def note_summary(metrics: dict[str, Any]) -> str:
    notes = metrics.get("notes", [])
    parts: list[str] = []
    if metrics.get("mode") == "dry_run":
        parts.append("dry-run")
    if any("axicli was not found" in note for note in notes):
        parts.append("preview fallback")
    if metrics.get("failure_count"):
        parts.append(f"{metrics['failure_count']} failures")
    return ", ".join(parts) or "ok"


def build_summary_table(results_dir: Path) -> str:
    lines = [
        "| Model | Valid JSON | Sanitized | SVG | Preview | Avg commands | Avg pen distance | Bounds violations | Avg latency | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model_key in MODEL_ORDER:
        metrics = load_model_metrics(results_dir, model_key)
        if not metrics:
            continue
        total = int(metrics.get("prompt_count", 0))
        latency = metrics.get("average_latency_ms")
        latency_text = "n/a" if latency is None else f"{latency} ms"
        lines.append(
            "| {model} | {valid} | {sanitized} | {svg} | {preview} | {commands} | {distance} mm | {bounds} | {latency} | {notes} |".format(
                model=metrics["display_name"],
                valid=fraction(int(metrics["valid_json"]), total),
                sanitized=fraction(int(metrics["sanitized_success"]), total),
                svg=fraction(int(metrics["svg_generated"]), total),
                preview=fraction(int(metrics["preview_generated"]), total),
                commands=metrics["average_command_count"],
                distance=metrics["average_estimated_pen_distance_mm"],
                bounds=metrics["page_bounds_violations"],
                latency=latency_text,
                notes=note_summary(metrics),
            )
        )
    return "\n".join(lines) + "\n"


def selected_models(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"unknown models: {', '.join(unknown)}")
    return values


def selected_prompts(raw: str, smoke: bool) -> list[DrawingPrompt]:
    prompts_by_slug = {prompt.slug: prompt for prompt in PROMPTS}
    if smoke:
        return [PROMPTS[0]]
    if raw == "all":
        return list(PROMPTS)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in values if item not in prompts_by_slug]
    if unknown:
        raise SystemExit(f"unknown prompts: {', '.join(unknown)}")
    return [prompts_by_slug[item] for item in values]


def run_benchmark(args: argparse.Namespace) -> int:
    results_dir = args.out_dir
    models = selected_models(args.models)
    prompts = selected_prompts(args.prompts, args.smoke)
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_key in models:
        model_dir = results_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        prompt_results = [run_one(model_key, prompt, model_dir, args.dry_run or args.smoke) for prompt in prompts]

        raw_responses = {
            item["prompt"]: {
                "request": item["request"],
                "raw_model_response": item.pop("raw_model_response"),
            }
            for item in prompt_results
        }
        sanitized_plans = {
            item["prompt"]: item.pop("sanitized_plan")
            for item in prompt_results
            if item.get("sanitized_plan")
        }
        aggregate = aggregate_model(model_key, prompt_results, args.dry_run or args.smoke)
        write_json(model_dir / "raw_model_responses.json", raw_responses)
        write_json(model_dir / "sanitized_plans.json", sanitized_plans)
        write_json(model_dir / "metrics.json", {"aggregate": aggregate, "prompts": prompt_results})
        print(f"wrote {model_dir / 'metrics.json'}")

    summary = build_summary_table(results_dir)
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"wrote {summary_path}")
    return 0


def run_summary(args: argparse.Namespace) -> int:
    summary = build_summary_table(args.out_dir)
    if args.write_summary:
        path = args.out_dir / "summary.md"
        path.write_text(summary, encoding="utf-8")
        print(f"wrote {path}")
    print(summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AxiDraw prompt plans without physical plotter hardware.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--prompts", default="all")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic fixture responses instead of APIs.")
    parser.add_argument("--smoke", action="store_true", help="Run one fixture prompt per model for a quick validation pass.")
    parser.add_argument("--summary-only", action="store_true", help="Only aggregate existing metrics into a markdown table.")
    parser.add_argument("--write-summary", action="store_true", help="Write summary.md when using --summary-only.")
    args = parser.parse_args()

    if args.summary_only:
        return run_summary(args)
    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
