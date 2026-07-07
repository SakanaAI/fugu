#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reachy_fugu.config import Settings, load_settings
from reachy_fugu.fugu_client import build_payload, extract_jsonish_answer, extract_response_text
from reachy_fugu.pipeline import answer_question
from reachy_fugu.schemas import (
    CONFIDENCE,
    EMOTIONS,
    PRIMITIVES,
    ValidationError,
    extract_json_object,
    fallback_gesture_plan,
    validate_gesture_plan,
)


QUESTIONS = [
    "What can you do?",
    "Tell me one fun fact about pufferfish.",
    "Can you explain what you are seeing?",
    "How would you help a child learn robotics?",
    "Say hello and wave.",
]

SMOKE_FIXTURES = [
    (
        "I can listen to typed or robot questions, answer briefly, speak back, and choose gentle Reachy gestures.",
        "happy",
        "nod",
        "I can listen, answer, speak, and move gently.",
    ),
    (
        "Pufferfish can inflate by taking in water or air, making themselves much harder for predators to swallow.",
        "curious",
        "curious_tilt",
        "Pufferfish can puff up like a round balloon.",
    ),
    (
        "In this simulation I do not receive camera frames, so I can only describe what the operator tells me.",
        "thinking",
        "curious_tilt",
        "I do not have a camera frame in this simulation.",
    ),
    (
        "I would let a child ask questions, show one safe movement at a time, and connect each motion to a simple idea.",
        "happy",
        "nod",
        "We can learn one safe robot idea at a time.",
    ),
    (
        "Hello, I am Reachy Mini. I am waving with a small friendly motion.",
        "excited",
        "antenna_pulse",
        "Hello, I am Reachy Mini.",
    ),
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str
    api_model: str
    effort: str
    env_key: str
    base_url_env: str
    default_base_url: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "fugu_ultra": ModelSpec(
        "fugu_ultra",
        "Fugu Ultra",
        "sakana",
        "fugu-ultra",
        "xhigh",
        "SAKANA_API_KEY",
        "SAKANA_BASE_URL",
        "https://api.sakana.ai/v1",
    ),
    "fugu": ModelSpec(
        "fugu",
        "Fugu",
        "sakana",
        "fugu",
        "xhigh",
        "SAKANA_API_KEY",
        "SAKANA_BASE_URL",
        "https://api.sakana.ai/v1",
    ),
    "gpt": ModelSpec(
        "gpt",
        "GPT-5.5",
        "openrouter",
        "openai/gpt-5.5",
        "xhigh",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
    "gemini": ModelSpec(
        "gemini",
        "Gemini 3.1 Pro",
        "openrouter",
        "google/gemini-3.1-pro-preview",
        "high",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
    "opus": ModelSpec(
        "opus",
        "Claude Opus 4.8",
        "openrouter",
        "anthropic/claude-opus-4.8",
        "max",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    ),
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@contextlib.contextmanager
def benchmark_environment() -> Any:
    old_fast_intents = os.environ.get("REACHY_FAST_LOCAL_INTENTS")
    old_hardware = os.environ.get("REACHY_HARDWARE_ENABLED")
    os.environ["REACHY_FAST_LOCAL_INTENTS"] = "0"
    os.environ["REACHY_HARDWARE_ENABLED"] = "0"
    try:
        yield
    finally:
        if old_fast_intents is None:
            os.environ.pop("REACHY_FAST_LOCAL_INTENTS", None)
        else:
            os.environ["REACHY_FAST_LOCAL_INTENTS"] = old_fast_intents
        if old_hardware is None:
            os.environ.pop("REACHY_HARDWARE_ENABLED", None)
        else:
            os.environ["REACHY_HARDWARE_ENABLED"] = old_hardware


class RawCapture:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    @property
    def latest(self) -> dict[str, Any]:
        return self.records[-1] if self.records else {}


def smoke_response(prompt_index: int) -> dict[str, Any]:
    answer, emotion, primitive, speech = SMOKE_FIXTURES[prompt_index % len(SMOKE_FIXTURES)]
    motion_args = {"duration": 0.8} if primitive == "antenna_pulse" else {}
    return {
        "answer": answer,
        "confidence": "partial",
        "citations": [],
        "emotion": emotion,
        "gesture_plan": [
            {"t_ms": 0, "primitive": "idle", "args": {}},
            {"t_ms": 300, "primitive": primitive, "args": motion_args},
            {"t_ms": 900, "primitive": "speak", "args": {"text": speech}},
            {"t_ms": 3300, "primitive": "idle", "args": {}},
        ],
    }


def smoke_call_factory(spec: ModelSpec, capture: RawCapture) -> Callable[[str, Settings], dict[str, Any]]:
    turn = {"index": 0}

    def call(prompt: str, settings: Settings) -> dict[str, Any]:
        started = time.monotonic()
        parsed = smoke_response(turn["index"])
        turn["index"] += 1
        raw_text = json.dumps(parsed, ensure_ascii=False)
        capture.add(
            {
                "provider": "smoke_fixture",
                "model": spec.api_model,
                "effort": spec.effort,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "raw_text": raw_text,
                "response_envelope": {"fixture": True},
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
        return parsed

    return call


def live_call_factory(spec: ModelSpec, capture: RawCapture) -> Callable[[str, Settings], dict[str, Any]]:
    def call(prompt: str, settings: Settings) -> dict[str, Any]:
        api_key = os.environ.get(spec.env_key)
        if not api_key:
            raise RuntimeError(f"{spec.env_key} is required for live {spec.key} runs")
        base_url = os.environ.get(spec.base_url_env, spec.default_base_url).rstrip("/")
        if spec.provider == "sakana":
            body = build_payload(prompt, settings)
            endpoint = "/chat/completions" if settings.fugu_api_mode == "chat_completions" else "/responses"
        else:
            body = {
                "model": spec.api_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": settings.max_output_tokens,
                "reasoning": {"effort": spec.effort},
            }
            endpoint = "/chat/completions"
        request = urllib.request.Request(
            f"{base_url}{endpoint}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=float(os.environ.get("REACHY_BENCHMARK_TIMEOUT", "60"))) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{spec.key} API returned HTTP {exc.code}: {detail[:300]}") from exc
        raw_text = extract_response_text(envelope)
        capture.add(
            {
                "provider": spec.provider,
                "model": spec.api_model,
                "effort": spec.effort,
                "latency_ms": int((time.monotonic() - started) * 1000),
                "raw_text": raw_text,
                "response_envelope": envelope,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
        try:
            return extract_json_object(raw_text)
        except (json.JSONDecodeError, ValueError):
            answer = extract_jsonish_answer(raw_text) or " ".join(raw_text.split())
            answer = answer or "I heard you, but the model did not return an answer. Please try again."
            return {
                "answer": answer,
                "confidence": "partial",
                "citations": [],
                "emotion": "neutral",
                "gesture_plan": fallback_gesture_plan(answer, "partial", "neutral"),
            }

    return call


def model_settings(spec: ModelSpec) -> Settings:
    settings = load_settings()
    return replace(
        settings,
        sakana_api_key=os.environ.get(spec.env_key),
        sakana_base_url=os.environ.get(spec.base_url_env, spec.default_base_url).rstrip("/"),
        fugu_api_mode="chat_completions",
        fugu_model=spec.api_model,
        fugu_dry_run=False,
        reasoning_effort=spec.effort,
        reachy_hardware_enabled=False,
        cache_dir=DEMO_ROOT / ".cache" / "benchmark" / spec.key,
    )


def raw_validation(raw_text: str, final_plan: list[dict[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    failure_reason = ""
    try:
        parsed = extract_json_object(raw_text)
        valid_json = True
    except (json.JSONDecodeError, ValueError) as exc:
        valid_json = False
        failure_reason = f"invalid_json: {exc}"

    unknown_primitives: list[str] = []
    raw_plan_valid = False
    schema_valid = False
    fallback_used = not valid_json
    if parsed:
        raw_plan = parsed.get("gesture_plan")
        if isinstance(raw_plan, list):
            unknown_primitives = sorted(
                {
                    str(step.get("primitive"))
                    for step in raw_plan
                    if isinstance(step, dict) and step.get("primitive") not in PRIMITIVES
                }
            )
        try:
            validate_gesture_plan(raw_plan)
            raw_plan_valid = True
        except (ValidationError, TypeError) as exc:
            failure_reason = failure_reason or f"invalid_gesture_plan: {exc}"
        confidence = str(parsed.get("confidence", "")).strip().lower()
        emotion = str(parsed.get("emotion", "")).strip().lower()
        schema_valid = (
            isinstance(parsed.get("answer"), str)
            and bool(str(parsed.get("answer")).strip())
            and confidence in CONFIDENCE
            and emotion in EMOTIONS
            and isinstance(parsed.get("citations"), list)
            and raw_plan_valid
        )
        if not schema_valid:
            fallback_used = True
            failure_reason = failure_reason or "schema_validation_failed"
        if raw_plan_valid and raw_plan != final_plan:
            fallback_used = True
            failure_reason = failure_reason or "gesture_plan_normalized"

    return {
        "valid_json": valid_json,
        "schema_valid": schema_valid,
        "unsafe_or_unknown_primitives": unknown_primitives,
        "fallback_used": fallback_used,
        "failure_reason": failure_reason,
        "parsed": parsed,
    }


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def audio_metadata(audio_path: Path | None) -> dict[str, Any]:
    if audio_path is None:
        return {"generated": False, "exists": False}
    exists = audio_path.exists()
    metadata: dict[str, Any] = {
        "generated": True,
        "exists": exists,
        "audio_blob_committed": False,
        "note": "Generated audio cache files are local runtime outputs and are not repository artifacts.",
        "path": relative_path(audio_path, DEMO_ROOT),
        "size_bytes": audio_path.stat().st_size if exists else 0,
    }
    if exists:
        try:
            with wave.open(str(audio_path), "rb") as wav:
                metadata.update(
                    {
                        "duration_seconds": round(wav.getnframes() / float(wav.getframerate()), 3)
                        if wav.getframerate()
                        else 0.0,
                        "sample_rate": wav.getframerate(),
                        "channels": wav.getnchannels(),
                    }
                )
        except (OSError, EOFError, wave.Error):
            metadata["duration_seconds"] = None
    return metadata


def run_turn(
    question: str,
    settings: Settings,
    fugu_call: Callable[[str, Settings], dict[str, Any]],
    capture: RawCapture,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        result = answer_question(text_question=question, audio_path=None, settings=settings, fugu_call=fugu_call)
    except Exception as exc:  # noqa: BLE001 - per-turn failures are benchmark output
        raw = capture.latest
        validation = raw_validation(str(raw.get("raw_text", "")), [])
        metric = {
            "question": question,
            "valid_json": validation["valid_json"],
            "schema_valid": False,
            "answer_nonempty": False,
            "answer_word_count": 0,
            "gesture_plan_valid": False,
            "unsafe_or_unknown_primitives": validation["unsafe_or_unknown_primitives"],
            "tts_success": False,
            "latency_ms": raw.get("latency_ms"),
            "fallback_used": True,
            "notes": "",
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
        return raw, validation["parsed"], {}, {}, metric

    validation = raw_validation(str(capture.latest.get("raw_text", "")), result.gesture_plan)
    audio = audio_metadata(result.audio_path)
    try:
        validate_gesture_plan(result.gesture_plan)
        plan_valid = True
    except ValidationError:
        plan_valid = False
    latency = result.debug.get("latency_ms") if isinstance(result.debug, dict) else None
    metric = {
        "question": question,
        "valid_json": validation["valid_json"],
        "schema_valid": validation["schema_valid"],
        "answer_nonempty": bool(result.answer.strip()),
        "answer_word_count": len(result.answer.split()),
        "gesture_plan_valid": plan_valid,
        "unsafe_or_unknown_primitives": validation["unsafe_or_unknown_primitives"],
        "tts_success": bool(audio.get("exists") and audio.get("size_bytes", 0) > 0),
        "latency_ms": latency,
        "fallback_used": validation["fallback_used"],
        "notes": "",
        "failure_reason": validation["failure_reason"],
    }
    generated_plan = {
        "question": question,
        "answer": result.answer,
        "confidence": result.confidence,
        "emotion": result.emotion,
        "gesture_plan": result.gesture_plan,
        "debug": result.debug,
    }
    return capture.latest, validation["parsed"], generated_plan, audio, metric


def aggregate_metrics(spec: ModelSpec, mode: str, metrics: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        latency["fugu"]
        for metric in metrics
        if isinstance((latency := metric.get("latency_ms")), dict) and isinstance(latency.get("fugu"), int)
    ]
    word_counts = [int(metric.get("answer_word_count", 0)) for metric in metrics]
    turns = len(metrics)
    return {
        "model_key": spec.key,
        "model_label": spec.label,
        "api_model": spec.api_model,
        "provider": spec.provider,
        "effort": spec.effort,
        "mode": mode,
        "turns": turns,
        "valid_json": sum(metric.get("valid_json") is True for metric in metrics),
        "schema_valid": sum(metric.get("schema_valid") is True for metric in metrics),
        "answer_nonempty": sum(metric.get("answer_nonempty") is True for metric in metrics),
        "gesture_plan_valid": sum(metric.get("gesture_plan_valid") is True for metric in metrics),
        "unsafe_or_unknown_primitives": sorted(
            {
                primitive
                for metric in metrics
                for primitive in metric.get("unsafe_or_unknown_primitives", [])
            }
        ),
        "tts_success": sum(metric.get("tts_success") is True for metric in metrics),
        "fallback_used": sum(metric.get("fallback_used") is True for metric in metrics),
        "answer_word_count_avg": round(statistics.mean(word_counts), 1) if word_counts else 0,
        "latency_ms_median": int(statistics.median(latencies)) if latencies else None,
        "failure_count": sum(1 for metric in metrics if metric.get("failure_reason")),
    }


def write_model_artifacts(
    model_dir: Path,
    spec: ModelSpec,
    mode: str,
    raw_records: list[dict[str, Any]],
    parsed_records: list[dict[str, Any]],
    generated_plans: list[dict[str, Any]],
    audio_records: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "raw_responses.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in raw_records),
        encoding="utf-8",
    )
    write_json(model_dir / "parsed_responses.json", parsed_records)
    write_json(model_dir / "gesture_plans.json", generated_plans)
    write_json(model_dir / "audio_metadata.json", audio_records)
    aggregate = aggregate_metrics(spec, mode, metrics)
    write_json(model_dir / "metrics.json", {"aggregate": aggregate, "turns": metrics})
    return aggregate


def render_summary(aggregates: list[dict[str, Any]], mode: str) -> str:
    headers = [
        "model",
        "run",
        "valid JSON",
        "schema valid",
        "answer",
        "gesture plan",
        "unknown primitives",
        "TTS",
        "fallbacks",
        "median model ms",
    ]
    rows = []
    for aggregate in aggregates:
        turns = aggregate["turns"]
        unknown = ", ".join(aggregate["unsafe_or_unknown_primitives"]) or "none"
        latency = aggregate["latency_ms_median"]
        rows.append(
            [
                aggregate["model_key"],
                f"{aggregate['model_label']} ({aggregate['effort']})",
                f"{aggregate['valid_json']}/{turns}",
                f"{aggregate['schema_valid']}/{turns}",
                f"{aggregate['answer_nonempty']}/{turns}",
                f"{aggregate['gesture_plan_valid']}/{turns}",
                unknown,
                f"{aggregate['tts_success']}/{turns}",
                f"{aggregate['fallback_used']}/{turns}",
                str(latency) if latency is not None else "n/a",
            ]
        )
    lines = [
        "# Reachy Mini Simulation Comparison",
        "",
        f"Mode: `{mode}`. Shared task: five typed user questions through the Reachy answer, schema, gesture, and TTS pipeline.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reachy Mini simulated model-comparison benchmark.")
    parser.add_argument("--mode", choices=["smoke", "live"], default="smoke")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--output-dir", type=Path, default=DEMO_ROOT / "benchmark_results")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def load_existing_aggregates(output_dir: Path, models: list[str]) -> list[dict[str, Any]]:
    return [
        json.loads((output_dir / model_key / "metrics.json").read_text(encoding="utf-8"))["aggregate"]
        for model_key in models
    ]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        aggregates = load_existing_aggregates(args.output_dir, args.models)
        mode = aggregates[0].get("mode", "existing") if aggregates else "existing"
        (args.output_dir / "summary.md").write_text(render_summary(aggregates, mode), encoding="utf-8")
        print(args.output_dir / "summary.md")
        return 0

    aggregates: list[dict[str, Any]] = []
    with benchmark_environment():
        for model_key in args.models:
            spec = MODEL_SPECS[model_key]
            settings = model_settings(spec)
            capture = RawCapture()
            fugu_call = smoke_call_factory(spec, capture) if args.mode == "smoke" else live_call_factory(spec, capture)
            raw_records: list[dict[str, Any]] = []
            parsed_records: list[dict[str, Any]] = []
            generated_plans: list[dict[str, Any]] = []
            audio_records: list[dict[str, Any]] = []
            metrics: list[dict[str, Any]] = []
            for question in QUESTIONS:
                raw, parsed, generated_plan, audio, metric = run_turn(question, settings, fugu_call, capture)
                raw_records.append({"question": question, **raw})
                parsed_records.append({"question": question, "parsed_response": parsed})
                generated_plans.append(generated_plan or {"question": question, "gesture_plan": []})
                audio_records.append({"question": question, **audio})
                metrics.append(metric)
            aggregate = write_model_artifacts(
                args.output_dir / spec.key,
                spec,
                args.mode,
                raw_records,
                parsed_records,
                generated_plans,
                audio_records,
                metrics,
            )
            aggregates.append(aggregate)
            print(
                f"{spec.key}: valid_json={aggregate['valid_json']}/{aggregate['turns']} "
                f"schema_valid={aggregate['schema_valid']}/{aggregate['turns']} "
                f"fallbacks={aggregate['fallback_used']}/{aggregate['turns']}"
            )

    (args.output_dir / "summary.md").write_text(render_summary(aggregates, args.mode), encoding="utf-8")
    print(args.output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
