from __future__ import annotations

import json
import re
from typing import Any


PRIMITIVES = {
    "idle",
    "curious_tilt",
    "nod",
    "bow",
    "celebration",
    "antenna_pulse",
    "look_to_worker",
    "speak",
    "sound",
}
EMOTIONS = {"curious", "thinking", "happy", "neutral", "excited"}
CONFIDENCE = {"grounded", "partial", "not_found"}
MAX_SPEAK_CHARS = 160
MAX_ANSWER_CHARS = 220
MAX_ANSWER_SENTENCES = 2


class ValidationError(ValueError):
    pass


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
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValidationError("Fugu response must be a JSON object")
    return parsed


def unwrap_nested_answer_json(raw: dict[str, Any]) -> dict[str, Any]:
    answer = raw.get("answer")
    if not isinstance(answer, str):
        return raw
    stripped = answer.strip()
    if not stripped.startswith("```") and not (stripped.startswith("{") and '"answer"' in stripped):
        return raw
    try:
        nested = extract_json_object(stripped)
    except (json.JSONDecodeError, ValueError):
        return raw
    nested_answer = nested.get("answer")
    if not isinstance(nested_answer, str) or not nested_answer.strip():
        return raw
    merged = dict(raw)
    merged.update(nested)
    return merged


def split_speech_chunks(text: str, limit: int = 135, max_chunks: int = 4) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ["I do not know from this knowledge base yet."]
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        words = sentence.split()
        if len(sentence) <= limit:
            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= limit:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = sentence
            continue
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > limit and current:
                chunks.append(current)
                current = word
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks[:max_chunks] or ["I do not know from this knowledge base yet."]


def shorten_spoken_answer(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", clean) if sentence]
    if len(sentences) > MAX_ANSWER_SENTENCES:
        clean = " ".join(sentences[:MAX_ANSWER_SENTENCES]).strip()
    if len(clean) <= max_chars:
        return clean
    trimmed = clean[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed or clean[:max_chars].rstrip()


def _number(value: Any, name: str, low: float, high: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric")
    numeric = float(value)
    if numeric < low or numeric > high:
        raise ValidationError(f"{name} out of range {low}..{high}")
    return numeric


def validate_step(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValidationError(f"gesture {index}: must be an object")
    t_ms = raw.get("t_ms")
    if not isinstance(t_ms, int) or isinstance(t_ms, bool) or t_ms < 0:
        raise ValidationError(f"gesture {index}: t_ms must be a non-negative integer")
    primitive = raw.get("primitive")
    if primitive not in PRIMITIVES:
        raise ValidationError(f"gesture {index}: primitive is not whitelisted: {primitive!r}")
    args = raw.get("args") or {}
    if not isinstance(args, dict):
        raise ValidationError(f"gesture {index}: args must be an object")

    allowed: set[str] = set()
    if primitive == "antenna_pulse":
        allowed = {"duration"}
        if "duration" in args:
            _number(args["duration"], f"gesture {index}: duration", 0.1, 2.0)
    elif primitive == "look_to_worker":
        allowed = {"index"}
        worker = args.get("index")
        if not isinstance(worker, int) or isinstance(worker, bool) or worker < 0 or worker > 4:
            raise ValidationError(f"gesture {index}: look_to_worker.index must be 0..4")
    elif primitive == "speak":
        allowed = {"text"}
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError(f"gesture {index}: speak.text must be a non-empty string")
        if len(text) > MAX_SPEAK_CHARS:
            raise ValidationError(f"gesture {index}: speak.text exceeds {MAX_SPEAK_CHARS} chars")
    elif primitive == "sound":
        allowed = {"emotion"}
        emotion = str(args.get("emotion", "neutral")).strip().lower()
        if emotion not in EMOTIONS:
            raise ValidationError(f"gesture {index}: sound.emotion is not supported")
        args = {"emotion": emotion}

    unknown = set(args) - allowed
    if unknown:
        raise ValidationError(f"gesture {index}: unexpected args for {primitive}: {sorted(unknown)}")
    return {"t_ms": t_ms, "primitive": primitive, "args": args}


def validate_gesture_plan(raw_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValidationError("gesture_plan must be a non-empty list")
    validated = [validate_step(step, i) for i, step in enumerate(raw_plan)]
    last_time = -1
    for i, step in enumerate(validated):
        if step["t_ms"] < last_time:
            raise ValidationError(f"gesture {i}: t_ms must be sorted")
        last_time = step["t_ms"]
    if validated[0]["primitive"] != "idle":
        raise ValidationError("gesture_plan must start with idle")
    if validated[-1]["primitive"] != "idle":
        raise ValidationError("gesture_plan must end with idle")
    if validated[-1]["t_ms"] > 20000:
        raise ValidationError("gesture_plan must stay under 20 seconds")
    return validated


def fallback_gesture_plan(answer: str, confidence: str = "grounded", emotion: str = "neutral") -> list[dict[str, Any]]:
    if confidence not in CONFIDENCE:
        confidence = "partial"
    if emotion not in EMOTIONS:
        emotion = "thinking" if confidence != "grounded" else "neutral"
    chunks = split_speech_chunks(answer)
    plan: list[dict[str, Any]] = [
        {"t_ms": 0, "primitive": "idle", "args": {}},
        {"t_ms": 300, "primitive": "curious_tilt" if confidence != "grounded" else "nod", "args": {}},
        {"t_ms": 850, "primitive": "sound", "args": {"emotion": emotion}},
    ]
    cursor = 1300
    for chunk in chunks:
        plan.append({"t_ms": cursor, "primitive": "speak", "args": {"text": chunk}})
        cursor += max(1400, min(3600, 500 + len(chunk) * 42))
        if cursor < 15000:
            plan.append({"t_ms": cursor, "primitive": "nod", "args": {}})
            cursor += 450
    plan.append({"t_ms": min(cursor + 400, 19000), "primitive": "idle", "args": {}})
    return validate_gesture_plan(plan)


def normalize_fugu_response(raw: dict[str, Any], fallback_citations: list[dict[str, str]]) -> dict[str, Any]:
    raw = unwrap_nested_answer_json(raw)
    answer = str(raw.get("answer") or "").strip()
    confidence = str(raw.get("confidence") or "partial").strip().lower()
    if confidence not in CONFIDENCE:
        confidence = "partial"
    if not answer:
        answer = "I do not know from this knowledge base yet."
        confidence = "not_found"
    shortened_answer = shorten_spoken_answer(answer)
    answer_was_shortened = shortened_answer != answer
    answer = shortened_answer

    citations: list[dict[str, str]] = []
    for citation in raw.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        title = str(citation.get("title") or "").strip()
        url = str(citation.get("url") or "").strip()
        if title and url:
            citations.append({"title": title, "url": url})
    if confidence != "not_found" and not citations:
        citations = fallback_citations

    emotion = str(raw.get("emotion") or ("thinking" if confidence != "grounded" else "neutral")).strip().lower()
    if emotion not in EMOTIONS:
        emotion = "thinking" if confidence != "grounded" else "neutral"

    try:
        gesture_plan = validate_gesture_plan(raw.get("gesture_plan"))
        if answer_was_shortened or not any(step["primitive"] == "speak" for step in gesture_plan):
            gesture_plan = fallback_gesture_plan(answer, confidence, emotion)
    except ValidationError:
        gesture_plan = fallback_gesture_plan(answer, confidence, emotion)

    return {
        "answer": answer,
        "confidence": confidence,
        "citations": citations,
        "emotion": emotion,
        "gesture_plan": gesture_plan,
    }
