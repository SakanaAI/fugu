from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .audio.stt import transcribe_audio
from .audio.tts import synthesize_speech
from .config import Settings
from .fugu_client import call_fugu_json
from .schemas import fallback_gesture_plan, normalize_fugu_response


class UserFacingError(RuntimeError):
    pass


FAST_LOCAL_INTENT_PHRASES = {
    "what can you do",
    "what do you do",
    "who are you",
    "introduce yourself",
}


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    confidence: str
    citations: list[dict[str, str]]
    emotion: str
    gesture_plan: list[dict[str, Any]]
    audio_path: Path
    debug: dict[str, Any]


FuguCallable = Callable[[str, Settings], dict[str, Any]]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _stop_requested(stop_event: Any | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _raise_if_stopped(stop_event: Any | None) -> None:
    if _stop_requested(stop_event):
        raise UserFacingError("Generation canceled.")


def _transcription_empty_error(status: str) -> str:
    if status == "no_local_transcriber":
        return "No local transcription backend is configured. Type the question in the text box."
    if status == "audio_decode_error":
        return "The captured audio could not be decoded. Try the robot mic again or type the question."
    if status == "transcription_service_error":
        return "Speech-to-text service failed. Check internet access or type the question."
    if status == "speech_recognition_empty_transcript":
        return "Speech-to-text did not hear a clear phrase. Try again closer to Reachy or type the question."
    return "Ask a question by voice or text."


def _fast_local_intents_enabled() -> bool:
    raw = os.environ.get("REACHY_FAST_LOCAL_INTENTS", "1")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _fast_local_intent_response(question: str) -> dict[str, Any] | None:
    if not _fast_local_intents_enabled():
        return None
    lowered = re.sub(r"\s+", " ", question.lower()).strip(" ?.!")
    if not any(phrase in lowered for phrase in FAST_LOCAL_INTENT_PHRASES):
        return None
    answer = (
        "I can listen through Reachy's microphone, answer with Fugu, speak back, "
        "and move gently while tracking you."
    )
    return {
        "answer": answer,
        "confidence": "partial",
        "citations": [],
        "emotion": "happy",
        "gesture_plan": fallback_gesture_plan(answer, "partial", "happy"),
    }


def fixture_fugu_response(question: str) -> dict[str, Any]:
    answer = "I can answer conversationally and move while I speak. Ask me a question, and I will use Fugu Mini to respond."
    return {
        "answer": answer,
        "confidence": "partial",
        "citations": [],
        "emotion": "happy",
        "gesture_plan": fallback_gesture_plan(answer, "partial", "happy"),
        "dry_run_question": question,
    }


def build_robot_prompt(question: str) -> str:
    return f"""You are Reachy Mini, a small friendly robot powered by Sakana Fugu.
Answer the user in one or two short spoken sentences.
Return only JSON with this schema:
{{
  "answer": "short spoken answer",
  "confidence": "partial",
  "citations": [],
  "emotion": "happy|curious|thinking|neutral|excited",
  "gesture_plan": [
    {{"t_ms": 0, "primitive": "idle", "args": {{}}}},
    {{"t_ms": 300, "primitive": "curious_tilt", "args": {{}}}},
    {{"t_ms": 900, "primitive": "speak", "args": {{"text": "short phrase"}}}},
    {{"t_ms": 3000, "primitive": "idle", "args": {{}}}}
  ]
}}

User question: {question}
"""


def _result_from_response(
    *,
    question: str,
    raw: dict[str, Any],
    settings: Settings,
    transcript_status: str,
    fugu_ms: int,
    total_started: float,
    fugu_source: str,
) -> AnswerResult:
    normalized = normalize_fugu_response(raw, [])
    tts_started = time.monotonic()
    audio = synthesize_speech(normalized["answer"], settings.cache_dir / "audio")
    tts_ms = _elapsed_ms(tts_started)
    return AnswerResult(
        question=question,
        answer=normalized["answer"],
        confidence=normalized["confidence"],
        citations=normalized["citations"],
        emotion=normalized["emotion"],
        gesture_plan=normalized["gesture_plan"],
        audio_path=audio,
        debug={
            "fugu": fugu_source,
            "transcription": transcript_status,
            "fugu_model": settings.fugu_model,
            "fugu_api_mode": settings.fugu_api_mode,
            "fugu_reasoning_effort": settings.reasoning_effort,
            "fugu_max_output_tokens": settings.max_output_tokens,
            "latency_ms": {
                "fugu": fugu_ms,
                "tts": tts_ms,
                "total": _elapsed_ms(total_started),
            },
        },
    )


def answer_question(
    *,
    text_question: str,
    audio_path: str | None,
    settings: Settings,
    fugu_call: FuguCallable = call_fugu_json,
    stop_event: Any | None = None,
) -> AnswerResult:
    total_started = time.monotonic()
    typed = " ".join((text_question or "").split())
    transcript = transcribe_audio(audio_path) if not typed else None
    question = typed or (transcript.text if transcript else "")
    if not question:
        raise UserFacingError(_transcription_empty_error(transcript.status if transcript else "no_audio"))
    _raise_if_stopped(stop_event)

    fast_intent = _fast_local_intent_response(question)
    if fast_intent is not None:
        _raise_if_stopped(stop_event)
        return _result_from_response(
            question=question,
            raw=fast_intent,
            settings=settings,
            transcript_status=transcript.status if transcript else "typed",
            fugu_ms=0,
            total_started=total_started,
            fugu_source="local_fast_intent",
        )

    prompt = build_robot_prompt(question)
    _raise_if_stopped(stop_event)
    fugu_started = time.monotonic()
    raw = fixture_fugu_response(question) if settings.fugu_dry_run else fugu_call(prompt, settings)
    fugu_ms = _elapsed_ms(fugu_started)
    _raise_if_stopped(stop_event)
    return _result_from_response(
        question=question,
        raw=raw,
        settings=settings,
        transcript_status=transcript.status if transcript else "typed",
        fugu_ms=fugu_ms,
        total_started=total_started,
        fugu_source="dry_run_fixture" if settings.fugu_dry_run else "live",
    )
