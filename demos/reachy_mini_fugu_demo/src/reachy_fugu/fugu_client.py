from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from .config import Settings
from .schemas import extract_json_object, fallback_gesture_plan


TIMEOUT_SECONDS = 20
JSONISH_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(?P<answer>(?:\\.|[^"\\])*)', re.DOTALL)
EMPTY_FUGU_ANSWER = "I heard you, but Fugu did not return an answer. Please try again."


def responses_input(prompt: str) -> str:
    return prompt


def chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def build_payload(prompt: str, settings: Settings) -> dict[str, Any]:
    if settings.fugu_api_mode == "chat_completions":
        return {
            "model": settings.fugu_model,
            "messages": chat_messages(prompt),
            "max_tokens": settings.max_output_tokens,
        }
    return {
        "model": settings.fugu_model,
        "input": responses_input(prompt),
        "reasoning": {"effort": settings.reasoning_effort},
        "max_output_tokens": settings.max_output_tokens,
        "stream": False,
    }


def fugu_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("FUGU_TIMEOUT_SECONDS", str(TIMEOUT_SECONDS))))
    except ValueError:
        return float(TIMEOUT_SECONDS)


def extract_response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list):
        parts: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
        if parts:
            return "".join(parts)
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                parts.append(text)
    if parts:
        return "".join(parts)
    message = response.get("message")
    if isinstance(message, str):
        return message
    return ""


def extract_jsonish_answer(text: str) -> str:
    match = JSONISH_ANSWER_RE.search(text)
    if not match:
        return ""
    raw_answer = match.group("answer")
    try:
        return str(json.loads(f'"{raw_answer}"')).strip()
    except json.JSONDecodeError:
        return raw_answer.replace('\\"', '"').strip()


def call_fugu_json(prompt: str, settings: Settings) -> dict[str, Any]:
    api_key = settings.require_api_key()
    body = build_payload(prompt, settings)
    endpoint = "/chat/completions" if settings.fugu_api_mode == "chat_completions" else "/responses"
    request = urllib.request.Request(
        f"{settings.sakana_base_url}{endpoint}",
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
        with urllib.request.urlopen(request, timeout=fugu_timeout_seconds()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Sakana Fugu API returned HTTP {exc.code}: {detail}", file=sys.stderr)
        raise RuntimeError(f"Sakana Fugu API returned HTTP {exc.code}. Check API key, base URL, and model settings.") from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError("Sakana Fugu API request failed. Check network access, base URL, and retry.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Sakana Fugu API returned a non-JSON response envelope.") from exc
    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(f"Sakana Fugu {settings.fugu_model} ({settings.fugu_api_mode}) returned in {elapsed_ms} ms", file=sys.stderr)
    text = extract_response_text(payload)
    try:
        return extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        answer = extract_jsonish_answer(text) or " ".join(text.split())
        if not answer:
            answer = EMPTY_FUGU_ANSWER
        return {
            "answer": answer,
            "confidence": "partial",
            "citations": [],
            "emotion": "neutral",
            "gesture_plan": fallback_gesture_plan(answer, "partial", "neutral"),
        }
