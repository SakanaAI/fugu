from __future__ import annotations

import importlib.util
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transcription:
    text: str
    status: str


def _speech_recognition_language() -> str:
    value = os.environ.get("TRANSCRIPTION_LANGUAGE", "en-US").strip() or "en-US"
    return "en-US" if value.lower() in {"auto", "detect"} else value


def _load_speech_recognition():
    import speech_recognition as sr  # type: ignore[import-not-found]

    return sr


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return 0.0 if rate <= 0 else wav.getnframes() / float(rate)
    except (OSError, EOFError, wave.Error):
        return 0.0


def _minimum_stt_audio_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_STT_MIN_AUDIO_SECONDS", "1.0")))
    except ValueError:
        return 1.0


def _transcribe_with_speech_recognition(path: Path) -> Transcription:
    if importlib.util.find_spec("speech_recognition") is None:
        return Transcription(text="", status="no_local_transcriber")
    duration = _wav_duration_seconds(path)
    if 0.0 < duration < _minimum_stt_audio_seconds():
        return Transcription(text="", status="captured_audio_too_short")
    sr = _load_speech_recognition()
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(str(path)) as source:
            audio = recognizer.record(source)
    except Exception as exc:  # noqa: BLE001 - decode/backend errors should not crash the operator UI
        print(f"Robot STT audio decode failed: {exc}", file=sys.stderr)
        return Transcription(text="", status="audio_decode_error")
    try:
        text = str(recognizer.recognize_google(audio, language=_speech_recognition_language()) or "").strip()
    except sr.UnknownValueError:
        return Transcription(text="", status="speech_recognition_empty_transcript")
    except sr.RequestError as exc:
        print(f"Robot STT speech_recognition request failed: {exc}", file=sys.stderr)
        return Transcription(text="", status="transcription_service_error")
    if not text:
        return Transcription(text="", status="speech_recognition_empty_transcript")
    return Transcription(text=text, status="speech_recognition:google")


def transcribe_audio(audio_path: str | None) -> Transcription:
    started = time.monotonic()

    def finish(text: str, status: str) -> Transcription:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        preview = text[:80].replace("\n", " ")
        print(f"Robot STT {status} in {elapsed_ms} ms: {preview!r}", file=sys.stderr)
        return Transcription(text=text, status=status)

    fixture = os.environ.get("TRANSCRIPTION_FIXTURE")
    if fixture:
        return finish(fixture.strip(), "fixture")
    if not audio_path:
        return finish("", "no_audio")
    path = Path(audio_path)
    if not path.exists():
        return finish("", "missing_audio")
    transcript = _transcribe_with_speech_recognition(path)
    return finish(transcript.text, transcript.status)
