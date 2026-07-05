from __future__ import annotations

import math
import re
import tempfile
import wave
from pathlib import Path


SOUND_EMOTIONS = {
    "happy",
    "sad",
    "curious",
    "thinking",
    "excited",
    "confused",
    "sleepy",
    "neutral",
}

_ASSET_BY_EMOTION = {
    "happy": "dance1.wav",
    "excited": "dance1.wav",
    "sad": "go_sleep.wav",
    "sleepy": "go_sleep.wav",
    "curious": "confused1.wav",
    "thinking": "confused1.wav",
    "confused": "confused1.wav",
    "neutral": "wake_up.wav",
}

_KEYWORDS_BY_EMOTION = {
    "sad": ("sad", "sorrow", "down", "droop", "cry", "teary", "gloom", "gentle sadness"),
    "sleepy": ("sleepy", "tired", "drowsy", "sleep", "yawn", "settle"),
    "excited": ("excited", "delighted", "celebrat", "dance", "aha", "brighten"),
    "happy": ("happy", "joy", "smile", "cheer", "pleased"),
    "curious": ("curious", "wonder", "puzzled", "question"),
    "thinking": ("thinking", "think", "thoughtful", "consider", "pause"),
    "confused": ("confused", "unsure", "uncertain"),
}

_TONE_BY_EMOTION = {
    "happy": (660, 880, 0.42),
    "excited": (740, 1040, 0.5),
    "sad": (330, 220, 0.62),
    "sleepy": (260, 180, 0.72),
    "curious": (520, 700, 0.46),
    "thinking": (420, 510, 0.5),
    "confused": (390, 470, 0.56),
    "neutral": (440, 560, 0.4),
}


def normalize_sound_emotion(value: str | None) -> str:
    emotion = re.sub(r"[^a-z]+", "", (value or "neutral").lower())
    return emotion if emotion in SOUND_EMOTIONS else "neutral"


def sound_asset_name(emotion: str | None) -> str:
    return _ASSET_BY_EMOTION[normalize_sound_emotion(emotion)]


def bundled_sound_path(emotion: str | None) -> Path | None:
    try:
        import reachy_mini  # type: ignore[import-not-found]
    except Exception:
        return None
    asset = Path(reachy_mini.__file__).resolve().parent / "assets" / sound_asset_name(emotion)
    return asset if asset.exists() else None


def synthetic_sound_path(emotion: str | None) -> Path:
    normalized = normalize_sound_emotion(emotion)
    path = Path(tempfile.gettempdir()) / f"reachy_{normalized}_cue.wav"
    if path.exists():
        return path

    start_hz, end_hz, duration = _TONE_BY_EMOTION[normalized]
    sample_rate = 22050
    frames = int(sample_rate * duration)
    amplitude = 9000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            progress = i / max(frames - 1, 1)
            hz = start_hz + (end_hz - start_hz) * progress
            envelope = math.sin(math.pi * progress)
            value = int(amplitude * envelope * math.sin(2 * math.pi * hz * i / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return path


def infer_sound_sequence(text: str, limit: int = 2) -> list[str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for emotion, keywords in _KEYWORDS_BY_EMOTION.items():
        positions = [lowered.find(keyword) for keyword in keywords if lowered.find(keyword) >= 0]
        if positions:
            matches.append((min(positions), emotion))
    matches.sort()

    sequence: list[str] = []
    for _, emotion in matches:
        if emotion == "thinking" and "curious" in sequence:
            continue
        if emotion == "happy" and "excited" in sequence:
            continue
        if emotion not in sequence:
            sequence.append(emotion)
        if len(sequence) >= limit:
            break
    return sequence


def fallback_sound_cues(text: str, max_t_ms: int = 14000) -> list[dict[str, object]]:
    emotions = infer_sound_sequence(text)
    if not emotions:
        return []

    if len(emotions) == 1:
        times = [250]
    elif len(emotions) == 2:
        times = [250, max(1600, min(max_t_ms - 1200, max_t_ms // 2))]
    else:
        times = [250, max(1800, max_t_ms // 2), max(3000, min(max_t_ms - 1200, int(max_t_ms * 0.78)))]

    return [
        {"t_ms": max(0, int(t_ms)), "primitive": "sound", "args": {"emotion": emotion}}
        for t_ms, emotion in zip(times, emotions, strict=False)
    ]


def add_fallback_sound_cues(cues: list[dict[str, object]], text: str) -> list[dict[str, object]]:
    if any(cue.get("primitive") == "sound" for cue in cues):
        return cues
    max_t_ms = max((int(cue.get("t_ms") or 0) for cue in cues), default=14000)
    sound_cues = fallback_sound_cues(text, max_t_ms=max(max_t_ms, 3000))
    if not sound_cues:
        return cues
    return sorted([*cues, *sound_cues], key=lambda cue: int(cue.get("t_ms") or 0))
