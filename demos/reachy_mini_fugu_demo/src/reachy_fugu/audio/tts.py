from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import wave
from array import array
from pathlib import Path


DEFAULT_TTS_VOICE = "Eddy (English (US))"
DEFAULT_TTS_RATE = "230"
FALLBACK_MACOS_VOICES = ("Zarvox", "Junior")


def _run(command: list[str]) -> bool:
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _wav_stats(path: Path) -> tuple[float, float]:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0, 0.0
            frames = wav.readframes(wav.getnframes())
            duration = wav.getnframes() / rate
            if not frames:
                return duration, 0.0
            if wav.getsampwidth() == 2:
                samples = array("h")
                samples.frombytes(frames)
                if samples.itemsize != 2:
                    samples.byteswap()
                peak = max(abs(value) for value in samples) / 32768.0
                return duration, peak
            peak = max(frames) / 255.0
            return duration, peak
    except (OSError, EOFError, wave.Error):
        return 0.0, 0.0


def _min_expected_duration(text: str) -> float:
    if len(text) <= 8:
        return 0.2
    return min(1.0, max(0.35, len(text) * 0.012))


def _usable_wav(path: Path, text: str) -> bool:
    if not path.exists():
        return False
    duration, peak = _wav_stats(path)
    return duration >= _min_expected_duration(text) and peak >= 0.003


def _remove_bad_wav(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _write_tone(path: Path, text: str) -> None:
    duration = max(0.8, min(4.0, 0.35 + len(text) * 0.025))
    sample_rate = 22050
    frames = int(sample_rate * duration)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(frames):
            progress = i / max(frames - 1, 1)
            envelope = math.sin(math.pi * progress)
            hz = 420 + 180 * math.sin(progress * math.pi * 2)
            value = int(7000 * envelope * math.sin(2 * math.pi * hz * i / sample_rate))
            wav.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))


def _cache_digest(text: str, backend: str, voice: str, rate: str) -> str:
    key = "\n".join([backend, voice, rate, text])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def _macos_voice_candidates() -> list[str]:
    configured = os.environ.get("REACHY_TTS_VOICE")
    candidates: list[str] = []
    if configured and configured.strip():
        candidates.append(configured.strip())
    candidates.extend([DEFAULT_TTS_VOICE, *FALLBACK_MACOS_VOICES])
    return list(dict.fromkeys(candidates))


def synthesize_speech(text: str, cache_dir: Path) -> Path:
    clean = " ".join(text.split())[:1200] or "I do not know from this knowledge base yet."
    cache_dir.mkdir(parents=True, exist_ok=True)

    say = shutil.which("say")
    afconvert = shutil.which("afconvert")
    rate = os.environ.get("REACHY_TTS_RATE", DEFAULT_TTS_RATE).strip() or DEFAULT_TTS_RATE
    if say and afconvert:
        for voice in _macos_voice_candidates():
            digest = _cache_digest(clean, "say", voice, rate)
            wav_path = cache_dir / f"reachy_fugu_{digest}.wav"
            if _usable_wav(wav_path, clean):
                return wav_path
            _remove_bad_wav(wav_path)
            aiff_path = cache_dir / f"reachy_fugu_{digest}.aiff"
            if _run([say, "-v", voice, "-r", rate, "-o", str(aiff_path), clean]) and _run(
                [afconvert, "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)]
            ) and _usable_wav(wav_path, clean):
                return wav_path
            _remove_bad_wav(wav_path)

    for command_name in ("espeak-ng", "espeak"):
        command = shutil.which(command_name)
        if command:
            digest = _cache_digest(clean, command_name, "", rate)
            wav_path = cache_dir / f"reachy_fugu_{digest}.wav"
            if _usable_wav(wav_path, clean):
                return wav_path
            _remove_bad_wav(wav_path)
            if _run([command, "-w", str(wav_path), "-s", rate, clean]) and _usable_wav(wav_path, clean):
                return wav_path
            _remove_bad_wav(wav_path)

    digest = _cache_digest(clean, "tone", "", rate)
    wav_path = cache_dir / f"reachy_fugu_{digest}.wav"
    if _usable_wav(wav_path, clean):
        return wav_path
    _remove_bad_wav(wav_path)
    _write_tone(wav_path, clean)
    return wav_path
