from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from reachy_fugu.audio.stt import transcribe_audio
from reachy_fugu.config import Settings


@dataclass(frozen=True)
class RobotTranscript:
    text: str
    status: str
    audio_path: Path | None = None
    duration_seconds: float = 0.0
    rms: float = 0.0


class RobotInputUnavailableError(RuntimeError):
    """Raised when the robot mic/camera media stream cannot be opened."""


class RobotTrackingSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.media = None

    def __enter__(self) -> "RobotTrackingSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.media is not None:
            self.media.close()

    def pose(self) -> dict[str, Any] | None:
        source = _tracking_source()
        if source in {"doa", "mic", "audio"}:
            return doa_tracking_pose(self.settings)
        camera_pose = self._camera_pose()
        if camera_pose is not None or source in {"camera", "vision"}:
            return camera_pose
        return doa_tracking_pose(self.settings)

    def _camera_pose(self) -> dict[str, Any] | None:
        if self.media is None:
            try:
                self.media = _open_robot_media(self.settings)
            except RuntimeError:
                return None
        if self.media is None:
            return None
        frame = self.media.get_frame()
        if frame is None:
            return None
        target = target_from_frame(np.asarray(frame))
        if target is None:
            return None
        u, v, confidence = target
        if confidence < 0.08:
            return None
        return _tracking_pose_from_frame_target(np.asarray(frame), u, v)


class RobotInputSession:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.media = None

    def __enter__(self) -> "RobotInputSession":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> None:
        if self.media is None:
            self.media = _open_robot_media(self.settings)

    def close(self) -> None:
        if self.media is None:
            return
        try:
            self.media.close()
        finally:
            self.media = None
            _acquire_daemon_media(self.settings)

    def record_question(self, duration_seconds: float = 7.0, stop_event: Any | None = None) -> RobotTranscript:
        self.open()
        if os.environ.get("REACHY_MIC_CAPTURE_MODE", "speech").strip().lower() in {"fixed", "simple"}:
            return _record_robot_question_fixed(
                self.media,
                duration_seconds,
                stop_event=stop_event,
                close_media=False,
            )
        return _record_robot_question_speech(
            self.media,
            duration_seconds,
            stop_event=stop_event,
            close_media=False,
        )


def _robot_connection(settings: Settings) -> tuple[str, int, str]:
    parsed = urlparse(settings.reachy_daemon_url)
    host = os.environ.get("REACHY_SDK_HOST") or parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    mode = os.environ.get("REACHY_SDK_CONNECTION_MODE")
    if mode:
        return host, port, mode
    if host in {"127.0.0.1", "localhost", "::1"}:
        return host, port, "localhost_only"
    return host, port, "network"


def _reachy_control_kwargs(settings: Settings) -> dict[str, Any]:
    host, port, mode = _robot_connection(settings)
    return {
        "host": host,
        "port": port,
        "connection_mode": mode,
        "media_backend": "no_media",
        "timeout": float(os.environ.get("REACHY_SDK_TIMEOUT_SECONDS", "5")),
    }


def _media_signalling_host(settings: Settings) -> str:
    explicit = os.environ.get("REACHY_MEDIA_SIGNALLING_HOST") or os.environ.get("REACHY_SIGNALLING_HOST")
    if explicit:
        return explicit
    parsed = urlparse(settings.reachy_daemon_url)
    host = parsed.hostname or "127.0.0.1"
    if host in {"localhost", "::1"}:
        return "127.0.0.1"
    return host


def _open_robot_media(settings: Settings):
    try:
        from reachy_mini.media.media_manager import MediaBackend, MediaManager
    except Exception as exc:  # noqa: BLE001 - optional robot SDK dependency
        raise RobotInputUnavailableError(
            "reachy-mini SDK media support is required for robot microphone and camera input"
        ) from exc

    backend_name = os.environ.get("REACHY_INPUT_MEDIA_BACKEND", "webrtc")
    try:
        backend = MediaBackend(backend_name.lower())
    except ValueError as exc:
        raise RuntimeError(f"Unsupported robot input media backend: {backend_name}") from exc

    attempts = max(1, int(os.environ.get("REACHY_MEDIA_OPEN_ATTEMPTS", "2")))
    delay_seconds = max(0.05, float(os.environ.get("REACHY_MEDIA_OPEN_RETRY_SECONDS", "0.25")))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return MediaManager(
                backend=backend,
                signalling_host=_media_signalling_host(settings),
                daemon_url=settings.reachy_daemon_url,
                log_level=os.environ.get("REACHY_MEDIA_LOG_LEVEL", "WARNING"),
            )
        except Exception as exc:  # noqa: BLE001 - SDK/GStreamer can raise several backend-specific exceptions
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(delay_seconds)
    raise RobotInputUnavailableError(f"Robot media stream unavailable through {backend.value}: {last_error}") from last_error


def _tracking_source() -> str:
    return os.environ.get("REACHY_TRACKING_SOURCE", "doa").strip().lower()


def _minimum_transcribe_rms() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_MIC_MIN_TRANSCRIBE_RMS", "0.003")))
    except ValueError:
        return 0.003


def _acquire_daemon_media(settings: Settings) -> None:
    request = urllib.request.Request(
        f"{settings.reachy_daemon_url}/api/media/acquire",
        data=b"{}",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0):
            return
    except (OSError, urllib.error.URLError):
        return


def _sample_rms(sample: np.ndarray) -> float:
    if sample.size == 0:
        return 0.0
    sample_float = _audio_sample_to_float(sample)
    return float(np.sqrt(np.mean(sample_float * sample_float)))


def _audio_sample_to_float(sample: np.ndarray) -> np.ndarray:
    array = np.asarray(sample)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return array.astype(np.float32) / scale

    sample_float = array.astype(np.float32)
    if sample_float.size == 0:
        return sample_float
    max_abs = float(np.max(np.abs(sample_float)))
    if max_abs > 1.5:
        scale = 32768.0 if max_abs <= 32768.0 else max_abs
        sample_float = sample_float / scale
    return sample_float


def _record_robot_question_fixed(
    media,
    duration_seconds: float,
    stop_event: Any | None = None,
    *,
    close_media: bool = True,
) -> RobotTranscript:
    samples: list[np.ndarray] = []
    sample_rate = 16000
    channels = 2
    record_seconds = float(os.environ.get("REACHY_MIC_FIXED_RECORD_SECONDS", "3.0"))
    max_wall_seconds = float(os.environ.get("REACHY_MIC_MAX_WALL_SECONDS", str(record_seconds + 2.0)))
    poll_seconds = float(os.environ.get("REACHY_MIC_POLL_SECONDS", "0.01"))
    silence_threshold = float(os.environ.get("REACHY_MIC_SILENCE_RMS_THRESHOLD", "0.0015"))
    try:
        media.start_recording()
        sample_rate = int(media.get_input_audio_samplerate() or 16000)
        channels = int(media.get_input_channels() or 2)
        started_at = time.monotonic()
        target_seconds = min(max(record_seconds, 1.0), max(duration_seconds, record_seconds))
        target_frames = int(sample_rate * target_seconds)
        collected_frames = 0
        deadline = started_at + max(target_seconds, max_wall_seconds)
        while time.monotonic() < deadline and collected_frames < target_frames:
            if stop_event is not None and stop_event.is_set():
                break
            sample = media.get_audio_sample()
            if sample is not None and sample.size:
                sample_float = _audio_sample_to_float(sample)
                samples.append(sample_float)
                collected_frames += int(sample_float.shape[0]) if sample_float.ndim >= 1 else 0
            time.sleep(max(0.001, min(0.04, poll_seconds)))
    finally:
        try:
            media.stop_recording()
        except Exception:
            pass
        if close_media:
            media.close()

    audio_path = _write_wav(samples, sample_rate, channels) if samples else None
    duration = 0.0
    mean_rms = 0.0
    if samples:
        merged = np.concatenate(samples, axis=0)
        duration = float(merged.shape[0]) / float(sample_rate)
        mean_rms = _sample_rms(merged)
    print(
        f"Robot mic fixed capture: duration={duration:.2f}s samples={len(samples)} rms={mean_rms:.5f} path={audio_path}",
        file=sys.stderr,
    )
    if stop_event is not None and stop_event.is_set():
        return RobotTranscript(
            text="",
            status="stopped",
            audio_path=audio_path,
            duration_seconds=duration,
            rms=mean_rms,
        )
    if audio_path is None or mean_rms < silence_threshold:
        return RobotTranscript(
            text="",
            status="no_speech_detected",
            audio_path=audio_path,
            duration_seconds=duration,
            rms=mean_rms,
        )
    transcript = transcribe_audio(str(audio_path))
    return RobotTranscript(
        text=transcript.text,
        status=transcript.status,
        audio_path=audio_path,
        duration_seconds=duration,
        rms=mean_rms,
    )


def _write_wav(samples: list[np.ndarray], sample_rate: int, channels: int) -> Path:
    merged = _audio_sample_to_float(np.concatenate(samples, axis=0))
    if merged.ndim == 1:
        merged = merged.reshape(-1, 1)
    if merged.shape[1] > 1:
        merged = merged.mean(axis=1, keepdims=True)
    channels = 1
    peak = float(np.max(np.abs(merged))) if merged.size else 0.0
    if peak > 0.0001 and peak < 0.35:
        merged = merged * min(20.0, 0.8 / peak)
    pcm = np.clip(merged, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype("<i2")
    tmp_dir = Path(tempfile.gettempdir()) / "reachy_fugu_robot_mic"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"robot_mic_{int(time.time() * 1000)}.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())
    return path


def _record_robot_question_speech(
    media,
    duration_seconds: float = 7.0,
    stop_event: Any | None = None,
    *,
    close_media: bool = True,
) -> RobotTranscript:
    samples: list[np.ndarray] = []
    pre_roll_chunks = max(1, int(os.environ.get("REACHY_MIC_PRE_ROLL_CHUNKS", "50")))
    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
    sample_rate = 16000
    channels = 2
    heard_speech = False
    first_speech_at: float | None = None
    noise_floor = float(os.environ.get("REACHY_MIC_INITIAL_NOISE_RMS", "0.0012"))
    min_threshold = float(os.environ.get("REACHY_MIC_RMS_THRESHOLD", "0.0035"))
    max_wait_seconds = float(os.environ.get("REACHY_MIC_WAIT_SECONDS", "2.5"))
    trailing_silence_seconds = float(os.environ.get("REACHY_MIC_TRAILING_SILENCE_SECONDS", "0.8"))
    min_speech_record_seconds = float(os.environ.get("REACHY_MIC_MIN_RECORD_SECONDS", "1.2"))
    min_transcribe_seconds = float(os.environ.get("REACHY_MIC_MIN_TRANSCRIBE_SECONDS", "1.35"))
    max_speech_record_seconds = float(os.environ.get("REACHY_MIC_MAX_SPEECH_SECONDS", "5.0"))
    poll_seconds = float(os.environ.get("REACHY_MIC_POLL_SECONDS", "0.02"))
    collected_frames = 0
    silence_frames_after_speech = 0
    try:
        media.start_recording()
        sample_rate = int(media.get_input_audio_samplerate() or 16000)
        channels = int(media.get_input_channels() or 2)
        started_at = time.monotonic()
        deadline = started_at + max_wait_seconds + max(1.0, duration_seconds)
        while time.monotonic() < deadline:
            if stop_event is not None and stop_event.is_set():
                break
            sample = media.get_audio_sample()
            if sample is not None and sample.size:
                sample_float = _audio_sample_to_float(sample)
                rms = _sample_rms(sample_float)
                threshold = max(min_threshold, noise_floor * 3.5)
                if rms >= threshold:
                    if not heard_speech:
                        samples.extend(pre_roll)
                        collected_frames += sum(int(chunk.shape[0]) if chunk.ndim >= 1 else 0 for chunk in pre_roll)
                        first_speech_at = time.monotonic()
                    heard_speech = True
                    silence_frames_after_speech = 0
                if heard_speech:
                    samples.append(sample_float)
                    sample_frames = int(sample_float.shape[0]) if sample_float.ndim >= 1 else 0
                    collected_frames += sample_frames
                    if rms < threshold:
                        silence_frames_after_speech += sample_frames
                else:
                    pre_roll.append(sample_float)
                    noise_floor = (noise_floor * 0.96) + (min(rms, noise_floor * 2.0) * 0.04)
                    if time.monotonic() - started_at >= max_wait_seconds:
                        break
                if heard_speech and first_speech_at is not None:
                    elapsed_wall_record = time.monotonic() - first_speech_at
                    collected_audio_seconds = float(collected_frames) / float(sample_rate)
                    captured_silence_seconds = float(silence_frames_after_speech) / float(sample_rate)
                    if collected_audio_seconds >= max_speech_record_seconds or elapsed_wall_record >= max_speech_record_seconds + 3.0:
                        break
                    if (
                        collected_audio_seconds >= min_speech_record_seconds
                        and collected_audio_seconds >= min_transcribe_seconds
                        and captured_silence_seconds >= trailing_silence_seconds
                    ):
                        break
            time.sleep(max(0.001, min(0.04, poll_seconds)))
    finally:
        try:
            media.stop_recording()
        except Exception:
            pass
        if close_media:
            media.close()

    if stop_event is not None and stop_event.is_set():
        audio_path = _write_wav(samples if samples else list(pre_roll), sample_rate, channels) if samples or pre_roll else None
        duration = 0.0
        mean_rms = 0.0
        if samples or pre_roll:
            merged = np.concatenate(samples if samples else list(pre_roll), axis=0)
            duration = float(merged.shape[0]) / float(sample_rate)
            mean_rms = _sample_rms(merged)
        print(
            f"Robot mic speech capture: stopped duration={duration:.2f}s "
            f"samples={len(samples) if samples else len(pre_roll)} rms={mean_rms:.5f} path={audio_path}",
            file=sys.stderr,
        )
        return RobotTranscript(
            text="",
            status="stopped",
            audio_path=audio_path,
            duration_seconds=duration,
            rms=mean_rms,
        )
    if not heard_speech:
        audio_path = _write_wav(list(pre_roll), sample_rate, channels) if pre_roll else None
        duration = 0.0
        mean_rms = 0.0
        if pre_roll:
            merged = np.concatenate(list(pre_roll), axis=0)
            duration = float(merged.shape[0]) / float(sample_rate)
            mean_rms = _sample_rms(merged)
        print(
            f"Robot mic speech capture: no speech duration={duration:.2f}s samples={len(pre_roll)} "
            f"rms={mean_rms:.5f} path={audio_path}",
            file=sys.stderr,
        )
        return RobotTranscript(
            text="",
            status="no_speech_detected",
            audio_path=audio_path,
            duration_seconds=duration,
            rms=mean_rms,
        )
    if not samples:
        raise RuntimeError("Robot microphone returned no speech samples")
    audio_path = _write_wav(samples, sample_rate, channels)
    merged = np.concatenate(samples, axis=0)
    duration = float(merged.shape[0]) / float(sample_rate)
    mean_rms = _sample_rms(merged)
    print(
        f"Robot mic speech capture: duration={duration:.2f}s samples={len(samples)} "
        f"rms={mean_rms:.5f} path={audio_path}",
        file=sys.stderr,
    )
    if mean_rms < _minimum_transcribe_rms():
        return RobotTranscript(
            text="",
            status="no_speech_detected",
            audio_path=audio_path,
            duration_seconds=duration,
            rms=mean_rms,
        )
    transcript = transcribe_audio(str(audio_path))
    return RobotTranscript(
        text=transcript.text,
        status=transcript.status,
        audio_path=audio_path,
        duration_seconds=duration,
        rms=mean_rms,
    )


def record_robot_question(settings: Settings, duration_seconds: float = 7.0, stop_event: Any | None = None) -> RobotTranscript:
    with RobotInputSession(settings) as session:
        return session.record_question(duration_seconds=duration_seconds, stop_event=stop_event)


def target_from_frame(frame: np.ndarray) -> tuple[int, int, float] | None:
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return None
    step_y = max(1, height // 90)
    step_x = max(1, width // 120)
    small = frame[::step_y, ::step_x, :3].astype(np.float32)
    # SDK camera frames are BGR.
    b = small[:, :, 0]
    g = small[:, :, 1]
    r = small[:, :, 2]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    saturation = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
    skin_like = (r > 72) & (g > 40) & (b > 28) & (r > g * 1.04) & (r > b * 1.18)
    yy, xx = np.indices(luminance.shape)
    center_bias = 1.0 - np.minimum(0.55, np.abs((xx / max(1, luminance.shape[1] - 1)) - 0.5) * 0.7)
    weight = np.maximum(0.0, luminance - float(luminance.mean()) + saturation * 0.25 + skin_like * 42.0)
    weight *= center_bias
    total = float(weight.sum())
    if total < 3500.0:
        return None
    x = float((xx * weight).sum() / total) * step_x
    y = float((yy * weight).sum() / total) * step_y
    confidence = min(1.0, total / 180000.0)
    return int(max(1, min(width - 2, x))), int(max(1, min(height - 2, y))), confidence


def _tracking_pose_from_frame_target(frame: np.ndarray, u: int, v: int) -> dict[str, Any]:
    height, width = frame.shape[:2]
    dx = (u / max(1.0, float(width))) - 0.5
    dy = (v / max(1.0, float(height))) - 0.5
    yaw = max(-34.0, min(34.0, dx * 68.0))
    pitch = max(-18.0, min(14.0, -dy * 36.0 - 3.0))
    return {
        "yaw": yaw,
        "body_yaw": yaw * 0.3,
        "pitch": pitch,
        "right_antenna": -24.0,
        "left_antenna": 24.0,
    }


def robot_tracking_session(settings: Settings) -> RobotTrackingSession:
    return RobotTrackingSession(settings)


def doa_tracking_pose(settings: Settings) -> dict[str, Any] | None:
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            f"{settings.reachy_daemon_url}/api/state/doa",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=0.7) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("speech_detected"):
        return None
    angle = payload.get("angle")
    if not isinstance(angle, (int, float)) or isinstance(angle, bool):
        return None
    yaw = max(-34.0, min(34.0, (float(angle) - (math.pi / 2.0)) * 180.0 / math.pi))
    return {
        "yaw": yaw,
        "body_yaw": yaw * 0.36,
        "pitch": -3.0,
        "right_antenna": -26.0,
        "left_antenna": 26.0,
    }
