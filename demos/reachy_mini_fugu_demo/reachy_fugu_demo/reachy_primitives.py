from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import urllib.error
import urllib.request

from .sounds import bundled_sound_path, normalize_sound_emotion, sound_asset_name, synthetic_sound_path


HEAD_ROLL_LIMIT = (-40.0, 40.0)
HEAD_PITCH_LIMIT = (-40.0, 40.0)
HEAD_YAW_LIMIT = (-65.0, 65.0)
BODY_YAW_LIMIT = (-55.0, 55.0)
ANTENNA_LIMIT = (-70.0, 70.0)

WORKER_YAWS = [-42.0, -20.0, 0.0, 20.0, 42.0]
REST_ANTENNAS = (-10.0, 10.0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Pose:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    body_yaw: float = 0.0
    right_antenna: float = REST_ANTENNAS[0]
    left_antenna: float = REST_ANTENNAS[1]

    def clamped(self) -> "Pose":
        return Pose(
            roll=clamp(self.roll, *HEAD_ROLL_LIMIT),
            pitch=clamp(self.pitch, *HEAD_PITCH_LIMIT),
            yaw=clamp(self.yaw, *HEAD_YAW_LIMIT),
            body_yaw=clamp(self.body_yaw, *BODY_YAW_LIMIT),
            right_antenna=clamp(self.right_antenna, *ANTENNA_LIMIT),
            left_antenna=clamp(self.left_antenna, *ANTENNA_LIMIT),
        )


class Backend(Protocol):
    def goto(self, pose: Pose, duration: float = 0.7, method: str = "minjerk") -> None:
        ...

    def speak(self, text: str) -> None:
        ...

    def sound(self, emotion: str = "neutral") -> None:
        ...

    def close(self) -> None:
        ...


class DryRunBackend:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def goto(self, pose: Pose, duration: float = 0.7, method: str = "minjerk") -> None:
        event = {
            "pose": pose.clamped().__dict__,
            "duration": round(duration, 3),
            "method": method,
        }
        self.events.append(event)
        print(f"[dry-run] goto {event}")

    def speak(self, text: str) -> None:
        self.events.append({"speak": text})
        print(f"[dry-run] speak: {text}")

    def sound(self, emotion: str = "neutral") -> None:
        normalized = normalize_sound_emotion(emotion)
        self.events.append({"sound": normalized})
        print(f"[dry-run] sound: {normalized}")

    def close(self) -> None:
        print("[dry-run] close")


class SDKBackend:
    def __init__(
        self,
        connection_mode: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.connection_mode = connection_mode
        self.host = host
        self.port = port
        self._mini = None
        self._create_head_pose = None

    def __enter__(self) -> "SDKBackend":
        try:
            from reachy_mini import ReachyMini
            from reachy_mini.utils import create_head_pose
        except Exception as exc:  # noqa: BLE001 - optional dependency
            detail = f"{type(exc).__name__}: {exc}"
            print(f"[reachy sdk] import failed: {detail}", file=sys.stderr)
            raise RuntimeError(f"reachy-mini SDK import failed ({detail})") from exc

        kwargs: dict[str, Any] = {"media_backend": os.environ.get("REACHY_MEDIA_BACKEND", "no_media")}
        if self.connection_mode:
            kwargs["connection_mode"] = self.connection_mode
        if self.host:
            kwargs["host"] = self.host
        if self.port:
            kwargs["port"] = self.port
        self._mini = ReachyMini(**kwargs)
        self._mini.__enter__()
        self._create_head_pose = create_head_pose
        # The SDK's no_media mode releases daemon media; this demo still plays
        # generated speech through the daemon speaker, so reacquire it explicitly.
        self._post_daemon_json("/api/media/acquire", {}, timeout=3.0)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def goto(self, pose: Pose, duration: float = 0.7, method: str = "minjerk") -> None:
        if self._mini is None or self._create_head_pose is None:
            raise RuntimeError("SDKBackend is not connected")
        p = pose.clamped()
        head = self._create_head_pose(
            roll=p.roll,
            pitch=p.pitch,
            yaw=p.yaw,
            degrees=True,
            mm=False,
        )
        antennas = [math.radians(p.right_antenna), math.radians(p.left_antenna)]
        try:
            self._mini.goto_target(
                head=head,
                antennas=antennas,
                body_yaw=math.radians(p.body_yaw),
                duration=max(0.2, float(duration)),
                method=method,
            )
        except TimeoutError as exc:
            print(f"[reachy motion cue] timeout; continuing playback: {exc}")

    def speak(self, text: str) -> None:
        spoken = " ".join(str(text).split())[:160]
        if not spoken:
            return
        print(f"[reachy speak cue] {spoken}")
        if self._play_speech_on_daemon(spoken):
            return
        if self._play_local_speech(spoken):
            print("[reachy speech cue] local say fallback")

    def _daemon_url(self) -> str:
        if self._mini is None:
            return ""
        explicit = getattr(self._mini, "_daemon_http_url", "")
        if explicit:
            return str(explicit).rstrip("/")
        client = getattr(self._mini, "client", None)
        host = getattr(client, "host", "")
        port = getattr(client, "port", "")
        if host and port:
            return f"http://{host}:{port}".rstrip("/")
        return ""

    def _post_daemon_json(self, path: str, payload: dict[str, Any], timeout: float = 2.0) -> bool:
        daemon_url = self._daemon_url()
        if not daemon_url:
            return False
        req = urllib.request.Request(
            f"{daemon_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except (OSError, urllib.error.URLError) as exc:
            print(f"[reachy daemon cue] {path} failed: {exc}")
            return False

    def _daemon_media_available(self) -> bool:
        daemon_url = self._daemon_url()
        if not daemon_url:
            return False
        try:
            with urllib.request.urlopen(f"{daemon_url}/api/media/status", timeout=1.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, ValueError):
            return True
        if payload.get("released"):
            self._post_daemon_json("/api/media/acquire", {})
            try:
                with urllib.request.urlopen(f"{daemon_url}/api/media/status", timeout=1.0) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except (OSError, urllib.error.URLError, ValueError):
                return True
        if payload.get("no_media") or not payload.get("available", True):
            print(f"[reachy sound cue] daemon media unavailable: {payload}")
            return False
        self._ensure_daemon_volume()
        return True

    def _ensure_daemon_volume(self) -> None:
        daemon_url = self._daemon_url()
        if not daemon_url:
            return
        try:
            minimum = int(os.environ.get("REACHY_SPEAKER_MIN_VOLUME", "65"))
        except ValueError:
            minimum = 65
        minimum = max(0, min(100, minimum))
        try:
            with urllib.request.urlopen(f"{daemon_url}/api/volume/current", timeout=1.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            volume = int(payload.get("volume", 0))
        except (OSError, urllib.error.URLError, ValueError, TypeError):
            return
        if volume < minimum:
            self._post_daemon_json("/api/volume/set", {"volume": minimum}, timeout=2.0)

    def _play_on_daemon(self, asset_name: str) -> bool:
        daemon_url = self._daemon_url()
        if not daemon_url or not self._daemon_media_available():
            return False
        return self._post_daemon_json("/api/media/play_sound", {"file": asset_name})

    def _upload_daemon_sound(self, path: Path, filename: str) -> bool:
        daemon_url = self._daemon_url()
        if not daemon_url or not self._daemon_media_available():
            return False
        boundary = f"----FuguReachy{hashlib.sha1(filename.encode()).hexdigest()[:12]}"
        payload = path.read_bytes()
        body = b"".join(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
                b"Content-Type: audio/wav\r\n\r\n",
                payload,
                f"\r\n--{boundary}--\r\n".encode("utf-8"),
            ]
        )
        req = urllib.request.Request(
            f"{daemon_url}/api/media/sounds/upload",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body)),
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0):
                return True
        except (OSError, urllib.error.URLError) as exc:
            print(f"[reachy speech cue] daemon upload failed: {exc}")
            return False

    def _speech_wav_path(self, text: str) -> Path | None:
        say = shutil.which("say")
        afconvert = shutil.which("afconvert")
        if not say or not afconvert:
            return None
        voice = os.environ.get("REACHY_TTS_VOICE", "Eddy (English (US))")
        rate = os.environ.get("REACHY_TTS_RATE", "230")
        digest = hashlib.sha1(f"{voice}:{rate}:{text}".encode("utf-8")).hexdigest()[:16]
        tmp_dir = Path(tempfile.gettempdir()) / "fugu_reachy_speech"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        aiff_path = tmp_dir / f"reachy_speech_{digest}.aiff"
        wav_path = tmp_dir / f"reachy_speech_{digest}.wav"
        if wav_path.exists():
            return wav_path
        try:
            subprocess.run(
                [say, "-v", voice, "-r", rate, "-o", str(aiff_path), text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [afconvert, "-f", "WAVE", "-d", "LEI16@22050", str(aiff_path), str(wav_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return wav_path
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[reachy speech cue] speech wav generation failed: {exc}")
            return None

    def _play_speech_on_daemon(self, text: str) -> bool:
        wav_path = self._speech_wav_path(text)
        if wav_path is None:
            return False
        if not self._upload_daemon_sound(wav_path, wav_path.name):
            return False
        if self._play_on_daemon(wav_path.name):
            print(f"[reachy speech cue] {wav_path.name} via daemon")
            return True
        return False

    def _play_local_speech(self, text: str) -> bool:
        if os.environ.get("REACHY_LOCAL_SPEECH_FALLBACK", "1").lower() in {"0", "false", "no"}:
            return False
        say = shutil.which("say")
        if not say:
            return False
        voice = os.environ.get("REACHY_TTS_VOICE", "Eddy (English (US))")
        rate = os.environ.get("REACHY_TTS_RATE", "230")
        try:
            subprocess.Popen(
                [say, "-v", voice, "-r", rate, text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError as exc:
            print(f"[reachy speech cue] local speech failed: {exc}")
            return False

    def _play_local_fallback(self, emotion: str) -> bool:
        if os.environ.get("REACHY_LOCAL_SOUND_FALLBACK", "1").lower() in {"0", "false", "no"}:
            return False
        path = bundled_sound_path(emotion) or synthetic_sound_path(emotion)
        player = shutil.which("afplay") or shutil.which("aplay")
        if not player:
            print(f"[reachy sound cue] no local audio player found for {path}")
            return False
        try:
            subprocess.Popen(
                [player, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except OSError as exc:
            print(f"[reachy sound cue] local playback failed: {exc}")
            return False

    def sound(self, emotion: str = "neutral") -> None:
        normalized = normalize_sound_emotion(emotion)
        asset_name = sound_asset_name(normalized)
        if self._play_on_daemon(asset_name):
            print(f"[reachy sound cue] {normalized} ({asset_name})")
            return
        media = getattr(self._mini, "media", None) if self._mini is not None else None
        audio = getattr(media, "audio", None) if media is not None else None
        if media is not None and audio is not None:
            try:
                media.play_sound(str(bundled_sound_path(normalized) or synthetic_sound_path(normalized)))
                print(f"[reachy sound cue] {normalized} via SDK media")
                return
            except Exception as exc:  # noqa: BLE001 - sound should not cancel movement playback
                print(f"[reachy sound cue] SDK media playback failed: {exc}")
        if self._play_local_fallback(normalized):
            print(f"[reachy sound cue] {normalized} via local fallback")
        else:
            print(f"[reachy sound cue] {normalized} skipped")

    def close(self) -> None:
        if self._mini is not None:
            try:
                try:
                    self._mini.goto_target(
                        head=self._create_head_pose(degrees=True, mm=False),
                        antennas=[math.radians(REST_ANTENNAS[0]), math.radians(REST_ANTENNAS[1])],
                        body_yaw=0.0,
                        duration=0.8,
                        method="minjerk",
                    )
                except TimeoutError as exc:
                    print(f"[reachy motion cue] idle timeout during close: {exc}")
            finally:
                try:
                    self._mini.__exit__(None, None, None)
                finally:
                    self._post_daemon_json("/api/media/acquire", {}, timeout=3.0)
                    self._mini = None


def create_backend(
    kind: str = "auto",
    connection_mode: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> Backend:
    if kind == "dry-run":
        return DryRunBackend()
    if kind == "sdk":
        return SDKBackend(connection_mode=connection_mode, host=host, port=port).__enter__()
    if kind != "auto":
        raise ValueError(f"unknown backend {kind!r}")
    try:
        return SDKBackend(connection_mode=connection_mode, host=host, port=port).__enter__()
    except Exception as exc:  # noqa: BLE001 - auto fallback should be forgiving
        print(f"[auto] falling back to dry-run backend: {exc}")
        return DryRunBackend()


class ReachyPrimitives:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.pose = Pose()

    def _move(self, pose: Pose, duration: float = 0.7, pause: float = 0.05) -> None:
        self.pose = pose.clamped()
        self.backend.goto(self.pose, duration=duration)
        if pause:
            time.sleep(pause)

    def idle(self) -> None:
        self._move(Pose(), duration=0.8)

    def nod(self) -> None:
        base = self.pose
        self._move(Pose(roll=base.roll, pitch=12, yaw=base.yaw, body_yaw=base.body_yaw), 0.35)
        self._move(Pose(roll=base.roll, pitch=-6, yaw=base.yaw, body_yaw=base.body_yaw), 0.35)
        self._move(Pose(roll=base.roll, pitch=0, yaw=base.yaw, body_yaw=base.body_yaw), 0.3)

    def look_to_worker(self, index: int) -> None:
        yaw = WORKER_YAWS[index]
        self._move(Pose(yaw=yaw, body_yaw=yaw * 0.35), duration=0.55)

    def antenna_pulse(self, duration: float = 0.5) -> None:
        duration = clamp(float(duration), 0.1, 2.0)
        base = self.pose
        self._move(
            Pose(
                roll=base.roll,
                pitch=base.pitch,
                yaw=base.yaw,
                body_yaw=base.body_yaw,
                right_antenna=-45,
                left_antenna=45,
            ),
            duration=duration / 2,
        )
        self._move(
            Pose(
                roll=base.roll,
                pitch=base.pitch,
                yaw=base.yaw,
                body_yaw=base.body_yaw,
                right_antenna=REST_ANTENNAS[0],
                left_antenna=REST_ANTENNAS[1],
            ),
            duration=duration / 2,
        )

    def curious_tilt(self) -> None:
        self._move(Pose(roll=15, pitch=-7, yaw=-12, body_yaw=-4), duration=0.6)

    def celebration(self) -> None:
        self.antenna_pulse(0.4)
        self._move(Pose(roll=-12, pitch=-4, yaw=24, body_yaw=12, right_antenna=-35, left_antenna=35), 0.45)
        self._move(Pose(roll=12, pitch=-4, yaw=-24, body_yaw=-12, right_antenna=-35, left_antenna=35), 0.45)
        self.nod()
        self.idle()

    def bow(self) -> None:
        self._move(Pose(pitch=22, right_antenna=-18, left_antenna=18), duration=0.7)
        self._move(Pose(), duration=0.7)

    def speak(self, text: str) -> None:
        self.backend.speak(text[:160])

    def sound(self, emotion: str = "neutral") -> None:
        self.backend.sound(normalize_sound_emotion(emotion))

    def run(self, primitive: str, args: dict[str, Any]) -> None:
        method = getattr(self, primitive)
        method(**args)

    def close(self) -> None:
        self.backend.close()
