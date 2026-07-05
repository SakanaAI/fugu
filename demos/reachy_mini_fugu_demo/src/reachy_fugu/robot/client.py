from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from reachy_fugu.config import Settings
from reachy_fugu.robot.status import get_robot_status
from reachy_fugu.schemas import validate_gesture_plan


MOTION_PRIMITIVES = {"idle", "curious_tilt", "nod", "bow", "celebration", "antenna_pulse", "look_to_worker"}
TrackingPoseProvider = Callable[[], dict[str, Any] | None]
StopCallback = Callable[[], None]


def _post_json(settings: Settings, path: str, payload: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{settings.reachy_daemon_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body) if body.strip() else {}


def _get_json(settings: Settings, path: str, timeout: float = 0.7) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{settings.reachy_daemon_url}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body) if body.strip() else {}
    return payload if isinstance(payload, dict) else {}


def _stop_requested(stop_event: Any | None) -> bool:
    return bool(stop_event is not None and stop_event.is_set())


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sdk_backend_kwargs(settings: Settings | None) -> dict[str, Any]:
    if settings is None:
        return {}
    parsed = urlparse(settings.reachy_daemon_url)
    host = os.environ.get("REACHY_SDK_HOST") or parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    mode = os.environ.get("REACHY_SDK_CONNECTION_MODE")
    if not mode:
        mode = "localhost_only" if host in {"127.0.0.1", "localhost", "::1"} else "network"
    return {"host": host, "port": port, "connection_mode": mode}


def _tracking_pose_from_payload(payload: dict[str, Any] | None):
    if not isinstance(payload, dict):
        return None
    from reachy_fugu_demo.reachy_primitives import Pose

    return Pose(
        roll=_number(payload.get("roll")),
        pitch=_number(payload.get("pitch")),
        yaw=_number(payload.get("yaw")),
        body_yaw=_number(payload.get("body_yaw")),
        right_antenna=_number(payload.get("right_antenna"), -10.0),
        left_antenna=_number(payload.get("left_antenna"), 10.0),
    ).clamped()


def _tracking_pose_from_robot_mic(settings: Settings):
    try:
        payload = _get_json(settings, "/api/state/doa")
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if not payload.get("speech_detected"):
        return None
    angle = payload.get("angle")
    if not isinstance(angle, (int, float)) or isinstance(angle, bool):
        return None

    # Reachy daemon reports 0=left, pi/2=front, pi=right. Convert to a
    # bounded yaw/body orientation instead of any uncontrolled translation.
    yaw = max(-34.0, min(34.0, (float(angle) - (math.pi / 2.0)) * 180.0 / math.pi))
    return _tracking_pose_from_payload(
        {
            "yaw": yaw,
            "body_yaw": yaw * 0.36,
            "pitch": -3.0,
            "right_antenna": -26.0,
            "left_antenna": 26.0,
        }
    )


def _upload_sound(settings: Settings, path: Path, filename: str) -> None:
    boundary = f"----ReachyFugu{int(time.time() * 1000)}"
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
    request = urllib.request.Request(
        f"{settings.reachy_daemon_url}/api/media/sounds/upload",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15.0):
        return


def _stop_robot_sound(settings: Settings) -> None:
    try:
        _post_json(settings, "/api/media/stop_sound", {}, timeout=1.0)
    except (OSError, urllib.error.URLError, ValueError):
        return


def _ensure_daemon_media_available(settings: Settings) -> None:
    try:
        status = _get_json(settings, "/api/media/status", timeout=0.7)
    except (OSError, urllib.error.URLError, ValueError):
        status = {}
    if status.get("available") is True and status.get("released") is False:
        return
    _post_json(settings, "/api/media/acquire", {}, timeout=3.0)


def play_on_robot(
    plan: list[dict[str, Any]],
    audio_path: Path,
    settings: Settings,
    *,
    tracking_pose_provider: TrackingPoseProvider | None = None,
    stop_event: Any | None = None,
) -> str:
    validated = validate_gesture_plan(plan)
    status = get_robot_status(settings, force=True)
    if not status.enabled:
        raise RuntimeError("Reachy hardware mode is disabled")
    if not status.connected:
        raise RuntimeError(status.message)
    if _stop_requested(stop_event):
        return "Robot playback canceled before audio started."

    try:
        _ensure_daemon_media_available(settings)
        try:
            _post_json(settings, "/api/motors/set_mode/enabled", {}, timeout=3.0)
        except (OSError, urllib.error.URLError):
            # Some daemon versions do not expose motor mode control. SDK playback
            # below will still fail loudly if the robot cannot move.
            pass
        _upload_sound(settings, audio_path, audio_path.name)
        _post_json(settings, "/api/media/play_sound", {"file": audio_path.name}, timeout=3.0)
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"Reachy media playback failed: {exc}") from exc

    if _stop_requested(stop_event):
        _stop_robot_sound(settings)
        return "Robot playback canceled before motion cues."

    executed = execute_motion_primitives(
        validated,
        settings=settings,
        tracking_pose_provider=tracking_pose_provider,
        stop_event=stop_event,
        on_stop=lambda: _stop_robot_sound(settings),
    )
    if _stop_requested(stop_event):
        _stop_robot_sound(settings)
        return f"Robot playback interrupted. Executed {executed} motion/tracking cues from {len(validated)} validated cues."
    return f"Robot audio playback started. Executed {executed} motion/tracking cues from {len(validated)} validated cues."


def execute_motion_primitives(
    plan: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    tracking_pose_provider: TrackingPoseProvider | None = None,
    stop_event: Any | None = None,
    on_stop: StopCallback | None = None,
) -> int:
    try:
        from reachy_fugu_demo.reachy_primitives import ReachyPrimitives, create_backend
    except Exception as exc:  # noqa: BLE001 - optional local hardware dependency
        raise RuntimeError("Reachy SDK primitive layer is unavailable for motion playback") from exc

    backend = create_backend("sdk", **_sdk_backend_kwargs(settings))
    reachy = ReachyPrimitives(backend)
    executed = 0
    started = time.monotonic()
    last_tracking_at = 0.0
    stop_notified = False

    def notify_stop() -> None:
        nonlocal stop_notified
        if stop_notified:
            return
        stop_notified = True
        if on_stop is not None:
            on_stop()

    def tracking_pose():
        if tracking_pose_provider is not None:
            pose = _tracking_pose_from_payload(tracking_pose_provider())
            if pose is not None:
                return pose
        if settings is not None:
            return _tracking_pose_from_robot_mic(settings)
        return None

    def apply_tracking(force: bool = False) -> bool:
        nonlocal last_tracking_at
        if _stop_requested(stop_event):
            notify_stop()
            return False
        now = time.monotonic()
        if not force and now - last_tracking_at < 0.55:
            return False
        pose = tracking_pose()
        if pose is None:
            return False
        reachy._move(pose, duration=0.35, pause=0.0)
        last_tracking_at = now
        return True

    def wait_until(target_seconds: float) -> bool:
        while True:
            if _stop_requested(stop_event):
                notify_stop()
                return False
            remaining = target_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return True
            apply_tracking()
            time.sleep(min(0.12, remaining))

    def animate_speech() -> None:
        if apply_tracking(force=True):
            reachy.antenna_pulse(0.22)
        else:
            reachy.nod()

    try:
        for step in plan:
            if _stop_requested(stop_event):
                notify_stop()
                break
            primitive = step["primitive"]
            if primitive not in MOTION_PRIMITIVES and primitive not in {"speak", "sound"}:
                continue
            target = step["t_ms"] / 1000.0
            if not wait_until(target):
                break
            if _stop_requested(stop_event):
                notify_stop()
                break
            if primitive == "speak":
                animate_speech()
            elif primitive == "sound":
                reachy.antenna_pulse(0.25)
            else:
                apply_tracking(force=False)
                reachy.run(primitive, step["args"])
            executed += 1
    finally:
        reachy.idle()
        reachy.close()
    return executed
