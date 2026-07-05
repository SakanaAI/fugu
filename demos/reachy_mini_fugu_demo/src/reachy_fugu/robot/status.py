from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from reachy_fugu.config import Settings


@dataclass(frozen=True)
class RobotStatus:
    connected: bool
    enabled: bool
    message: str
    daemon_url: str
    raw: dict[str, Any]


_CACHE: dict[str, tuple[float, RobotStatus]] = {}


def _daemon_json(base_url: str, path: str, timeout: float = 1.2) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body) if body.strip() else {}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _fallback_connected_status(settings: Settings, error: Exception) -> RobotStatus | None:
    try:
        doa = _daemon_json(settings.reachy_daemon_url, "/api/state/doa", timeout=0.7)
    except (OSError, urllib.error.URLError, ValueError):
        return None
    if "angle" not in doa and "speech_detected" not in doa:
        return None
    return RobotStatus(
        connected=True,
        enabled=True,
        message="Reachy connected",
        daemon_url=settings.reachy_daemon_url,
        raw={"doa": doa, "status_fallback": True, "status_error": str(error)},
    )


def get_robot_status(settings: Settings, *, force: bool = False) -> RobotStatus:
    cache_key = settings.reachy_daemon_url
    cached = _CACHE.get(cache_key)
    now = time.monotonic()
    if cached and not force and now - cached[0] < settings.reachy_status_cache_seconds:
        return cached[1]

    if not settings.reachy_hardware_enabled:
        status = RobotStatus(
            connected=False,
            enabled=False,
            message="Reachy hardware mode is off. Preview locally is available.",
            daemon_url=settings.reachy_daemon_url,
            raw={},
        )
        _CACHE[cache_key] = (now, status)
        return status

    try:
        daemon = _daemon_json(settings.reachy_daemon_url, "/api/daemon/status")
        media = _daemon_json(settings.reachy_daemon_url, "/api/media/status")
        try:
            motors = _daemon_json(settings.reachy_daemon_url, "/api/motors/status")
        except (OSError, urllib.error.URLError, ValueError) as exc:
            motors = {"error": str(exc)}
    except (OSError, urllib.error.URLError, ValueError) as exc:
        fallback = _fallback_connected_status(settings, exc)
        if fallback is not None:
            _CACHE[cache_key] = (now, fallback)
            return fallback
        status = RobotStatus(
            connected=False,
            enabled=True,
            message="Reachy is not connected. Open Reachy Mini Control, connect the robot, then reconnect.",
            daemon_url=settings.reachy_daemon_url,
            raw={"error": str(exc)},
        )
        _CACHE[cache_key] = (now, status)
        return status

    connected = daemon.get("state") == "running" and not daemon.get("error")
    message = "Reachy connected" if connected else "Reachy is not connected. Open Reachy Mini Control, connect the robot, then reconnect."
    status = RobotStatus(
        connected=connected,
        enabled=True,
        message=message,
        daemon_url=settings.reachy_daemon_url,
        raw={"daemon": daemon, "media": media, "motors": motors},
    )
    _CACHE[cache_key] = (now, status)
    return status
