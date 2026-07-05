from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    pass


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    sakana_api_key: str | None
    sakana_base_url: str = "https://api.sakana.ai/v1"
    fugu_api_mode: str = "chat_completions"
    fugu_model: str = "fugu"
    fugu_dry_run: bool = False
    reasoning_effort: str = "high"
    max_output_tokens: int = 512
    reachy_daemon_url: str = "http://127.0.0.1:8000"
    reachy_hardware_enabled: bool = False
    reachy_status_cache_seconds: int = 2
    cache_dir: Path = ROOT / ".cache"

    def require_api_key(self) -> str:
        if not self.sakana_api_key:
            raise ConfigError("SAKANA_API_KEY is required for live Fugu generation")
        return self.sakana_api_key


def load_settings() -> Settings:
    model = os.environ.get("FUGU_MODEL") or os.environ.get("FUGU_EXPRESSION_MODEL") or "fugu"
    api_mode = os.environ.get("FUGU_API_MODE", "chat_completions").strip().lower()
    if api_mode not in {"chat_completions", "responses"}:
        raise ConfigError("FUGU_API_MODE must be one of: chat_completions, responses")
    effort = os.environ.get("FUGU_REASONING_EFFORT", "high").strip().lower()
    if effort not in {"high", "xhigh", "max"}:
        raise ConfigError("FUGU_REASONING_EFFORT must be one of: high, xhigh, max")
    return Settings(
        sakana_api_key=os.environ.get("SAKANA_API_KEY"),
        sakana_base_url=os.environ.get("SAKANA_BASE_URL", "https://api.sakana.ai/v1").rstrip("/"),
        fugu_api_mode=api_mode,
        fugu_model=model.strip() or "fugu",
        fugu_dry_run=_bool_env("FUGU_DRY_RUN", False),
        reasoning_effort=effort,
        max_output_tokens=_int_env("FUGU_MAX_OUTPUT_TOKENS", 512, minimum=64),
        reachy_daemon_url=os.environ.get("REACHY_DAEMON_URL", "http://127.0.0.1:8000").rstrip("/"),
        reachy_hardware_enabled=_bool_env("REACHY_HARDWARE_ENABLED", False),
        reachy_status_cache_seconds=_int_env("REACHY_STATUS_CACHE_SECONDS", 2, minimum=1),
    )
