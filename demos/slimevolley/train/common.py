"""Shared helpers for the Slime Volley reproduction tool: the model registry, how to invoke each CLI
agent, and where a run lives. Imported by train.py, assess.py, and preflight.py."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TIMEOUT = 1800  # seconds for one agent turn


@dataclass(frozen=True)
class Model:
    """One wired model: which CLI runs it, the model id, and an optional codex profile."""

    cli: str  # "claude" | "gemini" | "codex"
    model: str
    profile: str | None = None


MODELS: dict[str, Model] = {
    "claude": Model("claude", "claude-opus-4-8"),
    "gemini": Model("gemini", "gemini-3.1-pro-preview"),
    "fugu": Model("codex", "fugu-ultra", profile="fugu"),
}


def run_directory(profile: str, override: str | None = None) -> str:
    """Where a profile's run lives. Default: ./run/<profile> next to these scripts."""
    return override or os.path.join(HERE, "run", profile)


def cli_command(model: Model) -> list[str]:
    """The argv that runs one agent turn for `model`, reading its prompt from stdin."""
    if model.cli == "claude":
        return [
            "claude",
            "-p",
            "--dangerously-skip-permissions",
            "--disallowedTools",
            "WebSearch WebFetch",
            "--effort",
            "max",
            "--model",
            model.model,
        ]
    if model.cli == "gemini":
        return ["gemini", "--skip-trust", "--approval-mode", "yolo", "-m", model.model, "-p", "-"]
    if model.cli == "codex":
        command = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "workspace-write"]
        if model.profile:
            command += ["-p", model.profile]
        return command + ["-c", "model_reasoning_effort=high", "-m", model.model, "-"]
    raise ValueError(f"unknown cli: {model.cli!r}")


def run_agent(model: Model, prompt: str, cwd: str, timeout: int = DEFAULT_TIMEOUT) -> int:
    """Run one agent turn in `cwd`, piping `prompt` to stdin and the transcript to transcript.txt.

    Returns the exit code; kills the agent if it runs past `timeout`.
    """
    if model.cli == "gemini":
        _write_gemini_settings(cwd, model.model)
    with open(os.path.join(cwd, "transcript.txt"), "w", encoding="utf-8") as transcript:
        agent = subprocess.Popen(
            cli_command(model),
            cwd=cwd,
            env=_agent_environment(model.cli),
            stdin=subprocess.PIPE,
            stdout=transcript,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            agent.communicate(prompt, timeout=timeout)
        except subprocess.TimeoutExpired:
            agent.kill()
            agent.communicate()
    return agent.returncode


def _agent_environment(cli: str) -> dict[str, str]:
    """Child env for the agent. Claude uses its logged-in CLI OAuth, so we drop any API-key vars."""
    env = dict(os.environ)
    if cli == "claude":
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            env.pop(key, None)
    return env


def _write_gemini_settings(cwd: str, model: str) -> None:
    """Ask Gemini for its full thinking budget, scoped to this run dir only."""
    settings = {
        "modelConfigs": {
            "overrides": [
                {
                    "match": {"model": model},
                    "modelConfig": {"generateContentConfig": {"thinkingConfig": {"thinkingBudget": -1}}},
                }
            ]
        }
    }
    gemini_dir = os.path.join(cwd, ".gemini")
    os.makedirs(gemini_dir, exist_ok=True)
    with open(os.path.join(gemini_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump(settings, handle)
