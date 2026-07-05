#!/usr/bin/env python3
"""Local browser dashboard for AxiDraw demo preview and guarded plotting."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
APP_STATIC = ROOT / "app" / "static"
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
SUMMARY = OUTPUTS / "simulation_summary.json"
GENERATED_SUMMARY = OUTPUTS / "generated_summary.json"
DISPLAY_MODEL_NAME = "Sakana Fugu"
MAX_CAMERA_DATA_URL_BYTES = 2_000_000
DEFAULT_SAKANA_BASE_URL = "https://api.sakana.ai/v1"
DEFAULT_ALLOWED_BASE_URLS = {
    DEFAULT_SAKANA_BASE_URL,
}
SUPPORTED_MODELS = {"fugu", "fugu-ultra"}
LOCAL_API_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_ORIGIN_RE = re.compile(r"^https?://(127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv(ROOT / ".env")


class JobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.kind = ""
        self.demo = ""
        self.started_at = 0.0
        self.finished_at = 0.0
        self.returncode: int | None = None
        self.output = ""

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            if self.process is not None and not running and self.returncode is None:
                self.returncode = self.process.returncode
                self.finished_at = time.time()
            return {
                "running": running,
                "kind": self.kind,
                "demo": self.demo,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "output": self.output[-12000:],
            }


JOB = JobState()


def pyaxidraw_available() -> bool:
    try:
        import pyaxidraw  # noqa: F401

        return True
    except Exception:
        return False


def python_path() -> str:
    return sys.executable


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_sakana_base_url(value: str | None = None) -> str:
    raw_value = (value or os.environ.get("SAKANA_BASE_URL") or DEFAULT_SAKANA_BASE_URL).strip()
    if not raw_value:
        raw_value = DEFAULT_SAKANA_BASE_URL
    candidate = raw_value.rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API base URL must be an http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("API base URL must not include username or password credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("API base URL must not include query strings or fragments")
    if parsed.scheme != "https" and parsed.hostname not in LOCAL_API_HOSTS:
        raise ValueError("API base URL must use https unless it is localhost")
    return candidate


def configured_sakana_base_url() -> str:
    return normalize_sakana_base_url(os.environ.get("SAKANA_BASE_URL", DEFAULT_SAKANA_BASE_URL))


def custom_base_urls_allowed() -> bool:
    return env_flag("SAKANA_ALLOW_CUSTOM_BASE_URLS")


def allowed_base_urls() -> list[str]:
    values = set(DEFAULT_ALLOWED_BASE_URLS)
    raw_extra = os.environ.get("SAKANA_ALLOWED_BASE_URLS", "")
    for item in raw_extra.split(","):
        item = item.strip()
        if item:
            values.add(normalize_sakana_base_url(item))
    values.add(configured_sakana_base_url())
    return sorted(values)


def validate_sakana_base_url(value: str | None = None) -> str:
    base_url = normalize_sakana_base_url(value)
    hostname = urlparse(base_url).hostname
    if hostname in LOCAL_API_HOSTS:
        return base_url
    if custom_base_urls_allowed() or base_url in allowed_base_urls():
        return base_url
    allowed = ", ".join(allowed_base_urls())
    raise ValueError(
        "API base URL is not allowed for server-key requests. "
        f"Allowed: {allowed}. Set SAKANA_ALLOWED_BASE_URLS or "
        "SAKANA_ALLOW_CUSTOM_BASE_URLS=true to test another host."
    )


def server_key_allowed_for_base_url(base_url: str) -> bool:
    return base_url in allowed_base_urls()


def fugu_settings(base_url: str | None = None) -> dict[str, str]:
    selected_base_url = normalize_sakana_base_url(base_url) if base_url else configured_sakana_base_url()
    return {
        "base_url": selected_base_url,
        "max_output_tokens": os.environ.get("FUGU_MAX_OUTPUT_TOKENS", "1536"),
        "stream": os.environ.get("FUGU_STREAM", "false"),
        "reasoning_effort": os.environ.get("FUGU_REASONING_EFFORT", ""),
        "store": os.environ.get("FUGU_STORE", ""),
    }


def test_sakana_models(base_url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body) if body else {}
            models = []
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                models = [
                    str(item.get("id"))
                    for item in data["data"]
                    if isinstance(item, dict) and item.get("id")
                ]
            return {"ok": True, "base_url": base_url, "models": models, "status": response.status}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API test failed with HTTP {exc.code}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API test could not reach {base_url}: {exc.reason}") from exc


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _runtime_output_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    marker = "/outputs/"
    if marker in value:
        return f"/outputs/{value.split(marker, 1)[1]}"
    if value.startswith("outputs/"):
        return f"/{value}"
    return value


def _normalize_demo_paths(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    for key in ("svg", "preview_svg", "report_json"):
        normalized[key] = _runtime_output_path(normalized.get(key))
    return normalized


def load_demos() -> list[dict[str, Any]]:
    return [_normalize_demo_paths(item) for item in [*_read_summary(SUMMARY), *_read_summary(GENERATED_SUMMARY)]]


def demo_by_slug(slug: str) -> dict[str, Any]:
    for item in load_demos():
        if item.get("demo") == slug:
            return item
    raise KeyError(slug)


def output_svg_for_demo(slug: str) -> Path:
    item = demo_by_slug(slug)
    raw_svg = str(item.get("svg") or "")
    if raw_svg.startswith("/outputs/"):
        svg = (OUTPUTS / raw_svg.removeprefix("/outputs/")).resolve()
    else:
        svg = Path(raw_svg)
    if not svg.is_absolute():
        svg = (ROOT / svg).resolve()
    else:
        svg = svg.resolve()
    if not svg.exists():
        raise FileNotFoundError(svg)
    if not (svg == OUTPUTS.resolve() or OUTPUTS.resolve() in svg.parents):
        raise ValueError("plot SVG must be inside outputs/")
    if svg.name.endswith(".axidraw_preview.svg"):
        raise ValueError("plot the generated SVG, not the preview SVG")
    return svg


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length") or "0")
    if not length:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def send_local_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    origin = handler.headers.get("Origin", "")
    if origin and LOCAL_ORIGIN_RE.match(origin):
        handler.send_header("access-control-allow-origin", origin)
        handler.send_header("vary", "Origin")


def write_json(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json; charset=utf-8")
    send_local_cors_headers(handler)
    handler.send_header("cache-control", "no-store")
    handler.send_header("content-length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def cache_control_for(resolved: Path) -> str:
    try:
        if resolved == APP_STATIC.resolve() / "index.html":
            return "no-cache"
        if resolved.is_relative_to(APP_STATIC.resolve()):
            return "public, max-age=31536000, immutable"
        if resolved.is_relative_to(OUTPUTS.resolve()):
            if resolved.suffix == ".svg" and re.search(r"_20\d{6}_\d{6}", resolved.stem):
                return "public, max-age=86400, immutable"
            return "no-cache"
    except ValueError:
        pass
    return "no-store"


def write_camera_data_url(value: str) -> Path:
    data_url = value.strip()
    if not data_url:
        raise ValueError("camera image is empty")
    if not data_url.startswith("data:image/"):
        raise ValueError("camera image must be a data:image/... URL")
    if len(data_url.encode("utf-8")) > MAX_CAMERA_DATA_URL_BYTES:
        raise ValueError("camera image is too large; use a lower capture resolution")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="camera_plot_frame_",
        suffix=".txt",
        dir=OUTPUTS,
        delete=False,
    )
    with handle:
        handle.write(data_url)
    return Path(handle.name)


def cleanup_files(paths: list[Path] | None) -> None:
    for cleanup_path in paths or []:
        try:
            cleanup_path.unlink(missing_ok=True)
        except OSError:
            pass


def start_job(
    kind: str,
    demo: str,
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    cleanup_paths: list[Path] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    with JOB.lock:
        if JOB.process is not None and JOB.process.poll() is None:
            raise RuntimeError(f"{JOB.kind} job is already running")
        JOB.kind = kind
        JOB.demo = demo
        JOB.started_at = time.time()
        JOB.finished_at = 0.0
        JOB.returncode = None
        note_text = "".join(f"# {note}\n" for note in notes or [])
        JOB.output = f"$ {' '.join(cmd)}\n{note_text}"
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        JOB.process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process = JOB.process

    def collect() -> None:
        assert process.stdout is not None
        parts: list[str] = []
        for line in process.stdout:
            parts.append(line)
            with JOB.lock:
                JOB.output += line
        returncode = process.wait()
        cleanup_files(cleanup_paths)
        with JOB.lock:
            JOB.returncode = returncode
            JOB.finished_at = time.time()
            JOB.output += f"\n[exit {returncode}]\n"

    threading.Thread(target=collect, daemon=True).start()
    return JOB.snapshot()


def start_job_or_cleanup(*args: Any, cleanup_paths: list[Path] | None = None, **kwargs: Any) -> dict[str, Any]:
    try:
        return start_job(*args, cleanup_paths=cleanup_paths, **kwargs)
    except Exception:
        cleanup_files(cleanup_paths)
        raise


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return None

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin and not LOCAL_ORIGIN_RE.match(origin):
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(204)
        send_local_cors_headers(self)
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.end_headers()

    def do_GET(self) -> None:
        try:
            self.handle_get()
        except Exception as exc:  # noqa: BLE001 - local dashboard should surface exact issue
            write_json(self, {"error": str(exc)}, status=500)

    def handle_get(self) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self.serve_file(APP_STATIC / "index.html")
        elif path in {"/app.js", "/styles.css"}:
            self.serve_file(APP_STATIC / path.removeprefix("/"))
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("cache-control", "max-age=86400")
            self.end_headers()
        elif path.startswith("/assets/"):
            self.serve_file(APP_STATIC / path.removeprefix("/"))
        elif path == "/api/demos":
            base_url = configured_sakana_base_url()
            write_json(
                self,
                {
                    "axicli": "axicli",
                    "plotter_enabled": pyaxidraw_available(),
                    "fugu_enabled": bool(os.environ.get("SAKANA_API_KEY")),
                    "base_url": base_url,
                    "allowed_base_urls": allowed_base_urls(),
                    "custom_base_url_allowed": custom_base_urls_allowed(),
                    "fugu_settings": fugu_settings(base_url),
                    "display_model": DISPLAY_MODEL_NAME,
                    "demos": load_demos(),
                    "job": JOB.snapshot(),
                },
            )
        elif path == "/api/health":
            base_url = configured_sakana_base_url()
            write_json(
                self,
                {
                    "ok": True,
                    "base_url": base_url,
                    "allowed_base_urls": allowed_base_urls(),
                    "custom_base_url_allowed": custom_base_urls_allowed(),
                    "plotter_enabled": pyaxidraw_available(),
                    "fugu_enabled": bool(os.environ.get("SAKANA_API_KEY")),
                    "fugu_settings": fugu_settings(base_url),
                    "local_vectorizer": "opencv-contours",
                    "models": sorted(SUPPORTED_MODELS),
                },
            )
        elif path == "/api/job":
            write_json(self, JOB.snapshot())
        elif path.startswith("/outputs/"):
            self.serve_file(OUTPUTS / path.removeprefix("/outputs/"))
        elif path.startswith("/static/"):
            self.serve_file(APP_STATIC / path.removeprefix("/static/"))
        else:
            write_json(self, {"error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            body = read_json_body(self)
            if self.path == "/api/preview":
                demo = str(body.get("demo") or "")
                svg = output_svg_for_demo(demo)
                cmd = [
                    python_path(),
                    str(SCRIPTS / "simulate_axidraw_preview.py"),
                    str(svg),
                ]
                snapshot = start_job("preview", demo, cmd)
                write_json(self, snapshot, status=202)
            elif self.path == "/api/test-api":
                requested_base_url = str(body.get("base_url") or "").strip()
                browser_api_key = str(body.get("api_key") or "").strip()
                server_api_key = os.environ.get("SAKANA_API_KEY", "").strip()
                try:
                    base_url = validate_sakana_base_url(requested_base_url or None)
                except ValueError as exc:
                    write_json(self, {"error": str(exc)}, status=400)
                    return
                server_key_eligible = bool(server_api_key and server_key_allowed_for_base_url(base_url))
                api_key = server_api_key if server_key_eligible else browser_api_key
                key_source = "server" if server_key_eligible else "browser"
                if not api_key:
                    if server_api_key:
                        write_json(
                            self,
                            {
                                "error": (
                                    "Browser API key is required for this API base URL; "
                                    "the server key is only sent to allowlisted hosts."
                                ),
                                "base_url": base_url,
                            },
                            status=400,
                        )
                        return
                    write_json(self, {"error": "API key is required to test this endpoint"}, status=400)
                    return
                try:
                    result = test_sakana_models(base_url, api_key)
                except RuntimeError as exc:
                    write_json(self, {"error": str(exc), "base_url": base_url}, status=502)
                    return
                result["key_source"] = key_source
                write_json(self, result)
            elif self.path == "/api/generate":
                prompt = str(body.get("prompt") or "").strip()
                model = str(body.get("model") or "fugu").strip()
                requested_base_url = str(body.get("base_url") or "").strip()
                browser_api_key = str(body.get("api_key") or "").strip()
                server_api_key = os.environ.get("SAKANA_API_KEY", "").strip()
                if model not in SUPPORTED_MODELS:
                    write_json(self, {"error": "model must be fugu or fugu-ultra"}, status=400)
                    return
                if len(prompt) < 8:
                    write_json(self, {"error": "prompt is too short"}, status=400)
                    return
                try:
                    base_url = validate_sakana_base_url(requested_base_url or None)
                except ValueError as exc:
                    write_json(self, {"error": str(exc)}, status=400)
                    return
                cmd = [
                    python_path(),
                    str(SCRIPTS / "generate_prompt_demo.py"),
                    "--model",
                    model,
                    "--request",
                    prompt,
                ]
                fugu_timeout = os.environ.get("FUGU_API_TIMEOUT_SECONDS", "45")
                fugu_max_output_tokens = os.environ.get("FUGU_MAX_OUTPUT_TOKENS", "1536")
                extra_env = {
                    "FUGU_API_TIMEOUT_SECONDS": fugu_timeout,
                    "FUGU_MAX_OUTPUT_TOKENS": fugu_max_output_tokens,
                    "FUGU_STREAM": os.environ.get("FUGU_STREAM", "false"),
                    "SAKANA_BASE_URL": base_url,
                }
                if os.environ.get("FUGU_REASONING_EFFORT", "").strip():
                    extra_env["FUGU_REASONING_EFFORT"] = os.environ["FUGU_REASONING_EFFORT"].strip()
                if "FUGU_STORE" in os.environ:
                    extra_env["FUGU_STORE"] = os.environ["FUGU_STORE"]
                server_key_eligible = bool(server_api_key and server_key_allowed_for_base_url(base_url))
                if server_key_eligible:
                    extra_env["SAKANA_API_KEY"] = server_api_key
                    key_source = "server API key provided"
                elif browser_api_key:
                    extra_env["SAKANA_API_KEY"] = browser_api_key
                    key_source = "browser API key provided"
                elif server_api_key:
                    write_json(
                        self,
                        {
                            "error": (
                                "Browser API key is required for this API base URL; "
                                "the server key is only sent to allowlisted hosts."
                            )
                        },
                        status=400,
                    )
                    return
                else:
                    key_source = "no API key provided; using fallback preview"
                snapshot = start_job_or_cleanup(
                    "generate",
                    DISPLAY_MODEL_NAME,
                    cmd,
                    extra_env=extra_env,
                    notes=[
                        key_source,
                        "Sakana API endpoint configured.",
                        "Input: text prompt",
                        (
                            "Fugu settings: "
                            f"stream={extra_env['FUGU_STREAM']}, "
                            f"max_output_tokens={extra_env['FUGU_MAX_OUTPUT_TOKENS']}, "
                            f"timeout={extra_env['FUGU_API_TIMEOUT_SECONDS']}s, "
                            f"reasoning={extra_env.get('FUGU_REASONING_EFFORT', 'none') or 'none'}"
                        ),
                    ],
                )
                write_json(self, snapshot, status=202)
            elif self.path == "/api/vectorize-camera":
                image_data_url = str(body.get("image_data_url") or "").strip()
                if not image_data_url:
                    write_json(self, {"error": "image_data_url is required"}, status=400)
                    return
                cleanup_requested = bool(body.get("cleanup"))
                image_path = write_camera_data_url(image_data_url)
                if cleanup_requested:
                    model = str(body.get("model") or "fugu").strip()
                    requested_base_url = str(body.get("base_url") or "").strip()
                    browser_api_key = str(body.get("api_key") or "").strip()
                    server_api_key = os.environ.get("SAKANA_API_KEY", "").strip()
                    prompt = str(body.get("prompt") or "").strip() or (
                        "Clean up the fast webcam contour plan while preserving the real subject geometry."
                    )
                    if model not in SUPPORTED_MODELS:
                        write_json(self, {"error": "model must be fugu or fugu-ultra"}, status=400)
                        return
                    try:
                        base_url = validate_sakana_base_url(requested_base_url or None)
                    except ValueError as exc:
                        write_json(self, {"error": str(exc)}, status=400)
                        return
                    cleanup_timeout = os.environ.get(
                        "FUGU_CLEANUP_API_TIMEOUT_SECONDS",
                        os.environ.get("FUGU_API_TIMEOUT_SECONDS", "90"),
                    )
                    cleanup_max_output_tokens = os.environ.get(
                        "FUGU_CLEANUP_MAX_OUTPUT_TOKENS",
                        "1024",
                    )
                    cleanup_stream = os.environ.get(
                        "FUGU_CLEANUP_STREAM",
                        os.environ.get("FUGU_STREAM", "false"),
                    )
                    extra_env = {
                        "FUGU_API_TIMEOUT_SECONDS": cleanup_timeout,
                        "FUGU_MAX_OUTPUT_TOKENS": cleanup_max_output_tokens,
                        "FUGU_STREAM": cleanup_stream,
                        "SAKANA_BASE_URL": base_url,
                    }
                    if os.environ.get("FUGU_REASONING_EFFORT", "").strip():
                        extra_env["FUGU_REASONING_EFFORT"] = os.environ["FUGU_REASONING_EFFORT"].strip()
                    if "FUGU_STORE" in os.environ:
                        extra_env["FUGU_STORE"] = os.environ["FUGU_STORE"]
                    server_key_eligible = bool(server_api_key and server_key_allowed_for_base_url(base_url))
                    if server_key_eligible:
                        extra_env["SAKANA_API_KEY"] = server_api_key
                        key_source = "server API key provided"
                    elif browser_api_key:
                        extra_env["SAKANA_API_KEY"] = browser_api_key
                        key_source = "browser API key provided"
                    elif server_api_key:
                        write_json(
                            self,
                            {
                                "error": (
                                    "Browser API key is required for this API base URL; "
                                    "the server key is only sent to allowlisted hosts."
                                )
                            },
                            status=400,
                        )
                        return
                    else:
                        key_source = "no API key provided; using fast contour fallback"
                    snapshot = start_job_or_cleanup(
                        "cleanup",
                        DISPLAY_MODEL_NAME,
                        [
                            python_path(),
                            str(SCRIPTS / "cleanup_contour_with_fugu.py"),
                            "--model",
                            model,
                            "--request",
                            prompt,
                            "--image-data-file",
                            str(image_path),
                        ],
                        extra_env=extra_env,
                        cleanup_paths=[image_path],
                        notes=[
                            key_source,
                            "Sakana API endpoint configured.",
                            "Input: fast webcam contour plan",
                            (
                                "Fugu cleanup settings: "
                                f"stream={extra_env['FUGU_STREAM']}, "
                                f"max_output_tokens={extra_env['FUGU_MAX_OUTPUT_TOKENS']}, "
                                f"timeout={extra_env['FUGU_API_TIMEOUT_SECONDS']}s, "
                                f"reasoning={extra_env.get('FUGU_REASONING_EFFORT', 'none') or 'none'}"
                            ),
                        ],
                    )
                    write_json(self, snapshot, status=202)
                    return
                snapshot = start_job_or_cleanup(
                    "vectorize",
                    "webcam contour",
                    [
                        python_path(),
                        str(SCRIPTS / "vectorize_camera_frame.py"),
                        "--image-data-file",
                        str(image_path),
                    ],
                    cleanup_paths=[image_path],
                    notes=[
                        "OpenCV contour vectorizer",
                        "No Fugu API call is used for this webcam mode",
                    ],
                )
                write_json(self, snapshot, status=202)
            elif self.path == "/api/plot":
                demo = str(body.get("demo") or "").strip()
                if not demo:
                    write_json(self, {"error": "demo is required"}, status=400)
                    return
                svg = output_svg_for_demo(demo)
                speed = max(1, min(100, int(body.get("speed") or os.environ.get("AXIDRAW_SPEED", "50"))))
                dry_run = bool(body.get("dry_run")) or env_flag("AXIDRAW_DRY_RUN")
                cmd = [
                    python_path(),
                    str(SCRIPTS / "plot_axidraw_svg.py"),
                    str(svg),
                    "--speed",
                    str(speed),
                ]
                if dry_run:
                    cmd.append("--dry-run")
                snapshot = start_job(
                    "plot",
                    demo,
                    cmd,
                    notes=[
                        "AxiDraw hardware plot command",
                        "Uses guarded pyaxidraw plot mode settings",
                        "Dry run: " + ("yes" if dry_run else "no"),
                    ],
                )
                write_json(self, snapshot, status=202)
            elif self.path == "/api/stop":
                with JOB.lock:
                    process = JOB.process
                    if process is None or process.poll() is not None:
                        write_json(self, JOB.snapshot())
                        return
                    os.killpg(process.pid, signal.SIGTERM)
                    JOB.output += "\n[terminate requested]\n"
                write_json(self, JOB.snapshot())
            else:
                write_json(self, {"error": "not found"}, status=404)
        except Exception as exc:  # noqa: BLE001 - local dashboard should surface exact issue
            write_json(self, {"error": str(exc)}, status=500)

    def serve_file(self, path: Path) -> None:
        try:
            resolved = path.resolve()
            allowed = (APP_STATIC.resolve(), OUTPUTS.resolve())
            if not any(resolved == root or root in resolved.parents for root in allowed):
                raise FileNotFoundError(path)
            if resolved.parent == OUTPUTS.resolve() and resolved.name.startswith("camera_plot_frame_"):
                raise FileNotFoundError(path)
            if not resolved.is_file():
                raise FileNotFoundError(path)
            data = resolved.read_bytes()
            mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
            if resolved.suffix == ".svg":
                mime = "image/svg+xml"
            self.send_response(200)
            self.send_header("content-type", mime)
            send_local_cors_headers(self)
            self.send_header("cache-control", cache_control_for(resolved))
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            write_json(self, {"error": "file not found"}, status=404)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8776")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AxiDraw Fugu demo: http://{args.host}:{args.port}/")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
