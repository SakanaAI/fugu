from __future__ import annotations

import html
import inspect
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_local_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()

import gradio as gr

from reachy_fugu.audio.tts import synthesize_speech
from reachy_fugu.config import ConfigError, load_settings
from reachy_fugu.pipeline import UserFacingError, answer_question
from reachy_fugu.robot.client import play_on_robot
from reachy_fugu.robot.inputs import (
    RobotInputUnavailableError,
    RobotInputSession,
    record_robot_question,
    robot_tracking_session,
)
from reachy_fugu.robot.status import get_robot_status


CSS = """
:root {
  --s-red: #d71920;
  --s-ink: #111111;
  --s-muted: #666666;
  --s-line: #e8e8e8;
}
.gradio-container {
  max-width: 1180px !important;
  color: var(--s-ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
#title h1 {
  font-size: 32px;
  line-height: 1.1;
  margin: 0 0 4px;
  letter-spacing: 0;
}
#title p {
  color: var(--s-muted);
  margin-top: 4px;
}
.mode-banner {
  border-left: 4px solid var(--s-red);
  padding: 10px 12px;
  background: #fff8f8;
  border-radius: 4px;
}
button.primary {
  background: var(--s-red) !important;
  border-color: var(--s-red) !important;
}
.robot-input-panel {
  border: 1px solid var(--s-line);
  border-radius: 6px;
  padding: 12px;
}
.robot-input-panel p {
  margin: 0 0 8px;
  color: var(--s-muted);
  font-size: 14px;
}
.hearing-status {
  margin-top: 8px;
  color: var(--s-muted);
  font-size: 14px;
}
.hearing-status.active {
  color: var(--s-ink);
}
.hearing-status.error {
  color: var(--s-red);
}
.turn-state {
  border: 1px solid var(--s-line);
  border-radius: 6px;
  padding: 8px 10px;
  margin: 8px 0;
  color: var(--s-muted);
  background: #fafafa;
}
.turn-state.active {
  color: var(--s-ink);
  background: #f7fbff;
}
.turn-state.error {
  color: var(--s-red);
  background: #fff8f8;
}
"""


def _blocks_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "title": "Reachy Mini x Sakana Fugu",
        "theme": gr.themes.Default(
            font=(gr.themes.GoogleFont("Source Sans Pro"), "sans-serif"),
            font_mono=("monospace",),
        ),
    }
    if "css" in inspect.signature(gr.Blocks).parameters:
        kwargs["css"] = CSS
    return kwargs


def _launch_kwargs(demo: gr.Blocks) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "server_name": os.environ.get("HOST", "0.0.0.0"),
        "server_port": int(os.environ.get("PORT", "7860")),
    }
    if "css" in inspect.signature(demo.launch).parameters and "css" not in _blocks_kwargs():
        kwargs["css"] = CSS
    return kwargs


def _settings():
    return load_settings()


def _warm_tts_cache() -> None:
    if os.environ.get("REACHY_TTS_WARMUP", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    try:
        started = time.monotonic()
        path = synthesize_speech("Hi, I am Reachy and I am ready.", _settings().cache_dir / "audio")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        print(f"Reachy TTS warmup ready in {elapsed_ms} ms: {path}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - warmup should not block the app
        print(f"Reachy TTS warmup skipped: {exc}", file=sys.stderr)


def _status_markdown(force: bool = False) -> tuple[str, Any, Any]:
    settings = _settings()
    status = get_robot_status(settings, force=force)
    css_class = "mode-banner"
    robot_update = gr.update(interactive=status.connected)
    return f'<div class="{css_class}">{status.message}</div>', robot_update, robot_update


def _robot_hearing_html() -> str:
    settings = _settings()
    if not settings.reachy_hardware_enabled:
        return '<div class="hearing-status">Robot mic: hardware mode off.</div>'
    payload, error = _robot_doa_payload(settings)
    if error is not None:
        exc = html.escape(str(error))
        return f'<div class="hearing-status error">Robot mic: unavailable ({exc}).</div>'
    if payload is None:
        return '<div class="hearing-status error">Robot mic: unavailable.</div>'
    angle = payload.get("angle")
    if payload.get("speech_detected") and isinstance(angle, (int, float)) and not isinstance(angle, bool):
        degrees = round(float(angle) * 180.0 / math.pi, 1)
        return f'<div class="hearing-status active">Robot mic: speech at {degrees} deg.</div>'
    return '<div class="hearing-status">Robot mic: listening.</div>'


def _robot_doa_payload(settings) -> tuple[dict[str, Any] | None, Exception | None]:
    try:
        request = urllib.request.Request(
            f"{settings.reachy_daemon_url}/api/state/doa",
            headers={"Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=0.7) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except (OSError, urllib.error.URLError, ValueError) as exc:
        return None, exc
    if not isinstance(payload, dict):
        return None, None
    return payload, None


def _robot_speech_detection(settings) -> tuple[bool, str | None]:
    payload, error = _robot_doa_payload(settings)
    if error is not None:
        return False, str(error)
    return bool(payload and payload.get("speech_detected")), None


class RobotConversationLoop:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._events: deque[str] = deque(maxlen=20)
        self._phase = "Idle. Press Start Robot Conversation when you are ready."
        self._phase_css = ""

    def start(self) -> str:
        settings = _settings()
        if not settings.reachy_hardware_enabled:
            self._append("Robot conversation cannot start because hardware mode is off.")
            return self.status_html()
        status = get_robot_status(settings, force=True)
        if not status.connected:
            self._append(f"Robot conversation cannot start: {status.message}")
            return self.status_html()
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._append("Robot conversation is already listening.")
                return self.status_html()
            self._stop.clear()
            self._phase = "Starting robot listener."
            self._phase_css = "active"
            self._thread = threading.Thread(target=self._run, name="reachy-fugu-conversation", daemon=True)
            self._thread.start()
        self._append("Robot conversation started. Listening through the robot mic.")
        return self.status_html()

    def stop(self) -> str:
        self._stop.set()
        self._set_phase("Stopping. Current capture or playback will cancel.")
        self._append("Robot conversation stop requested.")
        return self.status_html()

    def status_html(self) -> str:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            stopping = self._stop.is_set() and running
            events = list(self._events)
            phase = self._phase
            phase_css = self._phase_css
        css = "active" if running and not stopping else ""
        if not events:
            events = ["Robot conversation loop is idle."]
        items = "".join(f"<li>{html.escape(event)}</li>" for event in reversed(events[-8:]))
        state = "stopping" if stopping else ("running" if running else "stopped")
        return (
            f'<div class="hearing-status {css}">'
            f"Robot conversation: {state}."
            f'<div class="turn-state {html.escape(phase_css)}"><strong>Now:</strong> {html.escape(phase)}</div>'
            f"<ul>{items}</ul></div>"
        )

    def _append(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._events.append(f"{timestamp} {message}")

    def _set_phase(self, message: str, css_class: str = "") -> None:
        with self._lock:
            self._phase = message
            self._phase_css = css_class

    def _run(self) -> None:
        doa_failures = 0
        dead_audio_failures = 0
        require_fresh_speech = False
        last_reset_log_at = 0.0
        input_session = None

        def close_input_session() -> None:
            nonlocal input_session
            if input_session is None:
                return
            input_session.close()
            input_session = None

        self._set_phase("Ready, speak now. Reachy is listening through its microphone.", "active")
        while not self._stop.is_set():
            settings = _settings()
            if not settings.reachy_hardware_enabled:
                close_input_session()
                self._set_phase("Hardware mode is off. Enable Reachy hardware before listening.", "error")
                self._append("Hardware mode turned off; stopping robot conversation.")
                return
            speech_detected, doa_error = _robot_speech_detection(settings)
            if doa_error:
                doa_failures += 1
                self._set_phase("Checking robot microphone status.", "error")
                if doa_failures >= 10:
                    close_input_session()
                    self._set_phase("Robot mic status is unavailable. Check the Reachy daemon.", "error")
                    self._append(f"Robot mic status unavailable; stopping robot conversation ({doa_error}).")
                    return
            else:
                doa_failures = 0
            if require_fresh_speech and not speech_detected:
                require_fresh_speech = False
            if require_fresh_speech and speech_detected:
                self._set_phase("Waiting for the robot mic signal to reset before the next question.", "active")
                now = time.monotonic()
                if now - last_reset_log_at >= 3.0:
                    self._append("Waiting for a fresh speech signal before recording again.")
                    last_reset_log_at = now
                time.sleep(0.2)
                continue
            self._set_phase("Listening. Speak now; capture stops after you pause.", "active")
            self._append("Robot mic is listening for a question.")
            try:
                turn_started = time.monotonic()
                if input_session is None:
                    input_session = RobotInputSession(settings)
                transcript, capture_ms = _capture_robot_question_with_retries(
                    settings,
                    self._stop,
                    self._append,
                    input_session=input_session,
                )
                if self._stop.is_set() or transcript.status == "stopped":
                    close_input_session()
                    self._set_phase("Stopped during microphone capture.")
                    self._append("Current robot mic capture canceled.")
                    return
                if not transcript.text:
                    if transcript.status == "no_speech_detected" and not speech_detected:
                        self._set_phase("Ready, speak now. Reachy is listening through its microphone.", "active")
                        time.sleep(0.1)
                        continue
                    if _is_dead_audio_capture(transcript):
                        dead_audio_failures += 1
                        limit = _dead_audio_failure_limit()
                        if dead_audio_failures >= limit:
                            raise RobotInputUnavailableError(
                                f"robot mic returned dead audio {dead_audio_failures} times without a valid transcript"
                            )
                        self._append(
                            f"Robot mic returned dead audio ({dead_audio_failures}/{limit}); checking media again."
                        )
                    close_input_session()
                    self._set_phase("Ready, speak again. I did not catch a clear question.", "error")
                    require_fresh_speech = True
                    self._append(_empty_transcript_event_message(transcript))
                    time.sleep(0.6)
                    continue
                dead_audio_failures = 0
                close_input_session()
                self._set_phase("Thinking. Fugu is preparing the answer.", "active")
                self._append(f"Transcript in {capture_ms} ms{_transcript_capture_suffix(transcript)}: {transcript.text}")
                if self._stop.is_set():
                    self._set_phase("Stopped before Fugu generation.")
                    self._append("Stop requested before Fugu generation.")
                    return
                result = _answer_question_interruptible(transcript.text, settings, self._stop)
                if result is None:
                    self._set_phase("Stopped. Fugu generation will finish in the background and be ignored.")
                    self._append("Stop requested during Fugu generation; ignoring the late answer.")
                    return
                if self._stop.is_set():
                    self._set_phase("Stopped before robot playback.")
                    self._append("Stop requested; skipping robot playback.")
                    return
                latency = result.debug.get("latency_ms") if isinstance(result.debug, dict) else None
                fugu_ms = latency.get("fugu") if isinstance(latency, dict) else None
                tts_ms = latency.get("tts") if isinstance(latency, dict) else None
                elapsed_ms = int((time.monotonic() - turn_started) * 1000)
                if isinstance(fugu_ms, int) and isinstance(tts_ms, int):
                    self._append(f"Fugu answer ready in {elapsed_ms} ms (fugu {fugu_ms} ms, tts {tts_ms} ms); speaking.")
                else:
                    self._append(f"Fugu answer ready in {elapsed_ms} ms; speaking on Reachy.")
                self._set_phase("Speaking. Listening is paused so Reachy does not hear itself.", "active")
                with robot_tracking_session(settings) as tracking:
                    playback = play_on_robot(
                        result.gesture_plan,
                        result.audio_path,
                        settings,
                        tracking_pose_provider=tracking.pose,
                        stop_event=self._stop,
                    )
                self._append(playback)
                require_fresh_speech = True
                self._set_phase(
                    f"Paused {_post_speech_listen_cooldown_seconds():.1f}s after speech to avoid self-triggering."
                )
                _pause_after_robot_speech(settings, self._stop, self._append)
                if not self._stop.is_set():
                    self._set_phase("Ready, speak now. Reachy is listening through its microphone.", "active")
            except RobotInputUnavailableError as exc:
                close_input_session()
                self._set_phase("Robot mic or camera media is unavailable. Check Reachy media support.", "error")
                self._append(f"Robot input media unavailable; stopping conversation loop: {exc}")
                return
            except Exception as exc:
                close_input_session()
                self._set_phase("Turn failed. Ready to try again.", "error")
                self._append(f"Turn failed: {exc}")
            time.sleep(0.8)
        close_input_session()
        self._set_phase("Stopped.")
        self._append("Robot conversation stopped.")


_ROBOT_CONVERSATION = RobotConversationLoop()
_ANSWER_GENERATION_LOCK = threading.Lock()


def _empty_transcript_message(status: str) -> str:
    if status == "captured_audio_too_short":
        return "Robot mic heard a very short fragment. Keep speaking until the listening status changes."
    if status == "no_speech_detected":
        return "Robot mic did not detect a clear question. Speak toward Reachy after the listening status appears."
    if status == "speech_recognition_empty_transcript":
        return "Robot mic captured audio, but SpeechRecognition did not hear a clear phrase. Try again closer to Reachy."
    if status == "non_english_transcript":
        return "Robot mic heard unclear or non-English speech. Try again in English, closer to Reachy."
    if status == "low_confidence_transcript":
        return "Robot mic heard speech, but the transcription confidence was too low. Try again closer to Reachy."
    if status == "transcription_service_error":
        return "Robot mic captured audio, but the speech-to-text service failed. Check internet access or switch to typed fallback."
    return f"Robot mic captured audio, but local transcription was empty ({status}). Try again closer to the robot mic."


def _transcript_capture_suffix(transcript: Any) -> str:
    details: list[str] = []
    duration = float(getattr(transcript, "duration_seconds", 0.0) or 0.0)
    rms = float(getattr(transcript, "rms", 0.0) or 0.0)
    if duration:
        details.append(f"{duration:.1f}s audio")
    if rms:
        details.append(f"rms {rms:.4f}")
    return f" ({', '.join(details)})" if details else ""


def _empty_transcript_event_message(transcript: Any) -> str:
    return f"{_empty_transcript_message(getattr(transcript, 'status', ''))}{_transcript_capture_suffix(transcript)}"


def _empty_transcript_retry_attempts() -> int:
    try:
        return max(0, int(os.environ.get("REACHY_MIC_EMPTY_RETRY_ATTEMPTS", "1")))
    except ValueError:
        return 1


def _empty_transcript_retry_delay_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_MIC_EMPTY_RETRY_DELAY_SECONDS", "0.25")))
    except ValueError:
        return 0.25


def _post_speech_listen_cooldown_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_POST_SPEECH_LISTEN_COOLDOWN_SECONDS", "1.5")))
    except ValueError:
        return 1.5


def _post_speech_doa_clear_timeout_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_POST_SPEECH_DOA_CLEAR_TIMEOUT_SECONDS", "1.5")))
    except ValueError:
        return 1.5


def _pause_after_robot_speech(settings, stop_event: Any, append_event) -> None:
    seconds = _post_speech_listen_cooldown_seconds()
    if seconds > 0.0:
        append_event(f"Listening paused {seconds:.1f}s after Reachy speaks.")
        deadline = time.monotonic() + seconds
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))
    clear_timeout = _post_speech_doa_clear_timeout_seconds()
    if stop_event.is_set() or clear_timeout <= 0.0:
        return
    deadline = time.monotonic() + clear_timeout
    waiting_logged = False
    while not stop_event.is_set():
        speech_detected, error = _robot_speech_detection(settings)
        if not speech_detected:
            return
        if not waiting_logged:
            append_event("Waiting for robot mic speech signal to clear.")
            waiting_logged = True
        if error is not None or time.monotonic() >= deadline:
            append_event("Robot mic speech signal stayed active after cooldown; resuming carefully.")
            return
        time.sleep(0.1)


def _answer_question_interruptible(text_question: str, settings, stop_event: Any) -> Any | None:
    if not _ANSWER_GENERATION_LOCK.acquire(blocking=False):
        raise RuntimeError("Previous Fugu generation is still finishing; wait a moment before asking again.")
    done = threading.Event()
    cancelled = threading.Event()
    result_box: dict[str, Any] = {}

    def worker() -> None:
        try:
            result = answer_question(text_question=text_question, audio_path=None, settings=settings, stop_event=stop_event)
            if not cancelled.is_set() and not stop_event.is_set():
                result_box["result"] = result
        except Exception as exc:  # noqa: BLE001 - re-raised on the conversation thread
            if not cancelled.is_set() and not stop_event.is_set():
                result_box["error"] = exc
        finally:
            _ANSWER_GENERATION_LOCK.release()
            done.set()

    threading.Thread(target=worker, name="reachy-fugu-answer", daemon=True).start()
    while not done.wait(0.1):
        if stop_event.is_set():
            cancelled.set()
            return None
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("result")


def _retryable_empty_transcript(transcript: Any) -> bool:
    if getattr(transcript, "text", ""):
        return False
    return getattr(transcript, "status", "") in {"speech_recognition_empty_transcript", "captured_audio_too_short"}


def _dead_audio_failure_limit() -> int:
    try:
        return max(1, int(os.environ.get("REACHY_MIC_ZERO_AUDIO_FAILURE_LIMIT", "3")))
    except ValueError:
        return 3


def _dead_audio_max_seconds() -> float:
    try:
        return max(0.05, float(os.environ.get("REACHY_MIC_DEAD_AUDIO_MAX_SECONDS", "0.6")))
    except ValueError:
        return 0.6


def _dead_audio_rms_threshold() -> float:
    try:
        return max(0.0, float(os.environ.get("REACHY_MIC_DEAD_AUDIO_RMS_THRESHOLD", "0.001")))
    except ValueError:
        return 0.001


def _is_dead_audio_capture(transcript: Any) -> bool:
    if getattr(transcript, "text", ""):
        return False
    if getattr(transcript, "status", "") != "no_speech_detected":
        return False
    duration = float(getattr(transcript, "duration_seconds", 0.0) or 0.0)
    rms = float(getattr(transcript, "rms", 0.0) or 0.0)
    if duration <= 0.05 and rms <= 0.0001:
        return True
    return duration <= _dead_audio_max_seconds() and rms <= _dead_audio_rms_threshold()


def _zero_audio_failure_limit() -> int:
    return _dead_audio_failure_limit()


def _is_zero_audio_capture(transcript: Any) -> bool:
    return _is_dead_audio_capture(transcript)


def _retry_empty_transcript_message(transcript: Any) -> str:
    details: list[str] = []
    duration = float(getattr(transcript, "duration_seconds", 0.0) or 0.0)
    rms = float(getattr(transcript, "rms", 0.0) or 0.0)
    if duration:
        details.append(f"{duration:.1f}s")
    if rms:
        details.append(f"rms {rms:.4f}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"Robot mic caught audio but STT was empty{suffix}; listening again."


def _capture_robot_question_with_retries(
    settings,
    stop_event: Any,
    append_event,
    *,
    input_session: Any | None = None,
) -> tuple[Any, int]:
    started = time.monotonic()
    attempts = _empty_transcript_retry_attempts() + 1
    transcript = None
    for attempt in range(attempts):
        if input_session is not None:
            transcript = input_session.record_question(stop_event=stop_event)
        else:
            transcript = record_robot_question(settings, stop_event=stop_event)
        if stop_event.is_set() or getattr(transcript, "status", "") == "stopped":
            break
        if getattr(transcript, "text", ""):
            break
        if attempt < attempts - 1 and _retryable_empty_transcript(transcript):
            append_event(_retry_empty_transcript_message(transcript))
            time.sleep(_empty_transcript_retry_delay_seconds())
            continue
        break
    return transcript, int((time.monotonic() - started) * 1000)


def _render_result(result) -> tuple[str, str, str]:
    answer = f"**Confidence:** `{result.confidence}`\n\n{result.answer}"
    debug_payload = {**result.debug, "gesture_plan": result.gesture_plan}
    debug = json.dumps(debug_payload, indent=2, ensure_ascii=False)
    return answer, str(result.audio_path), debug


def preview_locally(text_question: str):
    yield "**Thinking...**", None, "{}"
    try:
        result = answer_question(text_question=text_question, audio_path=None, settings=_settings())
    except (ConfigError, UserFacingError, RuntimeError, ValueError) as exc:
        message = f"**Error:** {exc}"
        yield message, None, "{}"
        return
    yield _render_result(result)


def run_on_reachy(text_question: str):
    yield "**Checking Reachy...**", None, "{}"
    try:
        settings = _settings()
        status = get_robot_status(settings, force=True)
        if not status.connected:
            raise RuntimeError(status.message)
        yield "**Thinking...**", None, "{}"
        result = answer_question(text_question=text_question, audio_path=None, settings=settings)
        answer, audio, debug = _render_result(result)
        yield answer, audio, debug
        with robot_tracking_session(settings) as tracking:
            playback = play_on_robot(
                result.gesture_plan,
                result.audio_path,
                settings,
                tracking_pose_provider=tracking.pose,
            )
        debug_payload = json.loads(debug)
        debug_payload["robot_playback"] = playback
        yield answer, audio, json.dumps(debug_payload, indent=2, ensure_ascii=False)
    except (ConfigError, UserFacingError, RuntimeError, ValueError) as exc:
        message = f"**Error:** {exc}"
        yield message, None, "{}"


def start_robot_conversation_loop() -> str:
    return _ROBOT_CONVERSATION.start()


def stop_robot_conversation_loop() -> str:
    return _ROBOT_CONVERSATION.stop()


def robot_conversation_loop_html() -> str:
    return _ROBOT_CONVERSATION.status_html()


def create_app() -> gr.Blocks:
    with gr.Blocks(**_blocks_kwargs()) as demo:
        with gr.Row(elem_id="title"):
            gr.Markdown("# Reachy Mini x Sakana Fugu\nRobot-sourced conversation with Fugu answers and validated Reachy gestures.")

        robot_status = gr.HTML()

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=360):
                text_input = gr.Textbox(
                    label="Manual text fallback",
                    placeholder="What is Fugu?",
                    lines=2,
                    submit_btn=False,
                    elem_id="question_text",
                )
                with gr.Row():
                    preview_button = gr.Button("Preview Locally", variant="primary", elem_classes=["primary"])
                    robot_button = gr.Button("Run on Reachy", interactive=False, elem_id="run_on_reachy_button")
                    refresh_button = gr.Button("Reconnect", elem_id="reconnect_button")
                with gr.Group(elem_classes=["robot-input-panel"]):
                    gr.Markdown(
                        "Robot mic and robot camera tracking run automatically. "
                        "The browser is only the operator dashboard and manual text fallback."
                    )
                    with gr.Row():
                        robot_loop_start_button = gr.Button("Start Robot Conversation", variant="primary", interactive=False)
                        robot_loop_stop_button = gr.Button("Stop")
                    robot_loop_status = gr.HTML()
                robot_hearing = gr.HTML()

            with gr.Column(scale=7, min_width=420):
                answer_output = gr.Markdown(label="Answer")
                audio_output = gr.Audio(label="Response audio", type="filepath", autoplay=False)
                debug_output = gr.Code(language="json", label="Debug log")

        outputs = [answer_output, audio_output, debug_output]
        preview_button.click(preview_locally, inputs=[text_input], outputs=outputs)
        text_input.submit(preview_locally, inputs=[text_input], outputs=outputs)
        robot_button.click(run_on_reachy, inputs=[text_input], outputs=outputs)
        robot_loop_start_button.click(start_robot_conversation_loop, outputs=[robot_loop_status], queue=False)
        robot_loop_stop_button.click(stop_robot_conversation_loop, outputs=[robot_loop_status], queue=False)
        refresh_button.click(lambda: _status_markdown(True), outputs=[robot_status, robot_button, robot_loop_start_button], queue=False)
        demo.load(lambda: _status_markdown(False), outputs=[robot_status, robot_button, robot_loop_start_button], queue=False)
        demo.load(_robot_hearing_html, outputs=[robot_hearing], queue=False)
        demo.load(robot_conversation_loop_html, outputs=[robot_loop_status], queue=False)
        gr.Timer(1.0).tick(_robot_hearing_html, outputs=[robot_hearing], queue=False)
        gr.Timer(1.0).tick(robot_conversation_loop_html, outputs=[robot_loop_status], queue=False)
        gr.Timer(5.0).tick(
            lambda: _status_markdown(False),
            outputs=[robot_status, robot_button, robot_loop_start_button],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    _warm_tts_cache()
    app = create_app().queue()
    app.launch(**_launch_kwargs(app))
