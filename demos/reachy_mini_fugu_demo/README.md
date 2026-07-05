# Reachy Mini Fugu Demo

Minimal Gradio demo for robot-sourced conversation with Reachy Mini and Sakana Fugu. Reachy listens through the robot microphone, sends the transcribed question to Fugu, speaks the short answer back, and moves through validated gentle gestures while the browser shows the transcript, answer, audio, and plan.

![Reachy Mini Fugu robot demo](demo_media/reachy_mini_fugu_demo.gif)

## What It Does

- Uses Reachy Mini microphone input for the primary conversation path.
- Uses Reachy camera or direction-of-arrival tracking cues when available.
- Calls Sakana Fugu through the OpenAI-compatible chat completions endpoint.
- Synthesizes speech locally and plays it through Reachy's daemon speaker.
- Keeps the browser as an operator dashboard and manual text fallback.

## Run Locally

```bash
cd demos/reachy_mini_fugu_demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a no-key UI smoke test:

```bash
export FUGU_DRY_RUN=1
export REACHY_HARDWARE_ENABLED=0
python app.py
```

For robot mode:

```bash
export SAKANA_API_KEY=<your-api-key>
export REACHY_DAEMON_URL=http://127.0.0.1:8000
export REACHY_HARDWARE_ENABLED=1
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860).

## Reachy Setup

Start the Reachy daemon and keep `REACHY_DAEMON_URL` pointed at it. Set `REACHY_HARDWARE_ENABLED=1` only when the robot is connected and ready. The main robot flow is **Start Robot Conversation**; **Preview Locally** and the text box are fallback controls.

## Files

- `app.py` runs the Gradio dashboard and conversation loop.
- `src/reachy_fugu/` contains Fugu calls, speech, robot input, and playback glue.
- `reachy_fugu_demo/` contains the small Reachy primitive layer used by playback.
- `demo_media/reachy_mini_fugu_demo.gif` is the README demo preview.
