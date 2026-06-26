# Rubik's Cube Solver Benchmark (one-shot, code-writing)

A reproducible comparison of LLMs on a single, verifiable task: **write a
Rubik's cube solver from scratch, once.**

Each model is given the same prompt and must return a self-contained Python
module defining `solve(facelet: str) -> str`. The model spends tokens **once**
(writing the solver); we then run that frozen solver locally against a frozen
set of 300 scrambled cubes. The solving itself costs no tokens. Every returned
solution is re-verified with a trusted cube engine (`cube.py`), so a model
cannot win by claiming a cube is solved.

This isolates a clean signal: can the model produce **correct, efficient,
robust algorithmic code in one shot?**

## The task contract

- Input: a 54-character facelet string (faces in order `U R F D L B`, each read
  row-major). See `prompt.txt` for the exact, complete spec given to every model.
- Output: a space-separated WCA move string (`R U R' U2 F D' L2`); empty = already solved.
- Rules: standard library only, no cube/solver library, `solve()` must be a pure
  function (no I/O), reasonably fast (per-cube wall-clock cap).
- Scoring: **solve rate** (correctness, re-verified) and **mean turn count / HTM**
  (efficiency; God's number = 20 is the absolute upper bound on optimal length).

## Directory layout

```
prompt.txt              The exact prompt given to every model.
cube.py                 Trusted cube engine: applies moves, verifies solved, counts turns.
                        This is the ground truth. It also generates the eval set.
eval_cubes.json         The frozen 300-cube eval set (id 0 = "hero" cube for visuals).
build_eval_set.py       Regenerates eval_cubes.json deterministically from seeds.

rubik_compare.py        Full pipeline: holds the PROMPT, calls each model's API,
                        saves the returned solver, runs + verifies it. (Needs API keys.)
call_models_standalone.py  The six per-model API calls in isolation (no harness) --
                        the exact code we use to hit each model. (Needs API keys.)
solver_runner.py        Subprocess-side runner: imports one solver, runs it per cube
                        with a per-cube timeout, prints raw solutions (no judging).
rerun_saved.py          Reproduce from FROZEN solvers, NO API calls. <-- main entry point.
run_checkpointed.py     Resumable single-solver runner (one model, checkpointed JSONL).

results/
  <model>_solver.py     The exact solver code each model wrote (frozen).
  <model>_raw.txt       The raw model response it was extracted from.
  reference/            The result summaries WE measured (for comparison).
    first100_all5.json    All 5 core models on the first 100 cubes.
    new200_<model>.json   The 3 models that solve, on cubes 100..299.
    fable5_300.json       Fable 5 on the full 300-cube set.
    prelim20_incl_fable5.json  Early 20-cube run that includes Fable 5.
```

Models (key -> label): `fugu_ultra` (Fugu-Ultra), `fugu` (Fugu),
`gpt55` (GPT-5.5), `opus48` (Claude Opus 4.8), `gemini` (Gemini 3.1 Pro),
`fable5` (Fable 5).

## How each model's API is called (the exact generation code)

The numbers come from one API call per model (the model writes the solver once).
The exact call we used for every model is below — same prompt (`prompt.txt`),
each at its highest reliable reasoning effort, keys read from env vars, no retries
(`max_retries=0`, so a failure is visible and never silently re-billed). Two files
hold this code:

- **`call_models_standalone.py`** — the same six calls with no eval harness around
  them (a trivial prompt), so you can verify access and see each model's call
  pattern in isolation: `python3 call_models_standalone.py fugu_ultra`.
- **`rubik_compare.py`** — the full pipeline: identical call functions
  (`call_gpt55`, `call_opus48`, `call_fable5`, `call_gemini`, `call_fugu`,
  `call_fugu_ultra`), but with the real solver `PROMPT`, then it saves and runs the
  returned solver.

Env vars / SDKs: `OPENAI_API_KEY` (`openai`), `ANTHROPIC_API_KEY` (`anthropic`),
`GEMINI_API_KEY` (`google-genai`), `FUGU_API_KEY` (Fugu, OpenAI-compatible).

| key | API surface | model id | effort / params |
|---|---|---|---|
| `gpt55` | OpenAI Responses (`client.responses.create`) | `gpt-5.5` | `reasoning.effort="high"`, `max_output_tokens=64000` |
| `opus48` | Anthropic Messages stream | `claude-opus-4-8` | `thinking=adaptive`, `output_config.effort="high"`, `max_tokens=64000` |
| `fable5` | Anthropic Messages stream | `claude-fable-5` | `thinking=adaptive`, `output_config.effort="xhigh"`, `max_tokens=128000` |
| `gemini` | Google GenAI (`models.generate_content`) | `gemini-3.1-pro-preview` | `thinking_level=HIGH`, `max_output_tokens=65536` |
| `fugu` | OpenAI-compatible @ `https://api.sakana.ai/v1` | `fugu` | `temperature=0.7`, `max_tokens=64000` |
| `fugu_ultra` | OpenAI-compatible @ `https://api.sakana.ai/v1` (streamed) | `fugu-ultra` | `reasoning_effort="max"`, `max_tokens=64000` |

The exact code (verbatim from `call_models_standalone.py` / `rubik_compare.py`):

```python
# GPT-5.5 — OpenAI Responses API
import openai
client = openai.OpenAI(timeout=8000, max_retries=0)          # OPENAI_API_KEY
resp = client.responses.create(
    model="gpt-5.5", input=PROMPT,
    reasoning={"effort": "high"}, max_output_tokens=64000)
text = resp.output_text

# Claude Opus 4.8 — Anthropic Messages (streamed)
import anthropic
client = anthropic.Anthropic(timeout=8000, max_retries=0)    # ANTHROPIC_API_KEY
with client.messages.stream(
        model="claude-opus-4-8", max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},                    # "max" can starve output
        messages=[{"role": "user", "content": PROMPT}]) as stream:
    for _ in stream:
        pass
text = next(b.text for b in stream.get_final_message().content if b.type == "text")

# Fable 5 — Anthropic Messages (streamed); needs the 128k budget at xhigh
with client.messages.stream(
        model="claude-fable-5", max_tokens=128000,
        thinking={"type": "adaptive"},
        output_config={"effort": "xhigh"},
        messages=[{"role": "user", "content": PROMPT}]) as stream:
    for _ in stream:
        pass
text = next(b.text for b in stream.get_final_message().content if b.type == "text")

# Gemini 3.1 Pro — Google GenAI
from google import genai
from google.genai import types
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(timeout=8000 * 1000))
resp = client.models.generate_content(
    model="gemini-3.1-pro-preview", contents=PROMPT,
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
        max_output_tokens=65536))
text = next(p.text for p in resp.candidates[0].content.parts
            if getattr(p, "text", None) and not getattr(p, "thought", False))

# Fugu — OpenAI-compatible endpoint
client = openai.OpenAI(api_key=os.environ["FUGU_API_KEY"],
                       base_url="https://api.sakana.ai/v1", timeout=8000, max_retries=0)
resp = client.chat.completions.create(
    model="fugu", messages=[{"role": "user", "content": PROMPT}],
    temperature=0.7, max_tokens=64000)
text = resp.choices[0].message.content

# Fugu-Ultra — same endpoint, streamed, reasoning_effort high|max only
resp = client.chat.completions.create(
    model="fugu-ultra", messages=[{"role": "user", "content": PROMPT}],
    max_tokens=64000, extra_body={"reasoning_effort": "max"},
    stream=True, stream_options={"include_usage": True})
text = "".join(c.choices[0].delta.content or "" for c in resp if c.choices)
```

**Why these effort levels.** Each model runs at the highest effort that *reliably
returns output* for this task, not a uniform setting:
- `opus48` uses `high` (not `max`): at `max`, the model's reasoning consumes the
  whole `max_tokens` budget and returns an empty string for this prompt, so `high`
  is its best usable setting here. (Its 0/300 is a bug in the code it wrote at
  `high`, not an effort limitation.)
- `gpt55` uses `high` for the same reason — `xhigh` hung for tens of minutes.
- `fugu-ultra` accepts only `high` / `max` (we use `max`); `fable5` needs the 128k
  token budget to avoid starving at `xhigh`.

## How to reproduce the numbers (no API keys, no third-party packages)

The headline result needs **only Python 3 and its standard library** — the
solvers are frozen, so reproduction is pure local CPU. No model is called.

```bash
# run all frozen solvers on the 300-cube set, re-verify each solution with cube.py
RUBIK_CUBE_TIMEOUT=600 RUBIK_SUMMARY_OUT=summary_300.json \
    python3 rerun_saved.py
# -> writes results/summary_300.json
```

Use `RUBIK_CUBE_TIMEOUT=600` (not the 300 default): one Fugu-Ultra cube (#153)
genuinely needs ~305 s of optimal search and is a *false* timeout at a 300 s cap.
At 600 s it solves correctly in 19 moves, giving Fugu-Ultra a clean 300/300.

Per single model (resumable, useful for the slow ones):

```bash
RUBIK_CUBE_TIMEOUT=600 python3 run_checkpointed.py \
    fugu_ultra eval_cubes.json results/ckpt_fugu_ultra.jsonl results/summary_fugu_ultra_300.json
```

Sanity-check the engine itself: `python3 cube.py` runs its self-test.

## Results (300 cubes)

| Model        | Solve rate | Mean HTM | Notes |
|--------------|-----------:|---------:|-------|
| Fugu-Ultra   | 300 / 300  | 19.72    | most efficient; cube #153 needs 600 s cap (see above) |
| GPT-5.5      | 300 / 300  | 19.76    | |
| Fable 5      | 300 / 300  | 20.22    | two-phase (Kociemba) solver; fast & steady (max 4 s/cube), 0 timeouts |
| Fugu    | 300 / 300  | 21.15    | ~35x faster per cube than Ultra/GPT (max ~13 s vs ~250-305 s) |
| Claude Opus 4.8 | 0 / 300 | —        | written module raises `IndexError` at import (no usable `solve`) |
| Gemini 3.1 Pro  | 0 / 300 | —        | `solve()` raises `IndexError` at runtime on every cube |

Two of six frontier solvers do not run at all — the **robustness** axis is as
important as the move count. Among the four solvers that run, the move counts are
close (a band from ~19.7 to ~21.2), so the headline is robustness + Fugu's
speed, not a large efficiency gap.

300-cube result file for Fable 5: `results/reference/fable5_300.json`.

## Running fresh against the live model APIs (optional)

`rubik_compare.py` regenerates solvers by calling each model. This needs the
relevant SDKs (`openai`, `anthropic`, `google-genai`) and keys
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `FUGU_API_KEY`).
Note this re-rolls the one-shot solver; the frozen `results/<model>_solver.py`
files are the exact code behind the numbers above.

## TODO / known gaps

- **Cost & token transparency.** The current summaries record wall-clock
  (`api_sec`, `solve_sec`) but not input/output token counts or $ cost per model.
  Capturing usage from each API response would let the comparison report
  cost-per-solver, which is the most informative axis for a one-shot code task.
- Fable 5 300-cube run (above).
