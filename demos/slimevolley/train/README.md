# Slime Volley: train a policy

This is the "how it learns" side of the Slime Volley demo. An agentic model writes a control policy as Python
code, scores it against the built-in baseline, and improves it over a few iterations. It is deliberately small:
three scripts over one shared module.

## The scripts

- **`train.py --model <model> [--iters N]`** runs a learning loop (default 10 iterations) into `./run/<model>/`. Each
  iteration assembles a prompt (the one `INSTRUCTIONS.md` plus the note the agent wrote last time), spawns that
  model's CLI agent in `run/<model>/iter_NN/`, and carries `note.md` forward. It does not score anything; that is
  `assess.py`. It is **re-entrant**: re-running resumes from the last finished iteration, so raising `--iters` just
  adds more.
- **`assess.py`** is the single source of truth for whether a policy improved. It plays a policy against the
  baseline in the real solid-net env, 500 games across worker processes:
  - `uv run python assess.py --file path/to/policy.py` scores one file (works on the shipped `../policies/*.py` too).
  - `uv run python assess.py --model <model>` ranks every policy in that run and writes the best to `champion.py`.
- **`preflight.py`** spawns each model once to confirm its CLI, key, and config are in place. Run it first.

`common.py` holds the small shared pieces (the model registry, how to spawn each CLI, the run-dir convention).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (this directory is a uv project; `uv sync` installs the deps).
- The CLIs for the models you want to run: `claude`, `codex`, and/or `gemini`.

## What each model needs

| Model  | Needs |
|--------|-------|
| claude | a logged-in `claude` CLI (no API key) |
| gemini | `GEMINI_API_KEY` in your environment |
| fugu   | `SAKANA_API_KEY` + a codex `[profile.fugu]` |

Export the keys in your shell. Put codex profiles in `~/.codex/config.toml`, or keep a local `./.codex/config.toml`
(git-ignored) that `run.sh` uses via `CODEX_HOME`. Nothing private is committed.

## Run it

```
uv run python preflight.py                 # preflight: which models are runnable
./run.sh --model fugu                       # train: 10 iterations into ./run/fugu/
uv run python assess.py --model fugu        # rank the run and write ./run/fugu/champion.py
./run.sh --model fugu --iters 13            # optional: add 3 more iterations (resumes where it stopped)
```

Two ways to get a champion into the demo:

- **Permanent:** copy `run/<model>/champion.py` to `../policies/<id>.py`, add a row to `../policies.json`, rebuild
  (`cd .. && node web/build.mjs`). Baked into `slimevolley.html`; works from `file://`.
- **Live (local only):** each `assess --model` writes `run/inventory.json`; serve the dir over HTTP and trained
  champions appear in the picker automatically. `run/` is git-ignored, so this is ephemeral.

## The policy contract

A policy must be **self-contained** (numpy + the standard library only), because it runs in the browser via Pyodide.
Its signature:

```python
def make_policy(seat=0):
    def act(obs):           # obs: 12 floats, /10-scaled, side-relative
        return [forward, backward, jump]   # three buttons, each 0 or 1
    return act
```
