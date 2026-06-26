# Introduction

This directory contains the scripts for reproducing 6 demos inspired by Twitter users' experiences with Sakana Fugu. Each demo was run across the following 4 models, for a total of 24 codex sessions.

| key          | model                     | harness                                          | effort           |
| ------------ | ------------------------- | ------------------------------------------------ | ---------------- |
| `fugu-ultra` | Sakana Fugu Ultra         | `codex-fugu` (Sakana provider)[^codex]           | `xhigh`          |
| `gpt55`      | OpenAI GPT-5.5            | `codex` 0.142.0 then OpenRouter (Responses API)  | `xhigh`          |
| `opus48`     | Anthropic Claude Opus 4.8 | `codex` 0.142.0 then strip-proxy then OpenRouter | `xhigh`          |
| `gemini`     | Google Gemini 3.1 Pro     | `codex` 0.142.0 then OpenRouter                  | `high` (its max) |

[^codex]: `codex-fugu` installed from commit [`655675e`](https://github.com/SakanaAI/fugu/commit/655675ed7a4ebe14b8e712ab4d8a1b025510608f) (Codex 0.142.0).

## Environment setup

Run these steps from this directory.

1. Install `codex-fugu`.

   ```bash
   curl -fsSL https://sakana.ai/fugu/install | bash
   ```

   The one-line installer supports Ubuntu and macOS. On Windows, or if the install does not complete, follow this [guide](https://console.sakana.ai/get-started#manually-setting-up-codex).

   To maximize reproducibility, match the exact versions used for this sweep: Codex 0.142.0 and the `codex-fugu` build at commit `655675e`.

   The one-liner above always installs the latest `codex-fugu`, which may pin a newer Codex. For an exact match, install from that commit instead:

   ```bash
   git clone https://github.com/SakanaAI/fugu.git ~/.fugu
   git -C ~/.fugu checkout 655675ed7a4ebe14b8e712ab4d8a1b025510608f
   bash ~/.fugu/scripts/install.sh        # add --force if Codex is already at a different version
   ```

   The baseline models (`gpt55`, `opus48`, `gemini`) call the same Codex 0.142.0 binary through OpenRouter, so this single install covers all four harnesses. Newer Codex releases are forward-compatible and should also work, but they are not what these demos were produced with.

2. Export your OpenRouter key, which covers all three baseline models.

   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-..."
   ```

3. Install the proxy dependencies with uv.

   ```bash
   uv sync
   ```

## Run it

```bash
cd demos/users_demo

# whole sweep (starts proxy, keeper, monitor, launches all 6 demos, 5h cap per cell)
bash harness/run_sweep.sh 18000

# one demo by name (4 models concurrently)
bash harness/run_demo.sh demo5_crossy_road 18000

# one cell
bash harness/run_codex.sh gemini demo5_crossy_road/gemini prompts/demo5_crossy_road.txt 18000
```

A standalone `opus48` run needs the strip-proxy up first (`run_sweep.sh` starts it automatically). For a single Opus demo or cell, start it by hand with `PORT=9461 PROXY_PYTHON=<python> <python> harness/or_proxy.py &` and then `echo 9461 > harness/.proxy_port`.

Each cell writes into `<demo>/<model>/` the `index.html` (or a project tree), `events.jsonl` (agent transcript), `last_message.txt`, and `meta.json` (rc and timing).

## Monitor and stop

```bash
python3 harness/monitor.py          # one health snapshot
tail -f monitor.log                 # daemon snapshots (every 10 min)
touch STOP_MONITOR                  # stops the monitor daemon and proxy keeper
```

## Collect results

```bash
python3 harness/collect.py          # writes collected/SUMMARY.md and summary.json
```

Examine the deliverables directly by opening each `<demo>/<model>/index.html` in a browser, or by following that cell's instructions. Multi-file projects need a build or dev-server.

## Notes

- codex 0.142 only speaks the OpenAI Responses API (`wire_api="responses"`), and OpenRouter serves it, so GPT-5.5 and Gemini work straight through. Claude's extended-thinking blocks carry a signature that must round-trip byte-exact, and OpenRouter's Responses beta breaks it, giving `Invalid signature in thinking block` on turn 2. `harness/or_proxy.py` is a tiny localhost pass-through that strips the `reasoning` items out of each request's history (Opus still reasons fresh at xhigh each turn). Only `opus48` is routed through it. GPT and Gemini go direct.
- Effort is set via codex `model_reasoning_effort`, and codex caps at `xhigh`, so GPT-5.5 and Opus are requested at `xhigh` and Gemini at `high` (its top thinking level). Each run's session rollout under `~/.codex/sessions/` records the effort that was sent.
- The harness uses `--dangerously-bypass-approvals-and-sandbox` (full auto). Run only on a machine you are fine letting the agents execute shell commands on.
- Multi-file demos (Rocket League with Vite and Rapier, Trader Desk with a frontend and backend) produce project trees that need `npm install` and a build or dev-server. The other four are self-contained single HTML files that open directly.

## Demos

The six demos, each inspired by a Twitter showcase of Sakana Fugu and credited to its author.

### Demo 1 - Subway Surfers

Credit: [@CoinSh0t](https://x.com/CoinSh0t)

A single HTML file endless runner in the Subway Surfers style, built with Three.js. The character sprints down a multi-lane track, dodging trains and obstacles while the speed ramps up. A short, open-ended prompt that leans on each model's sense of game feel.

### Demo 2 - Rocket League

Credit: [@LLMJunky](https://x.com/LLMJunky)

A full browser game of car soccer, and by far the most detailed prompt in the set. It calls for Three.js rendering with a real physics engine (Rapier), a controllable rocket car with boost and double jump, an enclosed arena with goals and scoring, and arcade-style handling, all built as a multi-file Vite project.

### Demo 3 - Trader Desk

Credit: [@atomic_chat_hq](https://x.com/atomic_chat_hq)

A complete live trading desk with both a frontend and a backend, streaming real-time market data for eight symbols into a custom dark-theme UI. This is the only full-stack app in the set, so each model chooses its own server stack and wiring.

### Demo 4 - Procedural Terrain

Credit: [@omarsar0](https://x.com/omarsar0)

A single HTML file endless procedural terrain generator in Three.js. The camera glides over a landscape that is generated on the fly, so the comparison comes down to mesh quality, performance, and how convincing the terrain looks as it streams in.

### Demo 5 - Crossy Road

Credit: [@markksantos](https://x.com/markksantos)

A single HTML file Crossy Road style game in Three.js. The player hops a character across busy roads and rivers as traffic and hazards scroll by. Worth watching for how playable each version feels and whether the difficulty builds as you go.

### Demo 6 - Rube Goldberg

Credit: [@KinasRemek](https://x.com/KinasRemek)

A single HTML file chain-reaction machine that plays out on its own with no input. It links a rolling ball, falling dominoes, a lever, a spring launcher, a pendulum, and a pulley to test whether each model models real physics rather than faking the motion on fixed paths.

The reproduced results for all six demos are available in the [user-demo-v1 release](https://github.com/SakanaAI/fugu/releases/tag/user-demo-v1).
