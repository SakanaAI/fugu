# autoresearch

Artifacts from autonomous LLM research runs on [karpathy/autoresearch](https://github.com/karpathy/autoresearch). This folder contains **outputs only** — not the full experiment repos (`prepare.py`, `program.md`, git history, etc.).

## Background

[autoresearch](https://github.com/karpathy/autoresearch) gives a coding agent a single-GPU nanochat-style training setup. The agent edits `train.py`, trains for a **fixed 5-minute wall-clock budget**, checks **val_bpb** (validation bits per byte — lower is better), keeps improvements, and reverts failures. Human instructions live in `program.md`; the agent only modifies `train.py`.

## How to reproduce

1. **Clone the upstream repo**
   ```bash
   git clone https://github.com/karpathy/autoresearch.git
   cd autoresearch
   ```

2. **Rename the repo for each independent run**  
   Clone (or copy) once per agent session so git history stays isolated:
   ```bash
   mv autoresearch autoresearch_<id>   # e.g. autoresearch_opus48max
   cd autoresearch_<id>
   ```

3. **One-time setup** (requires a single NVIDIA GPU, Python 3.10+, [uv](https://github.com/astral-sh/uv))
   ```bash
   uv sync
   uv run prepare.py    # downloads data + trains tokenizer (~2 min)
   ```

4. **Run a coding agent**  
   Point your agent at `program.md` and start an experiment loop, e.g.:
   ```
   Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
   ```
   The agent autonomously edits `train.py`, runs training, logs results, and keeps or discards each change.

5. **Collect artifacts**  
   After a run finishes, copy the final `train.py`, `run.log`, and `results.tsv` into this demo layout (see below).

## Agents in this demo

Four coding agents were run with different models / reasoning settings:

| Folder | Agent | Model / settings |
|--------|-------|------------------|
| `fugu_ultra/` | Fugu Ultra (Codex wrapper) | — |
| `gemini31_pro/` | Gemini | Gemini 3.1 Pro, high reasoning effort |
| `gpt55_xhigh/` | Codex | GPT-5.5, xhigh reasoning effort |
| `opus48_max/` | Claude | Opus 4.8, max reasoning effort |

Each agent folder has numbered subfolders (`1/`, `2/`, `3/`) for **independent replication runs** of the same agent configuration.

Experiment counts per run (`results.tsv` rows, excluding header):

| Agent | Run 1 | Run 2 | Run 3 |
|-------|------:|------:|------:|
| `fugu_ultra` | 146 | 128 | 170 |
| `gemini31_pro` | 123 | 123 | 123 |
| `gpt55_xhigh` | 134 | 135 | 128 |
| `opus48_max` | 146 | 128 | 170 |

## Folder layout

```
autoresearch/
├── fugu_ultra/
│   ├── 1/   train.py  run.log  results.tsv
│   ├── 2/
│   └── 3/
├── gemini31_pro/
│   ├── 1/
│   ├── 2/
│   └── 3/
├── gpt55_xhigh/
│   ├── 1/
│   ├── 2/
│   └── 3/
└── opus48_max/
    ├── 1/
    ├── 2/
    └── 3/
```

## Artifact files

Each `{agent}/{run}/` directory holds three files from one autonomous research session.

### `train.py`

The training script the agent iterates on. In upstream autoresearch this is the **only file the agent may edit** — model architecture, optimizer (Muon + AdamW), hyperparameters, and training loop all live here. The copy in this folder is the **final version** at the end of that run (the best kept state on the agent's branch).

### `run.log`

Full stdout/stderr from the **last** training run:
```bash
uv run train.py > run.log 2>&1
```

The log includes kernel autotuning, step-by-step training output, and a summary block at the end. Extract the key metric with:
```bash
grep "^val_bpb:" run.log
grep "^peak_vram_mb:" run.log
```

Example summary (printed by `train.py` when a run succeeds):
```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

If `grep "^val_bpb:" run.log` returns nothing, the run crashed — inspect the tail of the log for a stack trace.

### `results.tsv`

Tab-separated ledger of **every experiment** the agent tried during that session. Columns:

| Column | Meaning |
|--------|---------|
| `commit` | Short git commit hash for that experiment |
| `val_bpb` | Validation bits per byte achieved (lower is better; `0.0` on crash) |
| `memory_gb` | Peak GPU memory in GB (`peak_vram_mb / 1024`) |
| `status` | `KEEP` — improvement, change retained; `DISCARD` — no improvement, reverted; `CRASH` — run failed |
| `description` | Short note on what the experiment changed |

The agent appends one row per iteration. The best `val_bpb` among `KEEP` rows is the final result for that run.

## Running `train.py` manually

To replay a saved `train.py` outside the agent loop, you need a full autoresearch checkout with data prepared:

```bash
cd autoresearch_<id>
uv sync
uv run prepare.py          # once, if ~/.cache/autoresearch_<id>/ is empty
cp /path/to/artifact/train.py .
uv run train.py > run.log 2>&1
grep "^val_bpb:" run.log
```

Training always stops after ~5 minutes of wall-clock training time (excluding startup/compilation), regardless of hardware.
