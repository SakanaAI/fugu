# Introduction

This is the official repository for Sakana Fugu. It contains our technical report, `Fugu_technical_report.pdf`, and the tooling to install and launch the Codex CLI with the Fugu models.

## Install Fugu into Codex

Run this one-line command (it works from any directory and leaves your current directory unchanged):

```bash
( git clone https://github.com/SakanaAI/fugu.git ~/.fugu && bash ~/.fugu/scripts/install.sh ) && source ~/.config/fugu/env
```

The command clones this repo to `~/.fugu`, installs and pins a tested Codex CLI version (currently 0.140.0), deploys the Fugu config that wires up the Sakana provider, and prompts once for your Sakana API key. Get an API key at https://platform.torafugu.app/api-keys. After installing, open a new shell (or run `source ~/.config/fugu/env`) so the key is loaded.

This one-line install supports Ubuntu and macOS. On Windows, or if the install does not complete, see https://console.sakana.ai/get-started for manual configuration.

## Run Codex with Fugu

```bash
codex-fugu
```

`codex-fugu` is the launcher installed alongside Codex, so you can run it from any directory. It runs `codex -p fugu` and keeps your config up to date. Use `/model` inside Codex to switch between `fugu` and `fugu-ultra`. To launch without the wrapper, run `codex -p fugu` directly.

## Installer flags

`bash ~/.fugu/scripts/install.sh [flag]`. Run with no flag to install and deploy.

| Flag | What it does |
| --- | --- |
| (none) | Install and pin the Codex CLI, then deploy the Fugu config |
| `--set-key` | Re-prompt for and store the Sakana API key, no redeploy |
| `--remove-config` | Cleanly undo the deployed config |
| `--pinned-version X.Y.Z` | Pin a specific Codex version instead of the default |
| `--force` | Deploy even if the installed Codex version does not match the target |
| `--dry-run` | Show what would happen and change nothing |
| `-y`, `--yes` | Assume yes, for non-interactive use |
| `-h`, `--help` | Full list of flags and environment variables |

Non-interactive install (for CI or provisioning):

```bash
SAKANA_API_KEY=your_key bash ~/.fugu/scripts/install.sh --yes
```

## Launcher flags

`codex-fugu` runs `codex -p fugu` and, at most once a day, checks this repo for config updates and offers to apply them. It never blocks launch, and any arguments you pass go straight to Codex.

| Flag | What it does |
| --- | --- |
| `--status` | Show the installed version, the pinned target, and update state |
| `--set-key` | Rotate the stored Sakana API key |
| `--check` | Check for a config update now instead of waiting for the daily check |
| `--recheck` | Re-enable update prompts you previously dismissed, then check |
| `--no-update` | Skip the update check for this launch |

Set `CODEX_FUGU_NO_UPDATE=1` to turn update checks off for good.
