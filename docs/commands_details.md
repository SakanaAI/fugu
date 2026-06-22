# Command reference

This reference covers the Fugu one-line install, the flags for the installer and the `codex-fugu` launcher, and how your Codex config is backed up, restored, and protected.

## The one-line install

The one-line install runs a small bootstrap script served at `https://sakana.ai/fugu/install`. The script clones this repository into `~/.fugu`, then runs `~/.fugu/scripts/install.sh`, which pins Codex, deploys the Fugu config, and stores your API key. Anything you pass after `bash` is forwarded straight to `install.sh`, so every installer flag below also works through the one-line command.

```bash
curl -fsSL https://sakana.ai/fugu/install | bash
```

The bootstrap reads two environment variables of its own. The installer's own variables (`SAKANA_API_KEY`, `CODEX_HOME`, `CODEX_INSTALL_DIR`, and the rest) pass straight through to it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FUGU_REPO_URL` | `https://github.com/SakanaAI/fugu.git` | clone source for the repo |
| `FUGU_HOME` | `~/.fugu` | directory the repo is cloned into |

Common forms:

| Situation | Command |
| --- | --- |
| Standard install | `curl -fsSL https://sakana.ai/fugu/install \| bash` |
| Pass installer flags | `curl -fsSL https://sakana.ai/fugu/install \| bash -s -- --yes` |
| Non-interactive or CI | `curl -fsSL https://sakana.ai/fugu/install \| SAKANA_API_KEY=your_key bash -s -- --yes` |
| Custom clone source | `curl -fsSL https://sakana.ai/fugu/install \| FUGU_REPO_URL=<url-or-path> bash` |

`bash -s --` passes the arguments that follow to the script, so `bash -s -- --yes` runs the installer with `--yes`. Place any environment variable right before `bash` so it reaches the installer rather than `curl`. Re-running the command reuses an existing `~/.fugu`, and ongoing updates are handled by `codex-fugu`, so a re-run is rarely needed.

An equivalent that needs no hosted endpoint and shows exactly what runs:

```bash
( git clone https://github.com/SakanaAI/fugu.git ~/.fugu && bash ~/.fugu/scripts/install.sh )
```

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

### Passing Codex arguments

The launcher flags above are read only when they come first. From the first other argument onward, everything is forwarded unchanged to `codex -p fugu`, so any Codex command, subcommand, prompt, or flag generally works through `codex-fugu`. The commands below are just examples:

```bash
codex-fugu resume                  # codex -p fugu resume
codex-fugu "fix the failing test"  # start a new session with a prompt
codex-fugu --no-update resume      # skip the update check, then resume
```

Because forwarding starts at the first non-launcher argument, put any launcher flag before the Codex arguments. Plain Codex flags such as `--model` or `--help` pass straight through, since the launcher only reacts to its own flags listed above.

## Codex version management and session resume

`codex-fugu` and the installer also manage your Codex version. The Fugu configs are verified against a specific Codex version, so on a mismatch the installer offers to switch your Codex binary to that version, and the launcher offers the same reconcile at most once a day. A switch happens only with your consent, either an interactive yes or `--force`.

Codex keeps a per-version session index, so `codex resume` lists different past sessions after a version switch. Your session transcripts under `~/.codex/sessions` are never deleted by a switch. Only which sessions `codex resume` enumerates changes.

Before any switch the installer saves your current session index (the `state`, `memories`, and `goals` `.sqlite` files) into the backup described below. To bring back your earlier `codex resume` list you can either run the Codex version that wrote those sessions, or restore the saved index from a backup:

```bash
cp -p ~/.codex-backups/codex-config-<timestamp>/*.sqlite* ~/.codex/
```

## Config backup, restore, and protection

Before switching the Codex version or making its first edit to `config.toml`, the installer saves a timestamped copy of your existing config to `~/.codex-backups/codex-config-<timestamp>/`. This location sits outside `~/.codex`, so a backup survives even a full `rm -rf ~/.codex`. Each backup holds your `config.toml`, any `*.config.toml`, `auth.json`, other catalog `*.json`, and `*.md` files, the session index (`state`, `memories`, and `goals` `.sqlite` files), plus a `MANIFEST.txt` and a `SHA256SUMS` for verification. The 10 most recent backups are kept. Use `CODEX_BACKUP_KEEP` and `CODEX_BACKUP_ROOT` to change the count and location, or `--no-backup` to skip the step.

To restore a backup, copy it back over your config directory and re-check it:

```bash
rsync -a --exclude MANIFEST.txt --exclude SHA256SUMS ~/.codex-backups/codex-config-<timestamp>/ ~/.codex/
codex doctor   # expect: config.toml parse: ok
```

Your provider settings go into `config.toml` inside managed `# >>> fugu:... >>>` markers, so a re-deploy replaces only that block and leaves the rest of your config untouched. After each edit the installer re-parses the file with `codex doctor`, and if it no longer parses, the change is rolled back automatically. The stored `auth.json` is kept at mode `0600` so your credentials stay private.
