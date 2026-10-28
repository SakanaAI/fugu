# Command reference

This reference covers the Fugu one-line install, the flags for the installer and the `codex-fugu` launcher, and how your Codex config is backed up, restored, and protected.

## The one-line install

The one-line install runs a small bootstrap script served at `https://sakana.ai/fugu/install`. The script clones this repository into `~/.fugu`, then asks what to set up, **Codex** (`codex-fugu`), **Claude Code** (`claude-fugu`), or **both**, and runs the matching installer: `~/.fugu/scripts/install.sh` (pins Codex, deploys the Fugu config, stores your API key) and/or `~/.fugu/scripts/install-claude.sh` (installs the Claude launcher and its settings file, reusing a production key that `install.sh` stored). Choose non-interactively with a `--codex`/`--claude`/`--both` flag or `FUGU_INSTALL_TARGET`; any other flag after `bash` is forwarded to the selected installer, so every installer flag below also works through the one-line command.

```bash
curl -fsSL https://sakana.ai/fugu/install | bash
```

The bootstrap reads a few environment variables of its own. The installers' own variables (`SAKANA_API_KEY`, `CODEX_HOME`, `CODEX_INSTALL_DIR`, and the rest) pass straight through.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FUGU_REPO_URL` | `https://github.com/SakanaAI/fugu.git` | clone source for the repo |
| `FUGU_HOME` | `~/.fugu` | directory the repo is cloned into |
| `FUGU_INSTALL_TARGET` | interactive menu (`codex` when headless) | which agent to install: `codex`, `claude`, or `both` (also `1`/`2`/`3`) |

Common forms:

| Situation | Command |
| --- | --- |
| Standard install (menu) | `curl -fsSL https://sakana.ai/fugu/install \| bash` |
| Claude Code only | `curl -fsSL https://sakana.ai/fugu/install \| bash -s -- --claude` |
| Both | `curl -fsSL https://sakana.ai/fugu/install \| bash -s -- --both` |
| Pass installer flags | `curl -fsSL https://sakana.ai/fugu/install \| bash -s -- --yes` |
| Non-interactive or CI | `curl -fsSL https://sakana.ai/fugu/install \| SAKANA_API_KEY=your_key FUGU_INSTALL_TARGET=both bash -s -- --yes` |
| Custom clone source | `curl -fsSL https://sakana.ai/fugu/install \| FUGU_REPO_URL=<url-or-path> bash` |

`bash -s --` passes the arguments that follow to the script. The bootstrap consumes `--codex`/`--claude`/`--both` to choose the target; every other flag (like `--yes`) goes to the selected installer (with `--both`, the Claude Code installer receives only `--yes` and `--dry-run`). Place any environment variable right before `bash` so it reaches the bootstrap rather than `curl`. Re-running the command reuses an existing `~/.fugu`, and ongoing updates are handled by `codex-fugu`, so a re-run is rarely needed.

**If your first install failed partway through**, re-running the one-liner will not help on its own: it reuses the `~/.fugu` it already cloned, so it runs the same copy of the installer again. Remove the clone and start clean:

```bash
rm -rf ~/.fugu && curl -fsSL https://sakana.ai/fugu/install | bash
```

To run the same steps without the hosted endpoint (cloning the repository yourself and seeing exactly what runs), clone it once and run the installer for what you want:

```bash
git clone https://github.com/SakanaAI/fugu.git ~/.fugu
bash ~/.fugu/scripts/install.sh          # Codex (codex-fugu)
bash ~/.fugu/scripts/install-claude.sh   # Claude Code (claude-fugu)
```

To set up both, run `install.sh` first; `install-claude.sh` then reuses the production (`fish_…`) API key it stored. A custom endpoint set with `--base-url` applies to Codex only: `claude-fugu` uses the production endpoint unless `CLAUDE_FUGU_BASE_URL` is set, and `install-claude.sh` reuses only a production key, so it asks for one when Codex is set up with another endpoint's key.

## Installer flags

Two installers ship in the repo: `install.sh` for Codex and `install-claude.sh` for Claude Code. Run either with no flag to install.

### Codex (`install.sh`)

`bash ~/.fugu/scripts/install.sh [flag]`. Run with no flag to install and deploy.

| Flag | What it does |
| --- | --- |
| (none) | Install and pin the Codex CLI, then deploy the Fugu config |
| `--set-key` | Re-prompt for and store the Sakana API key, checked against the endpoint the installer set up (production unless it saved and recorded a custom one). On an existing install, combine with `--base-url URL` to update the endpoint and matching key together; this replaces only the provider block, not the rest of the bundle. |
| `--reconfigure` | Reacquire the API key and redeploy. On a terminal (without `--yes` or `--base-url`), first offer the saved API base URL (production unless a custom one is saved; not asked when your own `[model_providers.sakana]` table sets the endpoint) |
| `--remove-config` | Cleanly undo the deployed config |
| `--pinned-version X.Y.Z` | Pin a specific Codex version instead of the default |
| `--model MODEL` | Set the profile's default model to a slug in the shipped catalog and keep it across updates; `--model default` follows the bundle's default again |
| `--base-url URL` | Set the Sakana API endpoint for Codex and redeploy, keeping a stored key that fits the new endpoint. `https://api.sakana.ai/v1` returns to production |
| `--force` | Deploy even if the installed Codex version does not match the target |
| `--dry-run` | Show what would happen and change nothing |
| `-y`, `--yes` | Assume yes, for non-interactive use |
| `-h`, `--help` | Full list of flags and environment variables |

Non-interactive install (for CI or provisioning); `--base-url` and `--model` are optional and work the same on a fresh or an existing install:

```bash
SAKANA_API_KEY=your_key bash ~/.fugu/scripts/install.sh --yes [--base-url URL] [--model MODEL]
```

An ordinary install uses `https://api.sakana.ai/v1` without adding an endpoint prompt. To use a dev or
self-hosted service, pass `--base-url URL`; the `SAKANA_BASE_URL` environment variable works too, but only on a
fresh install or together with `--reconfigure`. The URL is checked before Codex is installed or switched,
trailing slashes and default ports are dropped, any spelling of `https://api.sakana.ai/v1` counts as
production, and a plain `http://` URL to a host that is not local prints a warning because the key would
travel unencrypted. Only a custom URL is stored with the key in `~/.codex/.env`. Changing the endpoint this way
keeps a stored key that fits the new endpoint, which Codex then sends there, and the installer warns about it;
to change both together, run `install.sh --set-key --base-url URL`. On a terminal, `--reconfigure` offers the saved
URL before rotating the key (an exported `SAKANA_BASE_URL` becomes the offered default; with your own provider
table, which Fugu does not edit, it does not ask). The installer records the custom endpoint it saves in
`~/.codex/.fugu/state` and honours the `SAKANA_BASE_URL` line in `~/.codex/.env` only while it names that
endpoint. A line the installer did not write (for example one added for another tool) is ignored
with a warning and left in place, unless you pass `--base-url URL`, which replaces it and says so. If you edit or
delete the saved line, the installer stops before touching Codex and asks you to choose an endpoint with
`--base-url URL` (the message shows the saved one); that line is the installer's, so choosing production removes it.
A hand edit of the `base_url` in Fugu's provider block is undone by the next update, which renders the saved
endpoint again (production when none is saved, unless your `.env` line names the edited URL: that pair is kept).

The default model is `fugu-max`. Choose another catalog model with `--model MODEL`, or on a fresh install /
`--reconfigure` run with the `FUGU_MODEL` environment variable; the CLI flag wins if both are supplied. The installer records the
explicit selection so future config updates preserve it, until you run `--model default` to follow the bundle's
default again. If no model is selected, the bundle default is
not pinned, allowing future default changes to reach existing users. Runtime `/model` switching remains available.
Ordinary updates of an existing install ignore `SAKANA_BASE_URL` and `FUGU_MODEL` from the environment and
print a warning that names the matching flag (`--base-url` / `--model`); `--set-key` ignores them too, so
unrelated shell settings are never persisted. An exported `SAKANA_BASE_URL` that is already the endpoint in use is
ignored without a warning.

### Claude Code (`install-claude.sh`)

`bash ~/.fugu/scripts/install-claude.sh [flag]`. This installs the `claude-fugu` launcher, stores a Sakana API key, and writes the launcher's settings file; it does not install Codex or deploy any Codex config. With no flag it configures the key (reusing an existing one from `~/.claude/.fugu-env`, or a production key from `~/.codex/.env` if you already set up Codex, otherwise prompting), installs the launcher to `~/.local/bin/claude-fugu`, then writes its settings file `~/.claude/.fugu-settings.json`.

| Flag | What it does |
| --- | --- |
| (none) | Configure the Sakana API key, install the `claude-fugu` launcher, then write its settings file |
| `--set-key` | Reconfigure the stored Sakana API key only; does not install the launcher |
| `--reconfigure` | During install, overwrite an existing stored key instead of keeping it |
| `--remove` | Remove the installed `claude-fugu` launcher (the stored key and the settings file are kept) |
| `--dry-run` | Show what would happen and change nothing |
| `-y`, `--yes` | Assume yes, for non-interactive use (a key must already be available) |
| `-h`, `--help` | Full list of flags and environment variables |

It has no `--remove-config`, `--pinned-version`, `--force`, or `--no-backup`: it deploys no config bundle, never installs or version-pins Codex, and always backs up a settings file that differs from the defaults before resetting it.

## Launcher flags

> **The flags below are `codex-fugu`-only.** The `claude-fugu` launcher has one flag of its own, recognized only as the first argument: `claude-fugu --refresh-settings` resets its settings file `~/.claude/.fugu-settings.json` to the defaults (backing up your copy first if it differs) and exits without starting Claude Code. Otherwise every argument is forwarded straight to Claude Code, after that settings file (`claude --settings ~/.claude/.fugu-settings.json "$@"`). To reconfigure the Claude key or reinstall its launcher, use `install-claude.sh` (`--set-key`, `--reconfigure`, `--remove`).

`codex-fugu` runs `codex -p fugu` and, at most once an hour, checks this repo for config updates and offers to apply them. It never blocks launch, and any arguments you pass go straight to Codex.

| Flag | What it does |
| --- | --- |
| `--status` | Show the installed version, the pinned target, the default model (pinned or bundle default), the API endpoint, and update state |
| `--set-key` | Rotate the stored Sakana API key; `--set-key --base-url URL` moves the endpoint and its key together (any other argument after `--set-key`, apart from the launcher's own flags, is refused with exit 2 and is not printed back) |
| `--check` | Check for a config update now instead of waiting for the hourly check |
| `--recheck` | Re-enable update prompts you previously dismissed, then exit; the next launch checks again |
| `--no-update` | Skip the update check for this launch |

Set `CODEX_FUGU_NO_UPDATE=1` to turn update checks off for good.

### Launch notices

Now and then the launcher shows a short notice before it starts Codex, with two choices: "acknowledge and continue" (the default) and "acknowledge and never remind me again". Choosing the second one hides that notice for good. There is at most one notice at a time. Set `CODEX_FUGU_NO_NOTICE=1` to turn notices off entirely.

### Passing Codex arguments

The launcher flags above are read only when they come first. From the first other argument onward, everything is forwarded unchanged to `codex -p fugu`, so any Codex command, subcommand, prompt, or flag generally works through `codex-fugu`. The commands below are just examples:

```bash
codex-fugu resume                  # codex -p fugu resume
codex-fugu "fix the failing test"  # start a new session with a prompt
codex-fugu --no-update resume      # skip the update check, then resume
```

Because forwarding starts at the first non-launcher argument, put any launcher flag before the Codex arguments. Plain Codex flags such as `--model` or `--help` pass straight through, since the launcher only reacts to its own flags listed above.

## Version management and session resume

### Codex (`codex-fugu`)

`codex-fugu` and the installer also manage your Codex version. The Fugu configs are verified against a specific Codex version, so on a mismatch the installer offers to switch your Codex binary to that version, and the launcher offers the same reconcile on each interactive launch. A switch happens only with your consent, either an interactive yes or `--force`.

Codex keeps a per-version session index, so `codex resume` lists different past sessions after a version switch. Your session transcripts under `~/.codex/sessions` are never deleted by a switch. Only which sessions `codex resume` enumerates changes.

Before any switch the installer saves your current session index (the `state`, `memories`, and `goals` `.sqlite` files) into the backup described below. To bring back your earlier `codex resume` list you can either run the Codex version that wrote those sessions, or restore the saved index from a backup:

```bash
cp -p ~/.codex-backups/codex-config-<timestamp>/*.sqlite* ~/.codex/
```

### Claude Code (`claude-fugu`)

`claude-fugu` does not manage the Claude Code version or touch session state. It checks that `claude` is on your `PATH` (and exits with an install hint if it is not), creates its settings file `~/.claude/.fugu-settings.json` if it is missing (otherwise it leaves your edits alone and only updates the lines it owns, such as the API key when your key changes), then execs Claude Code with that file. It never pins or checks the Claude Code version, runs no update check, and keeps no session index; Claude Code manages its own `/model` selection and session resume, so there is no version-switch resume caveat like Codex's.

Because the launcher does not auto-update, you pick up new Fugu support for Claude Code (such as new Fugu models) by re-running the installer, which reinstalls the launcher and resets its settings file to the new defaults (backing up your copy first if it differs):

```bash
bash ~/.fugu/scripts/install-claude.sh
```

## Config backup, restore, and protection

### Codex (`install.sh`)

Before switching the Codex version or making its first edit to `config.toml`, the installer saves a timestamped copy of your existing config to `~/.codex-backups/codex-config-<timestamp>/`. This location sits outside `~/.codex`, so a backup survives even a full `rm -rf ~/.codex`. Each backup holds your `config.toml`, any `*.config.toml`, `auth.json`, other catalog `*.json`, and `*.md` files, the session index (`state`, `memories`, and `goals` `.sqlite` files), plus a `MANIFEST.txt` and a `SHA256SUMS` for verification. The 10 most recent backups are kept. Use `CODEX_BACKUP_KEEP` and `CODEX_BACKUP_ROOT` to change the count and location, or `--no-backup` to skip the step.

To restore a backup, copy it back over your config directory and re-check it:

```bash
rsync -a --exclude MANIFEST.txt --exclude SHA256SUMS ~/.codex-backups/codex-config-<timestamp>/ ~/.codex/
codex doctor   # expect: config.toml parse: ok
```

Your provider settings go into `config.toml` inside managed `# >>> fugu:... >>>` markers, so a re-deploy replaces only that block and leaves the rest of your config untouched. After each edit the installer re-parses the file with `codex doctor`, and if it no longer parses, the change is rolled back automatically. If you already have your own `[model_providers.sakana]` table outside those markers, the installer leaves it alone and warns; that table's endpoint stays in effect and your API key is still stored (checked against the production key format unless the installer saved that table's URL as your endpoint; pass `--base-url` with that URL once to save it), but asking for a different endpoint stops with an error, before anything is written (for `--base-url`, before Codex is installed or switched), until you merge that table by hand. The stored `auth.json` is kept at mode `0600` so your credentials stay private.

### Claude Code (`claude-fugu`)

`install-claude.sh` deploys no config bundle and never edits `~/.claude/settings.json` or any other Claude Code config except the launcher's own settings file. Claude Code is pointed at Fugu through the `ANTHROPIC_*` environment variables the launcher sets at launch and that settings file, which the launcher passes with `--settings`, so your own Claude Code settings are never modified. Under `~/.claude` the installer writes two files of its own, both mode `0600`: the API-key store `~/.claude/.fugu-env` (in a `0700` directory) and the settings file `~/.claude/.fugu-settings.json`, which holds the Fugu endpoint, model list, default model and effort level, and a copy of your API key. When the key store already exists, only the `SAKANA_API_KEY` line is rewritten and any other lines are preserved. If no key is stored yet, an existing production Codex key in `~/.codex/.env` is reused, read-only, and that file is never modified.

The settings file is yours to edit after install: a normal launch leaves your changes in place and only rewrites the lines the launcher owns, such as the API key if your key has changed. Re-running the installer, or running `claude-fugu --refresh-settings`, resets it to the launcher's defaults; if your copy differs, it is first saved as `~/.claude/.fugu-settings.json.bak.<timestamp>.<random>`, and copying that file back over `~/.claude/.fugu-settings.json` restores it. That backup is automatic, so there is no `--no-backup` flag. `install-claude.sh --remove` deletes only the launcher and keeps the key store and the settings file. Both of these files, and any settings backups, contain your API key, so to remove the key too, run `rm -f ~/.claude/.fugu-env ~/.claude/.fugu-settings.json*`.
