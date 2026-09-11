#!/usr/bin/env bash
set -euo pipefail

KEY_REGEX='^fish_[0-9a-f]{64}$'
SUPPORT_URL='https://console.sakana.ai/api-keys'
CLAUDE_FUGU_INSTALL_DIR="${CLAUDE_FUGU_INSTALL_DIR:-$HOME/.local/bin}"
CLAUDE_FUGU_ENV_FILE="${CLAUDE_FUGU_ENV_FILE:-$HOME/.claude/.fugu-env}"
CODEX_ENV_FILE="${FUGU_ENV_FILE:-${CODEX_HOME:-$HOME/.codex}/.env}"
ASSUME_YES=0
DRY_RUN=0
RECONFIGURE=0
ACTION=install

if [ -t 2 ]; then
  _C_RED=$'\033[31m'; _C_YEL=$'\033[33m'; _C_GRN=$'\033[32m'; _C_DIM=$'\033[2m'; _C_RST=$'\033[0m'
else
  _C_RED=''; _C_YEL=''; _C_GRN=''; _C_DIM=''; _C_RST=''
fi

log_info() { printf '%s[info]%s  %s\n' "$_C_DIM" "$_C_RST" "$*" >&2; }
log_warn() { printf '%s[warn]%s  %s\n' "$_C_YEL" "$_C_RST" "$*" >&2; }
log_error() { printf '%s[error]%s %s\n' "$_C_RED" "$_C_RST" "$*" >&2; }
log_ok() { printf '%s[ ok ]%s  %s\n' "$_C_GRN" "$_C_RST" "$*" >&2; }
die() { log_error "$1"; exit "${2:-1}"; }

print_usage() {
  cat <<EOF
Usage: install-claude.sh [OPTIONS]

Installs the claude-fugu launcher for Claude Code and stores a Sakana API key.
This installer does not install Codex or deploy Codex config.

Options:
  -y, --yes         Assume yes for non-interactive setup. Requires SAKANA_API_KEY
                    if no key is already configured.
      --set-key     Reconfigure only the stored Sakana API key, then exit.
      --reconfigure Overwrite an existing stored key during install.
      --remove      Remove the installed claude-fugu launcher, then exit.
                    (Keeps ~/.claude/.fugu-env and ~/.claude/.fugu-settings.json.)
      --dry-run     Print intended actions without writing files.
  -h, --help        Show this help and exit.

Environment:
  CLAUDE_FUGU_INSTALL_DIR  Launcher install dir (default: ~/.local/bin).
  CLAUDE_FUGU_ENV_FILE     0600 key store (default: ~/.claude/.fugu-env).
  SAKANA_API_KEY           Non-interactive key source.
EOF
}

parse_args() {
  ASSUME_YES="${CLAUDE_FUGU_ASSUME_YES:-${FUGU_ASSUME_YES:-0}}"
  DRY_RUN="${CLAUDE_FUGU_DRY_RUN:-0}"
  RECONFIGURE="${CLAUDE_FUGU_RECONFIGURE:-0}"
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -y|--yes) ASSUME_YES=1 ;;
      --set-key) ACTION=setkey ;;
      --reconfigure) RECONFIGURE=1 ;;
      --remove) ACTION=remove ;;
      --dry-run) DRY_RUN=1 ;;
      -h|--help) print_usage; exit 0 ;;
      *) print_usage >&2; die "Unknown argument: $1" ;;
    esac
    shift
  done
}

valid_key() { printf '%s' "${1:-}" | grep -qE "$KEY_REGEX"; }

read_key_file() {
  local f="$1" value
  [ -f "$f" ] || return 1
  value="$(sed -n -E 's/^(export[[:space:]]+)?SAKANA_API_KEY=//p' "$f" 2>/dev/null | tail -n1 | tr -d '[:space:]')"
  [ -n "$value" ] || return 1
  printf '%s' "$value"
}

persist_key() {
  local value="$1" tmp dir
  dir="$(dirname "$CLAUDE_FUGU_ENV_FILE")"
  if [ "$DRY_RUN" = 1 ]; then
    log_info "[dry-run] would persist SAKANA_API_KEY to ${CLAUDE_FUGU_ENV_FILE}"
    return 0
  fi
  mkdir -p "$dir"
  chmod 700 "$dir" 2>/dev/null || true
  tmp="$(mktemp "${TMPDIR:-/tmp}/claude-fugu-env.XXXXXX")"
  chmod 600 "$tmp" 2>/dev/null || true
  if [ -f "$CLAUDE_FUGU_ENV_FILE" ]; then
    grep -vE '^(export[[:space:]]+)?SAKANA_API_KEY=' "$CLAUDE_FUGU_ENV_FILE" > "$tmp" 2>/dev/null || true
  fi
  printf 'SAKANA_API_KEY=%s\n' "$value" >> "$tmp"
  mv -f "$tmp" "$CLAUDE_FUGU_ENV_FILE"
  chmod 600 "$CLAUDE_FUGU_ENV_FILE" 2>/dev/null || true
}

setup_key() {
  local value='' source='' tries=0
  if [ "$RECONFIGURE" != 1 ]; then
    if value="$(read_key_file "$CLAUDE_FUGU_ENV_FILE" 2>/dev/null)" && valid_key "$value"; then
      log_info "SAKANA_API_KEY is already configured in ${CLAUDE_FUGU_ENV_FILE}; skipping. Use --reconfigure to change."
      export SAKANA_API_KEY="$value"
      return 0
    fi
  fi

  if [ -n "${SAKANA_API_KEY:-}" ]; then
    valid_key "$SAKANA_API_KEY" || die "SAKANA_API_KEY is set but does not match the expected format."
    value="$SAKANA_API_KEY"
    source='environment'
  elif [ "$RECONFIGURE" != 1 ] && value="$(read_key_file "$CODEX_ENV_FILE" 2>/dev/null)" && valid_key "$value"; then
    source="$CODEX_ENV_FILE"
  elif [ "$ASSUME_YES" = 1 ]; then
    die "SAKANA_API_KEY is required but not set, and --yes was given. Export SAKANA_API_KEY=… or run interactively."
  else
    printf 'A Sakana API key is required for claude-fugu.\n' >&2
    printf 'Get or create one here: %s\n' "$SUPPORT_URL" >&2
    while :; do
      printf 'Paste your SAKANA_API_KEY (input hidden): ' >&2
      if ! IFS= read -rs value; then printf '\n' >&2; die 'No input received for SAKANA_API_KEY.'; fi
      printf '\n' >&2
      value="$(printf '%s' "$value" | tr -d '[:space:]')"
      valid_key "$value" && break
      tries=$((tries + 1)); value=''
      log_warn "That does not look like a valid SAKANA_API_KEY. Try again (${tries}/3)."
      [ "$tries" -ge 3 ] && die 'Too many invalid SAKANA_API_KEY attempts.'
    done
    source='prompt'
  fi

  persist_key "$value"
  export SAKANA_API_KEY="$value"
  if [ "$DRY_RUN" = 1 ]; then
    log_info "[dry-run] would configure SAKANA_API_KEY from ${source}."
  else
    log_ok "SAKANA_API_KEY configured from ${source} (stored 0600 in ${CLAUDE_FUGU_ENV_FILE})."
  fi
}

launcher_source() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

install_launcher() {
  local src dest tmp
  src="$(launcher_source)/claude-fugu"
  dest="$CLAUDE_FUGU_INSTALL_DIR/claude-fugu"
  [ -f "$src" ] || die "Launcher source not found: ${src}"
  if [ "$DRY_RUN" = 1 ]; then
    log_info "[dry-run] would install claude-fugu to ${dest}"
    return 0
  fi
  mkdir -p "$CLAUDE_FUGU_INSTALL_DIR"
  tmp="$(mktemp "$CLAUDE_FUGU_INSTALL_DIR/.claude-fugu.XXXXXX")"
  if cat "$src" > "$tmp" && chmod 755 "$tmp" 2>/dev/null && mv -f "$tmp" "$dest"; then
    log_ok "Installed claude-fugu launcher to ${dest}."
  else
    rm -f "$tmp" 2>/dev/null || true
    die "Could not install claude-fugu to ${dest}."
  fi
  command -v claude-fugu >/dev/null 2>&1 || log_warn "claude-fugu installed to ${CLAUDE_FUGU_INSTALL_DIR}, but that directory is not on PATH."
}

write_settings() {
  local dest="$CLAUDE_FUGU_INSTALL_DIR/claude-fugu"
  if [ "$DRY_RUN" = 1 ]; then
    log_info "[dry-run] would write the claude-fugu settings layer (claude-fugu --refresh-settings)"
    return 0
  fi
  if CLAUDE_FUGU_ENV_FILE="$CLAUDE_FUGU_ENV_FILE" bash "$dest" --refresh-settings; then
    log_ok "Settings layer ready (edit it to customize; re-run this installer or claude-fugu --refresh-settings to reset)."
  else
    log_warn "Could not write the settings layer; claude-fugu will create it on first run."
  fi
}

remove_launcher() {
  local dest="$CLAUDE_FUGU_INSTALL_DIR/claude-fugu"
  if [ ! -e "$dest" ]; then
    log_info "No claude-fugu launcher at ${dest}."
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    log_info "[dry-run] would remove ${dest}"
    return 0
  fi
  rm -f "$dest"
  log_ok "Removed ${dest}."
}

main() {
  parse_args "$@"
  case "$ACTION" in
    remove) remove_launcher; exit 0 ;;
    setkey) RECONFIGURE=1; setup_key; exit 0 ;;
  esac
  setup_key
  install_launcher
  write_settings
  if command -v claude >/dev/null 2>&1; then
    log_ok "Claude Code found. Run: claude-fugu"
  else
    log_warn "Claude Code is not on PATH. Install it first: https://claude.com/claude-code"
    log_ok "claude-fugu installer complete."
  fi
}

main "$@"
