#!/usr/bin/env bash
set -euo pipefail

readonly OFFICIAL_INSTALL_URL="https://chatgpt.com/codex/install.sh"
readonly RELEASE_INSTALL_BASE="https://github.com/openai/codex/releases/download"
readonly GH_LATEST_API="https://api.github.com/repos/openai/codex/releases/latest"
readonly NPM_REGISTRY_URL="https://registry.npmjs.org/@openai/codex"
readonly SUPPORT_URL="https://console.sakana.ai/get-started"
readonly DEFAULT_SAKANA_BASE_URL="https://api.sakana.ai/v1"

FUGU_OS="$(uname -s 2>/dev/null || echo unknown)"
_is_macos() { [ "$FUGU_OS" = "Darwin" ]; }

if ! _is_macos && command -v sha256sum >/dev/null 2>&1; then
  FUGU_SHA256="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  FUGU_SHA256="shasum -a 256"
elif command -v sha256sum >/dev/null 2>&1; then
  FUGU_SHA256="sha256sum"
else
  FUGU_SHA256=""
fi

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
CODEX_INSTALL_DIR="${CODEX_INSTALL_DIR:-$HOME/.local/bin}"
CODEX_BACKUP_ROOT="${CODEX_BACKUP_ROOT:-$HOME/.codex-backups}"
CODEX_BACKUP_KEEP="${CODEX_BACKUP_KEEP:-10}"

FUGU_CONFIGS_DIR="${FUGU_CONFIGS_DIR:-}"
FUGU_ENV_FILE="${FUGU_ENV_FILE:-$CODEX_HOME/.env}"
SAKANA_BASE_URL="${SAKANA_BASE_URL:-}"
FUGU_MODEL="${FUGU_MODEL:-}"
readonly DEFAULT_FUGU_MODEL="fugu-max"
readonly FUGU_KEY_PROMPT_TRIES=3
readonly BUNDLE_ID="configs"

HTTP_TOOL=""
RESULT_STATUS=""
LAST_BACKUP_DIR=""
ASSUME_YES=0
PINNED_VERSION_OVERRIDE=""
SKIP_BACKUP=0
DRY_RUN=0
CONFIG_NAME=""
CONFIG_ACTION=""
RECONFIGURE=0
BUNDLE_WARNINGS=0
BUNDLE_PROVIDER_DEPLOY_FAILED=0
BUNDLE_DIR=""
BUNDLE_CODEX_VERSION=""
BUNDLE_DEFAULT_MODEL=""
BUNDLE_MODEL_CATALOG=""
FUGU_MODEL_RESOLVED=0
FUGU_MODEL_EXPLICIT=0
FUGU_MODEL_CLI_EXPLICIT=0
SAKANA_BASE_URL_CLI_EXPLICIT=0
SAKANA_BASE_URL_BEFORE=""
SAKANA_BASE_URL_LINE_OWNED=0
SAKANA_BASE_URL_REQUESTED=0
SAKANA_BASE_URL_RECORD=""
FUGU_MODEL_PINNED=0
ENV_KEYS_TO_PERSIST=()
FORCE=0
_RUN_BACKUP_DONE=0

DEPLOYED_FORMAT=""
FORMAT_LATEST_ID=""
MIGRATE_KEEP_MARKERS=""
MIGRATE_KEEP_DESTS=""
FORMAT_SCHEMA=""; FORMAT_ID=""; FORMAT_LATEST=""; FORMAT_MIN=""; FORMAT_MAX=""
FORMAT_FILES=(); FORMAT_INJECTS=()

if [ -t 2 ]; then
  _C_RED=$'\033[31m'; _C_YEL=$'\033[33m'; _C_GRN=$'\033[32m'; _C_DIM=$'\033[2m'; _C_RST=$'\033[0m'
else
  _C_RED=""; _C_YEL=""; _C_GRN=""; _C_DIM=""; _C_RST=""
fi

log_info()  { printf '%s[info]%s  %s\n'  "$_C_DIM" "$_C_RST" "$*" >&2; }
log_warn()  { printf '%s[warn]%s  %s\n'  "$_C_YEL" "$_C_RST" "$*" >&2; }
log_error() { printf '%s[error]%s %s\n'  "$_C_RED" "$_C_RST" "$*" >&2; }
log_ok()    { printf '%s[ ok ]%s  %s\n'  "$_C_GRN" "$_C_RST" "$*" >&2; }

die() { log_error "$1"; exit "${2:-1}"; }

confirm() {
  local prompt="$1" reply=""
  [ "${ASSUME_YES:-0}" = "1" ] && return 0
  printf '%s [y/N]: ' "$prompt" >&2
  if IFS= read -r reply; then :; else reply=""; fi
  case "$reply" in
    [Yy] | [Yy][Ee][Ss]) return 0 ;;
    *) return 1 ;;
  esac
}

require_http_client() {
  if command -v curl >/dev/null 2>&1; then
    HTTP_TOOL="curl"
  elif command -v wget >/dev/null 2>&1; then
    HTTP_TOOL="wget"
  else
    die "Neither curl nor wget is available; cannot download Codex."
  fi
}

http_get() {
  local url="$1"
  if [ "$HTTP_TOOL" = "curl" ]; then
    curl -fsSL --max-time 30 "$url"
  else
    wget -qO- --timeout=30 "$url"
  fi
}

release_installer_url() { printf '%s/rust-v%s/install.sh' "$RELEASE_INSTALL_BASE" "$1"; }

print_usage() {
  cat <<EOF
Usage: install.sh [OPTIONS]

Ensures the Codex CLI is installed at the pinned version, then deploys the Fugu
config bundle (configs/) that wires up the Sakana provider.

Options:
  -y, --yes                 Assume "yes" to all prompts (non-interactive).
      --pinned-version X.Y.Z Pin a specific Codex version instead of the bundle's.
      --model MODEL         Set (and keep) the default Fugu model (must exist in fugu.json);
                            --model default follows the bundle's default again.
      --base-url URL        Set the Sakana API endpoint (http:// or https://);
                            https://api.sakana.ai/v1 returns to production.
      --no-backup           Do not back up existing config before switching.
      --dry-run             Resolve + detect + print intended actions only.
      --remove-config       Reverse the deployed config bundle, then exit.
      --set-key             Re-prompt for + persist the bundle's API key(s), then exit
                            (no version-pin/deploy; secure 0600 store in $CODEX_HOME/.env).
                            With --base-url URL, also move the provider block to URL
                            (config.toml only), so the key and endpoint change together.
      --reconfigure         Re-prompt / overwrite even if already configured; on a
                            terminal, also offers the API endpoint.
      --force               Deploy the bundle even if Codex isn't at the bundle's
                            target version (also authorizes a non-interactive switch).
  -h, --help                Show this help and exit.

Environment:
  CODEX_HOME                Codex config dir (default: ~/.codex).
  CODEX_INSTALL_DIR         Codex binary dir (default: ~/.local/bin).
  CODEX_BACKUP_ROOT         Where config backups are written (default: ~/.codex-backups).
  CODEX_BACKUP_KEEP         Number of backups to retain (default: 10).
  CODEX_LATEST_OVERRIDE     Force the resolved "latest" version (testing).
  FUGU_PINNED_VERSION / CODEX_RELEASE  Same as --pinned-version.
  FUGU_ASSUME_YES=1         Same as --yes.
  FUGU_FORCE=1              Same as --force.
  FUGU_CONFIGS_DIR          Bundles root (default: <repo>/configs).
  FUGU_ENV_FILE             0600 secret store, dotenvy KEY=VALUE (default: ~/.codex/.env).
  FUGU_MODEL                Default model; honoured on a fresh install or with --reconfigure
                            (an existing install ignores it: use --model).
  SAKANA_BASE_URL           API endpoint; honoured on a fresh install or with --reconfigure
                            (an existing install, --set-key and --remove-config ignore it).
  <PROVIDER_KEY>=...        Provide a bundle's API key non-interactively (e.g. SAKANA_API_KEY).
EOF
}

parse_args() {
  ASSUME_YES="${FUGU_ASSUME_YES:-0}"
  PINNED_VERSION_OVERRIDE="${FUGU_PINNED_VERSION:-${CODEX_RELEASE:-}}"
  SKIP_BACKUP="${FUGU_SKIP_BACKUP:-0}"
  DRY_RUN="${FUGU_DRY_RUN:-0}"
  RECONFIGURE="${FUGU_RECONFIGURE:-0}"
  FORCE="${FUGU_FORCE:-0}"
  CONFIG_NAME="$BUNDLE_ID"
  CONFIG_ACTION="install"
  [ -n "${FUGU_MODEL:-}" ] && FUGU_MODEL_EXPLICIT=1
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -y | --yes) ASSUME_YES=1 ;;
      --pinned-version) shift; [ "$#" -ge 1 ] || die "--pinned-version requires a value"; PINNED_VERSION_OVERRIDE="$1" ;;
      --pinned-version=*) PINNED_VERSION_OVERRIDE="${1#*=}" ;;
      --model) shift; [ "$#" -ge 1 ] || die "--model requires a value"; FUGU_MODEL="$1"; FUGU_MODEL_EXPLICIT=1; FUGU_MODEL_CLI_EXPLICIT=1 ;;
      --model=*) FUGU_MODEL="${1#*=}"; FUGU_MODEL_EXPLICIT=1; FUGU_MODEL_CLI_EXPLICIT=1 ;;
      --base-url) shift; [ "$#" -ge 1 ] || die "--base-url requires a value"; SAKANA_BASE_URL="$1"; SAKANA_BASE_URL_CLI_EXPLICIT=1 ;;
      --base-url=*) SAKANA_BASE_URL="${1#*=}"; SAKANA_BASE_URL_CLI_EXPLICIT=1 ;;
      --no-backup) SKIP_BACKUP=1 ;;
      --dry-run) DRY_RUN=1 ;;
      --remove-config) CONFIG_ACTION="remove" ;;
      --set-key) CONFIG_ACTION="setkey" ;;
      --reconfigure) RECONFIGURE=1 ;;
      --force) FORCE=1 ;;
      -h | --help) print_usage; exit 0 ;;
      *) print_usage >&2; die "Unknown argument: $1" ;;
    esac
    shift
  done

  if [ "$CONFIG_ACTION" = "install" ] && [ -f "$CODEX_HOME/.fugu/state" ] && [ "$RECONFIGURE" != "1" ]; then
    if [ "$FUGU_MODEL_CLI_EXPLICIT" != "1" ]; then
      [ -z "$FUGU_MODEL" ] || log_warn "Ignoring FUGU_MODEL from the environment on an existing install; use --model MODEL (or --reconfigure) to change the default model."
      FUGU_MODEL=""
      FUGU_MODEL_EXPLICIT=0
    fi
    if [ "$SAKANA_BASE_URL_CLI_EXPLICIT" != "1" ]; then
      [ -z "$SAKANA_BASE_URL" ] || _ambient_sakana_base_url_in_effect || \
        log_warn "Ignoring SAKANA_BASE_URL from the environment on an existing install; use --base-url URL (or --reconfigure) to change the endpoint."
      SAKANA_BASE_URL=""
    fi
  fi
  if [ "$CONFIG_ACTION" != "install" ]; then
    FUGU_MODEL=""; FUGU_MODEL_EXPLICIT=0
    if [ "$SAKANA_BASE_URL_CLI_EXPLICIT" != "1" ]; then
      [ -z "$SAKANA_BASE_URL" ] || [ "$CONFIG_ACTION" != "setkey" ] || _ambient_sakana_base_url_in_effect || \
        log_warn "Ignoring SAKANA_BASE_URL from the environment; use --set-key --base-url URL to change the endpoint and key together."
      SAKANA_BASE_URL=""
    fi
  fi
}

_ambient_sakana_base_url_in_effect() {
  [ "$(_canon_sakana_base_url "$SAKANA_BASE_URL")" = "$(_sakana_base_url_before)" ]
}

normalize_version() {
  local v="${1:-}"
  v="${v#rust-v}"
  v="${v#codex-cli }"
  v="${v#v}"
  v="$(printf '%s' "$v" | tr -d '[:space:]')"
  if printf '%s' "$v" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.]+)?$'; then
    printf '%s' "$v"
    return 0
  fi
  return 1
}

_ver_lt() { [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$2" ]; }

resolve_latest_version() {
  local json raw ver
  if [ -n "${PINNED_VERSION_OVERRIDE:-}" ]; then
    ver="$(normalize_version "$PINNED_VERSION_OVERRIDE")" \
      || die "Invalid --pinned-version / FUGU_PINNED_VERSION: '${PINNED_VERSION_OVERRIDE}'"
    printf '%s' "$ver"; return 0
  fi
  if [ -n "${CODEX_LATEST_OVERRIDE:-}" ]; then
    ver="$(normalize_version "$CODEX_LATEST_OVERRIDE")" \
      || die "Invalid CODEX_LATEST_OVERRIDE: '${CODEX_LATEST_OVERRIDE}'"
    printf '%s' "$ver"; return 0
  fi

  if json="$(http_get "$GH_LATEST_API" 2>/dev/null)"; then
    raw="$(printf '%s\n' "$json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"rust-v\([^"]*\)".*/\1/p' | head -n1)"
    if ver="$(normalize_version "$raw" 2>/dev/null)"; then printf '%s' "$ver"; return 0; fi
  fi
  log_warn "GitHub latest-version lookup failed or unparseable; trying the npm registry…"
  if json="$(http_get "$NPM_REGISTRY_URL" 2>/dev/null)"; then
    raw="$(printf '%s\n' "$json" | grep -o '"latest"[[:space:]]*:[[:space:]]*"[^"]*"' | head -n1 | sed -E 's/.*"([0-9][^"]*)".*/\1/')"
    if ver="$(normalize_version "$raw" 2>/dev/null)"; then printf '%s' "$ver"; return 0; fi
  fi
  die "Could not resolve the latest Codex version from official sources. Re-run with --pinned-version X.Y.Z."
}

codex_is_installed() { command -v codex >/dev/null 2>&1; }

_parse_codex_version() {
  local bin="${1:-codex}" out ver
  out="$("$bin" --version 2>/dev/null || true)"
  ver="$(printf '%s\n' "$out" | grep -oE 'codex-cli[[:space:]]+[0-9][0-9.]*' | head -n1 | awk '{print $2}' || true)"
  printf '%s' "$ver"
}

get_installed_version() {
  if [ -x "$CODEX_INSTALL_DIR/codex" ]; then _parse_codex_version "$CODEX_INSTALL_DIR/codex"
  else _parse_codex_version codex; fi
}

verify_installed_version() {
  local want="$1" got
  got="$(get_installed_version)"
  if [ -z "$got" ] && [ -x "$CODEX_INSTALL_DIR/codex" ]; then
    got="$(_parse_codex_version "$CODEX_INSTALL_DIR/codex")"
  fi
  [ "$got" = "$want" ] || die "Post-install verification failed: expected Codex ${want}, found '${got:-none}'."
  local path_ver; path_ver="$(_parse_codex_version codex)"
  if [ -z "$path_ver" ]; then
    log_warn "Codex installed to ${CODEX_INSTALL_DIR}, but it is not on your PATH."
    log_warn "Add it with:  export PATH=\"${CODEX_INSTALL_DIR}:\$PATH\"   (or open a new shell)."
  elif [ "$path_ver" != "$want" ]; then
    log_warn "Another Codex (${path_ver}) is ahead of ${CODEX_INSTALL_DIR} on your PATH and shadows the pinned ${want}."
    log_warn "codex-fugu uses the pinned ${want}; plain codex / codex -p fugu would use ${path_ver}."
    log_warn "Remove the other install or put ${CODEX_INSTALL_DIR} first on PATH."
  fi
}

explain_install_failure() {
  local version="$1" rc="$2" out="$3" src="$4" _self
  _self="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || echo .)/$(basename "${BASH_SOURCE[0]}")"
  log_warn "Codex ${version} install failed: the official installer (${src}) exited ${rc}. Its own output is above."
  case "$out" in
    *codex-code-mode-host* | *"chmod:"*)
      log_warn "Cause: the installer tried to set permissions on a file that the Codex ${version} package does not contain."
      log_warn "This is an upstream installer/packaging mismatch — not a network, rate-limit, permissions, or Fugu config problem."
      log_info "Fugu installs each pinned Codex with that release's own installer, so please report this with the output above: ${SUPPORT_URL}"
      ;;
    *"Could not fetch GitHub release metadata"* | *"rate limit"* | *"rate limited"*)
      log_warn "Cause: GitHub could not serve the release metadata (typically an API rate limit). This one IS transient."
      log_info "Wait a few minutes, then retry with:  bash ${_self}"
      ;;
    *"Could not find Codex package"*)
      log_warn "Cause: the installer could not resolve the release assets for Codex ${version}."
      log_warn "Usually this is GitHub rate-limiting the lookup — its unauthenticated API allows only 60 requests/hour per IP — which is transient. Less often, the release genuinely ships no package for this platform."
      log_info "Wait a few minutes, then retry with:  bash ${_self}"
      ;;
    *)
      log_warn "Fugu could not classify this failure; the installer's own output above is the authoritative error."
      ;;
  esac
  log_info "Nothing was deployed and your existing Codex config was not modified (any pre-switch backup is shown above)."
  die "Codex install aborted. More help: ${SUPPORT_URL}"
}

install_codex() {
  local version="$1"
  log_info "Installing Codex CLI ${version} via the official installer…"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    log_info "[dry-run] would install Codex ${version} into ${CODEX_INSTALL_DIR} (CODEX_HOME=${CODEX_HOME})."
    return 0
  fi
  export CODEX_HOME CODEX_INSTALL_DIR
  mkdir -p "$CODEX_INSTALL_DIR" "$CODEX_HOME"
  export TMPDIR="${TMPDIR:-/tmp}"
  export CODEX_NON_INTERACTIVE=1

  local _ic_rc=0 _ic_out="" _ic_src=""
  if [ -n "${CODEX_INSTALLER_CMD:-}" ]; then
    _ic_src="test-hook"
    _ic_out="$(CODEX_RELEASE="$version" "$CODEX_INSTALLER_CMD" "$version" 2>&1)" || _ic_rc=$?
    if [ -n "$_ic_out" ]; then printf '%s\n' "$_ic_out" >&2; fi
  else
    local _sh="${TMPDIR}/fugu-codex-installer.$$.sh" _log="${TMPDIR}/fugu-codex-installer.$$.log"
    if http_get "$(release_installer_url "$version")" > "$_sh" 2>/dev/null && [ -s "$_sh" ]; then
      _ic_src="rust-v${version}"
    elif http_get "$OFFICIAL_INSTALL_URL" > "$_sh" 2>/dev/null && [ -s "$_sh" ]; then
      _ic_src="latest"
      log_warn "Codex ${version} publishes no installer of its own; falling back to the latest official installer."
      log_warn "That installer is not guaranteed to support this pinned version — if the install fails, this is the likely cause."
    else
      rm -f "$_sh"
      die "Could not download a Codex installer for ${version}. Check your network and retry. More help: ${SUPPORT_URL}"
    fi
    if sh "$_sh" --release "$version" </dev/null 2>&1 | tee "$_log" >&2; then _ic_rc=0; else _ic_rc=$?; fi
    _ic_out="$(cat "$_log" 2>/dev/null || true)"
    rm -f "$_sh" "$_log"
  fi
  if [ "$_ic_rc" -ne 0 ]; then
    explain_install_failure "$version" "$_ic_rc" "$_ic_out" "$_ic_src"
  fi

  hash -r 2>/dev/null || true
  if _is_macos && command -v xattr >/dev/null 2>&1; then
    xattr -d com.apple.quarantine "$CODEX_INSTALL_DIR/codex" 2>/dev/null || true
  fi
  verify_installed_version "$version"
}

print_restore_instructions() {
  local dest="$1"
  cat >&2 <<EOF

Your existing Codex config was backed up before switching versions.
  Backup location : ${dest}
  Manifest        : ${dest}/MANIFEST.txt   (checksums: ${dest}/SHA256SUMS)

If the new Codex rejects your old config (e.g. a legacy [profiles.*] hard error),
restore it with either:
  rsync -a --exclude MANIFEST.txt --exclude SHA256SUMS "${dest}/" "${CODEX_HOME}/"
or per file:
  cp -p "${dest}/config.toml" "${CODEX_HOME}/"        2>/dev/null || true
  cp -p "${dest}"/*.config.toml "${CODEX_HOME}/"      2>/dev/null || true
  cp -p "${dest}"/*.json "${CODEX_HOME}/"             2>/dev/null || true
  cp -p "${dest}/auth.json" "${CODEX_HOME}/" && chmod 600 "${CODEX_HOME}/auth.json"
To restore the session/resume index (so 'codex resume' lists your pre-switch sessions again):
  cp -p "${dest}"/*.sqlite* "${CODEX_HOME}/"          2>/dev/null || true
Then verify:  codex doctor      (expect: config.toml parse: ok)
More help: docs/install_guide.md (section 4) or ${SUPPORT_URL}
EOF
}

prune_old_backups() {
  local root="$1" keep="${CODEX_BACKUP_KEEP:-10}" d
  [ -d "$root" ] || return 0
  ls -1d "$root"/codex-config-* 2>/dev/null | sort -r | tail -n "+$((keep + 1))" | while IFS= read -r d; do
    [ -d "$d" ] && [ -f "$d/MANIFEST.txt" ] && rm -rf "$d"
  done || true
}

backup_codex_config() {
  local from_ver="${1:-unknown}" to_ver="${2:-unknown}"
  LAST_BACKUP_DIR=""
  if [ "${SKIP_BACKUP:-0}" = "1" ]; then
    log_warn "Skipping config backup (--no-backup)."
    return 0
  fi
  if [ ! -d "$CODEX_HOME" ]; then
    log_info "No existing config to back up (CODEX_HOME=${CODEX_HOME} absent)."
    return 0
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    log_info "[dry-run] would back up existing config in ${CODEX_HOME} to ${CODEX_BACKUP_ROOT}."
    return 0
  fi

  local root="$CODEX_BACKUP_ROOT"
  mkdir -p "$root" 2>/dev/null \
    || die "Cannot create backup root ${root}; refusing to switch without a backup. Set CODEX_BACKUP_ROOT to a writable path."
  chmod 700 "$root" 2>/dev/null || true

  local ts base dest n
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  base="$root/codex-config-$ts"
  dest="$base"; n=1
  while ! mkdir "$dest" 2>/dev/null; do
    dest="$base-$(printf '%02d' "$n")"; n=$((n + 1))
    [ "$n" -gt 99 ] && die "Too many backups created this second under ${root}."
  done
  chmod 700 "$dest" 2>/dev/null || true

  local f bn copied=0 idx_copied=0
  for f in \
    "$CODEX_HOME"/config.toml \
    "$CODEX_HOME"/*.config.toml \
    "$CODEX_HOME"/auth.json \
    "$CODEX_HOME"/*.json \
    "$CODEX_HOME"/*.md
  do
    [ -f "$f" ] || continue
    bn="$(basename "$f")"
    [ "$bn" = "version.json" ] && continue
    [ -e "$dest/$bn" ] && continue
    cp -pL "$f" "$dest/$bn" 2>/dev/null || cp -p "$f" "$dest/$bn"
    copied=$((copied + 1))
  done

  for f in \
    "$CODEX_HOME"/state_*.sqlite* \
    "$CODEX_HOME"/memories_*.sqlite* \
    "$CODEX_HOME"/goals_*.sqlite*
  do
    [ -f "$f" ] || continue
    bn="$(basename "$f")"
    [ -e "$dest/$bn" ] && continue
    cp -pL "$f" "$dest/$bn" 2>/dev/null || cp -p "$f" "$dest/$bn"
    copied=$((copied + 1)); idx_copied=$((idx_copied + 1))
  done

  if [ $((copied + idx_copied)) -eq 0 ]; then
    rmdir "$dest" 2>/dev/null || true
    log_info "Nothing to back up (no user config found in ${CODEX_HOME}); skipping backup."
    return 0
  fi

  chmod 600 "$dest"/* 2>/dev/null || true

  {
    printf 'from_version=%s\n' "$from_ver"
    printf 'to_version=%s\n' "$to_ver"
    printf 'codex_home=%s\n' "$CODEX_HOME"
    printf 'created_utc=%s\n' "$ts"
    printf 'host=%s\n' "$(hostname 2>/dev/null || echo unknown)"
  } > "$dest/MANIFEST.txt"

  if [ -n "$FUGU_SHA256" ]; then
    ( cd "$dest" && for f in *; do
        { [ "$f" = "MANIFEST.txt" ] || [ "$f" = "SHA256SUMS" ]; } && continue
        $FUGU_SHA256 "$f"
      done > SHA256SUMS 2>/dev/null ) || true
  fi

  if [ -n "$FUGU_SHA256" ] && [ -s "$dest/SHA256SUMS" ]; then
    ( cd "$dest" && $FUGU_SHA256 -c SHA256SUMS >/dev/null 2>&1 ) \
      || die "Backup integrity check failed at ${dest}; refusing to switch versions."
  fi

  LAST_BACKUP_DIR="$dest"
  log_ok "Backed up ${copied} file(s) to ${dest}"
  [ "$idx_copied" -gt 0 ] && log_info "Included ${idx_copied} Codex session-index file(s) (state/memories/goals) so 'codex resume' history stays recoverable across the switch."
  print_restore_instructions "$dest"
  prune_old_backups "$root"
  return 0
}


_abs_codex_home() {
  if [ -d "$CODEX_HOME" ]; then ( cd "$CODEX_HOME" && pwd ); else printf '%s' "$CODEX_HOME"; fi
}

_is_codex_config() {
  case "$1" in
    "$(_abs_codex_home)"/config.toml | "$CODEX_HOME"/config.toml) return 0 ;;
    *) return 1 ;;
  esac
}

backup_once() {
  [ "${_RUN_BACKUP_DONE:-0}" = "1" ] && return 0
  backup_codex_config "${1:-bundle}" "${2:-bundle}"
  _RUN_BACKUP_DONE=1
}

bundles_root() {
  if [ -n "${FUGU_CONFIGS_DIR:-}" ]; then printf '%s' "$FUGU_CONFIGS_DIR"; return 0; fi
  local sdir
  sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  printf '%s' "$(cd "$sdir/.." && pwd)/configs"
}

resolve_bundle_dir() {
  local dir
  dir="$(bundles_root)"
  [ -d "$dir" ]           || die "No config bundle directory at ${dir}."
  [ -f "$dir/bundle.sh" ] || die "Config bundle at ${dir} is missing its bundle.sh manifest."
  printf '%s' "$dir"
}

load_manifest() {
  local dir="$1"
  BUNDLE_NAME=""; BUNDLE_DESC=""; BUNDLE_SCHEMA=""; BUNDLE_RUN_HINT=""; BUNDLE_CODEX_VERSION=""
  BUNDLE_DEFAULT_MODEL=""; BUNDLE_MODEL_CATALOG=""
  FILES=(); INJECTS=(); ENV_KEYS=()
  BUNDLE_DIR="$dir"
  . "$dir/bundle.sh"
  [ "${BUNDLE_SCHEMA:-0}" = "1" ] || die "Bundle '${BUNDLE_NAME:-?}' uses unsupported manifest schema '${BUNDLE_SCHEMA:-?}'."
  [ -n "${BUNDLE_NAME:-}" ]       || die "Bundle manifest is missing BUNDLE_NAME."
  if [ -n "${BUNDLE_CODEX_VERSION:-}" ]; then
    BUNDLE_CODEX_VERSION="$(normalize_version "$BUNDLE_CODEX_VERSION")" \
      || die "Bundle '${BUNDLE_NAME}' declares an invalid BUNDLE_CODEX_VERSION."
  else
    log_warn "Bundle '${BUNDLE_NAME}' does not declare BUNDLE_CODEX_VERSION; treating as unpinned (will use the resolved latest Codex version)."
  fi
}

resolve_fugu_model() {
  local saved="" catalog="" candidate=""
  [ "${FUGU_MODEL_RESOLVED:-0}" = "0" ] || return 0
  [ -n "${BUNDLE_MODEL_CATALOG:-}" ] || {
    [ "${FUGU_MODEL_EXPLICIT:-0}" = "0" ] || die "This config bundle does not support selecting a default model."
    return 0
  }
  catalog="$BUNDLE_DIR/$BUNDLE_MODEL_CATALOG"
  [ -f "$catalog" ] || die "Bundle model catalog not found: ${catalog}."

  if [ "${FUGU_MODEL_EXPLICIT:-0}" = "1" ]; then
    [ -n "${FUGU_MODEL:-}" ] || die "--model/FUGU_MODEL requires a non-empty model name."
    if [ "$FUGU_MODEL" = "default" ]; then
      log_info "Using the bundle's default Fugu model (not pinned)."
    else
      candidate="$FUGU_MODEL"
      FUGU_MODEL_PINNED=1
    fi
  elif [ -f "$(_state_path)" ]; then
    [ -r "$(_state_path)" ] || \
      log_warn "Cannot read $(_state_path); a default model chosen earlier with --model is not kept. Use --model MODEL to choose it again."
    saved="$(sed -n 's/^model=//p' "$(_state_path)" 2>/dev/null | tail -n1 || true)"
    if [ -n "$saved" ] && _single_line "$saved" && printf '%s' "$saved" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._-]*$' && \
       grep -qF "\"slug\": \"${saved}\"" "$catalog"; then
      candidate="$saved"
      FUGU_MODEL_PINNED=1
    elif [ -n "$saved" ]; then
      log_warn "Saved Fugu model is no longer available ('${saved}'); using the bundle default ${BUNDLE_DEFAULT_MODEL:-} instead."
    fi
  fi
  FUGU_MODEL="${candidate:-${BUNDLE_DEFAULT_MODEL:-}}"
  [ -n "$FUGU_MODEL" ] || die "Bundle '${BUNDLE_NAME}' does not declare a default model."
  _single_line "$FUGU_MODEL" || die "Invalid Fugu model name: a model name is a single line."
  printf '%s' "$FUGU_MODEL" | grep -qE '^[A-Za-z0-9][A-Za-z0-9._-]*$' \
    || die "Invalid Fugu model name: '${FUGU_MODEL}'."
  grep -qF "\"slug\": \"${FUGU_MODEL}\"" "$catalog" \
    || die "Unknown Fugu model '${FUGU_MODEL}'; choose a model listed in ${BUNDLE_MODEL_CATALOG}."
  FUGU_MODEL_RESOLVED=1
  log_info "Default Fugu model: ${FUGU_MODEL}"
}

load_manifest_for() {
  local d; d="$(resolve_bundle_dir)"; load_manifest "$d"
}

render_template() {
  local src="$1" dest="$2" render_model="${FUGU_MODEL:-$DEFAULT_FUGU_MODEL}"
  FUGU_TPL_HOME="$(_abs_codex_home)" FUGU_TPL_SAKANA_BASE_URL="$SAKANA_BASE_URL" FUGU_TPL_MODEL="$render_model" awk '
    function replace_all(line, token, value, out, i, n) {
      out = ""; n = length(token)
      while ((i = index(line, token)) > 0) {
        out = out substr(line, 1, i - 1) value
        line = substr(line, i + n)
      }
      return out line
    }
    BEGIN {
      home = ENVIRON["FUGU_TPL_HOME"]
      base_url = ENVIRON["FUGU_TPL_SAKANA_BASE_URL"]
      model = ENVIRON["FUGU_TPL_MODEL"]
    }
    { print replace_all(replace_all(replace_all($0, "{{CODEX_HOME}}", home), "{{SAKANA_BASE_URL}}", base_url), "{{FUGU_MODEL}}", model) }
  ' "$src" > "$dest"
}

_receipt_path() { printf '%s' "$CODEX_HOME/.fugu/installed/${1}.receipt"; }
record_installed_file() {
  local rp; rp="$(_receipt_path "$1")"
  mkdir -p "$(dirname "$rp")"; chmod 700 "$CODEX_HOME/.fugu" 2>/dev/null || true
  grep -qxF "file $2" "$rp" 2>/dev/null || printf 'file %s\n' "$2" >> "$rp"
  chmod 600 "$rp" 2>/dev/null || true
}
receipt_has_file() { local rp; rp="$(_receipt_path "$1")"; [ -f "$rp" ] && grep -qxF "file $2" "$rp"; }

record_installed_path() {
  local rp; rp="$(_receipt_path "$1")"
  mkdir -p "$(dirname "$rp")"; chmod 700 "$CODEX_HOME/.fugu" 2>/dev/null || true
  grep -qxF "path $2" "$rp" 2>/dev/null || printf 'path %s\n' "$2" >> "$rp"
  chmod 600 "$rp" 2>/dev/null || true
}

_state_path()     { printf '%s' "$CODEX_HOME/.fugu/state"; }
_decisions_path() { printf '%s' "$CODEX_HOME/.fugu/decisions"; }
_fugu_dir_init()  { mkdir -p "$CODEX_HOME/.fugu"; chmod 700 "$CODEX_HOME/.fugu" 2>/dev/null || true; }

detect_install_method() {
  local c; c="$(command -v codex 2>/dev/null || true)"
  [ -n "$c" ] || { printf 'unknown'; return 0; }
  case "$(file -bL "$c" 2>/dev/null || true)" in
    *ELF*|*Mach-O*) printf 'standalone'; return 0 ;;
  esac
  if head -n1 "$c" 2>/dev/null | grep -q 'node'; then printf 'npm'; else printf 'unknown'; fi
}

record_fugu_state() {
  [ "${DRY_RUN:-0}" = "1" ] && { log_info "[dry-run] would record $(_state_path)"; return 0; }
  local sp repo ref branch method tmp
  sp="$(_state_path)"; _fugu_dir_init
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo unknown)"
  ref="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
  branch="$(git -C "$repo" rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  method="$(detect_install_method)"
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-state.XXXXXX")"; chmod 600 "$tmp" 2>/dev/null || true
  {
    printf 'schema=1\n'
    printf 'bundle=%s\n'          "$1"
    printf 'deployed_target=%s\n' "$2"
    printf 'install_ref=%s\n'     "$ref"
    printf 'repo_dir=%s\n'        "$repo"
    printf 'branch=%s\n'          "$branch"
    printf 'install_method=%s\n'  "$method"
    printf 'deployed_format=%s\n' "${DEPLOYED_FORMAT:-}"
    if [ "${FUGU_MODEL_PINNED:-0}" = "1" ]; then
      printf 'model=%s\n'         "${FUGU_MODEL:-}"
    fi
    if [ -n "${SAKANA_BASE_URL_RECORD:-}" ]; then
      printf 'sakana_base_url=%s\n' "$SAKANA_BASE_URL_RECORD"
    fi
    printf 'updated_utc=%s\n'     "$(date -u +%Y%m%dT%H%M%SZ 2>/dev/null || echo unknown)"
  } > "$tmp"
  mv -f "$tmp" "$sp"; chmod 600 "$sp" 2>/dev/null || true
  log_info "Recorded Fugu state in ${sp}."
}

_state_set_sakana_base_url() {
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  local sp tmp; sp="$(_state_path)"
  [ -f "$sp" ] || return 0
  [ -r "$sp" ] || { log_warn "Cannot read ${sp}; its endpoint record was not updated."; return 0; }
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-state.XXXXXX")" || return 0
  chmod 600 "$tmp" 2>/dev/null || true
  { grep -v '^sakana_base_url=' "$sp" || true; [ -z "$1" ] || printf 'sakana_base_url=%s\n' "$1"; } > "$tmp"
  mv -f "$tmp" "$sp"; chmod 600 "$sp" 2>/dev/null || true
}

advance_fugu_install_ref() {
  local sp ref repo tmp; sp="$(_state_path)"
  [ -f "$sp" ] || return 0
  repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd || echo unknown)"
  ref="$(git -C "$repo" rev-parse HEAD 2>/dev/null || echo unknown)"
  [ "$ref" = "unknown" ] && return 0
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-state.XXXXXX")" || return 0
  chmod 600 "$tmp" 2>/dev/null || true
  if sed "s|^install_ref=.*|install_ref=${ref}|" "$sp" > "$tmp" 2>/dev/null; then
    mv -f "$tmp" "$sp"; chmod 600 "$sp" 2>/dev/null || true
    log_info "Advanced install_ref to ${ref} (launcher updated; bundle deploy unchanged)."
  else
    rm -f "$tmp" 2>/dev/null || true
  fi
}

record_mismatch_decision() {
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  local dp line; dp="$(_decisions_path)"; _fugu_dir_init
  line="never mismatch ${1} ${2}"
  grep -qxF "$line" "$dp" 2>/dev/null || printf '%s\n' "$line" >> "$dp"
  chmod 600 "$dp" 2>/dev/null || true
}

mismatch_declined() {
  local dp; dp="$(_decisions_path)"
  [ -f "$dp" ] && grep -qxF "never mismatch ${1} ${2}" "$dp"
}

install_bundle_file() {
  local src_rel="$1" dest_bn="$2" src dest tmp
  src="$BUNDLE_DIR/$src_rel"; dest="$CODEX_HOME/$dest_bn"
  [ -f "$src" ] || die "Bundle file missing: ${src}"
  mkdir -p "$CODEX_HOME"
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-file.XXXXXX")"
  render_template "$src" "$tmp"
  if [ -f "$dest" ] && cmp -s "$tmp" "$dest"; then
    log_info "Up to date: ${dest_bn} (identical)."
    rm -f "$tmp"; return 0
  fi
  if [ -e "$dest" ]; then
    log_warn "${dest_bn} exists and differs from the bundle version; backing up before overwrite."
    [ "${DRY_RUN:-0}" = "1" ] || backup_once "bundle:${CONFIG_NAME}" "bundle:${CONFIG_NAME}"
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would write ${dest}"; rm -f "$tmp"; return 0; fi
  cat "$tmp" > "$dest"; rm -f "$tmp"
  record_installed_file "$CONFIG_NAME" "$dest_bn"
  log_ok "Installed ${dest_bn} into ${CODEX_HOME}."
}

install_fugu_launcher() {
  local bundle="$1" src dest
  src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/codex-fugu"
  dest="$CODEX_INSTALL_DIR/codex-fugu"
  [ -f "$src" ] || { log_warn "Launcher source ${src} not found; skipping codex-fugu install."; return 0; }
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would install launcher to ${dest}"; return 0; fi
  mkdir -p "$CODEX_INSTALL_DIR"
  local tmp
  if tmp="$(mktemp "$CODEX_INSTALL_DIR/.codex-fugu.XXXXXX" 2>/dev/null)"; then
    if cat "$src" > "$tmp" && chmod 755 "$tmp" 2>/dev/null && mv -f "$tmp" "$dest"; then
      :
    else
      rm -f "$tmp" 2>/dev/null || true
      log_warn "Could not install launcher atomically to ${dest}; skipping codex-fugu install."; return 0
    fi
  else
    log_warn "Could not create a temp file in ${CODEX_INSTALL_DIR}; skipping codex-fugu install."; return 0
  fi
  record_installed_path "$bundle" "$dest"
  log_ok "Installed codex-fugu launcher to ${dest}."
  command -v codex-fugu >/dev/null 2>&1 || log_warn "codex-fugu installed to ${CODEX_INSTALL_DIR}, but it is not on your PATH."
}

_block_conflicts() {
  local file="$1" id="$2" pat="$3"
  [ -f "$file" ] || return 1
  [ -n "$pat" ]  || return 1
  awk -v mo="# >>> fugu:${id} >>>" -v mc="# <<< fugu:${id} <<<" -v pat="$pat" '
    $0 == mo  { inblk = 1; next }
    $0 == mc  { inblk = 0; next }
    !inblk && index($0, pat) { found = 1 }
    END { exit(found ? 0 : 1) }
  ' "$file"
}

_block_body_matches() {
  local file="$1" open="$2" close="$3" snippet="$4" cur
  grep -qxF "$open" "$file" 2>/dev/null || return 1
  cur="$(awk -v mo="$open" -v mc="$close" '
    $0 == mo { inb = 1; next } $0 == mc { inb = 0; next } inb { print }
  ' "$file")"
  [ "$cur" = "$snippet" ]
}

_config_parses_ok() {
  command -v codex >/dev/null 2>&1 || { log_info "codex not on PATH; skipping parse check."; return 0; }
  local out
  out="$(codex doctor 2>&1 || true)"
  if printf '%s\n' "$out" | grep -qiE 'config could not be loaded|could not load .*config|failed to (load|parse) .*config|config\.toml parse[[:space:]:]+(fail|error|invalid)|failed to parse .*config\.toml|invalid config'; then
    return 1
  fi
  return 0
}

_FUGU_AWK_TABLES='
function tbl_name(s,   n, t) {
  t = s; gsub(/"([^"\\]|\\.)*"|\047[^\047]*\047/, "\"\"", t); sub(/#.*$/, "", t)
  if (t !~ /^[ \t]*\[/ || t ~ /[=,]/) return ""
  n = s; sub(/^[ \t]*\[+[ \t]*/, "", n); sub(/[ \t]*\].*$/, "", n)
  return n
}
function foreign_hdr(s,   n) { n = tbl_name(s); return n != "" && n != id && index(n, id ".") != 1 }
'

inject_managed_block() {
  local file="$1" id="$2" snippet="$3"
  local open="# >>> fugu:${id} >>>" close="# <<< fugu:${id} <<<"

  local conflict; conflict="$(printf '%s\n' "$snippet" | grep -m1 -E '^\[[A-Za-z0-9_."-]+\]' || true)"
  if [ -n "$conflict" ] && _block_conflicts "$file" "$id" "$conflict"; then
    log_warn "Found an existing '${conflict}' in ${file} outside Fugu's managed markers; not editing it."
    log_warn "Please merge it manually — see ${SUPPORT_URL}"
    return 1
  fi

  if [ -f "$file" ] && _block_body_matches "$file" "$open" "$close" "$snippet"; then
    log_info "fugu:${id} already present in ${file} (unchanged)."
    return 0
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would inject fugu:${id} into ${file}"; return 0; fi

  if _is_codex_config "$file" && [ -f "$file" ]; then backup_once "bundle:${CONFIG_NAME}" "bundle:${CONFIG_NAME}"; fi

  local existed=0 snap=""
  if [ -f "$file" ]; then existed=1; snap="$(mktemp "${TMPDIR:-/tmp}/fugu-snap.XXXXXX")"; cat "$file" > "$snap"; fi

  mkdir -p "$(dirname "$file")"
  local tmp; tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-block.XXXXXX")"
  if [ -f "$file" ] && grep -qxF "$open" "$file"; then
    FUGU_SNIP="$snippet" awk -v mo="$open" -v mc="$close" -v id="$id" "$_FUGU_AWK_TABLES"'
      $0 == mo  { print mo; print ENVIRON["FUGU_SNIP"]; skip = 1; foreign = 0; tail = ""; next }
      $0 == mc  { if (skip) { skip = 0; print mc; if (tail != "") printf "\n%s", tail; next } }
      skip      { if (!foreign && foreign_hdr($0)) foreign = 1; if (foreign) tail = tail $0 "\n"; next }
      { print }
      END       { if (skip) { print mc; if (tail != "") printf "\n%s", tail } }
    ' "$file" > "$tmp"
  else
    [ -f "$file" ] && cat "$file" > "$tmp"
    [ -s "$tmp" ] && printf '\n' >> "$tmp"
    { printf '%s\n' "$open"; printf '%s\n' "$snippet"; printf '%s\n' "$close"; } >> "$tmp"
  fi
  cat "$tmp" > "$file"; rm -f "$tmp"
  log_ok "Wrote managed block fugu:${id} to ${file}."

  if _is_codex_config "$file" && ! _config_parses_ok; then
    log_error "codex doctor reports config.toml no longer parses; rolling back."
    if [ "$existed" = 1 ]; then cat "$snap" > "$file"; else rm -f "$file"; fi
    log_warn "Reverted ${file}. Please configure manually — see ${SUPPORT_URL}"
    [ -n "$snap" ] && rm -f "$snap"
    return 1
  fi
  [ -n "$snap" ] && rm -f "$snap"
  return 0
}

remove_managed_block() {
  local file="$1" id="$2" tmp
  local open="# >>> fugu:${id} >>>" close="# <<< fugu:${id} <<<"
  [ -f "$file" ]              || { log_info "No ${file}; nothing to remove."; return 0; }
  grep -qxF "$open" "$file"   || { log_info "No fugu:${id} block in ${file}."; return 0; }
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would remove fugu:${id} from ${file}"; return 0; fi
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-rm.XXXXXX")"
  awk -v mo="$open" -v mc="$close" -v id="$id" "$_FUGU_AWK_TABLES"'
    {
      if (inblk) {
        if ($0 == mc) { inblk = 0; if (tail != "") { if (sep) print ""; printf "%s", tail }; next }
        if (!foreign && foreign_hdr($0)) foreign = 1
        if (foreign) tail = tail $0 "\n"
        next
      }
      if ($0 == mo) { sep = (held && heldline == ""); if (held && heldline != "") print heldline; held = 0; inblk = 1; foreign = 0; tail = ""; next }
      if (held) print heldline
      heldline = $0; held = 1
    }
    END { if (held) print heldline; if (inblk && tail != "") { if (sep) print ""; printf "%s", tail } }
  ' "$file" > "$tmp"
  cat "$tmp" > "$file"; rm -f "$tmp"
  log_ok "Removed managed block fugu:${id} from ${file}."
}

key_already_persisted() {
  local var="$1" regex="$2" line val
  [ -f "$FUGU_ENV_FILE" ] || return 1
  line="$(grep -E "^(export )?${var}=" "$FUGU_ENV_FILE" 2>/dev/null | tail -n1)"
  [ -n "$line" ] || return 1
  val="${line#export }"; val="${val#${var}=}"
  val="${val#\'}"; val="${val%\'}"; val="${val#\"}"; val="${val%\"}"
  printf '%s' "$val" | grep -qE -- "$regex"
}

_export_from_env_file() {
  local var="${1?}" value=""
  value="$(read_persisted_env_value "$var" 2>/dev/null || true)"
  [ -n "$value" ] || return 0
  printf -v "$var" '%s' "$value"
  export "$var"
}

persist_api_key() {
  local var="$1" value="$2" dir old_umask tmp
  dir="$(dirname "$FUGU_ENV_FILE")"
  mkdir -p "$dir"
  old_umask="$(umask)"; umask 077
  [ -f "$FUGU_ENV_FILE" ] || : > "$FUGU_ENV_FILE"
  chmod 600 "$FUGU_ENV_FILE" 2>/dev/null || true
  tmp="$(mktemp "${dir}/.env.XXXXXX")"; chmod 600 "$tmp" 2>/dev/null || true
  grep -vE "^(export )?${var}=" "$FUGU_ENV_FILE" 2>/dev/null > "$tmp" || true
  printf '%s=%s\n' "$var" "$value" >> "$tmp"
  cat "$tmp" > "$FUGU_ENV_FILE"; rm -f "$tmp"
  umask "$old_umask"
}

remove_api_key() {
  local var="$1" tmp
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would remove ${var} from ${FUGU_ENV_FILE} (and the file itself if now empty)."; return 0; fi
  [ -f "$FUGU_ENV_FILE" ] || { log_info "No env file ${FUGU_ENV_FILE}."; return 0; }
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-envrm.XXXXXX")"; chmod 600 "$tmp" 2>/dev/null || true
  grep -vE "^(export )?${var}=" "$FUGU_ENV_FILE" > "$tmp" 2>/dev/null || true
  cat "$tmp" > "$FUGU_ENV_FILE"; rm -f "$tmp"
  log_ok "Removed ${var} from ${FUGU_ENV_FILE}."
  if [ ! -s "$FUGU_ENV_FILE" ]; then
    rm -f "$FUGU_ENV_FILE"
    log_info "Env file empty; removed it."
  fi
}

read_persisted_env_value() {
  local var="$1" line value
  [ -f "$FUGU_ENV_FILE" ] || return 1
  line="$(grep -E "^(export )?${var}=" "$FUGU_ENV_FILE" 2>/dev/null | tail -n1)"
  [ -n "$line" ] || return 1
  value="${line#export }"; value="${value#${var}=}"; value="${value%$'\r'}"
  case "$value" in
    \"*) value="${value#\"}"; value="${value%%\"*}" ;;
    \'*) value="${value#\'}"; value="${value%%\'*}" ;;
    *)   value="$(printf '%s' "$value" | sed 's/[[:space:]][[:space:]]*#.*$//;s/[[:space:]]*$//')" ;;
  esac
  printf '%s' "$value"
}

_single_line() { case "$1" in *$'\n'* | *$'\r'*) return 1 ;; esac; return 0; }

valid_sakana_base_url() {
  _single_line "$1" && printf '%s' "$1" | grep -qE '^https?://[A-Za-z0-9][A-Za-z0-9._:-]*(/[A-Za-z0-9._~:/-]*)?$'
}

_sakana_url_canon() {
  local url="$1" scheme rest hostport path host port=""
  case "$url" in *://*) ;; *) printf '%s' "$url"; return 0 ;; esac
  scheme="$(printf '%s' "${url%%://*}" | tr '[:upper:]' '[:lower:]')"
  rest="${url#*://}"; hostport="${rest%%/*}"; path="${rest#"$hostport"}"
  host="$(printf '%s' "${hostport%%:*}" | tr '[:upper:]' '[:lower:]')"
  case "$hostport" in *:*) port="${hostport#*:}" ;; esac
  { [ "$scheme" = https ] && [ "$port" = 443 ]; } && port=""
  { [ "$scheme" = http ] && [ "$port" = 80 ]; } && port=""
  while [ "${path%/}" != "$path" ]; do path="${path%/}"; done
  printf '%s://%s%s%s' "$scheme" "$host" "${port:+:$port}" "$path"
}
_sakana_url_host() { local c; c="$(_sakana_url_canon "$1")"; c="${c#*://}"; c="${c%%/*}"; printf '%s' "${c%%:*}"; }
_sakana_url_is_production_host() { [ "$(_sakana_url_host "$1")" = "$(_sakana_url_host "$DEFAULT_SAKANA_BASE_URL")" ]; }

_trim_sakana_base_url() {
  local value
  value="$(printf '%s' "$1" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  while [ "${value%/}" != "$value" ]; do value="${value%/}"; done
  printf '%s' "$value"
}

_canon_sakana_base_url() {
  local value
  value="$(_trim_sakana_base_url "$1")"
  case "$value" in *://*) value="$(printf '%s' "${value%%://*}" | tr '[:upper:]' '[:lower:]')://${value#*://}" ;; esac
  if valid_sakana_base_url "$value"; then _sakana_url_canon "$value"; fi
  return 0
}

_normalize_sakana_base_url() {
  local value
  value="$(_canon_sakana_base_url "$1")"
  [ -n "$value" ] || die "Invalid Sakana API base URL${2:+ $2} (expected http:// or https:// without spaces or quotes)."
  printf '%s' "$value"
}

_provider_table_base_url() {
  awk -v want="$1" '
    $0 == "# >>> fugu:model_providers.sakana >>>" { inm = 1; intbl = 0; next }
    $0 == "# <<< fugu:model_providers.sakana <<<" { inm = 0; intbl = 0; next }
    /^[ \t]*\[/ { t = $0; sub(/[ \t]*#.*$/, "", t); gsub(/[ \t\r]/, "", t)
                  intbl = (t == "[model_providers.sakana]"); next }
    intbl && ((want == "managed") == (inm == 1)) && /^[ \t]*base_url[ \t]*=/ {
      v = $0; sub(/^[^=]*=[ \t]*/, "", v); q = substr(v, 1, 1)
      if (q == "\"" || q == "\047") { v = substr(v, 2); i = index(v, q); if (i) { print substr(v, 1, i - 1); exit } }
    }' "$CODEX_HOME/config.toml" 2>/dev/null || true
}
_managed_block_present() { grep -qxF '# >>> fugu:model_providers.sakana >>>' "$CODEX_HOME/config.toml" 2>/dev/null; }

_sakana_base_url_before() {
  local url
  url="$(_canon_sakana_base_url "$(_provider_table_base_url managed)")"
  [ -n "$url" ] || url="$(_canon_sakana_base_url "$(_provider_table_base_url user)")"
  printf '%s' "${url:-$DEFAULT_SAKANA_BASE_URL}"
}

_recorded_sakana_base_url() {
  local url
  [ -f "$(_state_path)" ] || return 0
  url="$(_canon_sakana_base_url "$(sed -n 's/^sakana_base_url=//p' "$(_state_path)" 2>/dev/null | tail -n1)")"
  [ "$url" = "$DEFAULT_SAKANA_BASE_URL" ] || printf '%s' "$url"
}

_saved_sakana_base_url() {
  local recorded saved
  saved="$(_canon_sakana_base_url "$(read_persisted_env_value SAKANA_BASE_URL 2>/dev/null || true)")"
  [ -n "$saved" ] && [ "$saved" != "$DEFAULT_SAKANA_BASE_URL" ] || return 0
  recorded="$(_recorded_sakana_base_url)"
  if [ -n "$recorded" ]; then
    if [ "$saved" = "$recorded" ]; then printf '%s' "$saved"; fi
  elif [ "$saved" = "$(_canon_sakana_base_url "$(_provider_table_base_url managed)")" ]; then
    printf '%s' "$saved"
  fi
  return 0
}

_sakana_base_url_line_owned() {
  read_persisted_env_value SAKANA_BASE_URL >/dev/null 2>&1 || return 1
  [ -n "$(_recorded_sakana_base_url)" ] || [ -n "$(_saved_sakana_base_url)" ]
}

check_saved_sakana_base_url() {
  local recorded saved
  [ -z "$SAKANA_BASE_URL" ] || return 0
  recorded="$(_recorded_sakana_base_url)"
  [ -n "$recorded" ] || return 0
  saved="$(_saved_sakana_base_url)"
  if [ -z "$saved" ]; then
    die "The SAKANA_BASE_URL saved in ${FUGU_ENV_FILE} does not match the endpoint in ${CODEX_HOME}/config.toml (${recorded}); re-run with --base-url URL to choose one (e.g. --base-url ${recorded} to keep it)."
  fi
  if _managed_block_present && ! _bundle_renders_base_url; then
    die "This config bundle cannot render the custom endpoint ${saved} saved for this install; deploy the matching configs, or re-run with --base-url ${DEFAULT_SAKANA_BASE_URL} to return to production."
  fi
}

check_requested_endpoint_applicable() {
  local value
  [ -n "$SAKANA_BASE_URL" ] && [ "${DRY_RUN:-0}" != "1" ] || return 0
  value="$(_normalize_sakana_base_url "$SAKANA_BASE_URL")"
  [ "$value" != "$(_sakana_base_url_before)" ] || return 0
  _provider_inject_refused || return 0
  SAKANA_BASE_URL="$value"
  log_warn "Found an existing '[model_providers.sakana]' in ${CODEX_HOME}/config.toml outside Fugu's managed markers; not editing it."
  _endpoint_change_not_applied
}

_bundle_renders_base_url() {
  local entry rest marker snippet
  for entry in "${INJECTS[@]:-}"; do
    [ -n "$entry" ] || continue
    rest="${entry#*::}"; marker="${rest%%::*}"; snippet="${rest##*::}"
    [ "$marker" = "model_providers.sakana" ] || continue
    grep -qF '{{SAKANA_BASE_URL}}' "$BUNDLE_DIR/$snippet" 2>/dev/null && return 0
  done
  return 1
}

validate_requested_sakana_base_url() {
  local value where=""
  if [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" = "1" ] && [ -z "$SAKANA_BASE_URL" ]; then
    die "--base-url requires a non-empty URL."
  fi
  [ -n "$SAKANA_BASE_URL" ] || return 0
  [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" = "1" ] || where="in the SAKANA_BASE_URL environment variable"
  value="$(_normalize_sakana_base_url "$SAKANA_BASE_URL" "$where")"
  if [ "$value" != "$DEFAULT_SAKANA_BASE_URL" ] && ! _bundle_renders_base_url; then
    die "This config bundle does not support selecting an API endpoint."
  fi
}

load_sakana_base_url() {
  local url
  url="$(_recorded_sakana_base_url)"
  [ -n "$url" ] || url="$(_saved_sakana_base_url)"
  SAKANA_BASE_URL="${url:-$DEFAULT_SAKANA_BASE_URL}"
}

setup_sakana_base_url() {
  local saved="" value="" prompt_default
  if [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" = "1" ] && [ -z "$SAKANA_BASE_URL" ]; then
    die "--base-url requires a non-empty URL."
  fi
  check_saved_sakana_base_url
  saved="$(_saved_sakana_base_url)"
  SAKANA_BASE_URL_LINE_OWNED=0; ! _sakana_base_url_line_owned || SAKANA_BASE_URL_LINE_OWNED=1
  SAKANA_BASE_URL_BEFORE="$(_sakana_base_url_before)"
  SAKANA_BASE_URL_REQUESTED=0
  SAKANA_BASE_URL_RECORD="${saved:-$(_recorded_sakana_base_url)}"

  local from_env=0
  if [ -n "$SAKANA_BASE_URL" ]; then
    value="$SAKANA_BASE_URL"; SAKANA_BASE_URL_REQUESTED=1
    [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" = "1" ] || from_env=1
  elif [ -n "$saved" ]; then
    value="$saved"
    [ -n "$(_recorded_sakana_base_url)" ] || \
      log_warn "No endpoint record in $(_state_path); adopting ${saved}, which Fugu's provider block and the SAKANA_BASE_URL line in ${FUGU_ENV_FILE} both name, as this installer's custom endpoint."
  else
    value="$DEFAULT_SAKANA_BASE_URL"
    local stays="$SAKANA_BASE_URL_BEFORE"
    ! _managed_block_present || stays="$value"
    if read_persisted_env_value SAKANA_BASE_URL >/dev/null 2>&1 && \
       [ "$(_canon_sakana_base_url "$(read_persisted_env_value SAKANA_BASE_URL)")" != "$stays" ]; then
      log_warn "Ignoring the SAKANA_BASE_URL line in ${FUGU_ENV_FILE}: this installer did not save it (the endpoint stays ${stays}). Use --base-url URL to change the endpoint."
    fi
  fi

  if [ "${RECONFIGURE:-0}" = "1" ] && [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" != "1" ] && \
     [ "${ASSUME_YES:-0}" != "1" ] && [ -t 0 ] && \
     { _managed_block_present || [ -z "$(_provider_table_base_url user)" ]; }; then
    prompt_default="$value"
    printf 'Sakana API base URL [%s]: ' "$prompt_default" >&2
    if ! IFS= read -r value; then value=""; printf '\n' >&2; fi
    value="$(printf '%s' "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
    [ -z "$value" ] || SAKANA_BASE_URL_REQUESTED=1
    value="${value:-$prompt_default}"
  fi

  value="$(_normalize_sakana_base_url "$value")"
  [ "$value" != "$DEFAULT_SAKANA_BASE_URL" ] || { SAKANA_BASE_URL="$value"; export SAKANA_BASE_URL; _warn_env_endpoint_change "$from_env"; return 0; }
  if { [ "${SAKANA_BASE_URL_REQUESTED:-0}" = "1" ] || _managed_block_present; } && ! _bundle_renders_base_url; then
    die "This config bundle does not support selecting an API endpoint."
  fi
  case "$(_sakana_url_host "$value")" in
    localhost | 127.*) ;;
    *) [ "${value%%://*}" != http ] || log_warn "The Sakana API endpoint uses plain http://; the API key is sent unencrypted." ;;
  esac
  log_info "Sakana API endpoint: ${value}"
  SAKANA_BASE_URL="$value"
  export SAKANA_BASE_URL
  _warn_env_endpoint_change "$from_env"
}

_warn_env_endpoint_change() {
  [ "$1" = "1" ] && [ -f "$(_state_path)" ] && [ "$SAKANA_BASE_URL" != "${SAKANA_BASE_URL_BEFORE:-$DEFAULT_SAKANA_BASE_URL}" ] || return 0
  log_warn "Using SAKANA_BASE_URL from the environment (--reconfigure): the endpoint changes from ${SAKANA_BASE_URL_BEFORE} to ${SAKANA_BASE_URL}."
}

commit_sakana_base_url() {
  local saved=""
  saved="$(read_persisted_env_value SAKANA_BASE_URL 2>/dev/null || true)"
  if [ "$SAKANA_BASE_URL" != "$DEFAULT_SAKANA_BASE_URL" ]; then
    if [ -n "$saved" ] && [ "${SAKANA_BASE_URL_LINE_OWNED:-0}" != "1" ]; then
      if [ "$(_canon_sakana_base_url "$saved")" = "$SAKANA_BASE_URL" ]; then
        log_warn "Taking over the SAKANA_BASE_URL line in ${FUGU_ENV_FILE}, which this installer did not save: it now records ${SAKANA_BASE_URL} as its endpoint (--remove-config will remove the line)."
      else
        log_warn "Replacing the SAKANA_BASE_URL line in ${FUGU_ENV_FILE}, which this installer did not save, with ${SAKANA_BASE_URL}."
      fi
    fi
    [ "$saved" != "$SAKANA_BASE_URL" ] || return 0
    if [ "${DRY_RUN:-0}" = "1" ]; then
      log_info "[dry-run] would persist the custom SAKANA_BASE_URL to ${FUGU_ENV_FILE}"
    else
      persist_api_key "SAKANA_BASE_URL" "$SAKANA_BASE_URL"
    fi
  elif [ "${SAKANA_BASE_URL_LINE_OWNED:-0}" = "1" ]; then
    remove_api_key "SAKANA_BASE_URL"
  fi
}

_sakana_endpoint_changes() {
  [ "$SAKANA_BASE_URL" != "${SAKANA_BASE_URL_BEFORE:-$DEFAULT_SAKANA_BASE_URL}" ] || return 1
  [ "${SAKANA_BASE_URL_REQUESTED:-0}" = "1" ] || _managed_block_present
}

_provider_inject_refused() {
  local entry target rest marker snippet header
  for entry in "${INJECTS[@]:-}"; do
    [ -n "$entry" ] || continue
    target="${entry%%::*}"; rest="${entry#*::}"; marker="${rest%%::*}"; snippet="${rest##*::}"
    [ "$marker" = "model_providers.sakana" ] || continue
    header="$(grep -m1 -E '^\[[A-Za-z0-9_."-]+\]' "$BUNDLE_DIR/$snippet" 2>/dev/null || true)"
    [ -n "$header" ] && _block_conflicts "$CODEX_HOME/$target" "$marker" "$header" && return 0
  done
  return 1
}

_endpoint_change_not_applied() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    log_warn "[dry-run] the Fugu provider block could not be updated, so the endpoint change to ${SAKANA_BASE_URL} would fail."
    return 0
  fi
  die "The Fugu provider block was not updated, so the endpoint change to ${SAKANA_BASE_URL} was not applied; SAKANA_API_KEY and SAKANA_BASE_URL were left unchanged. Merge [model_providers.sakana] in ${CODEX_HOME}/config.toml (see ${SUPPORT_URL}) and re-run."
}

setup_api_key() {
  local var="$1" url="$2" regex="$3" provider="$4" existing value="" tries=0
  existing="$(printenv "$var" 2>/dev/null || true)"

  if [ "${RECONFIGURE:-0}" != "1" ] && key_already_persisted "$var" "$regex"; then
    log_info "${var} is already configured (in ${FUGU_ENV_FILE}); skipping. Use --reconfigure to change."
    _export_from_env_file "$var"
    return 0
  fi

  if [ -n "$existing" ]; then
    if _single_line "$existing" && printf '%s' "$existing" | grep -qE -- "$regex"; then
      value="$existing"; log_info "Using ${var} from the current environment."
    else
      die "${var} is set in the environment but does not match the expected format."
    fi
  elif [ "${ASSUME_YES:-0}" = "1" ]; then
    if [ "$var" = "SAKANA_API_KEY" ] && [ "${RECONFIGURE:-0}" != "1" ] && existing="$(read_persisted_env_value "$var" 2>/dev/null)" && \
       [ -n "$existing" ] && ! printf '%s' "$existing" | grep -qE -- "$regex"; then
      die "The ${var} stored in ${FUGU_ENV_FILE} does not match the key format for ${SAKANA_BASE_URL:-$DEFAULT_SAKANA_BASE_URL}. If it belongs to a custom endpoint, re-run with --base-url URL; otherwise rotate it with --set-key."
    fi
    die "${var} is required but not set, and --yes was given. Export ${var}=… or run interactively."
  else
    printf 'A %s API key is required for this bundle.\n' "$provider" >&2
    printf 'Get or create one here: %s\n' "$url" >&2
    while :; do
      printf 'Paste your %s (input hidden): ' "$var" >&2
      if ! IFS= read -rs value; then printf '\n' >&2; die "No input received for ${var}."; fi
      printf '\n' >&2
      value="$(printf '%s' "$value" | tr -d '[:space:]')"
      printf '%s' "$value" | grep -qE -- "$regex" && break
      tries=$((tries + 1)); value=""
      log_warn "That does not look like a valid ${var}. Try again (${tries}/${FUGU_KEY_PROMPT_TRIES})."
      [ "$tries" -ge "$FUGU_KEY_PROMPT_TRIES" ] && die "Too many invalid ${var} attempts."
    done
  fi

  printf -v "$var" '%s' "$value"
  export "$var"
  ENV_KEYS_TO_PERSIST+=("$var")
  unset value existing
  if [ "${DRY_RUN:-0}" = "1" ]; then
    log_info "[dry-run] would persist ${var} to ${FUGU_ENV_FILE}"
  elif [ "${CONFIG_ACTION:-install}" = "install" ]; then
    log_info "${var} validated; it will be committed after configuration deploys successfully."
  fi
}

commit_api_keys() {
  local var value
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  for var in "${ENV_KEYS_TO_PERSIST[@]:-}"; do
    [ -n "$var" ] || continue
    value="$(printenv "$var" 2>/dev/null || true)"
    persist_api_key "$var" "$value"
    log_ok "${var} configured (stored 0600 in ${FUGU_ENV_FILE}; Codex loads it automatically)."
  done
}

install_bundle_inject() {
  local entry="$1" target rest marker snippet tmp body
  target="${entry%%::*}"; rest="${entry#*::}"
  marker="${rest%%::*}";  snippet="${rest##*::}"
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-snip.XXXXXX")"
  render_template "$BUNDLE_DIR/$snippet" "$tmp"
  body="$(cat "$tmp")"; rm -f "$tmp"
  if ! inject_managed_block "$CODEX_HOME/$target" "$marker" "$body"; then
    BUNDLE_WARNINGS=$((BUNDLE_WARNINGS + 1))
    [ "$marker" = "model_providers.sakana" ] && BUNDLE_PROVIDER_DEPLOY_FAILED=1
  fi
  return 0
}

install_bundle_envkey() {
  local entry="$1" var rest url regex provider
  var="${entry%%::*}";    rest="${entry#*::}"
  url="${rest%%::*}";     rest="${rest#*::}"
  regex="${rest%%::*}";   provider="${rest##*::}"
  if [ "$var" = "SAKANA_API_KEY" ] && [ -n "$SAKANA_BASE_URL" ] && \
     ! _sakana_url_is_production_host "$SAKANA_BASE_URL"; then
    regex='^[A-Za-z0-9._~:/+=-]+$'
  fi
  setup_api_key "$var" "$url" "$regex" "$provider"
}

reconfigure_bundle_key() {
  local entry
  load_manifest_for
  load_sakana_base_url
  if [ "${#ENV_KEYS[@]}" -eq 0 ] || [ -z "${ENV_KEYS[0]:-}" ]; then
    log_info "The config bundle declares no API keys; nothing to configure."
    return 0
  fi
  RECONFIGURE=1
  for entry in "${ENV_KEYS[@]:-}"; do
    [ -n "$entry" ] || continue
    [ "${ASSUME_YES:-0}" = "1" ] || unset "${entry%%::*}"
    install_bundle_envkey "$entry"
  done
  commit_api_keys
  log_ok "API key configuration complete."
}

reconfigure_bundle_key_and_endpoint() {
  local entry rest marker provider_injects=0
  load_manifest_for
  CONFIG_NAME="$BUNDLE_ID"; BUNDLE_WARNINGS=0; BUNDLE_PROVIDER_DEPLOY_FAILED=0
  if [ ! -f "$(_state_path)" ]; then
    die "No Fugu install found in ${CODEX_HOME}; run install.sh (optionally with --base-url URL) first."
  fi
  [ -r "$(_state_path)" ] || die "Cannot read $(_state_path); fix its permissions, then re-run --set-key --base-url."
  validate_requested_sakana_base_url
  RECONFIGURE=1
  setup_sakana_base_url

  if [ "${#ENV_KEYS[@]}" -eq 0 ] || [ -z "${ENV_KEYS[0]:-}" ]; then
    die "The config bundle declares no API keys."
  fi
  ENV_KEYS_TO_PERSIST=()
  for entry in "${ENV_KEYS[@]:-}"; do
    [ -n "$entry" ] || continue
    [ "${ASSUME_YES:-0}" = "1" ] || unset "${entry%%::*}"
    install_bundle_envkey "$entry"
  done

  for entry in "${INJECTS[@]:-}"; do
    [ -n "$entry" ] || continue
    rest="${entry#*::}"; marker="${rest%%::*}"
    [ "$marker" = "model_providers.sakana" ] || continue
    provider_injects=$((provider_injects + 1))
    install_bundle_inject "$entry"
  done
  [ "$provider_injects" -gt 0 ] || die "This config bundle has no Sakana provider block to update."
  if [ "${BUNDLE_PROVIDER_DEPLOY_FAILED:-0}" = "1" ] && _sakana_endpoint_changes; then
    die "Fugu provider configuration was not updated; SAKANA_API_KEY and SAKANA_BASE_URL were left unchanged."
  fi

  commit_sakana_base_url
  commit_api_keys
  if [ "$SAKANA_BASE_URL" = "$DEFAULT_SAKANA_BASE_URL" ]; then _state_set_sakana_base_url ""
  else _state_set_sakana_base_url "$SAKANA_BASE_URL"; fi
  if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] API key and endpoint unchanged."
  else log_ok "API key and endpoint configuration complete."; fi
}

remove_bundle_file() {
  local src_rel="$1" dest_bn="$2" dest tmp
  dest="$CODEX_HOME/$dest_bn"
  [ -e "$dest" ] || { log_info "No ${dest_bn} to remove."; return 0; }
  tmp="$(mktemp "${TMPDIR:-/tmp}/fugu-cmp.XXXXXX")"
  render_template "$BUNDLE_DIR/$src_rel" "$tmp" 2>/dev/null || true
  if receipt_has_file "$CONFIG_NAME" "$dest_bn" || cmp -s "$tmp" "$dest"; then
    [ "${DRY_RUN:-0}" = "1" ] && { log_info "[dry-run] would remove ${dest}"; rm -f "$tmp"; return 0; }
    rm -f "$dest"; log_ok "Removed ${dest_bn} from ${CODEX_HOME}."
  else
    log_warn "${dest_bn} was modified since install; leaving it in place."
  fi
  rm -f "$tmp"
}

_clear_format_vars() {
  FORMAT_SCHEMA=""; FORMAT_ID=""; FORMAT_LATEST=""; FORMAT_MIN=""; FORMAT_MAX=""
  FORMAT_FILES=(); FORMAT_INJECTS=()
}

list_formats() {
  local dir="$1" d
  [ -d "$dir/formats" ] || return 0
  for d in "$dir"/formats/*/; do
    [ -f "${d}format.sh" ] || continue
    basename "$d"
  done | sort
}

source_format() {
  local dir="$1" id="$2" f="$1/formats/$2/format.sh"
  _clear_format_vars
  [ -f "$f" ] || die "Format manifest missing: ${f}"
  . "$f"
  [ "${FORMAT_SCHEMA:-0}" = "1" ] || die "Format '${id}' (${f}) uses unsupported schema '${FORMAT_SCHEMA:-?}'."
  [ -n "${FORMAT_ID:-}" ] || FORMAT_ID="$id"
}

select_latest_format() {
  local dir="$1" id count=0 latest=""
  FORMAT_LATEST_ID=""
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    source_format "$dir" "$id"
    if [ "${FORMAT_LATEST:-0}" = "1" ]; then count=$((count + 1)); latest="$id"; fi
  done < <(list_formats "$dir")
  [ "$count" -eq 1 ] || die "Bundle '${BUNDLE_NAME:-?}' must mark exactly one format FORMAT_LATEST=1 (found ${count})."
  FORMAT_LATEST_ID="$latest"
  source_format "$dir" "$latest"
}

_format_version_sanity_warn() {
  local got; got="$(get_installed_version)"
  [ -n "$got" ] || return 0
  if [ -n "${FORMAT_MIN:-}" ] && _ver_lt "$got" "$FORMAT_MIN"; then
    log_warn "Codex ${got} is below the '${FORMAT_LATEST_ID}' layout's minimum (${FORMAT_MIN}); it may not load this config."
  elif [ -n "${FORMAT_MAX:-}" ] && ! _ver_lt "$got" "$FORMAT_MAX"; then
    log_warn "Codex ${got} is at/above the '${FORMAT_LATEST_ID}' layout's ceiling (${FORMAT_MAX}); it may not load this config."
  fi
}

_migrate_old_inject() {
  local file="$1" marker="$2" snippet_file="$3" open="# >>> fugu:${2} >>>" header
  [ -f "$file" ] || return 0
  if grep -qxF "$open" "$file" 2>/dev/null; then
    if [ "${DRY_RUN:-0}" = "1" ]; then
      log_info "[dry-run] would migrate: remove old-format block fugu:${marker} from ${file}"; return 0
    fi
    _is_codex_config "$file" && backup_once "bundle:${CONFIG_NAME}" "bundle:${CONFIG_NAME}"
    remove_managed_block "$file" "$marker"
    log_info "Migrated: removed old-format block fugu:${marker} from $(basename "$file")."
    return 0
  fi
  header="$(grep -m1 -E '^\[[A-Za-z0-9_."-]+\]' "$snippet_file" 2>/dev/null || true)"
  [ -n "$header" ] || return 0
  if _block_conflicts "$file" "$marker" "$header"; then
    [ "${DRY_RUN:-0}" = "1" ] || { _is_codex_config "$file" && backup_once "bundle:${CONFIG_NAME}" "bundle:${CONFIG_NAME}"; }
    log_warn "Found a hand-added '${header}' in ${file} from an older Fugu config layout; the current layout no longer uses it and a newer Codex may reject it."
    log_warn "Please remove it manually (your config was backed up). See ${SUPPORT_URL}"
    BUNDLE_WARNINGS=$((BUNDLE_WARNINGS + 1))
  fi
}

migrate_old_formats() {
  local dir="$1" latest="$2" id entry target rest marker snippet
  while IFS= read -r id; do
    [ -n "$id" ] || continue
    [ "$id" = "$latest" ] && continue
    source_format "$dir" "$id"
    for entry in "${FORMAT_INJECTS[@]:-}"; do
      [ -n "$entry" ] || continue
      target="${entry%%::*}"; rest="${entry#*::}"; marker="${rest%%::*}"; snippet="${rest##*::}"
      printf '%s' "$MIGRATE_KEEP_MARKERS" | grep -qxF "$marker" && continue
      _migrate_old_inject "$CODEX_HOME/$target" "$marker" "$dir/$snippet"
    done
    for entry in "${FORMAT_FILES[@]:-}"; do
      [ -n "$entry" ] || continue
      printf '%s' "$MIGRATE_KEEP_DESTS" | grep -qxF "${entry##*::}" && continue
      remove_bundle_file "${entry%%::*}" "${entry##*::}"
    done
  done < <(list_formats "$dir")
  source_format "$dir" "$latest"
}

install_config_bundle() {
  local name="$BUNDLE_ID" dir entry rest m d has_formats=0
  dir="$(resolve_bundle_dir)"
  CONFIG_NAME="$name"; BUNDLE_WARNINGS=0; BUNDLE_PROVIDER_DEPLOY_FAILED=0; _RUN_BACKUP_DONE=0; DEPLOYED_FORMAT=""
  load_manifest "$dir"
  if [ -n "$(list_formats "$dir")" ]; then
    has_formats=1; select_latest_format "$dir"; DEPLOYED_FORMAT="$FORMAT_LATEST_ID"
  fi
  log_info "Loading config bundle '${BUNDLE_NAME}': ${BUNDLE_DESC:-}"

  resolve_fugu_model
  setup_sakana_base_url

  ENV_KEYS_TO_PERSIST=()
  for entry in "${ENV_KEYS[@]:-}"; do [ -n "$entry" ] && install_bundle_envkey "$entry"; done
  if [ "${DRY_RUN:-0}" != "1" ] && [ "${SAKANA_BASE_URL_REQUESTED:-0}" = "1" ] && _sakana_endpoint_changes && _provider_inject_refused; then
    log_warn "Found an existing '[model_providers.sakana]' in ${CODEX_HOME}/config.toml outside Fugu's managed markers; not editing it."
    _endpoint_change_not_applied
  fi
  if _sakana_endpoint_changes && [ -f "$FUGU_ENV_FILE" ] && \
     ! printf '%s\n' "${ENV_KEYS_TO_PERSIST[@]:-}" | grep -qxF SAKANA_API_KEY && \
     read_persisted_env_value SAKANA_API_KEY >/dev/null 2>&1; then
    log_warn "The endpoint changes to ${SAKANA_BASE_URL} but the stored SAKANA_API_KEY is kept and will be sent there. To change both together, run: install.sh --set-key --base-url ${SAKANA_BASE_URL}"
  fi

  for entry in "${FILES[@]:-}";   do [ -n "$entry" ] && install_bundle_file   "${entry%%::*}" "${entry##*::}"; done
  for entry in "${INJECTS[@]:-}"; do [ -n "$entry" ] && install_bundle_inject "$entry"; done

  if [ "$has_formats" = 1 ]; then
    _format_version_sanity_warn
    log_info "Config layout: deploying format '${FORMAT_LATEST_ID}' (latest); migrating off any older layout."
    for entry in "${FORMAT_FILES[@]:-}";   do [ -n "$entry" ] && install_bundle_file   "${entry%%::*}" "${entry##*::}"; done
    for entry in "${FORMAT_INJECTS[@]:-}"; do [ -n "$entry" ] && install_bundle_inject "$entry"; done
    MIGRATE_KEEP_MARKERS=""; MIGRATE_KEEP_DESTS=""
    for entry in "${INJECTS[@]:-}" "${FORMAT_INJECTS[@]:-}"; do
      [ -n "$entry" ] || continue
      rest="${entry#*::}"; m="${rest%%::*}"; MIGRATE_KEEP_MARKERS="${MIGRATE_KEEP_MARKERS}${m}"$'\n'
    done
    for entry in "${FILES[@]:-}" "${FORMAT_FILES[@]:-}"; do
      [ -n "$entry" ] || continue
      d="${entry##*::}"; MIGRATE_KEEP_DESTS="${MIGRATE_KEEP_DESTS}${d}"$'\n'
    done
    migrate_old_formats "$dir" "$FORMAT_LATEST_ID"
  fi

  if [ "${BUNDLE_PROVIDER_DEPLOY_FAILED:-0}" = "1" ]; then
    if [ "${SAKANA_BASE_URL_REQUESTED:-0}" = "1" ] && _sakana_endpoint_changes; then
      _endpoint_change_not_applied
    else
      if _provider_inject_refused; then
        log_warn "The Fugu provider block was not updated; your existing [model_providers.sakana] stays in effect."
      elif [ -f "$CODEX_HOME/config.toml" ]; then
        log_warn "The Fugu provider block was not updated; ${CODEX_HOME}/config.toml keeps its previous provider settings."
      else
        log_warn "The Fugu provider block was not updated; ${CODEX_HOME}/config.toml was not created."
      fi
      commit_api_keys
      if [ "${SAKANA_BASE_URL_REQUESTED:-0}" = "1" ]; then
        commit_sakana_base_url
        SAKANA_BASE_URL_RECORD=""
        [ "$SAKANA_BASE_URL" = "$DEFAULT_SAKANA_BASE_URL" ] || SAKANA_BASE_URL_RECORD="$SAKANA_BASE_URL"
      fi
    fi
  else
    commit_api_keys
    commit_sakana_base_url
    SAKANA_BASE_URL_RECORD=""
    [ "$SAKANA_BASE_URL" = "$DEFAULT_SAKANA_BASE_URL" ] || SAKANA_BASE_URL_RECORD="$SAKANA_BASE_URL"
  fi

  if [ "${BUNDLE_WARNINGS:-0}" -gt 0 ]; then
    log_warn "Config bundle '${name}' loaded with ${BUNDLE_WARNINGS} warning(s); see above and ${SUPPORT_URL}."
  else
    log_ok "Config bundle '${name}' loaded."
  fi
  if [ -n "${BUNDLE_RUN_HINT:-}" ]; then log_ok "Run it with:  ${BUNDLE_RUN_HINT}"; fi
}

remove_config_bundle() {
  local name="$BUNDLE_ID" dir entry target rest marker
  dir="$(resolve_bundle_dir)"
  CONFIG_NAME="$name"
  load_manifest "$dir"
  log_info "Removing config bundle '${name}'."
  local url_owned=0; ! _sakana_base_url_line_owned || url_owned=1
  for entry in "${INJECTS[@]:-}"; do
    [ -n "$entry" ] || continue
    target="${entry%%::*}"; rest="${entry#*::}"; marker="${rest%%::*}"
    remove_managed_block "$CODEX_HOME/$target" "$marker"
    if [ -f "$CODEX_HOME/$target" ] && ! grep -qE '[^[:space:]]' "$CODEX_HOME/$target"; then
      [ "${DRY_RUN:-0}" = "1" ] || rm -f "$CODEX_HOME/$target"
      log_info "Removed now-empty ${target}."
    fi
  done
  for entry in "${FILES[@]:-}";    do [ -n "$entry" ] && remove_bundle_file "${entry%%::*}" "${entry##*::}"; done
  if [ -n "$(list_formats "$dir")" ]; then
    local fid
    while IFS= read -r fid; do
      [ -n "$fid" ] || continue
      source_format "$dir" "$fid"
      for entry in "${FORMAT_INJECTS[@]:-}"; do
        [ -n "$entry" ] || continue
        target="${entry%%::*}"; rest="${entry#*::}"; marker="${rest%%::*}"
        remove_managed_block "$CODEX_HOME/$target" "$marker"
        if [ -f "$CODEX_HOME/$target" ] && ! grep -qE '[^[:space:]]' "$CODEX_HOME/$target"; then
          [ "${DRY_RUN:-0}" = "1" ] || rm -f "$CODEX_HOME/$target"
          log_info "Removed now-empty ${target}."
        fi
      done
      for entry in "${FORMAT_FILES[@]:-}"; do
        [ -n "$entry" ] && remove_bundle_file "${entry%%::*}" "${entry##*::}"
      done
    done < <(list_formats "$dir")
  fi
  for entry in "${ENV_KEYS[@]:-}"; do [ -n "$entry" ] && remove_api_key "${entry%%::*}"; done
  [ "$url_owned" != 1 ] || remove_api_key "SAKANA_BASE_URL"
  local rp p line; rp="$(_receipt_path "$name")"
  if [ -f "$rp" ]; then
    while IFS= read -r line; do
      case "$line" in
        path\ *) p="${line#path }"
          if [ -e "$p" ]; then
            if [ "${DRY_RUN:-0}" = "1" ]; then log_info "[dry-run] would remove ${p}"; else rm -f "$p"; log_ok "Removed ${p}."; fi
          fi ;;
      esac
    done < "$rp"
  fi
  local sp; sp="$(_state_path)"
  if [ -f "$sp" ] && grep -qxF "bundle=$name" "$sp" 2>/dev/null; then
    [ "${DRY_RUN:-0}" = "1" ] || rm -f "$sp"
    log_info "Removed Fugu state for ${name}."
  fi
  [ "${DRY_RUN:-0}" = "1" ] || rm -f "$(_receipt_path "$name")"
  log_ok "Config bundle '${name}' removed."
}

note_resume_caveat() {
  log_warn "Switching Codex versions changes which past sessions 'codex resume' lists (Codex keeps a per-version session index)."
  log_warn "Your transcripts are not deleted. The current session index is saved to the backup below. To list older sessions again, run the Codex version that wrote them or restore the saved index."
}

ensure_codex_version() {
  local pinned="$1" installed=""

  if ! codex_is_installed; then
    log_info "Codex is not currently installed."
    install_codex "$pinned"
    log_ok "Codex ${pinned} installed."
    RESULT_STATUS="installed"
    return 0
  fi

  installed="$(get_installed_version)"

  if [ -z "$installed" ]; then
    log_warn "Codex is present but its version could not be determined (it may be broken)."
    note_resume_caveat
    if confirm "Reinstall/repair Codex ${pinned} now?"; then
      backup_codex_config "unknown" "$pinned"
      install_codex "$pinned"
      log_ok "Codex ${pinned} (re)installed."
      RESULT_STATUS="repaired"
    else
      log_warn "Continuing with an unverifiable Codex installation."
      RESULT_STATUS="proceed_warn"
    fi
    return 0
  fi

  if [ "$installed" = "$pinned" ]; then
    log_ok "Codex ${pinned} is already installed."
    RESULT_STATUS="already"
    return 0
  fi

  log_warn "Installed Codex ${installed} differs from the pinned version ${pinned}."
  local dtgt="${BUNDLE_CODEX_VERSION:-$pinned}"
  if [ "${FORCE:-0}" != "1" ] && mismatch_declined "$installed" "$dtgt"; then
    log_warn "You previously chose to keep Codex ${installed} for this mismatch; not asking again."
    log_warn "Run 'codex-fugu --recheck' to clear that choice and be prompted again, or re-run with --force to switch now."
    RESULT_STATUS="declined"
    return 0
  fi
  local do_switch=0 human_declined=0
  if [ "${FORCE:-0}" = "1" ]; then
    log_warn "--force: switching Codex ${installed} -> ${pinned} non-interactively."
    note_resume_caveat
    do_switch=1
  elif [ "${ASSUME_YES:-0}" = "1" ]; then
    log_warn "Not switching Codex under --yes (that would change your binary). Re-run interactively to be prompted, or pass --force."
  else
    note_resume_caveat
    if confirm "Fugu models are optimized for Codex ${pinned}, but you currently have ${installed}. Switch to Codex ${pinned} now?"; then
      do_switch=1
    else
      human_declined=1
    fi
  fi
  if [ "$do_switch" = "1" ]; then
    backup_codex_config "$installed" "$pinned"
    install_codex "$pinned"
    log_ok "Switched to Codex ${pinned}."
    RESULT_STATUS="switched"
  else
    log_warn "Keeping Codex ${installed}. These Fugu configs are built for ${pinned}; on a different Codex they may not be fully applied."
    [ "$human_declined" = "1" ] && [ -f "$(_state_path)" ] && record_mismatch_decision "$installed" "$dtgt"
    RESULT_STATUS="declined"
  fi
  return 0
}

deploy_gate_ok() {
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  [ "${FORCE:-0}" = "1" ] && { log_warn "--force: deploying bundle despite any Codex version mismatch."; return 0; }
  [ -n "${BUNDLE_CODEX_VERSION:-}" ] || return 0
  local got; got="$(get_installed_version)"
  [ "$got" = "$BUNDLE_CODEX_VERSION" ] && return 0
  if [ -n "$got" ] && _ver_lt "$BUNDLE_CODEX_VERSION" "$got"; then
    log_error "Refusing to deploy bundle '${CONFIG_NAME}': your Codex ${got} is newer than this bundle's target ${BUNDLE_CODEX_VERSION} (configs are verified for an exact Codex version)."
    log_warn  "To deploy these configs, downgrade Codex to ${BUNDLE_CODEX_VERSION} by re-running with --force, or update to configs that target ${got} (when available)."
  else
    log_error "Refusing to deploy bundle '${CONFIG_NAME}': Codex is '${got:-unreadable}', but the bundle targets ${BUNDLE_CODEX_VERSION}."
    log_warn  "Switch Codex to ${BUNDLE_CODEX_VERSION} by re-running with --force (or re-run interactively, which prompts again unless you previously declined), then retry."
  fi
  log_warn  "See ${SUPPORT_URL}"
  return 1
}

run_next_stage() {
  local pinned="$1"
  case "${CONFIG_ACTION:-}" in
    install)
      if deploy_gate_ok; then
        install_config_bundle
        record_fugu_state     "$BUNDLE_ID" "${BUNDLE_CODEX_VERSION:-$pinned}"
        install_fugu_launcher "$BUNDLE_ID"
      else
        if [ "${FUGU_MODEL_EXPLICIT:-0}" = "1" ] || [ -n "${SAKANA_BASE_URL:-}" ]; then
          log_warn "The requested default model / API endpoint was not applied because the config bundle was not deployed."
        fi
        if [ -f "$(_state_path)" ]; then
          install_fugu_launcher "$BUNDLE_ID"
          advance_fugu_install_ref
        fi
      fi
      ;;
    *) : ;;
  esac
}

print_success() {
  local pinned="$1"
  case "${RESULT_STATUS:-}" in
    declined | proceed_warn)
      local got; got="$(get_installed_version 2>/dev/null || true)"
      log_warn "Bootstrap finished: the config bundle was not deployed because Codex ${got:-(unreadable)} does not match the configs' target ${pinned}."
      log_ok   "Fugu bootstrap complete (with warnings)."
      ;;
    *)
      log_ok "Installation succeeded — Codex CLI ${pinned} is ready. Fugu bootstrap complete."
      ;;
  esac
}

main() {
  parse_args "$@"

  case "${CONFIG_ACTION:-}" in
    remove) remove_config_bundle; exit 0 ;;
    setkey)
      if [ "${SAKANA_BASE_URL_CLI_EXPLICIT:-0}" = "1" ]; then
        reconfigure_bundle_key_and_endpoint
      else
        reconfigure_bundle_key
      fi
      exit $?
      ;;
  esac

  require_http_client
  local pinned=""
  if [ "${CONFIG_ACTION:-}" = "install" ]; then
    load_manifest_for
    resolve_fugu_model
    validate_requested_sakana_base_url
    check_saved_sakana_base_url
    check_requested_endpoint_applicable
    if [ -z "${PINNED_VERSION_OVERRIDE:-}" ] && [ -n "${BUNDLE_CODEX_VERSION:-}" ]; then
      pinned="$BUNDLE_CODEX_VERSION"
      log_info "Pinning to the config bundle's target Codex ${pinned}."
    fi
  fi
  [ -n "$pinned" ] || pinned="$(resolve_latest_version)"
  log_info "Pinned Codex version: ${pinned}"

  ensure_codex_version "$pinned"
  run_next_stage "$pinned"
  print_success "$pinned"
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
