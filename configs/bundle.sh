
BUNDLE_NAME="configs"
BUNDLE_DESC="Sakana Fugu models (fugu / fugu-ultra), optimized — stream-resilience hardening + generic agent-conduct base_instructions; use with: codex -p fugu"
BUNDLE_SCHEMA=1

BUNDLE_CODEX_VERSION="0.142.4"

FILES=(
  "files/fugu.json::fugu.json"
)

INJECTS=(
  "config.toml::model_providers.sakana::injects/model_providers.sakana.toml"
)

ENV_KEYS=(
  "SAKANA_API_KEY::https://platform.torafugu.app/api-keys::^fish_[0-9a-f]{64}\$::Sakana API"
)

BUNDLE_RUN_HINT="codex-fugu"
