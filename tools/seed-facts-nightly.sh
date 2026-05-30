#!/usr/bin/env bash
# Nightly safety-net re-seed of Mnemo Facts from the facts YAML.
#
# The post-commit hook catches edits made on this machine. This catches the
# drift it can't see: facts committed from another machine, or Mnemo being
# restarted / reset so the table needs re-asserting. Run it once a day.
#
# Crontab example (3:10 AM — adjust to taste):
#     10 3 * * * SEED_FACTS_REPO=$HOME/my-brain \
#                /path/to/mnemo-cortex/tools/seed-facts-nightly.sh
#
# Configure via env:
#     SEED_FACTS_REPO   repo holding seed-facts.yaml; pulled before seeding
#                       (default: current directory, no pull)
#     SEED_FACTS_YAML   YAML filename relative to the repo (default: seed-facts.yaml)
#     SEED_FACTS_PY     path to seed-facts.py (default: <this script's dir>/seed-facts.py)
#     SEED_FACTS_LOG    log file (default: ~/.mnemo-seed-facts.log)
#     MNEMO_URL         Mnemo base URL (default: http://127.0.0.1:50001)
#     MNEMO_AUTH_TOKEN  X-API-KEY, only if the service enforces auth
#
# Exits non-zero if the seeder fails, so a cron-failure mailer / wrapper can
# alert. Detection + assertion only — never commits or pushes anything.

set -uo pipefail

HOOK_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="${SEED_FACTS_REPO:-$PWD}"
YAML="${SEED_FACTS_YAML:-seed-facts.yaml}"
SEED_PY="${SEED_FACTS_PY:-$HOOK_DIR/seed-facts.py}"
LOG="${SEED_FACTS_LOG:-$HOME/.mnemo-seed-facts.log}"

PY_BIN="python3"
if [ -x "$HOOK_DIR/../.venv/bin/python" ]; then
  PY_BIN="$HOOK_DIR/../.venv/bin/python"
fi

{
  echo "===== $(date -Iseconds) ====="
  if [ -n "${SEED_FACTS_REPO:-}" ] && [ -d "$REPO/.git" ]; then
    if ! git -C "$REPO" pull --quiet --ff-only; then
      echo "[WARN] git pull failed — seeding from local state anyway"
    fi
  fi
  "$PY_BIN" "$SEED_PY" --yaml "$REPO/$YAML"
} >> "$LOG" 2>&1
