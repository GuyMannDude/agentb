#!/usr/bin/env bash
# Git post-commit hook: re-seed Mnemo Facts whenever the facts YAML changes.
#
# Keeps the Facts table in step with the source of truth — commit an edit to
# seed-facts.yaml and the new/changed facts land in Mnemo automatically.
#
# Install (once, in the repo that holds your seed-facts.yaml):
#     ln -sf /path/to/mnemo-cortex/tools/seed-facts-post-commit.sh \
#            "$(git rev-parse --show-toplevel)/.git/hooks/post-commit"
#
# Configure via env (export before committing, or set in the hook's shell):
#     SEED_FACTS_YAML   filename to watch, relative to repo root
#                       (default: seed-facts.yaml)
#     SEED_FACTS_PY     path to seed-facts.py
#                       (default: <this script's dir>/seed-facts.py)
#     MNEMO_URL         Mnemo base URL (default: http://127.0.0.1:50001)
#     MNEMO_AUTH_TOKEN  X-API-KEY, only if the service enforces auth
#
# Non-fatal on failure: the commit already succeeded, so a seeding hiccup
# (Mnemo down, etc.) prints a warning and exits 0 — never a scary red hook.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
YAML="${SEED_FACTS_YAML:-seed-facts.yaml}"
HOOK_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SEED_PY="${SEED_FACTS_PY:-$HOOK_DIR/seed-facts.py}"

# Only fire when the YAML was part of this commit.
if ! git diff-tree --no-commit-id --name-only -r HEAD | grep -qx "$YAML"; then
  exit 0
fi

echo "[post-commit] $YAML changed → re-seeding Mnemo Facts..."

# Pick an interpreter: a Mnemo venv if present, else python3.
PY_BIN="python3"
if [ -x "$HOOK_DIR/../.venv/bin/python" ]; then
  PY_BIN="$HOOK_DIR/../.venv/bin/python"
fi

# 30s timeout — the seeder is a few seconds normally; if Mnemo is down we don't
# want to block the terminal.
if timeout 30 "$PY_BIN" "$SEED_PY" --yaml "$REPO_ROOT/$YAML" 2>&1 | tail -8; then
  echo "[post-commit] seed complete."
else
  echo "[post-commit] WARNING: seeder failed (commit still succeeded). Run manually:"
  echo "             $PY_BIN $SEED_PY --yaml $REPO_ROOT/$YAML"
fi

exit 0
