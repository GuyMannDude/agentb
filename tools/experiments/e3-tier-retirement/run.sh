#!/bin/bash
# E3 tier probe — one read-only /context sweep per tenant, using the
# Experiment One probe unchanged. Set MNEMO_URL, EXP1_QUERIES (your own query
# list; ours stays out of the public repo) and list your tenants below.
# Output: <tenant>/recalls.jsonl next to each probe copy (gitignored).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PROBE=$HERE/../ranker-exp1/probe.py
export MNEMO_URL=${MNEMO_URL:-http://127.0.0.1:50001}
export EXP1_QUERIES=${EXP1_QUERIES:?path to a query list, one per line}
for t in "$@"; do
  mkdir -p "$HERE/$t" && cp "$PROBE" "$HERE/$t/probe.py"
  echo "=== tenant $t ==="
  MNEMO_AGENT_ID=$t python3 "$HERE/$t/probe.py"
done
echo ALLDONE
