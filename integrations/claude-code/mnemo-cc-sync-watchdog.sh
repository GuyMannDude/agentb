#!/usr/bin/env bash
# mnemo-cc-sync-watchdog — health check for mnemo-cc-sync.service.
#
# Verifies:
#   1. The systemd service is active
#   2. The newest Claude Code session JSONL has no unsynced backlog that the
#      sync should already have flushed (byte_offset in the offset file has
#      kept up with the file on disk)
#
# Why backlog instead of mtime/last-post: an open-but-idle Claude Code
# terminal gets hourly housekeeping appends that bump the JSONL mtime without
# producing anything postable. The sync correctly consumes those lines and
# advances byte_offset WITHOUT posting, so "mtime fresh but no recent post"
# is normal, not a failure (2026-07-08: 13 false Discord pages in one day).
# A genuinely stuck sync shows up here as backlog bytes that survive past the
# idle-flush window. Tradeoff: a sync that wedges mid-session isn't flagged
# until the session next goes quiet for FLUSH_GRACE_S — acceptable, sessions
# pause constantly and the old heuristic's false pages cost more than the
# shorter detection gap bought.
#
# Exits non-zero on failure so cron / scheduler can alert. Designed to plug
# into any monitoring tool that watches command exit codes — CronAlarm,
# systemd OnFailure=, healthchecks.io, or a plain cron + email.
#
# Configuration (env vars, all optional):
#   MNEMO_CC_SERVICE        systemd unit name (default: mnemo-cc-sync.service)
#   MNEMO_CC_OFFSET_FILE    Sync offset state file (default: ~/.mnemo-cc/cc-sync.offset.json)
#   MNEMO_CC_SESSIONS_DIR   Where Claude Code stores .jsonl files (default: ~/.claude/projects)
#   MNEMO_CC_FLUSH_GRACE_S  Seconds an idle JSONL may hold unsynced bytes before
#                           that counts as stuck (default: 600 = idle-flush 300s
#                           + several 60s sync cycles of margin)

set -e

SERVICE=${MNEMO_CC_SERVICE:-mnemo-cc-sync.service}
OFFSET=${MNEMO_CC_OFFSET_FILE:-$HOME/.mnemo-cc/cc-sync.offset.json}
SESSIONS_DIR=${MNEMO_CC_SESSIONS_DIR:-$HOME/.claude/projects}
FLUSH_GRACE_S=${MNEMO_CC_FLUSH_GRACE_S:-600}

# 1) Service must be active
if ! systemctl --user is-active --quiet "$SERVICE"; then
    echo "WATCHDOG FAIL: $SERVICE is not active"
    systemctl --user status "$SERVICE" --no-pager 2>&1 | head -10
    exit 1
fi

# 2) If no session JSONL exists, nothing to sync. Service-active alone passes.
LATEST_JSONL=$(find "$SESSIONS_DIR" -name "*.jsonl" -printf "%T@ %p\n" 2>/dev/null \
               | sort -rn | head -1 | cut -d' ' -f2-)
if [ -z "$LATEST_JSONL" ]; then
    echo "OK: $SERVICE active, no Claude Code sessions to sync"
    exit 0
fi

if [ ! -f "$OFFSET" ]; then
    echo "WATCHDOG FAIL: sessions exist but no offset file at $OFFSET — sync has never completed a cycle"
    exit 1
fi

NOW=$(date +%s)
JSONL_MTIME=$(stat -c %Y "$LATEST_JSONL")
JSONL_AGE=$((NOW - JSONL_MTIME))
JSONL_SIZE=$(stat -c %s "$LATEST_JSONL")
REL_PATH=${LATEST_JSONL#"$SESSIONS_DIR"/}

SYNCED_OFFSET=$(python3 -c "
import json, sys
state = json.load(open('$OFFSET'))
entry = state.get('files', {}).get('$REL_PATH')
print(entry['byte_offset'] if entry else -1)
")

# Newest JSONL not registered yet: the sync scans every cycle, so a file this
# old with no entry means the scanner isn't picking it up.
if [ "$SYNCED_OFFSET" -lt 0 ]; then
    if [ "$JSONL_AGE" -gt "$FLUSH_GRACE_S" ]; then
        echo "WATCHDOG FAIL: $REL_PATH is ${JSONL_AGE}s old but has no offset entry — sync isn't scanning it"
        exit 1
    fi
    echo "OK: $SERVICE active, $REL_PATH is new (${JSONL_AGE}s) — registration pending"
    exit 0
fi

BACKLOG=$((JSONL_SIZE - SYNCED_OFFSET))

# Fully consumed — sync has kept up, regardless of what last touched the file
if [ "$BACKLOG" -le 0 ]; then
    echo "OK: $SERVICE active, no unsynced backlog on $REL_PATH"
    exit 0
fi

# Backlog on a recently-written file is normal: messages accumulate between
# 60s sync cycles, and sub-minimum batches defer until the 300s idle-flush.
if [ "$JSONL_AGE" -le "$FLUSH_GRACE_S" ]; then
    echo "OK: $SERVICE active, ${BACKLOG}B pending on $REL_PATH (written ${JSONL_AGE}s ago — within flush grace)"
    exit 0
fi

echo "WATCHDOG FAIL: ${BACKLOG}B unsynced on $REL_PATH despite ${JSONL_AGE}s of idle — sync is stuck"
echo "  service: $(systemctl --user is-active $SERVICE)"
echo "  offset:  $OFFSET (byte_offset=$SYNCED_OFFSET, file size=$JSONL_SIZE)"
exit 1
