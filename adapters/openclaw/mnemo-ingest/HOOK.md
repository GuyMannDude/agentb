---
name: mnemo-ingest
description: "Live Wire: automatically captures every prompt/response exchange to Mnemo Cortex's /ingest endpoint. No manual saves. No lost sessions."
metadata:
  openclaw:
    emoji: "⚡"
    events: ["command:new", "session:message"]
    requires:
      bins: ["curl"]
---

# ⚡ Mnemo Cortex Live Wire Hook

Automatically captures every prompt/response exchange to Mnemo Cortex.
If Anthropic pulls the plug, if your session crashes, if the power goes out —
every conversation up to the last exchange is already on disk.

## Setup

1. Install Mnemo Cortex:
   ```bash
   pip install mnemo-cortex
   mnemo-cortex init
   mnemo-cortex start
   ```

2. Copy this hook to your OpenClaw workspace:
   ```bash
   cp -r adapters/openclaw/mnemo-ingest ~/.openclaw/workspace/hooks/
   ```

3. Enable it:
   ```bash
   openclaw hooks enable mnemo-ingest
   ```

4. Set your agent ID (optional, for multi-tenant isolation):
   ```bash
   export MNEMO_AGENT_ID=rocky
   ```

5. Restart the gateway:
   ```bash
   openclaw gateway restart
   ```

## How It Works

After every assistant response, this hook sends the prompt/response pair
to Mnemo Cortex's `/ingest` endpoint. The write takes <5ms and is
append-only, so it never slows down your agent.

On session start (`/new`), the hook also queries `/sessions/recent` to
inject recent conversation context into the bootstrap — so your agent
remembers what happened last session without you having to tell it.

## Environment Variables

- `MNEMO_URL` — Mnemo Cortex server URL (default: `http://localhost:50001`)
- `MNEMO_AGENT_ID` — Agent ID for multi-tenant isolation (default: none)
- `MNEMO_AUTH_TOKEN` — API auth token if configured (default: none)
