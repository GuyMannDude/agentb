# ChatGPT (Custom GPT Actions) — Mnemo Gate

Give a ChatGPT Custom GPT persistent memory backed by your own Mnemo Cortex
server. ChatGPT calls two REST Actions — `recallMemory` and `saveMemory` —
over public HTTPS; a small authenticated gate forwards them to your private
Mnemo server, pinned to one memory tenant.

> 📖 **Full step-by-step guide with the OpenAI plan disclaimer, HTTPS setup,
> and troubleshooting: [docs/install-chatgpt.md](../../docs/install-chatgpt.md).**

## Why a gate instead of exposing Mnemo directly

Your Mnemo API key grants full access to every agent's memory. ChatGPT should
never hold that key. The gate:

- exposes **only two routes** (`/recall`, `/save`) — everything else is 404
- holds its **own bearer token**; the Mnemo key never leaves the gate process
- **pins every request to one tenant** (`MNEMO_GATE_AGENT_ID`, default
  `chatgpt`) — a caller-supplied `agent_id` is silently ignored
- forces `source=user` and a `chatgpt-gate` tag on every save
- rate-limits (10 requests/hour by default), caps request bodies at 8KB,
  restricts categories, and audit-logs every request (snippet only, rotated
  at 5MB)
- returns generic errors on upstream failures — no internal details leak

## Secrets

Two token files, both outside the repo:

| File | Holds |
|---|---|
| `~/.mnemo-gate/token` | The gate's public bearer token (what ChatGPT sends). Generate ≥32 random chars. |
| `~/.mnemo-auth-token` | Your Mnemo server API key (what the gate sends upstream). |

Never commit either. On Windows the paths are under `%USERPROFILE%`.
Restrict file permissions to your user.

## Run

The gate needs Python 3.11+ with `fastapi`, `uvicorn`, `httpx`, and
`pydantic` — all already installed if this machine runs Mnemo Cortex.

Linux/macOS:

```bash
python -m uvicorn server:create_app --factory \
  --host 127.0.0.1 --port 50002 --no-access-log
```

Windows (and auto-start via Task Scheduler):

```powershell
powershell -ExecutionPolicy Bypass -File .\run-gate.ps1
powershell -ExecutionPolicy Bypass -File .\install-task.ps1
```

The gate listens only on `127.0.0.1:50002`. Publish it with Tailscale Funnel
or any HTTPS reverse proxy (see the [full guide](../../docs/install-chatgpt.md)).

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MNEMO_GATE_UPSTREAM_URL` | `http://127.0.0.1:50001` | Your Mnemo server. |
| `MNEMO_GATE_AGENT_ID` | `chatgpt` | The one tenant this gate reads and writes. |
| `MNEMO_GATE_TOKEN_FILE` | `~/.mnemo-gate/token` | Public bearer token file. |
| `MNEMO_GATE_UPSTREAM_TOKEN_FILE` | `~/.mnemo-auth-token` | Mnemo API key file. |
| `MNEMO_GATE_RATE_LIMIT` | `10` | Authorized requests per hour. |
| `MNEMO_GATE_AUDIT_FILE` | `~/.mnemo-gate/audit.jsonl` | Audit log location. |
| `MNEMO_GATE_AUDIT_ROTATE_BYTES` | `5242880` | Audit rotation threshold. |

## Tests

```bash
python -m pytest -q tests/test_chatgpt_gate.py
```

The suite proves: 401 without the token, Mnemo-key rejection on the public
side, 404 for unknown routes, forced tenant/source/tag fields, category
rejection, rate limiting, 8KB request caps, generic upstream errors, and one
audit line per request.
