# 🧠 AgentB

**Drop-in memory coprocessor for AI agents.**

Give any AI agent persistent memory, context retrieval, and response validation — in four HTTP calls. No cloud lock-in. Runs on Ollama, OpenAI, Anthropic, Google, or OpenRouter.

```
Your Agent                    AgentB (port 50001)
    │                              │
    │── POST /context ────────────▶│  "What do you remember about Easter?"
    │◀── memory chunks ────────────│  L1 cache → L2 index → L3 scan
    │                              │
    │── POST /preflight ──────────▶│  "Check my draft response"
    │◀── PASS / ENRICH / WARN ─────│  Cross-references memory
    │                              │
    │── POST /writeback ──────────▶│  "Archive this session"
    │◀── confirmed ────────────────│  Indexed for future recall
```

## Why AgentB?

AI agents forget everything between sessions. AgentB fixes that.

- **4 endpoints.** Context retrieval, preflight validation, session archiving, health check.
- **Any LLM.** Ollama (free/local), OpenAI, Anthropic, Google Gemini, OpenRouter.
- **Any embedding model.** Ollama, OpenAI, HuggingFace, Google.
- **L1/L2/L3 cache hierarchy.** Pre-built bundles → semantic search → full scan. Fast recall.
- **Framework adapters.** OpenClaw hook, Agent Zero skill, or raw HTTP from anything.
- **Zero cloud lock-in.** Runs fully local with Ollama, or use any API provider.
- **One config file.** `agentb.yaml` — pick your providers and go.

## Quick Start

### Option 1: Install Script
```bash
curl -fsSL https://raw.githubusercontent.com/GuyMannDude/agentb/main/install.sh | bash
# Edit config:
nano ~/.config/agentb/agentb.yaml
# Start:
agentb
```

### Option 2: Docker
```bash
git clone https://github.com/GuyMannDude/agentb.git
cd agentb
cp agentb.yaml.example agentb.yaml
# Edit agentb.yaml with your provider settings
docker compose up -d
```

### Option 3: Manual
```bash
git clone https://github.com/GuyMannDude/agentb.git
cd agentb
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp agentb.yaml.example agentb.yaml
python -m agentb.server
```

## Configuration

Edit `agentb.yaml`:

```yaml
# Free local setup (requires Ollama)
reasoning:
  provider: ollama
  model: qwen2.5:32b-instruct
  api_base: http://localhost:11434

embedding:
  provider: ollama
  model: nomic-embed-text
  api_base: http://localhost:11434
```

```yaml
# Cloud setup (works anywhere)
reasoning:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}

embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
```

```yaml
# Mixed (free embeddings, cheap reasoning)
reasoning:
  provider: openrouter
  model: nousresearch/hermes-3-llama-3.1-405b:free
  api_key: ${OPENROUTER_API_KEY}

embedding:
  provider: ollama
  model: nomic-embed-text
  api_base: http://localhost:11434
```

See [agentb.yaml.example](agentb.yaml.example) for all options.

## API

### `GET /health`
Returns system status, provider health, cache sizes.

### `POST /context`
Search memory for relevant context before your agent responds.
```json
{"prompt": "What was our Easter pricing?", "max_results": 5}
```

### `POST /preflight`
Validate your agent's draft response against memory.
```json
{"prompt": "user question", "draft_response": "agent's draft answer"}
```
Returns: `PASS` | `ENRICH` | `WARN` | `BLOCK` with reasoning.

### `POST /writeback`
Archive a session for future recall.
```json
{
  "session_id": "2026-03-05-planning",
  "summary": "Discussed Q2 product roadmap",
  "key_facts": ["Launch date set for April 15", "Budget approved at $5K"],
  "projects_referenced": ["Q2 Launch"],
  "decisions_made": ["Go with Shopify Plus"]
}
```

## Framework Adapters

| Framework | Adapter | Setup |
|-----------|---------|-------|
| **OpenClaw** | Bootstrap hook | `cp adapters/openclaw/agentb-context ~/.openclaw/workspace/hooks/` |
| **Agent Zero** | Skill file | Copy `adapters/agent-zero/SKILL-AGENTB.md` to Agent Zero skills |
| **Any framework** | HTTP/curl | See `adapters/generic/INTEGRATION.md` |

## Architecture

```
┌─────────────────────────────────────────┐
│              AgentB Server              │
│                                         │
│  ┌──────────┐  ┌──────────┐            │
│  │ Reasoning │  │Embedding │  Pluggable │
│  │ Provider  │  │ Provider │  Backends  │
│  └────┬─────┘  └────┬─────┘            │
│       │              │                   │
│  ┌────┴──────────────┴─────┐            │
│  │     Cache Hierarchy     │            │
│  │  L1: Pre-built bundles  │  Fast      │
│  │  L2: Semantic index     │  ↓         │
│  │  L3: Full memory scan   │  Slow      │
│  └────────────┬────────────┘            │
│               │                          │
│  ┌────────────┴────────────┐            │
│  │    Storage Backend      │            │
│  │  JSON / SQLite / PG     │            │
│  └─────────────────────────┘            │
└─────────────────────────────────────────┘
```

## Supported Providers

| Provider | Reasoning | Embedding | Cost |
|----------|-----------|-----------|------|
| **Ollama** | ✅ Any model | ✅ nomic-embed-text | Free (local) |
| **OpenAI** | ✅ GPT-4o-mini, GPT-4o | ✅ text-embedding-3-small | ~$0.15/M tokens |
| **Anthropic** | ✅ Claude Sonnet/Haiku | ❌ | ~$0.25/M tokens |
| **OpenRouter** | ✅ Any model | ✅ Any model | Varies (free tier available) |
| **Google** | ✅ Gemini Flash/Pro | ✅ embedding-001 | Free tier available |
| **HuggingFace** | ❌ | ✅ Any model | Free (local) or API |

## Created By

Guy Hoffman, Rocky Moltman 🦞, and Opie (Claude) — built for [Project Sparks](https://projectsparks.ai).

## License

MIT
