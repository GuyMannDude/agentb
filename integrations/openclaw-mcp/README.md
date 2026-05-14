# Deprecated path — moved to `../mcp-bridge/`

The Mnemo Cortex MCP bridge moved to [`../mcp-bridge/`](../mcp-bridge/) in version 2.8.1 (2026-05-13).

This directory is kept as a **back-compat symlink** so existing MCP client configs that point at `integrations/openclaw-mcp/server.js` keep working unchanged. Please update your config to the new path when convenient — this stub will be removed in a future major version.

## Why the rename?

The bridge code at this path was never OpenClaw-specific. It's the generic Node.js MCP server that **every** Mnemo Cortex integration spawns (Claude Desktop, Claude Code, OpenClaw, LM Studio, AnythingLLM, Agent Zero, Hermes Agent, Ollama Desktop, etc.). The old `openclaw-mcp` directory name was a historical leftover that misled new users into thinking the bridge was tied to one host. The new `mcp-bridge` path tells the truth.

## Migration — one-line config swap

In your MCP client config (Claude Desktop `claude_desktop_config.json`, OpenClaw `openclaw.json`, LM Studio `mcp.json`, AnythingLLM `anythingllm_mcp_servers.json`, etc.), change any path containing:

```
integrations/openclaw-mcp/server.js
```

to:

```
integrations/mcp-bridge/server.js
```

That's the entire migration. No env var changes, no tool name changes, no functional difference — only the path.

## What's at this directory now

- `server.js` — symlink to `../mcp-bridge/server.js`
- `package.json` — symlink to `../mcp-bridge/package.json`
- `README.md` — this deprecation notice

Everything else (code, CHANGELOG, tests, the actual `npm install` target) lives in [`../mcp-bridge/`](../mcp-bridge/).

## Windows users without symlink support

Most recent Git for Windows installs handle symlinks fine (the default since Git 2.10 + `core.symlinks=true`). If your install doesn't and `node integrations/openclaw-mcp/server.js` fails with an unreadable file, do the one-line config swap above immediately — no symlink fallback for you.

## See also

- [../mcp-bridge/README.md](../mcp-bridge/README.md) — the canonical bridge docs.
- [../mcp-bridge/CHANGELOG.md](../mcp-bridge/CHANGELOG.md) — full release history, including the 2.8.1 rename entry.
