# Mnemo Cortex on macOS

Install guide for macOS 12+ on Apple Silicon (M1–M4) or Intel. Tested by resolving the
full dependency tree as prebuilt macOS wheels (Python 3.12/3.13/3.14, arm64 + x86_64) —
nothing compiles on your machine. This guide is part of our macOS beta: if anything below
doesn't match what you see, please [file an issue](https://github.com/GuyMannDude/mnemo-cortex/issues)
with the exact command and output.

---

## ⚠️ Read this first: which Python you use matters

Mnemo Cortex stores vectors in SQLite via the [`sqlite-vec`](https://github.com/asg017/sqlite-vec)
extension. Loading a SQLite extension requires a Python whose `sqlite3` module was built
with extension support — and on macOS, **two common Pythons were not**:

| Python | `sqlite-vec` works? |
|---|---|
| **Homebrew** (`brew install python`) | ✅ yes |
| **uv-managed** (`uv python install`) | ✅ yes |
| MacPorts | ✅ yes |
| ❌ Apple system Python (`/usr/bin/python3`, Xcode CLT) | **no** — extension loading compiled out |
| ❌ python.org installer | **no** — same limitation |

**10-second preflight** — run this with the Python you plan to use:

```bash
python3 -c "import sqlite3; c = sqlite3.connect(':memory:'); c.enable_load_extension(True); print('✅ this Python can load SQLite extensions')"
```

If you see `AttributeError: 'sqlite3.Connection' object has no attribute 'enable_load_extension'`,
that Python cannot run Mnemo Cortex. Install Homebrew Python and use it explicitly
(`$(brew --prefix)/bin/python3`).

---

## Prerequisites

```bash
# Homebrew itself: https://brew.sh
brew install python git
brew install ollama          # recommended: free, private, local models
brew install node            # only if you'll run an MCP-bridge integration
                             # (Claude Desktop, Claude Code, LM Studio, ...)
```

Start Ollama and pull the embedding model (the vector index is locked to
`nomic-embed-text`, 768 dimensions):

```bash
brew services start ollama   # keeps Ollama running across reboots
ollama pull nomic-embed-text
ollama pull qwen3:8b         # or any tool-capable reasoning model you prefer
```

## Step 1 — Install

```bash
git clone https://github.com/GuyMannDude/mnemo-cortex.git
cd mnemo-cortex
$(brew --prefix)/bin/python3 -m venv .venv    # explicitly Homebrew Python
source .venv/bin/activate
pip install -e .
```

Run the preflight from the venv to confirm the whole chain:

```bash
python -c "import sqlite3, sqlite_vec; c = sqlite3.connect(':memory:'); c.enable_load_extension(True); sqlite_vec.load(c); print('✅ sqlite-vec', c.execute('select vec_version()').fetchone()[0])"
```

## Step 2 — Configure

```bash
mnemo-cortex init
```

The wizard writes `~/.config/agentb/agentb.yaml` (data lands in `~/.agentb/`).
Two macOS notes on the result:

- **Bind address.** The default binds `0.0.0.0` (all interfaces). On a laptop that
  joins coffee-shop Wi-Fi, set it to loopback unless you know you want LAN access:

  ```yaml
  server:
    host: 127.0.0.1
    port: 50001
  ```

  If other machines need to reach it, prefer a [Tailscale](https://tailscale.com)
  address over `0.0.0.0`, and set `server.auth_token`.

- **Config lives at `~/.config/agentb/agentb.yaml`** on macOS too (not
  `~/Library/Application Support`). The `AGENTB_CONFIG` env var overrides the path.

## Step 3 — Run and verify

```bash
mnemo-cortex start        # background; or: mnemo-cortex start --foreground
curl -s http://127.0.0.1:50001/health
mnemo-cortex status
```

`/health` should report the version and green components. Save a memory and recall it:

```bash
mnemo-cortex test
```

## Step 4 — Start at login (launchd)

macOS uses launchd where Linux uses systemd. A template LaunchAgent ships at
[`deploy/macos/com.mnemo-cortex.server.plist`](../deploy/macos/com.mnemo-cortex.server.plist).
It runs the server in the foreground under launchd's supervision (do **not** point it at
`mnemo-cortex start`, which daemonizes and would confuse launchd's process tracking).

```bash
# From the repo root, with the venv at .venv — substitutes your real paths:
sed -e "s|@REPO@|$(pwd)|g" -e "s|@HOME@|$HOME|g" \
    deploy/macos/com.mnemo-cortex.server.plist \
    > ~/Library/LaunchAgents/com.mnemo-cortex.server.plist

mkdir -p ~/Library/Logs/mnemo-cortex
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/com.mnemo-cortex.server.plist
```

Verify:

```bash
launchctl print gui/$UID/com.mnemo-cortex.server | head -20
curl -s http://127.0.0.1:50001/health
tail -f ~/Library/Logs/mnemo-cortex/server.log
```

Stop / uninstall:

```bash
launchctl bootout gui/$UID/com.mnemo-cortex.server
rm ~/Library/LaunchAgents/com.mnemo-cortex.server.plist
```

> If you use the LaunchAgent, don't also use `mnemo-cortex start`/`stop` — launchd
> restarts the server when it dies, so `stop` alone won't keep it down. Use
> `launchctl bootout` to stop it for real.

## Step 5 — Connect your agent (MCP bridge)

The bridge is pure JavaScript (no native modules) and works unchanged on macOS:

```bash
cd integrations/mcp-bridge && npm install
```

Then point your MCP host at `integrations/mcp-bridge/server.js` — per-host walkthroughs
(Claude Desktop `.mcpb` bundle, Claude Code, LM Studio, AnythingLLM, OpenClaw, llama.cpp)
are in the [main README](../README.md#-use-with-any-local-llm). Claude Desktop config on
macOS lives at `~/Library/Application Support/Claude/claude_desktop_config.json`.

---

## Known limitations on macOS (beta)

- **`mnemo-cortex doctor` / health service checks are systemd-aware only** — the
  "Services" section reports `SKIP: systemctl not available` on macOS. That's expected,
  not a failure. The launchd equivalent is `launchctl print gui/$UID/com.mnemo-cortex.server`.
- **The OpenClaw session watcher service unit is Linux-only** (systemd). The watcher
  itself runs fine in the foreground; a launchd template for it will follow after the
  server template survives beta.
- **No Homebrew formula yet.** `sqlite-vec` publishes wheels but no sdist, and
  mnemo-cortex isn't on PyPI yet, so a classic formula can't build from source.
  `git clone` + venv (above) is the supported path for now.

## Troubleshooting

**`AttributeError: ... no attribute 'enable_load_extension'`** — you're on Apple's or
python.org's Python. See the warning at the top; rebuild the venv with Homebrew Python.

**`sqlite3.OperationalError` mentioning `vec0` or `vec_version`** — the sqlite-vec
extension didn't load. Rerun the Step 1 preflight inside the venv.

**Embeddings fail / recall returns nothing** — check Ollama: `curl -s http://localhost:11434/api/tags`
should list `nomic-embed-text`. If you installed Ollama.app instead of Homebrew, it must
be running (menu-bar icon).

**Port 50001 already in use** — `lsof -nP -iTCP:50001 | grep LISTEN` to find the owner.
Change the port in `agentb.yaml` (`server.port`) or set `MNEMO_PORT`.

**LaunchAgent won't start** — check `~/Library/Logs/mnemo-cortex/server.log` first, then
`launchctl print gui/$UID/com.mnemo-cortex.server` (look at `last exit code`). The most
common cause is a stale `@REPO@`/`@HOME@` placeholder from skipping the `sed` step.

**"cannot be opened because the developer cannot be verified"** — shouldn't happen for
pip-installed wheels (Gatekeeper quarantine applies to browser downloads, not pip), but if
you copied the repo from a downloaded zip: `xattr -dr com.apple.quarantine .venv`.
