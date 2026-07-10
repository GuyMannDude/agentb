# Building `mnemo-cortex.mcpb`

The bundle is the canonical Claude Desktop install path. Rebuild it whenever the mcp-bridge bridge or its deps change.

## Prerequisites

```bash
sudo npm install -g @anthropic-ai/mcpb
```

## Build steps

```bash
cd mnemo-cortex
BUILD=integrations/mcpb-build

# Fresh stage from the canonical bridge.
# server.js imports sibling modules AND lazily reads ./package.json for its
# version string — miss any of these and the bundle crashes at startup or on
# the first tool call (package.json ENOENT found the hard way, 2026-07-09).
rm -rf "$BUILD"
mkdir -p "$BUILD/server"
cp integrations/mcp-bridge/server.js \
   integrations/mcp-bridge/boot-budget.js \
   integrations/mcp-bridge/dump.js \
   integrations/mcp-bridge/harness-tool-filter.js \
   integrations/mcp-bridge/package.json \
   "$BUILD/server/"
cp -r integrations/mcp-bridge/node_modules "$BUILD/server/"

# Icon (downscaled hero card)
convert docs/mnemo-cortex-card-v1.png -resize 512x512 "$BUILD/icon.png"

# Manifest — copy from previous bundle and update version + tool list as needed
cp integrations/claude-desktop/manifest.json "$BUILD/manifest.json"
# (or hand-edit "$BUILD/manifest.json")

# Validate then pack
cd "$BUILD"
mcpb validate manifest.json
mcpb pack . ../claude-desktop/mnemo-cortex.mcpb

# Cleanup
cd -
rm -rf "$BUILD"
```

## Versioning

The `manifest.json` `version` field should match the bridge's `package.json` version. Bump both together.

## Testing the bundle

Unzip the `.mcpb` to a tmp dir and try to launch the server with the env vars Claude Desktop will inject:

```bash
mkdir -p /tmp/mcpb-test && unzip -q integrations/claude-desktop/mnemo-cortex.mcpb -d /tmp/mcpb-test
cd /tmp/mcpb-test
MNEMO_URL=http://localhost:50001 MNEMO_AGENT_ID=test \
  timeout 3 node server/server.js < /dev/null
# expect: [mnemo-mcp] Connected to Mnemo Cortex (...)
```

The `Connected` line is NOT sufficient on an auth-enabled server — `/health` is
unauthenticated, so a bundle with broken auth still prints it. Also exercise a
real tool call over MCP stdio (initialize → `tools/call` `mnemo_recall`) both
with and without `MNEMO_AUTH_TOKEN`: expect a result with the token and a loud
`401: Unauthorized` error without it. A crash or hang here means a staging file
is missing (see the `cp` list above).

## Notes

- The `.mcpb` is a zip archive. Anything in the bundle dir gets included unless `.mcpbignore` excludes it.
- `mcpb pack` automatically prunes obvious dev-only files (test dirs, source maps, `.md` files inside `node_modules`, etc.) — that's why a 26 MB staging dir packs to ~3.6 MB.
- `manifest_version` is on the [MCPB spec](https://github.com/modelcontextprotocol/mcpb). Bump when Anthropic releases a new schema version.
