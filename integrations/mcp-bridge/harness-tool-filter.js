// Registration-time enforcement for harness-specific MCP tool allow-lists.
// An unset or empty HARNESS_ENABLED_TOOLS preserves the historical behavior:
// every tool is registered.

export function createHarnessToolGate(rawEnabledTools) {
  const enabledTools = new Set(
    (rawEnabledTools || "")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean)
  );
  const active = enabledTools.size > 0;
  const registered = [];
  const skipped = [];

  return {
    register(registerTool, name, schema, handler) {
      if (active && !enabledTools.has(name)) {
        skipped.push(name);
        return undefined;
      }
      registered.push(name);
      return registerTool(name, schema, handler);
    },

    startupNotice() {
      if (!active) return null;
      const warning = registered.length === 0
        ? " WARNING: allow-list matched no known tools."
        : "";
      const skippedNames = skipped.length > 0 ? skipped.join(", ") : "(none)";
      return `[mnemo-mcp] HARNESS_ENABLED_TOOLS enforced — skipped tools: ${skippedNames}.${warning}\n`;
    },
  };
}
