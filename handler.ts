import { HookHandler } from "openclaw/plugin-sdk";
import { execSync } from "child_process";

const AGENTB_URL = process.env.AGENTB_URL || "http://localhost:50001";

const handler: HookHandler = async (event) => {
  try {
    // Verify AgentB is reachable
    const healthRaw = execSync(`curl -sf ${AGENTB_URL}/health`, {
      timeout: 5000,
      encoding: "utf-8",
    });
    const health = JSON.parse(healthRaw);
    if (health.status !== "ok" && health.status !== "degraded") return;

    // Build query from recent messages or default to general context
    const recentMessages = (event.messages || []).slice(-3).join("\n").trim();
    const query = recentMessages || "recent project status and active tasks";

    // Escape for shell
    const safeQuery = query.replace(/"/g, '\\"').replace(/\n/g, "\\n");

    const contextRaw = execSync(
      `curl -sf -X POST ${AGENTB_URL}/context ` +
        `-H "Content-Type: application/json" ` +
        `-d '{"prompt": "${safeQuery}", "max_results": 3}'`,
      { timeout: 10000, encoding: "utf-8" }
    );

    const ctx = JSON.parse(contextRaw);
    if (!ctx.chunks || ctx.chunks.length === 0) return;

    const contextText = ctx.chunks
      .map((c: any) => `[${c.cache_tier} | relevance: ${c.relevance}]\n${c.content}`)
      .join("\n\n---\n\n");

    // Inject into bootstrap files
    if (event.context.bootstrapFiles) {
      event.context.bootstrapFiles.push({
        basename: "AGENTB-CONTEXT.md",
        content: [
          "# 🧠 AgentB Memory Context",
          "_Auto-injected from AgentB memory coprocessor._",
          "",
          contextText,
          "",
          "---",
          `_${ctx.total_found} chunks found in ${ctx.latency_ms}ms. ` +
            `Cache hits: L1=${ctx.cache_hits.L1}, L2=${ctx.cache_hits.L2}, L3=${ctx.cache_hits.L3}_`,
        ].join("\n"),
      });
    }
  } catch (err) {
    // AgentB unreachable — proceed without memory context
    console.error(
      "[agentb-context] AgentB unreachable:",
      err instanceof Error ? err.message : String(err)
    );
  }
};

export default handler;
