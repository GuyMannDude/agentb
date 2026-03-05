import { HookHandler } from "openclaw/plugin-sdk";
import { execSync } from "child_process";

const MNEMO_URL = process.env.MNEMO_URL || "http://localhost:50001";
const AGENT_ID = process.env.MNEMO_AGENT_ID || "";
const AUTH_TOKEN = process.env.MNEMO_AUTH_TOKEN || "";

function curlHeaders(): string {
  let h = '-H "Content-Type: application/json"';
  if (AUTH_TOKEN) {
    h += ` -H "X-API-KEY: ${AUTH_TOKEN}"`;
  }
  return h;
}

function agentParam(): string {
  return AGENT_ID ? `"agent_id": "${AGENT_ID}",` : "";
}

/**
 * Mnemo Cortex Live Wire Hook for OpenClaw
 * 
 * Fires on two events:
 * 
 * 1. agent:bootstrap — Injects recent conversation context from Mnemo
 *    into the agent's bootstrap files. The agent wakes up knowing what
 *    happened last session.
 * 
 * 2. command:new — When the user issues /new, saves the outgoing session
 *    to Mnemo via /writeback. The conversation is archived before the
 *    session resets.
 */
const handler: HookHandler = async (event) => {
  try {
    // ── Verify Mnemo is reachable ──
    const healthRaw = execSync(
      `curl -sf ${MNEMO_URL}/health`,
      { timeout: 3000, encoding: "utf-8" }
    );
    const health = JSON.parse(healthRaw);
    if (health.status !== "ok" && health.status !== "degraded") return;

    // ── agent:bootstrap — Inject recent context ──
    if (event.type === "agent" && event.action === "bootstrap") {
      // Get recent conversation context
      const agentQuery = AGENT_ID ? `?agent_id=${AGENT_ID}&n=15` : "?n=15";
      const recentRaw = execSync(
        `curl -sf ${curlHeaders()} "${MNEMO_URL}/sessions/recent${agentQuery}"`,
        { timeout: 5000, encoding: "utf-8" }
      );
      const recent = JSON.parse(recentRaw);
      const contextText = recent.context || "";

      if (!contextText.trim()) {
        // No recent context — try semantic search with general query
        const searchPayload = JSON.stringify({
          prompt: "recent project status, active tasks, and last session summary",
          agent_id: AGENT_ID || undefined,
          max_results: 3,
        });

        try {
          const searchRaw = execSync(
            `curl -sf -X POST ${MNEMO_URL}/context ${curlHeaders()} -d '${searchPayload.replace(/'/g, "'\\''")}'`,
            { timeout: 8000, encoding: "utf-8" }
          );
          const searchResult = JSON.parse(searchRaw);

          if (searchResult.chunks && searchResult.chunks.length > 0) {
            const chunks = searchResult.chunks
              .map((c: any) => `[${c.cache_tier} | ${c.relevance}] ${c.content}`)
              .join("\n\n---\n\n");

            if (event.context.bootstrapFiles) {
              event.context.bootstrapFiles.push({
                basename: "MNEMO-CONTEXT.md",
                content: [
                  "# ⚡ Mnemo Cortex — Memory Context",
                  "_Auto-injected at session start._",
                  "",
                  chunks,
                  "",
                  `_${searchResult.total_found} chunks, ${searchResult.latency_ms}ms_`,
                ].join("\n"),
              });
            }
          }
        } catch {
          // Semantic search failed — proceed without
        }
        return;
      }

      // Inject recent conversation context
      if (event.context.bootstrapFiles) {
        event.context.bootstrapFiles.push({
          basename: "MNEMO-CONTEXT.md",
          content: [
            "# ⚡ Mnemo Cortex — Recent Conversations",
            "_Auto-injected at session start. This is what happened recently._",
            "",
            contextText,
            "",
            "---",
            "_Use `/context` search for specific topics. Use `/ingest` to capture exchanges._",
          ].join("\n"),
        });
      }
      return;
    }

    // ── command:new — Archive the outgoing session ──
    if (event.type === "command" && event.action === "new") {
      // Extract conversation from the session entry
      const sessionEntry = event.context.sessionEntry;
      if (!sessionEntry) return;

      const messages = event.messages || [];
      if (messages.length === 0) return;

      // Build a basic session summary from the messages
      const sessionId = event.context.sessionId || `session-${Date.now()}`;
      const transcript = messages.slice(-30).join("\n");

      // Ingest the raw messages first (Live Wire backup)
      // Parse alternating user/assistant messages
      for (let i = 0; i < messages.length - 1; i += 2) {
        const prompt = messages[i] || "";
        const response = messages[i + 1] || "";
        if (!prompt.trim()) continue;

        try {
          const ingestPayload = JSON.stringify({
            prompt: prompt.slice(0, 2000),
            response: response.slice(0, 2000),
            agent_id: AGENT_ID || undefined,
          });

          execSync(
            `curl -sf -X POST ${MNEMO_URL}/ingest ${curlHeaders()} -d '${ingestPayload.replace(/'/g, "'\\''")}'`,
            { timeout: 3000, encoding: "utf-8" }
          );
        } catch {
          // Best-effort — don't block session reset
        }
      }

      // Also do a writeback with whatever summary we can build
      try {
        const writebackPayload = JSON.stringify({
          session_id: sessionId,
          summary: `OpenClaw session with ${messages.length} messages. Last topic: ${messages[messages.length - 1]?.slice(0, 200) || "unknown"}`,
          key_facts: [],
          projects_referenced: [],
          decisions_made: [],
          agent_id: AGENT_ID || undefined,
        });

        execSync(
          `curl -sf -X POST ${MNEMO_URL}/writeback ${curlHeaders()} -d '${writebackPayload.replace(/'/g, "'\\''")}'`,
          { timeout: 5000, encoding: "utf-8" }
        );
      } catch {
        // Best-effort
      }

      // Push a confirmation message
      event.messages.push(
        "⚡ Mnemo Cortex: Session archived. Your memories are safe."
      );
      return;
    }

  } catch (err) {
    // Mnemo is unreachable — proceed silently
    console.error(
      "[mnemo-ingest] Mnemo Cortex unreachable:",
      err instanceof Error ? err.message : String(err)
    );
  }
};

export default handler;
