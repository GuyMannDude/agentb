const DEFAULT_LIMIT = 500;
const DEFAULT_SHOWN = 12;
const DEFAULT_TIMEOUT_MS = 3_000;

export function formatUnrepliedBusSummary(messages, agent, shown = DEFAULT_SHOWN) {
  const rows = Array.isArray(messages) ? messages : [];
  const visible = rows.slice(0, shown);
  const lines = visible.map(
    (message) =>
      `- #${message.id} from ${message.from}: ${String(message.subject || "(no subject)")}`
  );
  const omitted = rows.length - visible.length;
  if (omitted > 0) {
    lines.push(`- ... ${omitted} more unreplied message(s); call disco-bus inbox(filter="unreplied").`);
  }
  if (!lines.length) lines.push("- None.");
  return (
    `# BUS INBOX - UNREPLIED (${agent})\n\n` +
    `Unreplied bus messages: ${rows.length}. This is reply-chain state, not read state.\n\n` +
    lines.join("\n")
  );
}

export async function fetchUnrepliedBusSummary({
  dispatcher,
  agent,
  fetchImpl = fetch,
  timeoutMs = DEFAULT_TIMEOUT_MS,
  limit = DEFAULT_LIMIT,
  shown = DEFAULT_SHOWN,
}) {
  if (!dispatcher) return null;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const base = dispatcher.replace(/\/$/, "");
  const url =
    `${base}/mesh/inbox/${encodeURIComponent(agent)}?` +
    new URLSearchParams({ limit: String(limit), filter: "unreplied" });
  try {
    const response = await fetchImpl(url, { signal: controller.signal });
    if (!response.ok) {
      return (
        `# WARNING: BUS INBOX CHECK FAILED\n\n` +
        `Dispatcher returned HTTP ${response.status}; unreplied state is UNKNOWN. ` +
        `Call disco-bus inbox(filter="unreplied") manually.`
      );
    }
    return formatUnrepliedBusSummary(await response.json(), agent, shown);
  } catch (error) {
    const reason = error?.name === "AbortError" ? "timed out" : error?.message || "failed";
    return (
      `# WARNING: BUS INBOX CHECK FAILED\n\n` +
      `Dispatcher ${reason}; unreplied state is UNKNOWN. ` +
      `Call disco-bus inbox(filter="unreplied") manually.`
    );
  } finally {
    clearTimeout(timer);
  }
}
