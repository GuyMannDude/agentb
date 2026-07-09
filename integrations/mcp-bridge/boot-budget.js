// Per-section byte budgets for the agent_startup boot block.
//
// The old scheme capped each brain file at a flat 40KB, left the dream
// brief and Mnemo context uncapped, and let the total float — CC's boot
// hit 73KB on 2026-07-09 and diverted to a file instead of landing
// inline (the MCP host caps inline tool results; ~45KB total is safely
// under it). Every section now has its own byte budget, sized so the
// WORST-CASE total (all sections maxed + header/freshness/separator
// overhead) stays below BOOT_TARGET. Anything cut is one tool call away
// — the truncation notice says exactly which tool re-reads it in full.

// Budgets count UTF-16 code units (.length), not UTF-8 bytes — for these
// near-ASCII brain files the two track within a few percent, and the ~1.1KB
// margin under BOOT_TARGET absorbs the difference.
export const BOOT_TARGET = 45_000;

// Overhead outside the budgeted sections: identity header (~1.1KB),
// lane-freshness banner (~0.4KB), `\n\n---\n\n` separators.
export const BOOT_OVERHEAD = 2_000;

export const STARTUP_BUDGETS = {
  lane: 11_000,        // the agent's own continuity — biggest slice
  "CLAUDE.md": 6_500,  // cross-agent operating doc / session ritual
  "active.md": 10_000, // the board; board rules keep it ~9KB
  "people.md": 2_000,
  "doctrines.md": 5_500,
  mnemo: 2_000,        // recent Mnemo context chunks
  dream: 3_500,        // overnight dream brief
};

// Cap a boot-block section to its budget. Sections are ordered
// most-important-first (newest-first lanes, priority-first board), so
// keeping the top and cutting the tail loses the least. `hint` names
// the tool that fetches the full content.
export function capSection(text, budget, hint) {
  if (text.length <= budget) return text;
  return (
    text.slice(0, budget) +
    `\n\n…[truncated ${text.length - budget} of ${text.length} chars — ` +
    `top kept; ${hint}]…\n`
  );
}
