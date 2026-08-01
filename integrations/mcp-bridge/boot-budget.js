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
// lane-freshness banner (~0.4KB), `\n\n---\n\n` separators, and the cut
// manifest (~0.9KB worst case — every section cut, measured not guessed).
// The manifest is counted here on purpose: it is emitted on every boot, so
// leaving it out of the invariant would let the block exceed BOOT_TARGET
// while the test that exists to prevent exactly that kept passing.
export const BOOT_OVERHEAD = 2_900;

export const STARTUP_BUDGETS = {
  lane: 11_000,        // the agent's own continuity — biggest slice
  "CLAUDE.md": 6_500,  // cross-agent operating doc / session ritual
  "active.md": 10_000, // the board; board rules keep it ~9KB
  "people.md": 2_000,
  "doctrines.md": 5_500,
  mnemo: 2_000,        // recent Mnemo context chunks
  dream: 3_500,        // overnight dream brief
};

// ── Cut audit (Guy's rule, 2026-07-30: "Nothing gets cut! New rule. If
// something is going to be cut then I am notified before any more.")
//
// Until now a cut announced itself only at the END of the section it
// truncated — i.e. inside the payload, ~11,000 chars into a lane, in the
// one place a reader skims past. Both CC and Opie received those notices
// for 20+ days and neither ever acted on one. An announcement buried in a
// payload nobody diffs is furniture, not an alarm.
//
// Cuts are now collected per boot and reported at the TOP of the block,
// where they cannot themselves be truncated, with an explicit instruction
// to tell Guy. The record is emitted whether or not anything was cut —
// "nothing was withheld" is a real result and is the only way to tell a
// healthy boot from a dead reporter.
let bootCuts = [];

export function beginBootAudit() {
  bootCuts = [];
}

export function getBootCuts() {
  return bootCuts.slice();
}

// Cap a boot-block section to its budget. Sections are ordered
// most-important-first (newest-first lanes, priority-first board), so
// keeping the top and cutting the tail loses the least. `hint` names
// the tool that fetches the full content. `label` attributes the cut in
// the manifest — without it a cut is recorded as "unnamed section",
// which is a bug worth seeing rather than hiding.
export function capSection(text, budget, hint, label) {
  if (text.length <= budget) return text;
  const withheld = text.slice(budget);
  const headings = [...text.matchAll(/^#{1,6}\s+(.+)$/gm)];
  const identifiers = headings
    .filter((m, index) => {
      const sectionEnd = index + 1 < headings.length ? headings[index + 1].index : text.length;
      return sectionEnd > budget;
    })
    .map((m) => m[1].trim())
    .concat(
      [...withheld.matchAll(/\b(?:memory_id|id)[:=]\s*([a-f0-9]{8,64})\b/gi)]
        .map((m) => m[1])
    );
  bootCuts.push({
    section: label || "unnamed section",
    actual: text.length,
    delivered: budget,
    dropped: text.length - budget,
    hint,
    dropped_identifiers: [...new Set(identifiers)],
  });
  return (
    text.slice(0, budget) +
    `\n\n…[truncated ${text.length - budget} of ${text.length} chars — ` +
    `top kept; ${hint}]…\n`
  );
}

// Render the manifest that leads the boot block. Kept deliberately small
// (~40 chars/section) so it never competes with the content it describes.
export function formatCutManifest() {
  if (bootCuts.length === 0) {
    return "✅ **BOOT COMPLETE — nothing was withheld from this boot.**";
  }
  const dropped = bootCuts.reduce((n, c) => n + c.dropped, 0);
  const actual = bootCuts.reduce((n, c) => n + c.actual, 0);
  const rows = bootCuts
    .map(
      (c) =>
        `| ${c.section} | ${c.actual.toLocaleString()} | ${c.delivered.toLocaleString()} | ` +
        `**${c.dropped.toLocaleString()}** | ${Math.round((100 * c.delivered) / c.actual)}% |`
    )
    .join("\n");
  return (
    `🚨 **${dropped.toLocaleString()} CHARACTERS WERE WITHHELD FROM THIS BOOT ` +
    `(${Math.round((100 * dropped) / actual)}% of ${bootCuts.length} file(s)).**\n\n` +
    `| section | actual | delivered | **withheld** | kept |\n` +
    `|---|---:|---:|---:|---:|\n${rows}\n\n` +
    `**Guy's standing rule (2026-07-30): he is to be NOTIFIED BEFORE ANY MORE IS CUT.** ` +
    `If this boot is the first you have seen these numbers, tell him — do not treat this ` +
    `table as boot furniture. You are reading a partial brain: anything you conclude from ` +
    `a file above may be contradicted by the part you were not given. ` +
    `\`read_brain_file\` fetches any of them in full.`
  );
}
