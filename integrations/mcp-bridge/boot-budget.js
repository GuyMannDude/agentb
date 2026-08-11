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

// Budgets count UTF-16 code units (.length), not UTF-8 bytes. This turns out to
// be exactly right rather than approximately right: the host measures the same
// way (see below), so the two sides agree unit-for-unit and the old worry about
// UTF-8 divergence does not apply.
//
// ── MEASURED 2026-08-11 (S218), replacing a guess ──────────────────────────
// 45,000 was never measured. It was inferred from one incident (a 73KB boot
// diverting to a file on 2026-07-09) and set low enough to feel safe, which
// cost every agent real continuity every morning — CC's boot on 2026-08-11
// withheld 1,946 chars, 49% of one section, against a ceiling nobody had
// checked.
//
// The host's actual rule, read out of the Claude Code binary (2.1.227) and
// confirmed empirically:
//
//   MAX_MCP_OUTPUT_TOKENS defaults to 25,000 TOKENS — not characters.
//   The check runs in two stages:
//     1. A cheap estimate, Math.round(text.length / 4). If that is <= 25,000
//        * 0.5 = 12,500, the result is returned inline WITHOUT ANY FURTHER
//        CHECK. That makes 50,000 UTF-16 units a hard guarantee, independent
//        of how token-dense the content is.
//     2. Above 50,000 chars it performs a REAL token count and diverts to a
//        file only if that exceeds 25,000 tokens.
//
// Empirical confirmation, both ends:
//   44,800 chars (that morning's boot)        -> inline
//   73,103 chars (active-archive-2026-Q2.md)  -> "exceeds maximum allowed
//                                                tokens", saved to a file
// The 73,103 result reproduces the 2026-07-09 incident exactly, and puts our
// content's real density under ~2.9 chars/token — markdown, emoji and code
// fences are expensive, so the naive length/4 estimate FLATTERS us. That is
// precisely why we stop at the stage-1 guarantee instead of chasing the true
// stage-2 ceiling near ~70,000: stage 1 cannot be tipped by a session that
// happens to be emoji-heavy, and stage 2 can.
//
// 49,000 leaves 1,000 units of margin below the 50,000 guarantee.
export const BOOT_TARGET = 49_000;

// Overhead outside the budgeted sections: identity header (~1.1KB),
// lane-freshness banner (~0.4KB), `\n\n---\n\n` separators, and the cut
// manifest (~0.9KB worst case — every section cut, measured not guessed).
// The manifest is counted here on purpose: it is emitted on every boot, so
// leaving it out of the invariant would let the block exceed BOOT_TARGET
// while the test that exists to prevent exactly that kept passing.
export const BOOT_OVERHEAD = 2_900;

// The +4,000 units bought by the measurement above, distributed 2026-08-11.
// It goes where the cuts were actually landing, not evenly: `lane` because all
// five agents sit within ~200 units of it and have been trimming continuity to
// pay for new entries, and `mnemo` because it was the section CC's boot cut 49%
// off that morning (3,946 actual against a 2,000 budget). `active.md` gets
// nothing on purpose — board-check.py already forces it to shrink, so handing
// it room would just relieve a pressure that is doing useful work.
export const STARTUP_BUDGETS = {
  lane: 12_500,        // the agent's own continuity — biggest slice (was 11,000)
  "CLAUDE.md": 7_000,  // cross-agent operating doc / session ritual (was 6,500)
  "active.md": 10_000, // the board; board rules keep it ~9KB — deliberately unchanged
  "people.md": 2_200,  // was 2,000
  "doctrines.md": 6_000, // was 5,500
  mnemo: 3_300,        // recent Mnemo context chunks (was 2,000 — the 49% cut)
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
