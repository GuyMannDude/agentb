// Tests for the boot-block byte budgets. No Mnemo server needed —
// exercises boot-budget.js in isolation. Run: node boot-budget.test.js
//
// Style matches dump.test.js: homemade runner, plain console output.

import {
  BOOT_TARGET,
  BOOT_OVERHEAD,
  STARTUP_BUDGETS,
  capSection,
  beginBootAudit,
  getBootCuts,
  formatCutManifest,
} from "./boot-budget.js";

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.message}`);
    failed++;
  }
}

console.log("\n── capSection ──\n");

test("under budget → returned unchanged", () => {
  const text = "short section";
  const out = capSection(text, 1000, "use read_brain_file");
  if (out !== text) throw new Error("text was modified");
});

test("exactly at budget → returned unchanged", () => {
  const text = "x".repeat(500);
  const out = capSection(text, 500, "use read_brain_file");
  if (out !== text) throw new Error("text was modified at exact boundary");
});

test("over budget → top slice kept, notice appended", () => {
  const text = "TOP-" + "x".repeat(1000);
  const out = capSection(text, 100, "use read_brain_file for the full file");
  if (!out.startsWith("TOP-")) throw new Error("top of section not kept");
  if (!out.includes("truncated 904 of 1004 chars"))
    throw new Error(`wrong counts in notice: ${out.slice(100, 200)}`);
  if (!out.includes("use read_brain_file for the full file"))
    throw new Error("hint missing from notice");
});

test("over budget → capped length stays near budget", () => {
  const text = "x".repeat(50_000);
  const out = capSection(text, 4_000, "hint");
  // budget + truncation notice; notice must stay small
  if (out.length > 4_000 + 200)
    throw new Error(`capped output too long: ${out.length}`);
});

console.log("\n── budget invariant ──\n");

test("worst-case boot block lands inline (< BOOT_TARGET)", () => {
  const sum = Object.values(STARTUP_BUDGETS).reduce((a, b) => a + b, 0);
  // + ~150 chars of truncation notice per section, worst case all truncate
  const notices = Object.keys(STARTUP_BUDGETS).length * 200;
  const worstCase = sum + notices + BOOT_OVERHEAD;
  if (worstCase >= BOOT_TARGET)
    throw new Error(
      `budgets sum to ${sum} + notices ${notices} + overhead ${BOOT_OVERHEAD} = ${worstCase} ≥ target ${BOOT_TARGET}`
    );
  console.log(`         worst-case boot: ${worstCase} bytes (target < ${BOOT_TARGET})`);
});

test("every boot section has a budget", () => {
  for (const key of ["lane", "CLAUDE.md", "active.md", "people.md", "doctrines.md", "bus", "dream"]) {
    if (!(key in STARTUP_BUDGETS)) throw new Error(`missing budget: ${key}`);
    if (!(STARTUP_BUDGETS[key] > 0)) throw new Error(`non-positive budget: ${key}`);
  }
});


// ── cut audit (Guy's rule, 2026-07-30: nothing gets cut silently) ──

console.log("\n── cut audit ──\n");

test("clean boot reports explicitly, not by silence", () => {
  beginBootAudit();
  capSection("tiny", 1000, "hint", "a.md");
  if (getBootCuts().length !== 0) throw new Error("recorded a cut that did not happen");
  const m = formatCutManifest();
  if (!m.includes("nothing was withheld"))
    throw new Error("a clean boot must SAY it is clean: " + m);
});

test("a cut is recorded with true numbers and attributed to its section", () => {
  beginBootAudit();
  capSection("x".repeat(5000), 1000, "hint", "active.md");
  const cuts = getBootCuts();
  if (cuts.length !== 1) throw new Error("expected exactly 1 cut, got " + cuts.length);
  const c = cuts[0];
  if (c.section !== "active.md") throw new Error("section not attributed: " + c.section);
  if (c.actual !== 5000 || c.delivered !== 1000 || c.dropped !== 4000)
    throw new Error("wrong arithmetic: " + JSON.stringify(c));
});

test("an unlabelled cut is visible as a defect, not silently anonymous", () => {
  beginBootAudit();
  capSection("x".repeat(3000), 100, "hint");
  if (getBootCuts()[0].section !== "unnamed section")
    throw new Error("unlabelled cuts must be identifiable");
});

test("beginBootAudit clears the previous boot (no cross-boot leakage)", () => {
  beginBootAudit();
  capSection("x".repeat(3000), 100, "hint", "one.md");
  beginBootAudit();
  if (getBootCuts().length !== 0) throw new Error("cuts leaked across boots");
});

test("manifest names every cut section and the true total", () => {
  beginBootAudit();
  capSection("x".repeat(3000), 1000, "hint", "lane.md");
  capSection("y".repeat(2500), 500, "hint", "doctrines.md");
  const m = formatCutManifest();
  for (const need of ["lane.md", "doctrines.md", "4,000"]) {
    if (!m.includes(need)) throw new Error(`manifest missing ${need}: ${m}`);
  }
});

test("the manifest itself is small enough to never need capping", () => {
  beginBootAudit();
  for (const f of Object.keys(STARTUP_BUDGETS)) {
    capSection("x".repeat(200_000), STARTUP_BUDGETS[f] || 1000, "hint", f);
  }
  const m = formatCutManifest();
  if (m.length > 1500)
    throw new Error(`manifest is ${m.length} chars — it must not compete with content`);
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
