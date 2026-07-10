// Tests for registration-time HARNESS_ENABLED_TOOLS enforcement.
// No Mnemo server needed. Run: node harness-tool-filter.test.js

import { createHarnessToolGate } from "./harness-tool-filter.js";

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

function exercise(rawEnabledTools) {
  const registered = [];
  const gate = createHarnessToolGate(rawEnabledTools);
  const registerTool = (name) => registered.push(name);
  for (const name of ["mnemo_recall", "mnemo_save", "agent_startup"]) {
    gate.register(registerTool, name, {}, () => {});
  }
  return { registered, notice: gate.startupNotice() };
}

console.log("\n── HARNESS_ENABLED_TOOLS registration gate ──\n");

test("unset registers every tool and emits no notice", () => {
  const out = exercise(undefined);
  if (out.registered.join(",") !== "mnemo_recall,mnemo_save,agent_startup")
    throw new Error(`registered: ${out.registered.join(",")}`);
  if (out.notice !== null) throw new Error(`unexpected notice: ${out.notice}`);
});

test("empty registers every tool and emits no notice", () => {
  const out = exercise("  , ");
  if (out.registered.join(",") !== "mnemo_recall,mnemo_save,agent_startup")
    throw new Error(`registered: ${out.registered.join(",")}`);
  if (out.notice !== null) throw new Error(`unexpected notice: ${out.notice}`);
});

test("subset registers only allow-listed tools and lists skipped names", () => {
  const out = exercise(" mnemo_save, agent_startup ");
  if (out.registered.join(",") !== "mnemo_save,agent_startup")
    throw new Error(`registered: ${out.registered.join(",")}`);
  if (!out.notice.includes("mnemo_recall"))
    throw new Error(`skipped tool missing from notice: ${out.notice}`);
  if (out.notice.includes("WARNING"))
    throw new Error(`unexpected warning: ${out.notice}`);
});

test("unknown-only allow-list registers nothing and warns without crashing", () => {
  const out = exercise("not_a_real_tool");
  if (out.registered.length !== 0)
    throw new Error(`registered: ${out.registered.join(",")}`);
  if (!out.notice.includes("WARNING: allow-list matched no known tools"))
    throw new Error(`warning missing: ${out.notice}`);
  for (const name of ["mnemo_recall", "mnemo_save", "agent_startup"]) {
    if (!out.notice.includes(name)) throw new Error(`skipped name missing: ${name}`);
  }
});

console.log(`\n${passed} passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
