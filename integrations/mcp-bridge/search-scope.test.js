// Tests for search-scope.js — mnemo_search's tenant-scoping decision.
// Run: node search-scope.test.js
//
// Style matches boot-budget.test.js: homemade runner, plain console output.

import { searchScope } from "./search-scope.js";

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

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

test("separate mode, no agent_id → self", () => {
  const s = searchScope({ shareActive: false, requestedAgent: undefined, selfAgent: "cc" });
  assert(s.agentId === "cc", `expected cc, got ${s.agentId}`);
  assert(s.selfScopedFallback === false, "separate mode is not a degraded scope");
});

test("separate mode ignores a requested agent (privacy gate holds)", () => {
  const s = searchScope({ shareActive: false, requestedAgent: "opie", selfAgent: "cc" });
  assert(s.agentId === "cc", `expected cc, got ${s.agentId}`);
});

test("share mode, explicit agent_id → that agent", () => {
  const s = searchScope({ shareActive: true, requestedAgent: "opie", selfAgent: "cc" });
  assert(s.agentId === "opie", `expected opie, got ${s.agentId}`);
  assert(s.selfScopedFallback === false, "explicit scope is not a fallback");
});

// The #1941 regression: share mode with no agent_id used to send NO
// agent_id at all, and the server answers agent-less queries with a
// silent empty 200. The request must never leave the bridge unscoped.
test("share mode, no agent_id → self-scoped, never absent (#1941)", () => {
  const s = searchScope({ shareActive: true, requestedAgent: undefined, selfAgent: "opie" });
  assert(s.agentId === "opie", `expected opie, got ${s.agentId}`);
  assert(s.selfScopedFallback === true, "caller must be told the scope degraded to self");
});

test("share mode, empty-string agent_id → treated as absent", () => {
  const s = searchScope({ shareActive: true, requestedAgent: "", selfAgent: "cc" });
  assert(s.agentId === "cc", `expected cc, got ${s.agentId}`);
  assert(s.selfScopedFallback === true, "empty string is not a tenant");
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
