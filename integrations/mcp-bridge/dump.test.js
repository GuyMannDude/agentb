// Tests for the Developer Dump writer. No Mnemo server needed —
// exercises dump.js in isolation against a tmpdir. Run: node dump.test.js
//
// Style matches test.js: homemade runner, plain console output.

import { existsSync, mkdtempSync, readFileSync, rmSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DumpWriter, listDumps, DUMP_SCHEMA_VERSION } from "./dump.js";

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.message}`);
    if (err.stack) console.log(err.stack.split("\n").slice(1, 4).join("\n"));
    failed++;
  }
}

async function testAsync(name, fn) {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.message}`);
    if (err.stack) console.log(err.stack.split("\n").slice(1, 4).join("\n"));
    failed++;
  }
}

function freshDir() {
  return mkdtempSync(join(tmpdir(), "mnemo-dump-test-"));
}

function readLines(path) {
  return readFileSync(path, "utf8").trim().split("\n").map((l) => JSON.parse(l));
}

console.log("\nTesting Developer Dump writer (no Mnemo server required)\n");
console.log("── Core writer ──\n");

// 1. mode: off → no files written, no dir created
test("mode=off: writes are no-ops, no files created", () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", {
    enabled: false,
    dir,
    retentionDays: 0,
  });
  w.write({ ts: "x", kind: "tool_call", tool: "test" });
  // The fresh dir exists (mkdtemp made it) but no agent subdir
  if (existsSync(join(dir, "rocky"))) {
    throw new Error("agent dir was created when disabled");
  }
  rmSync(dir, { recursive: true, force: true });
});

// 2. mode: on → header written on first write, event follows
test("mode=on: first write creates file with header + event line", () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });
  w.write({
    ts: "2026-05-15T20:00:00.000Z",
    kind: "tool_call",
    agent_id: "rocky",
    tool: "mnemo_save",
    params: { summary: "hello" },
    response: { ok: true },
    latency_ms: 12,
    ok: true,
  });

  const today = new Date().toISOString().slice(0, 10);
  const path = join(dir, "rocky", `${today}.jsonl`);
  if (!existsSync(path)) throw new Error(`expected file at ${path}`);

  const lines = readLines(path);
  if (lines.length !== 2) {
    throw new Error(`expected 2 lines (header + event), got ${lines.length}`);
  }
  if (lines[0].kind !== "header") throw new Error("first line not header");
  if (lines[0].schema_version !== DUMP_SCHEMA_VERSION) {
    throw new Error("header missing schema_version");
  }
  if (lines[0].agent_id !== "rocky") throw new Error("header agent_id wrong");
  if (lines[1].kind !== "tool_call") throw new Error("second line not tool_call");
  if (lines[1].tool !== "mnemo_save") throw new Error("event tool wrong");
  if (lines[1].schema_version !== DUMP_SCHEMA_VERSION) {
    throw new Error("event missing schema_version");
  }

  rmSync(dir, { recursive: true, force: true });
});

// 3. Two agents → two separate files
test("two agents write to separate files", () => {
  const dir = freshDir();
  const a = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });
  const b = new DumpWriter("opie", { enabled: true, dir, retentionDays: 0 });
  a.write({ ts: "x", kind: "tool_call", tool: "foo" });
  b.write({ ts: "x", kind: "tool_call", tool: "bar" });

  const today = new Date().toISOString().slice(0, 10);
  if (!existsSync(join(dir, "rocky", `${today}.jsonl`))) {
    throw new Error("rocky file missing");
  }
  if (!existsSync(join(dir, "opie", `${today}.jsonl`))) {
    throw new Error("opie file missing");
  }
  rmSync(dir, { recursive: true, force: true });
});

// 4. Day rollover: writing to a different date creates a new file with a
//    fresh header. Exercises _ensureFile directly to avoid mocking the
//    system clock.
test("day rollover: new date → new file with fresh header", () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });
  // Yesterday
  w._ensureFile("2026-05-14");
  // Today
  w._ensureFile("2026-05-15");

  const y = join(dir, "rocky", "2026-05-14.jsonl");
  const t = join(dir, "rocky", "2026-05-15.jsonl");
  if (!existsSync(y) || !existsSync(t)) {
    throw new Error("expected both day files");
  }
  const yLines = readLines(y);
  const tLines = readLines(t);
  if (yLines.length !== 1 || yLines[0].kind !== "header") {
    throw new Error("yesterday file missing header");
  }
  if (tLines.length !== 1 || tLines[0].kind !== "header") {
    throw new Error("today file missing header");
  }
  rmSync(dir, { recursive: true, force: true });
});

// 5. Write failure: unwritable dir → loud stderr, keeps running
test("write failure: stderr message, no crash, second call still no-throw", () => {
  const dir = freshDir();
  // Make the dump dir unwritable so mkdir/appendFile fail
  chmodSync(dir, 0o500);
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });

  // Capture stderr
  const original = process.stderr.write;
  const captured = [];
  process.stderr.write = (s) => {
    captured.push(String(s));
    return true;
  };

  try {
    w.write({ ts: "x", kind: "tool_call", tool: "foo" });
    w.write({ ts: "x", kind: "tool_call", tool: "bar" }); // second call must not throw
  } finally {
    process.stderr.write = original;
    chmodSync(dir, 0o700); // restore so rmSync works
    rmSync(dir, { recursive: true, force: true });
  }

  const msg = captured.join("");
  if (!msg.includes("[MNEMO DUMP FAIL]")) {
    throw new Error(`expected loud stderr, got: ${msg.slice(0, 200)}`);
  }
});

console.log("\n── Tool-handler wrap ──\n");

// 6. wrap() captures successful handler invocation
await testAsync("wrap: successful handler → ok:true, latency captured", async () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });

  const handler = async ({ query }) => ({
    content: [{ type: "text", text: `result for ${query}` }],
  });
  const wrapped = w.wrap("mnemo_recall", handler);
  const res = await wrapped({ query: "hello" });
  if (!res.content[0].text.includes("result for hello")) {
    throw new Error("handler return value mangled");
  }

  const today = new Date().toISOString().slice(0, 10);
  const lines = readLines(join(dir, "rocky", `${today}.jsonl`));
  const event = lines[1]; // [0] header, [1] event
  if (event.tool !== "mnemo_recall") throw new Error("tool name wrong");
  if (event.ok !== true) throw new Error("expected ok:true");
  if (typeof event.latency_ms !== "number") throw new Error("missing latency_ms");
  if (event.params.query !== "hello") throw new Error("params not captured");

  rmSync(dir, { recursive: true, force: true });
});

// 7. wrap() captures isError response as ok:false with error text extracted
await testAsync("wrap: handler returns {isError:true} → ok:false, error captured", async () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });

  const handler = async () => ({
    content: [{ type: "text", text: "Recall error: server down" }],
    isError: true,
  });
  const wrapped = w.wrap("mnemo_recall", handler);
  await wrapped({ query: "x" });

  const today = new Date().toISOString().slice(0, 10);
  const lines = readLines(join(dir, "rocky", `${today}.jsonl`));
  const event = lines[1];
  if (event.ok !== false) throw new Error("expected ok:false");
  if (!event.error || !event.error.includes("server down")) {
    throw new Error(`expected error text extracted, got: ${event.error}`);
  }

  rmSync(dir, { recursive: true, force: true });
});

// 8. wrap() lets thrown errors propagate but still logs ok:false
await testAsync("wrap: handler throws → dump records ok:false AND error re-throws", async () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });

  const handler = async () => {
    throw new Error("boom");
  };
  const wrapped = w.wrap("mnemo_recall", handler);

  let caught = null;
  try {
    await wrapped({ query: "x" });
  } catch (err) {
    caught = err;
  }
  if (!caught || caught.message !== "boom") {
    throw new Error("error should have re-thrown");
  }

  const today = new Date().toISOString().slice(0, 10);
  const lines = readLines(join(dir, "rocky", `${today}.jsonl`));
  const event = lines[1];
  if (event.ok !== false) throw new Error("expected ok:false");
  if (event.error !== "boom") throw new Error(`expected error 'boom', got ${event.error}`);

  rmSync(dir, { recursive: true, force: true });
});

// 9. wrap() is a no-op passthrough when disabled
await testAsync("wrap: disabled writer returns handler unchanged (zero overhead)", async () => {
  const w = new DumpWriter("rocky", { enabled: false, dir: "/tmp", retentionDays: 0 });
  const handler = async () => "raw";
  const wrapped = w.wrap("anything", handler);
  if (wrapped !== handler) {
    throw new Error("disabled wrap() should return the original handler reference");
  }
});

console.log("\n── listDumps ──\n");

// 10. listDumps returns rows sorted by mtime descending
test("listDumps returns entries with bytes + mtime", () => {
  const dir = freshDir();
  const w = new DumpWriter("rocky", { enabled: true, dir, retentionDays: 0 });
  w.write({ ts: "x", kind: "tool_call", tool: "foo" });

  const rows = listDumps({ enabled: true, dir, retentionDays: 0 });
  if (rows.length !== 1) throw new Error(`expected 1 row, got ${rows.length}`);
  if (rows[0].agent_id !== "rocky") throw new Error("agent_id wrong");
  if (rows[0].bytes <= 0) throw new Error("bytes should be > 0");
  if (!rows[0].mtime) throw new Error("mtime missing");

  rmSync(dir, { recursive: true, force: true });
});

console.log(`\n${passed} passed, ${failed} failed.\n`);
if (failed > 0) process.exitCode = 1;
