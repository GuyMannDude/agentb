import assert from "node:assert/strict";
import { fetchUnrepliedBusSummary, formatUnrepliedBusSummary } from "./bus-startup.js";

const formatted = formatUnrepliedBusSummary(
  [
    { id: 2, from: "CC", subject: "second" },
    { id: 1, from: "Opie", subject: "first" },
  ],
  "Cody",
  1
);
assert.match(formatted, /Unreplied bus messages: 2/);
assert.match(formatted, /#2 from CC: second/);
assert.match(formatted, /1 more unreplied/);
assert.doesNotMatch(formatted, /#1 from Opie/);

let requested = "";
const fetched = await fetchUnrepliedBusSummary({
  dispatcher: "http://dispatcher.invalid/",
  agent: "Cody",
  fetchImpl: async (url) => {
    requested = String(url);
    return { ok: true, json: async () => [{ id: 9, from: "CC", subject: "work" }] };
  },
});
assert.match(requested, /\/mesh\/inbox\/Cody\?/);
assert.match(requested, /filter=unreplied/);
assert.match(requested, /limit=500/);
assert.match(fetched, /#9 from CC: work/);

const failed = await fetchUnrepliedBusSummary({
  dispatcher: "http://dispatcher.invalid",
  agent: "Cody",
  fetchImpl: async () => ({ ok: false, status: 503 }),
});
assert.match(failed, /BUS INBOX CHECK FAILED/);
assert.match(failed, /state is UNKNOWN/);

console.log("bus-startup tests passed");
