// Write protection for write_brain_file.
//
// CLAUDE.md (the cross-agent operating doc) is refused for everyone —
// unconditionally, so a spoofy MNEMO_AGENT_ID=CLAUDE can't unlock it;
// edit it deliberately on disk instead.
//
// Lane-protected files may only be written by the agent whose lane they
// are, matching the lane-candidate convention used at startup
// (`<agent>.md` or `<agent>-session.md`).

const ALWAYS_REFUSED = ["CLAUDE.md"];
const LANE_PROTECTED = ["cc-session.md"];

export function refusesBrainWrite(filename, agentId) {
  if (ALWAYS_REFUSED.includes(filename)) return true;
  if (!LANE_PROTECTED.includes(filename)) return false;
  return ![`${agentId}.md`, `${agentId}-session.md`].includes(filename);
}
