// Which tenant does a mnemo_search actually query?
//
// The server answers agent-less /context requests with a silent empty
// 200 (bus #1941) — so the bridge must ALWAYS name a tenant. Share mode
// with no explicit agent_id used to send no agent_id at all, which is
// how a share-mode Desktop session lost a whole afternoon of searches
// to zero-result reads under green health. Until the server ships an
// explicit absent-agent_id contract (the server half of #1941), an
// unscoped share-mode search self-scopes and the caller is told so.
// When that server contract lands, re-enabling true unscoped search is
// a deliberate change here, not a cleanup.

export function searchScope({ shareActive, requestedAgent, selfAgent }) {
  if (shareActive && requestedAgent) {
    return { agentId: requestedAgent, selfScopedFallback: false };
  }
  return {
    agentId: selfAgent,
    // Only a share-mode caller was ever promised "omit = all agents";
    // separate mode always meant self, so nothing degraded there.
    selfScopedFallback: shareActive,
  };
}
