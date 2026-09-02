"""
Mnemo Cortex — composite recall ranking (v4.1)
==============================================
Before this, /context returned results in tier order, ranked inside each tier
by raw vector similarity alone. The quality audit (2026-06-09) showed what
that does in practice: a hand-written doctrine at similarity 0.57 loses every
top-5 slot to near-identical session-noise chunks at 0.73+. Similarity knows
what *matches*; it doesn't know what *matters*.

The composite score blends four signals, each in [0, 1]:

  similarity  — what the tiers already computed (semantic match)
  recency     — exponential decay over age; yesterday beats last quarter
  importance  — category prior: a doctrine outranks a session log at equal
                similarity; perpetual categories carry the most weight
  access      — log-scaled recall frequency; memories that keep getting used
                keep earning rank (and one lucky recall can't dominate)

Weights are config (RankingConfig). Similarity keeps the majority share on
purpose — the other signals break ties and re-order the band of plausible
matches; they must never make an irrelevant memory win.

Chunks with no age/category/access data get neutral values, not penalties —
pre-v3 records must not sink just for being old-format (every existing memory
stays accessible).
"""
from __future__ import annotations

import math
from typing import Optional

from agentb.config import RankingConfig

# Category priors. Perpetual categories (never decay) are also the ones whose
# *content* earns permanence: doctrine, incident, decision, identity, idea. The
# floor is session_log — when a caller explicitly opts INTO seeing logs they
# still rank below distilled knowledge at equal similarity. `idea` sits just
# below identity: a captured creative connection can't be re-derived from the
# environment the way topology can, but it never outranks the rules and
# postmortems that keep the system safe.
CATEGORY_IMPORTANCE: dict[Optional[str], float] = {
    "doctrine": 1.0,
    "incident": 0.95,
    "decision": 0.95,
    "identity": 0.90,
    "idea": 0.85,
    "relationship": 0.80,
    "topology": 0.75,
    "current_state": 0.75,
    "unknown": 0.40,
    "session_log": 0.20,
    None: 0.50,  # uncategorized / pre-v3 — neutral, not punished
}


def composite_score(
    *,
    similarity: float,
    age_days: Optional[float],
    category: Optional[str],
    access_count: int,
    cfg: RankingConfig,
) -> float:
    sim = max(0.0, min(1.0, similarity))

    if age_days is None:
        recency = 0.5  # unknown age — neutral
    else:
        recency = math.exp(-max(0.0, age_days) / cfg.recency_half_life_days * math.log(2))

    importance = CATEGORY_IMPORTANCE.get(category, 0.50)

    # log2(1+n) saturating at ~6 accesses-worth of signal: frequently-used
    # memories rise, but rank can't be bought by access count alone.
    access = min(1.0, math.log2(1 + max(0, access_count)) / math.log2(1 + 6))

    return (
        cfg.w_similarity * sim
        + cfg.w_recency * recency
        + cfg.w_importance * importance
        + cfg.w_access * access
    )


# ── Pool similarity normalisation (Experiment One, 2026-09-01) ─────────────
# The composite's similarity input was the tier's raw relevance. For VEC hits
# that is 1/(1+L2), which on unit vectors squashes the whole on-topic band
# into ~0.52-0.58: the 0.55-weighted similarity term then spans ~0.015 inside
# a served pool while recency spans up to 0.20 — the docstring's "majority
# share" was numerically inert. Measured over 187 live recalls
# (tools/experiments/ranker-exp1/RESULTS.md): Spearman(served rank, raw
# similarity) = +0.05; the most-similar hit was served first 15% of the time.
#
# Fix, two steps:
#   1. Put every tier on cosine. L1/L2/L3 already are. VEC's 1/(1+d) is
#      inverted to d and mapped through cos = 1 - d²/2, exact for unit vectors
#      — the Ollama /api/embed path (nomic-embed-text) returns L2 norm
#      1.000000 (probed 2026-09-01). Other providers are NOT normalised at the
#      boundary; there the mapping stays monotone (order preserved) but the
#      magnitudes drift and the clamp below can collapse far hits into ties.
#      HOT's fixed 0.75 was chosen to outrank every VEC hit on the old scale;
#      on cosine it sits just above the on-topic band, so it keeps that role
#      (HOT is session_log and hidden unless a caller opts in).
#   2. Anchor on the pool's BEST hit: similarity = 1 - (top - cos)/SPAN,
#      clamped at 0. Relative in anchor (survives a shift of the cosine band;
#      a RESCALED band means retuning SPAN), fixed in magnitude (a hair-width
#      gap stays a hair-width gap, so the tie-band
#      contract — category/recency re-order near-equal matches — survives).
#      Anchoring on the top rather than min-max over the pool matters because
#      /context ranks the full overfetch pool (30+ candidates with an
#      off-topic tail): a min-max span would stretch to the tail and
#      re-compress the on-topic band — the very inertness being removed.
#      Replay on the recorded pools: SPAN 0.20 → rho +0.70, top-1 agreement
#      74%; immune to an injected outlier by construction (min-max, measured,
#      fell to +0.50); 0.15 already let a two-item near-tie be decided by noise.
SIMILARITY_SPAN = 0.20  # cosine distance below the pool's best hit at which the term reaches 0


def _to_cosine(relevance: float, tier: str) -> float:
    if tier != "VEC":
        return relevance  # L1/L2/L3 are cosine; HOT is the fixed sentinel (see above)
    if relevance <= 0.0:
        return 0.0
    d = 1.0 / relevance - 1.0
    return max(-1.0, min(1.0, 1.0 - d * d / 2.0))


def pool_similarities(hits: list[tuple[float, str]]) -> list[float]:
    """Map a candidate pool's (relevance, cache_tier) pairs to [0, 1]
    similarities for composite_score: tiers unified on cosine, then each hit
    scored by its cosine distance below the pool's best hit over
    SIMILARITY_SPAN. The best hit is always 1.0; order within the pool is
    preserved; the pool's bottom never influences the top."""
    cos = [_to_cosine(r, t) for r, t in hits]
    if not cos:
        return []
    top = max(cos)
    return [max(0.0, 1.0 - (top - c) / SIMILARITY_SPAN) for c in cos]


# ── Explore mode (v4.8, the serendipity lens; rescaled in E4, 2026-09) ─────
# Focus recall answers "what matches best right now"; explore answers "what
# does this remind the store of". Three deliberate inversions of focus logic:
#   adjacency — prefer the band just BELOW the top hit (near, not identical);
#   no recency — a three-year-old connection is exactly as interesting;
#   novelty  — rarely-recalled memories rise (the half-forgotten one is the
#              serendipitous one). Explore results still bump access counts,
#              so repeated exploring naturally rotates through the idea space.
#
# The band constants are FRACTIONS OF SIMILARITY_SPAN, applied to the same
# pool-normalised similarity focus mode scores on (pool_similarities: the
# pool's best hit is 1.0, a hit SIMILARITY_SPAN cosine below it is 0.0, every
# tier on one cosine scale). Until E4 they were magnitudes on VEC's raw
# 1/(1+L2) relevance — relative in anchor, absolute in size — so an embedder
# change silently retuned the lens, and an L3 hit (raw cosine) entering the
# same pool set a top the VEC band could not reach and the floor zeroed the
# whole pool (snag-mnemo-explore-constants-raw-scale). The values below were
# set by the explore harness (tests/recall/test_explore_harness.py, grid in
# tools/experiments/e4-explore-rescale/sweep.py) against explore's own
# criterion — differential over focus, precision, anti-collapse — not ported
# from the raw numbers (0.03 / 0.05 / 0.08).
# Sweep result (2026-09-02, 16 explore queries over the E2 world, 159 grid
# points): the target sits a small step below the top — on real embedder
# geometry the on-topic band is ~0.07 cosine wide, so "one step sideways" is
# 0.01 cosine, and the adjacency term works as a band-pass 0.06 cosine wide
# that novelty and importance then re-order. Selection rule, fixed before
# the pick: most explore-only adjacent finds subject to precision >= focus
# mode's own precision on the same queries (0.525) and Jaccard divergence
# from focus >= 0.5. Runner-up (floor 0.9) found more but let precision fall
# below focus's.
EXPLORE_OFFSET = 0.05   # target = top - offset: "one step sideways"
EXPLORE_SCALE = 0.30    # how fast adjacency falls off around the target
EXPLORE_FLOOR = 0.80    # sim below top - floor is the noise band: hard zero
W_EXPLORE_ADJACENCY = 0.55
W_EXPLORE_IMPORTANCE = 0.30
W_EXPLORE_NOVELTY = 0.15


def explore_score(
    *,
    similarity: float,
    top_similarity: float,
    category: Optional[str],
    access_count: int,
) -> float:
    sim = max(0.0, min(1.0, similarity))
    top = max(sim, min(1.0, top_similarity))

    if sim <= 0.0 or sim < top - EXPLORE_FLOOR:
        # noise band — serendipity is adjacency, not randomness. A hit at 0.0
        # sits a full SIMILARITY_SPAN below the pool's best: off-topic by the
        # pool's own geometry, whatever the floor says.
        return 0.0

    target = top - EXPLORE_OFFSET
    adjacency = max(0.0, 1.0 - abs(sim - target) / EXPLORE_SCALE)

    importance = CATEGORY_IMPORTANCE.get(category, 0.50)

    # Inverse of the focus access signal, same log-scaled saturation.
    access = min(1.0, math.log2(1 + max(0, access_count)) / math.log2(1 + 6))
    novelty = 1.0 - access

    return (
        W_EXPLORE_ADJACENCY * adjacency
        + W_EXPLORE_IMPORTANCE * importance
        + W_EXPLORE_NOVELTY * novelty
    )
