# Experiment One — is the recall ranker listening to similarity?

**Date:** 2026-09-01 · **Server:** Mnemo Cortex 4.15.0 (live) · **Embedder:** ollama/nomic-embed-text · **Tenant:** one agent's own memories

Read-only probe: 187 realistic queries (doctrine titles, incident headlines,
task slugs, natural questions) → one real `/context` call each, focus mode,
10 results, `session_log` excluded. Logged per hit: served position, tier,
raw relevance, category, age. The query list and data file stay out of the
public repo (they name the team's memory topics — gitignored); `probe.py`
reproduces the reading against any Mnemo instance with your own query list
(`EXP1_QUERIES=path`, token from `~/.mnemo-auth-token`).

## Reading (analyze.py)

| measure | value |
|---|---|
| Spearman(served rank, raw relevance), mean over queries | **+0.046** |
| … median / p10 / p90 | +0.03 / −0.40 / +0.47 |
| queries with rho > 0.6 | 5 % |
| queries with rho < 0 | 47 % |
| most-similar hit served first | **15 %** (uniform over slots 1–10 otherwise) |
| similarity-term spread inside a pool (0.55 × range), median | 0.015 |
| recency-term spread inside a pool, median | 0.103 |
| importance-term spread, median | 0.037 |
| raw relevance, all hits p10/p50/p90 | 0.527 / 0.553 / 0.583 |
| cache_hits over 1,865 served chunks | VEC 99.7 %, L1 0.3 %, HOT/L2/L3/MEM0 0 |

Verdict: the ranker was a recency + access + category sort. The docstring's
"similarity keeps the majority share" was true of the weight and false of the
numbers, because `1/(1+L2)` compresses the whole on-topic band into ~0.05.

## Replay of candidate fixes (replay.py, sweep.py, sweep2.py)

Each recorded pool re-ranked offline (composite without the access term).
Caveat: pools are the served top-10, so this measures re-ordering inside the
served band, not what a different band would have contained.

| variant | mean rho | top-1 = max-sim | 2-item tie-band contract |
|---|---|---|---|
| A current, `1/(1+d)` raw | +0.08 | 18 % | holds |
| B cosine only | +0.26 | 30 % | holds |
| D cosine + min-max, floor 0 | +0.90 | 93 % | **broken** (noise beats doctrine) |
| D, floor 0.10 | +0.84 | 85 % | broken |
| D, floor 0.15 | +0.78 | 82 % | broken (Δ −0.002) |
| D, floor 0.20 | +0.71 | 74 % | holds (Δ +0.021) |
| D, floor 0.30 | +0.60 | 63 % | holds |
| **E cosine, top-anchored, span 0.20 (shipped)** | **+0.70** | **74 %** | holds (Δ +0.021) |
| E, span 0.15 | +0.74 | 82 % | broken (Δ −0.002) |
| E, span 0.30 | +0.60 | 63 % | holds |

Outlier robustness (sweep2.py): one off-topic candidate (cosine 0.30) injected
into every pool — D floor 0.20 falls to rho **+0.50**, E span 0.20 stays
**+0.70**. That decided it: `/context` ranks the full overfetch pool, whose
tail min-max would stretch to. 171 of 187 recorded pools had a cosine range
under 0.20, which is why D and E score the same on clean pools.

Cosine conversion is exact on this stack: the live Ollama `/api/embed`
(nomic-embed-text) returns L2 norm 1.000000 (probed 2026-09-01), so
`cos = 1 − d²/2`. Other providers are not normalised at the boundary — order
survives, magnitudes drift. Per-pool cosine range in the served band: median
0.071, p10 0.029, p90 0.185; the 0.20 span ≈ p90, so a clear-standout hit
gets full similarity dominance, a near-tie stays a near-tie, and anything
more than 0.20 below the best hit contributes nothing.

## Side effects and negative space

- `/context` bumps access counts on served memories; 1,865 bumps landed on the
  probed tenant. Same effect as ordinary use, reinforcing already-served order.
- Not measured: whether the *selected ten* would change (needs the full
  overfetch pool), explore mode, the L1/L2/HOT tiers' own scales (they served
  0.3 % of hits), recall@k against ground truth (that is E2, the Q&A harness).
