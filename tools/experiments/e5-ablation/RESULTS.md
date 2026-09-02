# E5 — explore weight + band-width ablation — RESULTS (2026-09-02, S296)

Spec: `brain/spec-mnemo-explore-e5-ablation.md` (Opie). Builder: CC. Grid: `sweep.py`,
60 runs (H0 + 9 weight sets × 6 widths) over the 23-query explore fixture set
(16 E4 queries + Opie's e18–e24; e17 unservable, see fixtures `_about`).
Offset 0.05 / floor 0.80 fixed. Baselines MEASURED in-run, never ported from the 16.

## Verdict: NO feasible point. The shipped lens stays. The rule was not loosened.

Selection rule (fixed before the sweep): max divergence s.t. precision ≥ focus's
(0.4783), adjMRR ≥ shipped (0.5232), adjR@5 ≥ shipped (0.4609), zero
divergence-class collapses. **The shipped lens fails its own rule** — it collapses
e03, e12, e23 — and every point that clears the collapses regresses adjMRR by
0.05–0.12. The two constraints are in direct conflict on this world.

## Hypotheses

- **H2 (CC): band WIDTH matters more than weight — FALSIFIED.** Widening
  EXPLORE_SCALE at the shipped weights *lowers* divergence monotonically
  (0.530 → 0.374 from 0.30 to 1.00) and adds collapses. A wide band is a
  similarity ranker: the narrowness of the band-pass is what makes explore
  differ from focus. e03 does not lift off 0 at any width at the shipped weights.
- **H1 (Opie): novelty drives the divergence edge — HALF TRUE, and it is not
  free.** Reallocating adjacency → novelty raises divergence monotonically
  (0.530 → 0.721 at adj 0.05) while adjMRR (0.523 → 0.401) and precision
  (0.483 → 0.343) fall monotonically. Novelty buys divergence by surfacing
  rarely-recalled items, and in this world those are often noise. Zeroing
  novelty alone costs 0.035 divergence; zeroing importance alone costs 0.006;
  together they are worth the lens's whole 0.11 edge over H0 (they interact).
- **H0 (similarity-only): explore beats it on divergence (0.530 vs 0.419) and
  ONLY there.** H0 is cleaner (precision 0.522 vs 0.483) and finds adjacent
  items better (adjMRR 0.569 vs 0.523, adjR@5 0.512 vs 0.461) — the E4
  finding, reproduced on the larger set.

## Does the adjacency TARGET term earn its 0.55? — the direct answer

**It is not inert, and it is not the best tool for its job.** Zeroing it
collapses precision to 0.326 (adjMRR 0.454): the weight is what keeps the lens
inside the on-topic band. But raw similarity does that job *better* (H0), so the
0.55 is paying for band-keeping that the input already provides. Explore's real
edge — 0.11 divergence — is importance + novelty re-ordering *within* the band,
bought with −0.04 precision and −0.05 adjMRR against H0. That is the measured
price of serendipity at the shipped constants. Whether to pay it is a product
question, not a constant.

## The per-query anti-collapse floor does NOT land

Under the computed definition (adjacency has any member outside focus@5),
**20 of 23 queries are divergence-class** — the class is too loose to gate on;
it is nearly the whole set. Three collapse at the shipped point (e03, e12, e23)
and nothing clears them without an adjMRR regression. Opie's coverage
predictions e19 and e22 compute as divergence-class (computed wins, per #3188).
Recommendation: redefine the class as "adjacency has a member outside focus@5
**that sits inside the band (span ≥ 0.65)**" — a member the lens *could* serve.
m-backup (span 0.350) fails that test; e03 becomes coverage-class and the
collapse is honest, not a miss.

## What changed in the repo

- `tests/recall/explore_fixtures.json`: 23 queries; gate re-baselined on the 23
  at the shipped lens (fixture change, not lens change — disclosed in `_about`).
- `tests/recall/test_explore_harness.py`: integrity check — no expected/adjacent
  id may be `session_log` (hidden by `/context` by default).
- `tools/experiments/e5-ablation/sweep.py`: the grid. Re-run lands on this table.
- Constants: **unchanged.** `agentb/ranking.py` is byte-identical to 4.16.0.

## Grid (abridged — full table: run `sweep.py`)

| lens | adj/imp/nov | scale | diverg | adjMRR | adjR@5 | prec | collapsed |
|---|---|---|---|---|---|---|---|
| H0 sim-only | – | any | 0.419 | 0.569 | 0.512 | 0.522 | e03 e13 e22 |
| shipped | .55/.30/.15 | 0.30 | 0.530 | 0.523 | 0.461 | 0.483 | e03 e12 e23 |
| shipped | .55/.30/.15 | 1.00 | 0.374 | 0.469 | 0.417 | 0.491 | e03 e11 e13 e22 e23 |
| adj=0 | 0/.67/.33 | 0.30 | 0.711 | 0.454 | 0.331 | 0.326 | e12 |
| imp=0 | .79/0/.21 | 0.30 | 0.524 | 0.517 | 0.475 | 0.529 | e03 e12 e13 e23 |
| nov=0 | .65/.35/0 | 0.30 | 0.495 | 0.489 | 0.464 | 0.483 | e03 |
| adj→nov | .35/.30/.35 | 0.50 | 0.570 | 0.457 | 0.421 | 0.474 | — |
| adj→nov | .05/.30/.65 | 0.30 | 0.721 | 0.407 | 0.337 | 0.343 | e12 |
