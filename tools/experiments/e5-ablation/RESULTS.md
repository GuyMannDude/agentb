# E5 — explore weight + band-width ablation — RESULTS (2026-09-02, S296; class redefined S297)

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

## The per-query anti-collapse floor — redefined (S297, Opie #3194), and it lands

The first cut (S296) — divergence-class = "adjacency has any member outside
focus@5" — made **20 of 23** queries divergence-class: too loose to gate on.
Redefinition, accepted by Opie in #3194: the outside-focus member must also sit
**inside the shipped band (span ≥ 0.65)** — a member the lens could actually
serve (H2 showed widening the band to reach the others destroys divergence).
Span is `pool_similarities` INPUT geometry, shared by focus and explore, so the
class is not read off the explore output under test. Computed in `sweep.py`
the way `/context` does (L2 on the seeded vectors → 1/(1+d) → anchored on the
pool's best hit, session_log never in the pool); cross-check: m-backup on e03
computes 0.350, the value the S296 E4 probe measured by a different route.

**Tally under the new definition: 3 divergence-class / 20 coverage** — e02
(d-archive, span 0.772), e08 (d-secrets, 0.962), e23 (d-archive, 0.835).
Opie's "non-trivial-but-real" range was 3–8; this is the floor of it. Small but
it has a tooth: **e23 collapses at the shipped point** (the lens serves focus's
five and misses a servable adjacent member at span 0.835) and stays collapsed
at every width. e03 and e12 become coverage-class (m-backup 0.350, d-git-add
0.549 — below the band): their collapses are honest, not misses. Next-nearest
misses: e18 d-brain-wins 0.558, e10 m-pip 0.546 — a band edge at 0.55 would
make them divergence-class; 0.65 is the shipped band's own edge, not tuned.
Label disagreements (computed wins, per #3188): e18, e20, e24 predicted
divergence, compute coverage.

**Selection re-run under the new class: still NO feasible point — the verdict
stands.** With e03/e12 reclassified, two grid points clear the guardrails and
zero collapses: imp=0 at width 0.80 and 1.00 (divergence 0.408, adjMRR 0.529,
adjR@5 0.483, precision 0.535). Both sit BELOW the H0 sim-only control on
divergence (0.419): they are focus with a novelty tiebreak, not a lens. The
docstring's H0 control ("explore must beat it on divergence or it is not
earning its complexity") had been stated since S296 but not branched on in
code — it is now the fifth feasibility term. Before the redefinition no point
reached it, so the S296 table's verdict is unchanged by the fix.

## What changed in the repo

- `tests/recall/explore_fixtures.json`: 23 queries; gate re-baselined on the 23
  at the shipped lens (fixture change, not lens change — disclosed in `_about`).
- `tests/recall/test_explore_harness.py`: integrity check — no expected/adjacent
  id may be `session_log` (hidden by `/context` by default).
- `tools/experiments/e5-ablation/sweep.py`: the grid. Re-run lands on this table.
  S297: `query_spans` + span-aware `classify` (band edge `BAND_MIN` captured at
  import from the shipped constants), per-query class/span print, H0 control
  enforced in the selection rule. No runtime change; live 4.16.0 unaffected.
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
