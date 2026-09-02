"""E5 — explore weight + band-width ablation (spec: brain/spec-mnemo-explore-e5-ablation.md).

Which knobs produce explore's edge (divergence over focus + explore-only finds)?
Axes: A) weights adjacency/importance/novelty on the simplex — zero-ablations and an
adjacency->novelty reallocation grid; B) EXPLORE_SCALE (band width). OFFSET 0.05 and
FLOOR 0.80 fixed (E4 probe: the floor first bites at span<0.20, below the band).
Control H0: similarity-only lens — explore must beat it on divergence or it is not
earning its complexity.

Per-query CLASS is COMPUTED from focus@5 (focus is fixed, not under test): a query is
divergence-class when its adjacency set has a member outside focus@5 that also sits
INSIDE the shipped band (span >= BAND_MIN) — a member the lens could actually serve;
else coverage. Span is the pool-normalised similarity (pool_similarities) over the
default-visible world: INPUT geometry shared by focus and explore, not the explore
output under test. (Redefined per Opie #3194; the first cut — any member outside
focus@5 — made 20/23 queries divergence-class and could not gate.)
Opie's fixture 'class' field is a prediction and is reported where it disagrees.

Selection rule, fixed before the sweep: maximise divergence subject to
  precision >= focus's precision on the same queries,
  adj MRR and adj recall@5 >= the SHIPPED lens on the same queries (no regression),
  zero divergence-class collapses (no divergence-class query with divergence 0),
  divergence > the H0 sim-only control at the same scale (stated above since S296;
  enforced in code since the class redefinition — before it no point reached this test).
Baselines are measured in this run, never ported from the 16-query E4 numbers.

    python tools/experiments/e5-ablation/sweep.py            # full grid (~60 runs)
    python tools/experiments/e5-ablation/sweep.py --scale 0.3 0.5 --adj 0.55 0.25
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import agentb.server  # noqa: E402
from agentb import ranking  # noqa: E402
from agentb.config import RankingConfig  # noqa: E402
from agentb.provenance import DEFAULT_HIDDEN_CATEGORIES  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402
from tests.recall.embed_fixtures import cache_key, load_cache, load_explore_fixtures, load_fixtures  # noqa: E402
from tests.recall.test_explore_harness import run_explore_harness, summarize  # noqa: E402
from tests.recall.test_recall_harness import _make_client, _seed  # noqa: E402

_orig_init = TestClient.__init__


def _loopback_init(self, *a, **kw):  # same as tests/conftest.py: unauthenticated = loopback only
    kw.setdefault("client", ("127.0.0.1", 50000))
    _orig_init(self, *a, **kw)


TestClient.__init__ = _loopback_init

SHIPPED = (0.55, 0.30, 0.15)
IMPORTANCE_FIXED = 0.30
# Inside the SHIPPED band = the adjacency term is live: |span - (1 - OFFSET)| < SCALE
# -> span > 1 - OFFSET - SCALE = 0.65. Captured at import, before the grid mutates SCALE:
# class is a property of the query under the shipped lens, not of the point under test.
BAND_MIN = round(1.0 - ranking.EXPLORE_OFFSET - ranking.EXPLORE_SCALE, 4)


def query_spans(fixtures, explore, vectors):
    """Per query: memory_id -> span, computed the way /context does — L2 on the seeded
    vectors -> 1/(1+d) relevance -> pool_similarities -> anchored on the pool's best hit —
    over the default-visible world (DEFAULT_HIDDEN_CATEGORIES never enter the pool)."""
    world = [m for m in fixtures["memories"] if m["category"] not in DEFAULT_HIDDEN_CATEGORIES]
    docs = [(m["id"], vectors[cache_key("document", m["summary"])]) for m in world]
    spans = {}
    for q in explore["queries"]:
        qv = vectors[cache_key("query", q["prompt"])]
        sims = ranking.pool_similarities([(1.0 / (1.0 + math.dist(qv, v)), "VEC") for _, v in docs])
        spans[q["id"]] = {mid: sim for (mid, _), sim in zip(docs, sims)}
    return spans


def classify(rows, spans):
    """divergence-class = an adjacent member outside focus@5 that sits inside the band
    (span >= BAND_MIN); computed, not labelled."""
    out = {}
    for r in rows:
        outside = set(r["adjacent"]) - set(r["focus"])
        out[r["id"]] = "divergence" if any(spans[r["id"]].get(m, 0.0) >= BAND_MIN for m in outside) else "coverage"
    return out


def run_point(fixtures, explore, vectors, spans, weights, scale, sim_only=False):
    ranking.W_EXPLORE_ADJACENCY, ranking.W_EXPLORE_IMPORTANCE, ranking.W_EXPLORE_NOVELTY = weights
    ranking.EXPLORE_SCALE = scale
    saved = agentb.server.explore_score
    if sim_only:
        agentb.server.explore_score = lambda **kw: max(kw["similarity"], 1e-9)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            index_path = _seed(p, fixtures, vectors)
            with _make_client(p, vectors, RankingConfig()) as client:
                rows = run_explore_harness(client, index_path, fixtures, explore)
    finally:
        agentb.server.explore_score = saved
    s = summarize(rows)
    cls = classify(rows, spans)
    s["collapsed"] = sorted(r["id"] for r in rows if cls[r["id"]] == "divergence" and r["divergence"] == 0.0)
    s["classes"] = cls
    s["rows"] = rows
    return s


def focus_metrics(rows):
    n = len(rows)
    prec = 0.0
    for r in rows:
        on = set(r["adjacent"]) | set(r.get("expected", [])) | set(r.get("ok", []))
        prec += len(set(r["focus"]) & on) / len(r["focus"]) if r["focus"] else 0.0
    return prec / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", nargs="+", type=float, default=[0.55, 0.45, 0.35, 0.25, 0.15, 0.05],
                    help="adjacency weight; importance fixed 0.30; novelty = 0.70 - adj")
    ap.add_argument("--scale", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.65, 0.80, 1.00])
    args = ap.parse_args()

    fixtures, explore, vectors = load_fixtures(), load_explore_fixtures(), load_cache()["vectors"]
    # score_query keeps adjacent/note only; carry expected/ok for focus precision
    byid = {q["id"]: q for q in explore["queries"]}

    def enrich(s):
        for r in s["rows"]:
            r["expected"] = byid[r["id"]]["expected"]
            r["ok"] = byid[r["id"]].get("ok", [])
        return s

    spans = query_spans(fixtures, explore, vectors)
    base = enrich(run_point(fixtures, explore, vectors, spans, SHIPPED, ranking.EXPLORE_SCALE))
    focus_prec = focus_metrics(base["rows"])
    labels = {q["id"]: q.get("class") for q in explore["queries"]}
    disagree = [q for q, c in base["classes"].items() if labels[q] not in (None, c)]
    n_div = sum(1 for c in base["classes"].values() if c == "divergence")
    n_loose = sum(1 for r in base["rows"] if set(r["adjacent"]) - set(r["focus"]))
    print(f"queries {base['queries']}  computed classes (outside focus@5 AND span >= {BAND_MIN}): "
          f"divergence {n_div} / coverage {base['queries'] - n_div}  [first-cut 'any outside focus@5': {n_loose}]"
          f"  label disagreements (computed wins): {disagree}")
    for r in base["rows"]:
        outside = sorted(set(r["adjacent"]) - set(r["focus"]))
        best = max(((spans[r["id"]].get(m, 0.0), m) for m in outside), default=(0.0, "-"))
        print(f"    {r['id']} {base['classes'][r['id']]:>10}  best outside-focus adjacent {best[1]:<16} span {best[0]:.3f}")
    print(f"BASELINES measured here — focus precision {focus_prec:.4f}; shipped lens "
          f"(0.55/0.30/0.15, scale {ranking.EXPLORE_SCALE}): adjMRR {base['adj_mrr']:.4f} "
          f"adjR@5 {base['adj_recall']:.4f} precision {base['precision']:.4f} divergence {base['divergence']:.4f} "
          f"differential {base['differential']:.4f} collapsed {base['collapsed']}")
    guard = {"precision": focus_prec, "adj_mrr": base["adj_mrr"], "adj_recall": base["adj_recall"]}

    weight_sets = [("shipped", SHIPPED),
                   ("adj=0", (0.0, 0.30 / 0.45, 0.15 / 0.45)),
                   ("imp=0", (0.55 / 0.70, 0.0, 0.15 / 0.70)),
                   ("nov=0", (0.55 / 0.85, 0.30 / 0.85, 0.0))]
    for adj in args.adj:
        if abs(adj - 0.55) > 1e-9:
            weight_sets.append((f"adj->nov {adj:.2f}", (adj, IMPORTANCE_FIXED, round(0.70 - adj, 4))))

    hdr = (f"{'lens':>16} {'adj':>5} {'imp':>5} {'nov':>5} {'scale':>5} | {'diverg':>6} {'diff':>5} "
           f"{'adjMRR':>6} {'adjR@5':>6} {'prec':>5} collapsed")
    print(hdr)
    results = []
    for scale in args.scale:
        h0 = run_point(fixtures, explore, vectors, spans, SHIPPED, scale, sim_only=True)
        s = h0
        print(f"{'H0 sim-only':>16} {'-':>5} {'-':>5} {'-':>5} {scale:>5.2f} | {s['divergence']:>6.3f} "
              f"{s['differential']:>5.2f} {s['adj_mrr']:>6.3f} {s['adj_recall']:>6.3f} {s['precision']:>5.3f} {s['collapsed']}")
        for name, w in weight_sets:
            s = run_point(fixtures, explore, vectors, spans, w, scale)
            ok = (s["precision"] >= guard["precision"] and s["adj_mrr"] >= guard["adj_mrr"]
                  and s["adj_recall"] >= guard["adj_recall"] and not s["collapsed"]
                  and s["divergence"] > h0["divergence"])
            results.append((name, w, scale, s, ok))
            print(f"{name:>16} {w[0]:>5.2f} {w[1]:>5.2f} {w[2]:>5.2f} {scale:>5.2f} | {s['divergence']:>6.3f} "
                  f"{s['differential']:>5.2f} {s['adj_mrr']:>6.3f} {s['adj_recall']:>6.3f} {s['precision']:>5.3f} "
                  f"{s['collapsed']}{'  <- feasible' if ok else ''}")
    feasible = [r for r in results if r[4]]
    if feasible:
        name, w, scale, s, _ = max(feasible, key=lambda r: r[3]["divergence"])
        print(f"\nSELECTION (max divergence s.t. guardrails + zero collapses + beats H0): {name} weights {w} scale {scale} "
              f"-> divergence {s['divergence']:.4f} differential {s['differential']:.3f} adjMRR {s['adj_mrr']:.3f} "
              f"precision {s['precision']:.3f}")
    else:
        print("\nSELECTION: NO feasible point — the shipped lens itself fails the guardrails "
              "(collapses or a regression). Report the negative; do not loosen the rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
