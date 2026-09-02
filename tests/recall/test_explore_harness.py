"""E4 — the explore harness: explore mode's own gate.

Focus asks "is the best match first" (recall@5 + MRR, test_recall_harness).
Explore asks a different question — "is the adjacent, rarely-recalled memory
surfaced" — so it cannot ride focus's fixtures: scored by best-match-first,
explore would only pass by collapsing into focus. This file gives the
serendipity lens its own queries (explore_fixtures.json, over the SAME
fictional memories and the SAME cached production vectors) and its own
three-part criterion:

  differential — adjacency-set items explore@5 serves that focus@5 does not
                 (the number that proves explore earns its existence);
  precision    — served items that are on-topic (bullseye, adjacent or ok):
                 explore keeps near-but-not-top, it does not open the gates;
  divergence   — Jaccard distance between explore@5 and focus@5: if explore
                 equals focus, E4 failed even when the other numbers look fine.

adj_mrr rides along as the rank-sensitive twin of differential: the set
metrics cannot see a regression that keeps every adjacent item in the window
and pushes it lower, so a one-rank-demotion control must be able to fail
adj_mrr alone (the hard-subset lesson from E2, #3167/#3169).

Gate rule: floors are MEASURED, never tuned. Floor = the first honest run
minus one unit of the metric's granularity over the fixture set — for the
means over N queries that is one rank step on one query (adj_mrr: 0.5/N;
precision: (1/TOP_K)/N; differential: 1/N; divergence: (1/TOP_K)/N) — so a
single rank move is churn and a second is a regression. A floor with zero
headroom is a ratchet.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import agentb.server
from agentb import ranking
from agentb.config import RankingConfig
from tests.recall.embed_fixtures import (
    CACHE_VERSION, EMBED_MODEL, EMBEDDINGS, cache_key, load_cache,
    load_explore_fixtures, load_fixtures,
)
from tests.recall.test_recall_harness import (
    REGEN, TOP_K, _make_client, _reset_access, _seed,
)

N_STEP = {  # one unit of granularity per metric, over one query
    "differential": 1.0,
    "adj_mrr": 0.5,
    "precision": 1.0 / TOP_K,
    "divergence": 1.0 / TOP_K,
}


@pytest.fixture(scope="module")
def world():
    fixtures = load_fixtures()
    explore = load_explore_fixtures()
    if not EMBEDDINGS.exists():
        pytest.fail(f"{EMBEDDINGS.name} missing — run {REGEN}")
    cache = load_cache()
    assert cache["model"] == EMBED_MODEL, (
        f"embedding cache built by {cache['model']!r}, harness expects {EMBED_MODEL!r} — run {REGEN} --all")
    assert cache.get("cache_version") == CACHE_VERSION, (
        f"embedding cache is v{cache.get('cache_version')}, harness expects v{CACHE_VERSION} — run {REGEN} --all")
    return fixtures, explore, cache["vectors"]


# ── serving + scoring ───────────────────────────────────────────────────────

def _serve(client, index_path: Path, fixtures: dict, prompt: str, mode: str) -> list[str]:
    _reset_access(index_path, fixtures)
    body = {"prompt": prompt, "max_results": TOP_K}
    if mode == "explore":
        body["mode"] = "explore"
    r = client.post("/context", json=body)
    assert r.status_code == 200, f"{mode} {prompt!r}: {r.text}"
    return [c["memory_id"] for c in r.json()["chunks"]]


def score_query(q: dict, explore: list[str], focus: list[str]) -> dict:
    adjacent = set(q["adjacent"])
    on_topic = adjacent | set(q["expected"]) | set(q.get("ok", []))
    e, f = set(explore), set(focus)
    hit_ranks = [i + 1 for i, m in enumerate(explore) if m in adjacent]
    return {
        "id": q["id"],
        "adj_recall": len(e & adjacent) / len(adjacent),
        "adj_rr": 1.0 / hit_ranks[0] if hit_ranks else 0.0,
        "differential": len((e & adjacent) - f),
        "precision": len(e & on_topic) / len(explore) if explore else 0.0,
        "divergence": 1.0 - len(e & f) / len(e | f) if (e | f) else 0.0,
        "explore": explore,
        "focus": focus,
        "adjacent": q["adjacent"],
        "note": q["note"],
    }


def run_explore_harness(client, index_path: Path, fixtures: dict, explore_fixtures: dict) -> list[dict]:
    rows = []
    for q in explore_fixtures["queries"]:
        served_focus = _serve(client, index_path, fixtures, q["prompt"], "focus")
        served_explore = _serve(client, index_path, fixtures, q["prompt"], "explore")
        rows.append(score_query(q, served_explore, served_focus))
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "queries": n,
        "adj_recall": sum(r["adj_recall"] for r in rows) / n,
        "adj_mrr": sum(r["adj_rr"] for r in rows) / n,
        "differential": sum(r["differential"] for r in rows) / n,
        "precision": sum(r["precision"] for r in rows) / n,
        "divergence": sum(r["divergence"] for r in rows) / n,
        "empty": [r["id"] for r in rows if not r["explore"]],
    }


def report(rows: list[dict], summary: dict, title: str) -> str:
    lines = [f"── {title} ──",
             f"differential {summary['differential']:.3f}/query   adj MRR {summary['adj_mrr']:.3f}   "
             f"adj recall@{TOP_K} {summary['adj_recall']:.3f}   precision {summary['precision']:.3f}   "
             f"divergence {summary['divergence']:.3f}   n={summary['queries']}"
             + (f"   EMPTY: {summary['empty']}" if summary["empty"] else "")]
    for r in rows:
        lines.append(f"{r['id']}  diff={r['differential']}  rr={r['adj_rr']:.2f}  "
                     f"prec={r['precision']:.2f}  div={r['divergence']:.2f}  "
                     f"explore={r['explore']}  focus={r['focus']}  adjacent={r['adjacent']}  {r['note']}")
    return "\n".join(lines)


def floors_from(summary: dict) -> dict:
    """The headroom rule made executable: one granularity unit on one query."""
    n = summary["queries"]
    return {f"{k}_min": round(summary[k] - N_STEP[k] / n, 4)
            for k in ("differential", "adj_mrr", "precision", "divergence")}


def _run(tmp_path: Path, world) -> tuple[list[dict], dict]:
    fixtures, explore, vectors = world
    index_path = _seed(tmp_path, fixtures, vectors)
    with _make_client(tmp_path, vectors, RankingConfig()) as client:
        rows = run_explore_harness(client, index_path, fixtures, explore)
    return rows, summarize(rows)


def _assert_gate(summary: dict, gate: dict, text: str) -> None:
    for k in ("differential", "adj_mrr", "precision", "divergence"):
        assert gate.get(f"{k}_min") is not None, (
            f"gate {k}_min unset in explore_fixtures.json — record the measured baseline first")
        assert summary[k] >= gate[f"{k}_min"], (
            f"explore {k} {summary[k]:.3f} fell below gate {gate[f'{k}_min']}\n{text}")


# ── the gate ────────────────────────────────────────────────────────────────

def test_explore_fixture_integrity(world):
    fixtures, explore, vectors = world
    ids = {m["id"] for m in fixtures["memories"]}
    seen = set()
    for q in explore["queries"]:
        assert q["id"] not in seen, f"duplicate query id {q['id']}"
        seen.add(q["id"])
        for key in ("expected", "adjacent", "ok"):
            unknown = [m for m in q.get(key, []) if m not in ids]
            assert not unknown, f"{q['id']}.{key} names unknown ids {unknown}"
        assert q["adjacent"], f"{q['id']} has an empty adjacency set — nothing to surface"
        assert not set(q["adjacent"]) & set(q["expected"]), (
            f"{q['id']}: an id cannot be both the bullseye and adjacent to it")
        assert cache_key("query", q["prompt"]) in vectors, f"{q['id']}: no vector — run {REGEN}"


def test_explore_gate(tmp_path, world):
    _, explore, _ = world
    rows, summary = _run(tmp_path, world)
    text = report(rows, summary, "E4 explore harness — live lens")
    print("\n" + text + f"\nfloors this run would set: {floors_from(summary)}")
    assert not summary["empty"], f"explore served NOTHING for {summary['empty']} — the floor ate the pool\n{text}"
    _assert_gate(summary, explore["gate"], text)


def _demote_first_adjacent(row: dict) -> dict:
    """Same served list, first adjacent id one slot lower, still inside the
    window. Set metrics cannot see it; adj_mrr must."""
    served = list(row["explore"])
    idx = next((i for i, m in enumerate(served) if m in row["adjacent"]), None)
    if idx is not None and idx + 1 < len(served):
        served[idx], served[idx + 1] = served[idx + 1], served[idx]
    q = {"id": row["id"], "adjacent": row["adjacent"], "expected": [], "ok": [], "note": row["note"]}
    # expected/ok are folded back through precision below; keep it exact:
    scored = score_query(q, served, row["focus"])
    scored["precision"] = row["precision"]  # unchanged by a swap inside the window
    return scored


def test_demotion_control_fails_only_the_rank_gate(tmp_path, world):
    """Every query's first adjacent item one rank lower inside the window.
    differential / precision / divergence are set metrics and must still
    PASS — that is the masking — and adj_mrr must FAIL. If the set gates
    also fail, adj_mrr adds nothing and this test says so."""
    _, explore, _ = world
    gate = explore["gate"]
    rows, _ = _run(tmp_path, world)
    demoted = [_demote_first_adjacent(r) for r in rows]
    summary = summarize(demoted)
    print("\n" + report(demoted, summary, "control — first adjacent one rank lower"))
    for k in ("differential", "precision", "divergence"):
        assert summary[k] >= gate[f"{k}_min"], (
            f"set gate {k} ALSO failed ({summary[k]:.3f} < {gate[f'{k}_min']}) — adj_mrr is not adding anything")
    assert summary["adj_mrr"] < gate["adj_mrr_min"], (
        f"adj MRR {summary['adj_mrr']:.3f} passed the gate {gate['adj_mrr_min']} with every first "
        "adjacent item demoted — the rank gate cannot catch the regression it was built for")


def test_gate_can_fail_without_the_adjacency_term(tmp_path, world, monkeypatch):
    """Control: explore scored on importance + novelty alone (adjacency
    weight 0) must fall below the gate — adj_mrr and precision specifically.

    Measured 2026-09-02: this control lands EXACTLY on the differential floor
    (0.375 — it loses one explore-only find, and the floor's headroom is one),
    so differential does not discriminate it; adj MRR (0.506 < 0.580) and
    precision (0.381 < 0.544) do. differential is the existence claim — a
    lens that finds nothing focus misses has no reason to exist — and the
    other floors are what catch a lens that drifted to noise. Keep all four.
    """
    _, explore, _ = world
    gate = explore["gate"]
    monkeypatch.setattr(ranking, "W_EXPLORE_ADJACENCY", 0.0)
    rows, summary = _run(tmp_path, world)
    print("\n" + report(rows, summary, "control — adjacency weight 0"))
    assert summary["adj_mrr"] < gate["adj_mrr_min"], (
        f"control adj MRR {summary['adj_mrr']:.3f} passed the gate — the harness cannot fail")
    assert summary["precision"] < gate["precision_min"], (
        f"control precision {summary['precision']:.3f} passed the gate — the noise floor cannot fail")


def test_gate_can_fail_when_explore_collapses_into_focus(tmp_path, world, monkeypatch):
    """Control: a 'lens' that ranks by similarity alone is a best-match
    ranker wearing explore's hat. divergence must fall below its floor.

    Measured 2026-09-02: this control finds adjacent items at least as well
    as the live lens (adj MRR 0.630 vs 0.611, adj recall 0.548 vs 0.506) and
    is as clean (precision 0.562) — the band floor plus similarity is what
    locates the adjacency band; the target term does not. What the live lens
    adds is divergence from focus (0.512 vs 0.418, the novelty + importance
    re-order) and one more explore-only find over 16 queries (0.438 vs
    0.375). That is explore's measured edge: modest, and real."""
    _, explore, _ = world
    gate = explore["gate"]
    monkeypatch.setattr(agentb.server, "explore_score", lambda **kw: max(kw["similarity"], 1e-9))
    rows, summary = _run(tmp_path, world)
    print("\n" + report(rows, summary, "control — similarity-only lens"))
    assert summary["divergence"] < gate["divergence_min"], (
        f"collapsed lens still diverged {summary['divergence']:.3f} ≥ {gate['divergence_min']} — "
        "the anti-collapse gate cannot fail")
