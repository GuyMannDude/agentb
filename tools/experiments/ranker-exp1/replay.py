#!/usr/bin/env python3
"""Offline replay of E1 candidates over the recorded pools.

Caveat printed with the results: each pool is the served top-10 (post-composite),
not the full VEC candidate list, so this measures re-ordering inside the served
band, not what a different band would have contained.

Variants:
  A  current:  rel = 1/(1+d), raw into 0.55*sim
  B  cosine:   cos = 1 - d^2/2 (unit vectors), raw into 0.55*sim
  C  minmax:   rel min-max normalised over the pool (trajectory.py pattern)
  D  cos+minmax
"""
import json, math, statistics as st
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze import spearman, recency, IMP, W_SIM, W_REC, W_IMP, W_ACC  # noqa

recs = [json.loads(l) for l in (Path(__file__).parent / "recalls.jsonl").read_text().splitlines()]

def to_cos(rel):
    d = 1.0 / rel - 1.0
    return max(-1.0, min(1.0, 1.0 - d * d / 2.0))

def minmax(xs):
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 1.0 for x in xs]

def score(hits, sims, access=None):
    out = []
    for h, s in zip(hits, sims):
        a = 0.0 if access is None else access[h["id"]]
        acc = min(1.0, math.log2(1 + a) / math.log2(7))
        out.append(W_SIM * s + W_REC * recency(h["age"]) + W_IMP * IMP.get(h["cat"], .5) + W_ACC * acc)
    return out

variants = {
    "A current (1/(1+d) raw)":     lambda rel: rel,
    "B cosine raw":                lambda rel: [to_cos(r) for r in rel],
    "C rel min-max":               lambda rel: minmax(rel),
    "D cosine + min-max":          lambda rel: minmax([to_cos(r) for r in rel]),
}

print(f"pools: {len(recs)}  (served top-10 only — see caveat in file header)\n")
print(f"{'variant':28s} {'mean rho':>9s} {'median':>7s} {'top1=maxsim':>12s} {'rho>0.6':>8s}  sim-term spread (median)")
for name, fn in variants.items():
    rhos, top1, spreads = [], [], []
    for r in recs:
        hs = r["hits"]
        if len(hs) < 3: continue
        rel = [h["rel"] for h in hs]
        sims = fn(rel)
        comp = score(hs, sims)
        order = sorted(range(len(hs)), key=lambda i: -comp[i])
        rank_score = [0] * len(hs)
        for pos, i in enumerate(order): rank_score[i] = -pos
        rhos.append(spearman(rank_score, rel))
        top1.append(order[0] == max(range(len(hs)), key=lambda i: rel[i]))
        spreads.append(W_SIM * (max(sims) - min(sims)))
    print(f"{name:28s} {st.mean(rhos):+9.3f} {st.median(rhos):+7.3f} {sum(top1)/len(top1):>11.0%} "
          f"{sum(x > .6 for x in rhos)/len(rhos):>8.0%}  {st.median(spreads):.3f}")

# what does cosine do to the band itself?
allrel = [h["rel"] for r in recs for h in r["hits"]]
allcos = [to_cos(x) for x in allrel]
q = lambda xs, p: sorted(xs)[min(len(xs)-1, int(p*len(xs)))]
print(f"\nband, all served hits:  rel p10/p50/p90 = {q(allrel,.1):.3f}/{q(allrel,.5):.3f}/{q(allrel,.9):.3f}"
      f"   cosine p10/p50/p90 = {q(allcos,.1):.3f}/{q(allcos,.5):.3f}/{q(allcos,.9):.3f}")
# per-pool cosine range
ranges = [max(to_cos(h["rel"]) for h in r["hits"]) - min(to_cos(h["rel"]) for h in r["hits"]) for r in recs if len(r["hits"]) >= 3]
print(f"per-pool cosine range: median {st.median(ranges):.3f}  p10 {q(ranges,.1):.3f}  p90 {q(ranges,.9):.3f}")
