#!/usr/bin/env python3
"""Experiment One analysis — Spearman(final rank, raw similarity) per recall."""
import json, math, statistics as st
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
recs = [json.loads(l) for l in (HERE / "recalls.jsonl").read_text().splitlines()]

# live defaults (agentb/config.py RankingConfig, repo == live 4.15.0)
W_SIM, W_REC, W_IMP, W_ACC, HALF = 0.55, 0.20, 0.15, 0.10, 30.0
IMP = {"doctrine": 1.0, "incident": .95, "decision": .95, "identity": .90, "idea": .85,
       "relationship": .80, "topology": .75, "current_state": .75, "unknown": .40,
       "session_log": .20, None: .50}

def avg_ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks

def spearman(a, b):
    if len(a) < 3: return None
    ra, rb = avg_ranks(a), avg_ranks(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else None

def recency(age):
    return 0.5 if age is None else math.exp(-max(0.0, age) / HALF * math.log(2))

def composite(h, access=0):
    acc = min(1.0, math.log2(1 + access) / math.log2(7))
    return W_SIM * h["rel"] + W_REC * recency(h["age"]) + W_IMP * IMP.get(h["cat"], .5) + W_ACC * acc

rhos, rhos_recon, top1_same, sim_spread, rec_spread, imp_spread = [], [], [], [], [], []
tiers, cats, hits_hist, all_rel, n_hits = Counter(), Counter(), Counter(), [], []
top1_sim_rank = []   # where the highest-similarity hit landed in the served order
for r in recs:
    hs = r["hits"]; n_hits.append(len(hs))
    for k, v in r["cache_hits"].items(): hits_hist[k] += v
    for h in hs:
        tiers[h["tier"]] += 1; cats[h["cat"]] += 1; all_rel.append(h["rel"])
    if len(hs) < 3: continue
    served = [-h["pos"] for h in hs]        # higher = served earlier
    rel = [h["rel"] for h in hs]
    rho = spearman(served, rel)
    if rho is not None: rhos.append(rho)
    rr = spearman(served, [composite(h) for h in hs])
    if rr is not None: rhos_recon.append(rr)
    best_sim = max(hs, key=lambda h: h["rel"])
    top1_same.append(best_sim["pos"] == 1)
    top1_sim_rank.append(best_sim["pos"])
    sim_spread.append(W_SIM * (max(rel) - min(rel)))
    recs_ = [recency(h["age"]) for h in hs]
    rec_spread.append(W_REC * (max(recs_) - min(recs_)))
    imps = [IMP.get(h["cat"], .5) for h in hs]
    imp_spread.append(W_IMP * (max(imps) - min(imps)))

def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

print(f"queries: {len(recs)}   hits: {sum(n_hits)}   pools scored (n>=3): {len(rhos)}")
print(f"latency ms: median {st.median([r['latency_ms'] for r in recs]):.0f}")
print()
print("== Spearman(served rank, raw relevance) per query ==")
print(f"  mean {st.mean(rhos):+.3f}  median {st.median(rhos):+.3f}  "
      f"p10 {q(rhos,.1):+.3f}  p90 {q(rhos,.9):+.3f}")
print(f"  share of queries with rho > 0.6: {sum(x > .6 for x in rhos)/len(rhos):.0%}   "
      f"rho < 0.2: {sum(x < .2 for x in rhos)/len(rhos):.0%}   rho < 0: {sum(x < 0 for x in rhos)/len(rhos):.0%}")
print(f"  top-1 served == top-1 by similarity: {sum(top1_same)/len(top1_same):.0%}")
print(f"  where the most-similar hit lands (served position): "
      + ", ".join(f"#{k}:{v}" for k, v in sorted(Counter(top1_sim_rank).items())))
print()
print("== Reconstructed composite (access=0) vs served order ==")
print(f"  mean rho {st.mean(rhos_recon):+.3f}  median {st.median(rhos_recon):+.3f}  "
      f"(1.0 = formula reproduces server; gap = access term + unknowns)")
print()
print("== Signal spread inside each served pool (weighted, median over queries) ==")
print(f"  similarity term: {st.median(sim_spread):.3f}   recency term: {st.median(rec_spread):.3f}   "
      f"importance term: {st.median(imp_spread):.3f}")
print(f"  raw relevance over ALL hits: min {min(all_rel):.3f}  p10 {q(all_rel,.1):.3f}  "
      f"median {st.median(all_rel):.3f}  p90 {q(all_rel,.9):.3f}  max {max(all_rel):.3f}")
print()
print("== cache_hits histogram (E3) ==")
tot = sum(hits_hist.values())
for k, v in hits_hist.most_common(): print(f"  {k:5s} {v:5d}  {v/tot:.1%}")
print()
print("== served categories ==")
for k, v in cats.most_common(): print(f"  {str(k):14s} {v:5d}  {v/sum(cats.values()):.1%}")
print()
print("== pool sizes ==", dict(sorted(Counter(n_hits).items())))
