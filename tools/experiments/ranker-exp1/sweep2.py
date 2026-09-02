import json, statistics as st, io, contextlib, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import spearman, recency, IMP, W_SIM, W_REC, W_IMP
    from replay import to_cos
recs = [json.loads(l) for l in (Path(__file__).parent / "recalls.jsonl").read_text().splitlines()]
def top_anchored(cos, span):
    hi = max(cos); return [max(0.0, 1.0 - (hi - c) / span) for c in cos]
def floored(cos, floor):
    lo, hi = min(cos), max(cos); s = max(hi - lo, floor); return [(c - lo) / s for c in cos]
ranges = [max(to_cos(h["rel"]) for h in r["hits"]) - min(to_cos(h["rel"]) for h in r["hits"]) for r in recs]
print(f"pools with cosine range < 0.20: {sum(x < .2 for x in ranges)}/{len(ranges)}")
print(f"{'variant':>22s} {'mean rho':>9s} {'top1':>5s} {'rho>0.6':>8s}  tie-band(0.98 vs 0.955)")
for name, fn in [("floored 0.20", lambda c: floored(c, .20)), ("top-anchored 0.15", lambda c: top_anchored(c, .15)),
                 ("top-anchored 0.20", lambda c: top_anchored(c, .20)), ("top-anchored 0.25", lambda c: top_anchored(c, .25)),
                 ("top-anchored 0.30", lambda c: top_anchored(c, .30))]:
    rhos, top1 = [], []
    for r in recs:
        hs = r["hits"]; rel = [h["rel"] for h in hs]; sims = fn([to_cos(x) for x in rel])
        comp = [W_SIM*s + W_REC*recency(h["age"]) + W_IMP*IMP.get(h["cat"], .5) for h, s in zip(hs, sims)]
        order = sorted(range(len(hs)), key=lambda i: -comp[i]); rs = [0]*len(hs)
        for p, i in enumerate(order): rs[i] = -p
        rhos.append(spearman(rs, rel)); top1.append(order[0] == max(range(len(hs)), key=lambda i: rel[i]))
    s = fn([0.98, 0.955]); d = (W_SIM*s[1] + W_IMP*1.0) - (W_SIM*s[0] + W_IMP*0.4)
    print(f"{name:>22s} {st.mean(rhos):+9.3f} {sum(top1)/len(top1):>5.0%} {sum(x>.6 for x in rhos)/len(rhos):>8.0%}  {'doctrine' if d>0 else 'NOISE'} Δ{d:+.3f}")
# outlier robustness: append one far-off candidate (cos 0.30) to every pool, re-measure served-band ordering
print("\nwith one off-topic outlier (cos 0.30) added to each pool:")
for name, fn in [("floored 0.20", lambda c: floored(c, .20)), ("top-anchored 0.20", lambda c: top_anchored(c, .20))]:
    rhos = []
    for r in recs:
        hs = r["hits"]; rel = [h["rel"] for h in hs]; cos = [to_cos(x) for x in rel] + [0.30]
        sims = fn(cos)[:len(hs)]
        comp = [W_SIM*s + W_REC*recency(h["age"]) + W_IMP*IMP.get(h["cat"], .5) for h, s in zip(hs, sims)]
        order = sorted(range(len(hs)), key=lambda i: -comp[i]); rs = [0]*len(hs)
        for p, i in enumerate(order): rs[i] = -p
        rhos.append(spearman(rs, rel))
    print(f"{name:>22s} {st.mean(rhos):+9.3f}")
