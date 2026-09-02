import json, math, statistics as st
from pathlib import Path as _P
from pathlib import Path
import sys; sys.path.insert(0, str(_P(__file__).parent))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    from analyze import spearman, recency, IMP, W_SIM, W_REC, W_IMP
from replay import to_cos
recs = [json.loads(l) for l in (_P(__file__).parent / "recalls.jsonl").read_text().splitlines()]
def norm(cos, floor):
    lo, hi = min(cos), max(cos); span = max(hi - lo, floor)
    return [(c - lo) / span if span > 0 else 1.0 for c in cos]
print(f"{'floor':>6s} {'mean rho':>9s} {'median':>7s} {'top1=maxsim':>12s} {'rho>0.6':>8s} {'sim spread med':>15s}  tie-band test (doc 0.955 vs noise 0.98, 2-pool)")
for floor in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    rhos, top1, spreads = [], [], []
    for r in recs:
        hs = r["hits"]
        if len(hs) < 3: continue
        rel = [h["rel"] for h in hs]; cos = [to_cos(x) for x in rel]
        sims = norm(cos, floor)
        comp = [W_SIM*s + W_REC*recency(h["age"]) + W_IMP*IMP.get(h["cat"], .5) for h, s in zip(hs, sims)]
        order = sorted(range(len(hs)), key=lambda i: -comp[i])
        rs = [0]*len(hs)
        for p, i in enumerate(order): rs[i] = -p
        rhos.append(spearman(rs, rel)); top1.append(order[0] == max(range(len(hs)), key=lambda i: rel[i]))
        spreads.append(W_SIM*(max(sims)-min(sims)))
    # existing test scenario: pool of 2, doctrine cos 0.955, unknown noise cos 0.98, same age
    s = norm([0.98, 0.955], floor)
    noise = W_SIM*s[0] + W_IMP*0.4; doc = W_SIM*s[1] + W_IMP*1.0
    print(f"{floor:6.2f} {st.mean(rhos):+9.3f} {st.median(rhos):+7.3f} {sum(top1)/len(top1):>11.0%} "
          f"{sum(x>.6 for x in rhos)/len(rhos):>8.0%} {st.median(spreads):>15.3f}  {'doctrine wins' if doc > noise else 'NOISE wins'} (Δ{doc-noise:+.3f})")
