#!/usr/bin/env python3
"""Mnemo Experiment One — READ-ONLY probe of /context.

For each query: one real /context call (focus mode, tenant cc, 10 results).
Logs per hit: position, cache_tier, relevance (raw), category, age_days,
memory_id. Never writes. The only server side effect is the access bump
/context always performs on served memories (documented in the report).
"""
import json, os, sys, time, urllib.request
from pathlib import Path

URL = os.environ.get("MNEMO_URL", "http://localhost:50001")
TOKEN = (Path.home() / ".mnemo-auth-token").read_text().strip()
HERE = Path(__file__).parent
OUT = HERE / "recalls.jsonl"
MAXR = 10

def call(prompt):
    body = json.dumps({
        "prompt": prompt, "agent_id": os.environ.get("MNEMO_AGENT_ID", "default"), "mode": "focus",
        "max_results": MAXR, "exclude_categories": ["session_log"],
    }).encode()
    req = urllib.request.Request(f"{URL}/context", data=body, method="POST",
        headers={"Content-Type": "application/json", "X-API-KEY": TOKEN})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

QUERIES = Path(os.environ.get("EXP1_QUERIES", HERE / "queries.txt"))
queries = [q.strip() for q in QUERIES.read_text().splitlines() if q.strip()]
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        done.add(json.loads(line)["query"])
print(f"{len(queries)} queries, {len(done)} already done", flush=True)

with OUT.open("a") as f:
    for i, q in enumerate(queries, 1):
        if q in done:
            continue
        try:
            d = call(q)
        except Exception as e:
            print(f"[{i}] FAIL {q!r}: {e}", flush=True)
            continue
        rec = {
            "query": q, "latency_ms": d["latency_ms"], "cache_hits": d["cache_hits"],
            "provider": d.get("provider_used"),
            "hits": [{
                "pos": p, "tier": c["cache_tier"], "rel": c["relevance"],
                "cat": c.get("category"), "age": c.get("age_days"),
                "id": c.get("memory_id"), "src": c.get("provenance_source"),
            } for p, c in enumerate(d["chunks"], 1)],
        }
        f.write(json.dumps(rec) + "\n"); f.flush()
        rels = [h["rel"] for h in rec["hits"]]
        print(f"[{i}/{len(queries)}] n={len(rels)} rel={min(rels or [0]):.3f}-{max(rels or [0]):.3f} "
              f"tiers={d['cache_hits']} {d['latency_ms']:.0f}ms", flush=True)
        time.sleep(0.3)
print("done", flush=True)
