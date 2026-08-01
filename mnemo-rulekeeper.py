#!/usr/bin/env python3
"""Nightly advisory-only memory near-duplicate scanner."""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import httpx
from agentb.config import get_agent_data_dir, load_config
from agentb.rulekeeper import format_advisory, scan_tenant
from agentb.vec import VecStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--agent")
    parser.add_argument("--output", default=os.getenv(
        "MNEMO_RULEKEEPER_ADVISORY", str(Path.home() / ".mnemo-dreams" / "rulekeeper-latest.md")))
    args = parser.parse_args()
    cfg = load_config(args.config)
    agents = [args.agent] if args.agent else (list(cfg.agents) or [None])
    blocks, findings, overall = [], {}, 0
    for agent_id in agents:
        data_dir = get_agent_data_dir(cfg, agent_id)
        store = VecStore(data_dir / "vec_index.sqlite")
        try:
            result = scan_tenant(data_dir / "memory", store,
                window_days=cfg.dedup.nightly_window_days, top_k=cfg.dedup.top_k,
                cosine_threshold=cfg.dedup.cosine_threshold,
                overlap_threshold=cfg.dedup.overlap_threshold,
                min_tokens=cfg.dedup.min_tokens)
        finally:
            store.close()
        name = agent_id or "default"
        findings[name] = result
        blocks.append(format_advisory(name, result))
        overall = max(overall, result["exit_code"])
    report = "\n\n".join(blocks) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="ascii", errors="replace")
    print(report, end="")

    bus_url, bus_from = os.getenv("MNEMO_DREAM_BUS_URL", ""), os.getenv("MNEMO_DREAM_BUS_FROM", "")
    if bus_url and bus_from:
        for agent_id, result in findings.items():
            if not result["pairs"]:
                continue
            envelope = {"mesh_version": "0.5", "from": bus_from,
                        "to": agent_id.capitalize(),
                        "subject": f"rulekeeper-near-duplicates-{agent_id}",
                        "body": {"source": "rulekeeper", "advisory_only": True,
                                 "report": result}}
            try:
                response = httpx.post(f"{bus_url}/mesh/ping", json=envelope, timeout=10)
                if response.status_code not in (200, 201, 202):
                    print(f"bus ping {agent_id}: HTTP {response.status_code}", file=sys.stderr)
            except httpx.HTTPError as exc:
                print(f"bus ping {agent_id}: {exc}", file=sys.stderr)
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
