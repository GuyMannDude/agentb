"""Disk-truth category filtering, and the E3 contract for the L1/L2 caches.

v4.0.2: the category is canonical *on disk* (`memory/<id>.json`), and the
reclassification migration rewrote only those files — not the L1/L2 tier
caches — so `session_log` leaked past `/context` through their stale
metadata. The fix re-reads disk-truth per hit (`resolve_disk_truth`).

v4.16 (E3): L1 and L2 no longer feed `/context` at all. The first two tests
keep the v4.0.2 seeds (a memory whose only cache residue is a stale L1 or
L2 entry, VEC empty) and pin the new contract: the default filter still
holds on disk-truth, and when the caller opts in the memory is reached by
the L3 escape hatch — never by the retired tier. The remaining tests
unit-test the L1 plumbing and the helper directly.
"""
from __future__ import annotations

import json
import time

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, ResilientProviderConfig, ProviderConfig,
    CacheConfig, ServerConfig, ClassificationConfig, DEFAULT_PERSONAS,
)
from agentb.cache import ContextChunk, resolve_disk_truth

# Fixed 768-dim vector — fake provider embeds everything to this, so the query
# matches every stored memory and the metadata filter alone decides what returns.
VEC = [0.0] * 768
VEC[0] = 1.0
_STATUS = {"primary": "fake", "active": "fake", "failed_over": False,
           "circuit_open": False, "primary_retry_in": None, "fallback_count": 0}


class FakeEmbedding:
    active_label = "fake/embed"
    @property
    def status(self): return _STATUS
    async def embed(self, text, *, use_breaker=True, task_type="document"): return list(VEC)
    async def health_check(self): return True


class FakeReasoning:
    active_label = "fake/reason"
    @property
    def status(self): return _STATUS
    async def generate(self, prompt, system="", max_tokens=2048, *, use_breaker=True): return "topology"
    async def health_check(self): return True


@pytest.fixture
def client(tmp_path):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic-embed-text")),
        cache=CacheConfig(), server=ServerConfig(host="127.0.0.1", port=50099),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=False),
        personas=dict(DEFAULT_PERSONAS),
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as c:
            yield c


def _seed_l2_and_disk(tmp_path, memory_id, summary, disk_category, cached_category):
    """Seed a memory into L2's on-disk index (with a stale cached category) and
    its memory JSON (with the canonical/reclassified category), writing files
    directly so the request thread builds the tenant fresh. VEC is left empty on
    purpose, so L2 is the tier under test."""
    base = tmp_path / "agents" / "default"
    mem_dir = base / "memory"
    l2_dir = base / "cache" / "l2"
    mem_dir.mkdir(parents=True, exist_ok=True)
    l2_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (mem_dir / f"{memory_id}.json").write_text(json.dumps({
        "id": memory_id, "summary": summary, "key_facts": [],
        "category": disk_category, "source": "tool", "created_at": now,
    }))
    (l2_dir / "index.json").write_text(json.dumps([{
        "id": memory_id, "content": summary, "source": f"session:{memory_id}",
        "embedding": list(VEC), "created_at": now,
        "metadata": {"memory_id": memory_id, "provenance_source": "tool",
                     "category": cached_category},
    }]))


def test_l2_residue_is_not_a_recall_tier(client, tmp_path):
    # Cached as 'topology' (pre-migration) but reclassified to 'session_log' on disk.
    _seed_l2_and_disk(tmp_path, "m-stale", "raw auto-sync activity dump",
                      disk_category="session_log", cached_category="topology")

    # Default recall must exclude it on disk-truth, whatever the stale L2 cache says.
    r = client.post("/context", json={"prompt": "activity", "max_results": 5})
    assert r.status_code == 200, r.text
    cats = [c.get("category") for c in r.json()["chunks"]]
    assert "session_log" not in cats
    assert r.json()["cache_hits"]["L2"] == 0

    # Opt back in (exclude_categories=[]) → the memory is reachable, but only
    # through the L3 disk-walk: VEC is empty and L2 is no longer a tier.
    r2 = client.post("/context", json={"prompt": "activity", "max_results": 5,
                                       "exclude_categories": []})
    assert r2.status_code == 200, r2.text
    assert "session_log" in [c.get("category") for c in r2.json()["chunks"]]
    hits = r2.json()["cache_hits"]
    assert hits["L3"] >= 1
    assert hits["L2"] == 0 and hits["L1"] == 0 and hits["HOT"] == 0 and hits["VEC"] == 0


def test_resolve_disk_truth_overrides_stale_chunk_category(tmp_path):
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "abc.json").write_text(json.dumps({
        "id": "abc", "category": "session_log", "source": "tool",
        "created_at": time.time(),
    }))
    # Chunk carries the stale category the migration left behind.
    chunk = ContextChunk("x", "l2-memory", 0.9, "L2",
                         memory_id="abc", category="topology", provenance_source=None)
    out = resolve_disk_truth(chunk, mem_dir)
    assert out.category == "session_log"   # disk wins
    assert out.provenance_source == "tool"

    # No memory_id (legacy entry), clean content → untouched, never raises.
    bare = ContextChunk("y", "l1-cache", 0.9, "L1")
    assert resolve_disk_truth(bare, mem_dir).category is None
    # No memory_id but auto-capture-shaped content → tagged session_log so the
    # default two-tier hiding applies to legacy cache entries too (v4.1).
    noisy = ContextChunk("[AUTO-CAPTURE] 3 tool calls: ...", "l2-memory", 0.9, "L2")
    assert resolve_disk_truth(noisy, mem_dir).category == "session_log"
    # memory_id with NO file on disk = deleted memory → dropped (v4.1).
    # The old no-op here is how purged [AUTO-CAPTURE] rows kept resurfacing
    # through L2 after the June-9 dedup sweep.
    ghost = ContextChunk("z", "l2-memory", 0.9, "L2", memory_id="nope")
    assert resolve_disk_truth(ghost, mem_dir) is None
