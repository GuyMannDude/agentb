"""v4.18.2: a caller-supplied writeback timestamp is the memory's birth.

Before the fix both the JSON record and the vec index stamped created_at=now,
so an imported or carried 30-day-old memory reported age_days≈0 and every
recency term treated it as newborn. Presence check: an old timestamp yields
an old age. Absence check: no timestamp still yields a newborn. Control: a
garbage timestamp is not an error, it is 'now'.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, ResilientProviderConfig, ProviderConfig,
    CacheConfig, ServerConfig, ClassificationConfig, DEFAULT_PERSONAS,
)

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
        cache=CacheConfig(), server=ServerConfig(host="127.0.0.1", port=50098),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=False),
        personas=dict(DEFAULT_PERSONAS),
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as c:
            yield c, tmp_path


def _writeback(client, summary, timestamp=None):
    body = {"session_id": "s-ts", "summary": summary, "key_facts": [],
            "category": "topology", "source": "user", "force": True}
    if timestamp is not None:
        body["timestamp"] = timestamp
    r = client.post("/writeback", json=body)
    assert r.status_code == 200, r.text
    return r.json()["memory_id"]


def _record(tmp_path, memory_id):
    matches = list(tmp_path.rglob(f"{memory_id}.json"))
    assert len(matches) == 1, matches
    return json.loads(matches[0].read_text())


def test_old_timestamp_yields_old_age(client):
    c, tmp = client
    thirty_days = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    mid = _writeback(c, "artforge alive in the workshop, thirty days back", thirty_days)
    rec = _record(tmp, mid)
    assert 29.9 < (time.time() - rec["created_at"]) / 86400 < 30.1
    r = c.post("/context", json={"prompt": "artforge", "max_results": 5})
    chunk = next(ch for ch in r.json()["chunks"] if ch["memory_id"] == mid)
    assert chunk["age_days"] is not None and 29.5 < chunk["age_days"] < 30.5


def test_no_timestamp_is_newborn(client):
    c, tmp = client
    mid = _writeback(c, "a memory born right now")
    assert (time.time() - _record(tmp, mid)["created_at"]) < 5


def test_garbage_timestamp_is_now_not_error(client):
    c, tmp = client
    mid = _writeback(c, "a memory with a broken clock", "not-a-date")
    assert (time.time() - _record(tmp, mid)["created_at"]) < 5


def test_z_suffix_and_naive_stamps_parse():
    from agentb.server import _epoch_from_iso
    z = _epoch_from_iso("2026-08-05T00:00:00Z")
    naive = _epoch_from_iso("2026-08-05T00:00:00")
    assert z == naive == datetime(2026, 8, 5, tzinfo=timezone.utc).timestamp()


# ── v4.18.3 ─────────────────────────────────────────────────────────────

def test_context_age_is_served_from_the_vec_tier(client):
    """The age assertion above must come from the vec index, not the JSON
    record on disk — otherwise a regressed vec column would pass unnoticed."""
    c, tmp = client
    stamp = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    mid = _writeback(c, "vec tier age probe", stamp)
    r = c.post("/context", json={"prompt": "vec tier age probe", "max_results": 5})
    chunk = next(ch for ch in r.json()["chunks"] if ch["memory_id"] == mid)
    assert chunk["cache_tier"] == "VEC", chunk["cache_tier"]
    assert 9.5 < chunk["age_days"] < 10.5


def test_same_stamp_different_content_are_two_memories(client):
    """Before 4.18.3 the id was session_id:ts — the second write silently
    replaced the first (200 OK, one file). Imports reuse one stamp constantly."""
    c, tmp = client
    stamp = "2026-08-05T00:00:00+00:00"
    a = _writeback(c, "first memory under a shared import stamp", stamp)
    b = _writeback(c, "second memory under the same import stamp", stamp)
    assert a != b
    assert len(list(tmp.rglob(f"{a}.json"))) == 1 and len(list(tmp.rglob(f"{b}.json"))) == 1


def test_identical_resend_is_idempotent(client):
    c, tmp = client
    stamp = "2026-08-05T00:00:00+00:00"
    a = _writeback(c, "an identical re-send", stamp)
    b = _writeback(c, "an identical re-send", stamp)
    assert a == b
    assert len(list(tmp.rglob(f"{a}.json"))) == 1


def test_future_stamp_is_clamped_to_now(client):
    c, tmp = client
    mid = _writeback(c, "a memory from the year 3000", "3000-01-01T00:00:00Z")
    rec = _record(tmp, mid)
    assert (time.time() - rec["created_at"]) < 5
    r = c.post("/context", json={"prompt": "year 3000", "max_results": 5})
    chunk = next(ch for ch in r.json()["chunks"] if ch["memory_id"] == mid)
    assert chunk["age_days"] >= 0


def test_fallback_is_visible_on_the_record(client):
    c, tmp = client
    mid = _writeback(c, "a memory with a broken clock, flagged", "1785888000")
    rec = _record(tmp, mid)
    assert rec.get("created_at_fallback") is True
    good = _writeback(c, "a memory with a good clock", "2026-08-05T00:00:00Z")
    assert "created_at_fallback" not in _record(tmp, good)


def test_ingested_at_is_arrival_not_birth(client):
    c, tmp = client
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    rec = _record(tmp, _writeback(c, "born a month ago, arrived now", old))
    assert (time.time() - rec["ingested_at"]) < 5
    assert rec["ingested_at"] - rec["created_at"] > 29 * 86400


def test_rulekeeper_scans_backdated_arrivals():
    """A 30-day-old memory that ARRIVED today is inside the 7-day window."""
    import json as _json
    from unittest.mock import MagicMock
    from agentb.rulekeeper import scan_tenant
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        mem_dir = Path(d) / "memory"; mem_dir.mkdir()
        now = time.time()
        for i, (created, ingested) in enumerate([(now - 30 * 86400, now), (now - 30 * 86400, None)]):
            rec = {"id": f"m{i}", "summary": "the same six token summary here", "key_facts": [],
                   "created_at": created}
            if ingested is not None:
                rec["ingested_at"] = ingested
            (mem_dir / f"m{i}.json").write_text(_json.dumps(rec))
        seen = []
        vec = MagicMock()
        vec.count.return_value = 2
        def _get(mid):
            seen.append(mid); return [1.0] * 768
        vec.get_embedding.side_effect = _get
        vec.search.return_value = []
        scan_tenant(mem_dir, vec, window_days=7)
        assert seen == ["m0"], seen   # arrived today → scanned; legacy old record → skipped
