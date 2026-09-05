"""v4.18.2: a caller-supplied writeback timestamp is the memory's birth.

Before the fix both the JSON record and the vec index stamped created_at=now,
so an imported or carried 30-day-old memory reported age_days≈0 and every
recency term treated it as newborn. Presence check: an old timestamp yields
an old age. Absence check: no timestamp still yields a newborn. Control: a
garbage timestamp is not an error, it is 'now'.
"""
from __future__ import annotations

import json
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
