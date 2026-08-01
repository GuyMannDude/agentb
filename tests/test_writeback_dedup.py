"""Write-time near-duplicate contract: bounded, explainable, lane-local."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, AgentConfig, CacheConfig, ClassificationConfig,
    DEFAULT_PERSONAS, ProviderConfig, ResilientProviderConfig, ServerConfig,
)
from agentb.rulekeeper import format_advisory, scan_tenant
from agentb.vec import VecStore

VEC = [0.0] * 768
VEC[0] = 1.0
STATUS = {"primary": "fake", "active": "fake", "failed_over": False,
          "circuit_open": False, "primary_retry_in": None, "fallback_count": 0}


class FakeEmbedding:
    active_label = "fake/embed"
    status = STATUS
    async def embed(self, text, *, use_breaker=True, task_type="document"):
        return list(VEC)
    async def health_check(self): return True


class FakeReasoning:
    active_label = "fake/reason"
    status = STATUS
    async def generate(self, *args, **kwargs): return "doctrine"
    async def health_check(self): return True


@pytest.fixture
def world(tmp_path):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama")),
        cache=CacheConfig(), classification=ClassificationConfig(enabled=False),
        server=ServerConfig(host="127.0.0.1", port=50098),
        data_dir=str(tmp_path / "default"), personas=dict(DEFAULT_PERSONAS),
        agents={
            "cody": AgentConfig(data_dir=str(tmp_path / "cody")),
            "cc": AgentConfig(data_dir=str(tmp_path / "cc")),
        },
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as client:
            yield client, tmp_path


TEXT = "Bus receipts to Guy must be the final line with recipient and ping number"


def save(client, session, *, agent="cody", text=TEXT, **extra):
    body = {"session_id": session, "summary": text, "agent_id": agent,
            "category": "doctrine", "source": "inferred", **extra}
    return client.post("/writeback", json=body)


def test_interactive_duplicate_is_held_with_scores(world):
    client, _ = world
    first = save(client, "first").json()
    held = save(client, "second").json()
    assert first["status"] == "archived"
    assert held["status"] == "held"
    assert held["memory_id"] == ""
    assert held["near_duplicates"][0]["id"] == first["memory_id"]
    assert held["near_duplicates"][0]["cosine"] >= 0.80
    assert held["near_duplicates"][0]["overlap"] >= 0.55
    assert "receipt" in held["near_duplicates"][0]["shared_tokens"]


def test_force_inserts_and_records_near_dup(world):
    client, root = world
    first = save(client, "first").json()
    forced = save(client, "forced", force=True).json()
    assert forced["status"] == "archived"
    stored = json.loads((root / "cody" / "memory" / f"{forced['memory_id']}.json").read_text())
    assert stored["near_dup_of"] == [first["memory_id"]]


def test_supersede_keeps_history_but_demotes_old(world):
    client, root = world
    first = save(client, "first").json()
    newer = save(client, "newer", supersedes=[first["memory_id"]]).json()
    old_path = root / "cody" / "memory" / f"{first['memory_id']}.json"
    assert old_path.exists()
    assert json.loads(old_path.read_text())["superseded_by"] == newer["memory_id"]
    recalled = client.post("/context", json={"prompt": TEXT, "agent_id": "cody",
                                               "exclude_categories": []}).json()
    assert first["memory_id"] not in {c.get("memory_id") for c in recalled["chunks"]}


def test_auto_capture_never_blocks(world):
    client, root = world
    first = save(client, "first").json()
    auto = save(client, "auto", category="session_log").json()
    assert auto["status"] == "archived"
    stored = json.loads((root / "cody" / "memory" / f"{auto['memory_id']}.json").read_text())
    assert stored["near_dup_of"] == [first["memory_id"]]


def test_short_text_never_holds(world):
    client, _ = world
    save(client, "short-one", text="alpha beta gamma")
    assert save(client, "short-two", text="alpha beta gamma").json()["status"] == "archived"


def test_cross_lane_isolation(world):
    client, _ = world
    save(client, "cc-first", agent="cc")
    assert save(client, "cody-first", agent="cody").json()["status"] == "archived"


def test_allowlist_pair_proceeds_clean(world):
    client, root = world
    first = save(client, "first").json()
    (root / "cody" / "dedup-allow.txt").write_text(
        f"allowed|{first['memory_id']}\n", encoding="utf-8")
    allowed = save(client, "allowed").json()
    assert allowed["status"] == "archived"
    assert allowed["near_duplicates"] == []


def test_rulekeeper_blind_store_is_loud(tmp_path):
    store = VecStore(tmp_path / "vec_index.sqlite")
    try:
        result = scan_tenant(tmp_path / "memory", store)
    finally:
        store.close()
    assert result["exit_code"] == 2
    assert "BLIND STORE" in format_advisory("cody", result)


def test_rulekeeper_reports_without_mutating(world):
    client, root = world
    first = save(client, "first").json()
    forced = save(client, "forced", force=True).json()
    memory_dir = root / "cody" / "memory"
    before = {p.name: p.read_bytes() for p in memory_dir.glob("*.json")}
    store = VecStore(root / "cody" / "vec_index.sqlite")
    try:
        result = scan_tenant(memory_dir, store)
    finally:
        store.close()
    after = {p.name: p.read_bytes() for p in memory_dir.glob("*.json")}
    assert result["exit_code"] == 1
    assert result["pairs"][0]["new_id"] in {first["memory_id"], forced["memory_id"]}
    assert before == after
    assert "nothing merged or demoted" in format_advisory("cody", result)
