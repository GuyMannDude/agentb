"""v4.8 recall mode=explore — the serendipity lens.

Contract: explore prefers the similarity band ADJACENT to the pool's top hit
(one step sideways, not the bullseye), ignores recency entirely, favors
rarely-recalled memories, hard-zeroes the noise band, and works even with
composite ranking disabled. Focus mode must be byte-for-byte unchanged.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, CacheConfig, ClassificationConfig, ProviderConfig,
    RankingConfig, ResilientProviderConfig, ServerConfig, DEFAULT_PERSONAS,
)
from agentb.ranking import EXPLORE_FLOOR, EXPLORE_OFFSET, explore_score

from tests.test_ranking import (  # reuse the endpoint harness
    FakeEmbedding, FakeReasoning, VEC_A, _seed_vec_memory,
)


# ── unit: explore_score geometry ────────────────────────────────────────────

def test_adjacent_beats_bullseye():
    top = 1.0  # E4: the lens reads pool-normalised similarity; the best hit is 1.0
    at_target = explore_score(similarity=top - EXPLORE_OFFSET, top_similarity=top,
                              category=None, access_count=0)
    at_top = explore_score(similarity=top, top_similarity=top,
                           category=None, access_count=0)
    assert at_target > at_top, "one step sideways must out-score the exact match"


def test_noise_band_is_hard_zero():
    top = 1.0
    s = explore_score(similarity=top - EXPLORE_FLOOR - 0.01, top_similarity=top,
                      category="idea", access_count=0)
    assert s == 0.0, "below the floor is noise, not serendipity"


def test_beyond_the_pool_span_is_noise_whatever_the_floor(monkeypatch):
    # E4: a hit a full SIMILARITY_SPAN below the pool's best normalises to
    # 0.0 — off-topic by the pool's own geometry, so it is zeroed even if a
    # future floor were loosened past 1.0. Loosen it here, or the floor cut
    # zeroes the case first and the guard is never reached (review, E4).
    from agentb import ranking
    monkeypatch.setattr(ranking, "EXPLORE_FLOOR", 1.5)
    s = explore_score(similarity=0.0, top_similarity=1.0, category="idea", access_count=0)
    assert s == 0.0


def test_novelty_prefers_rarely_recalled():
    fresh = explore_score(similarity=0.95, top_similarity=1.0,
                          category="idea", access_count=0)
    worn = explore_score(similarity=0.95, top_similarity=1.0,
                         category="idea", access_count=50)
    assert fresh > worn


def test_idea_outranks_session_log_in_explore():
    idea = explore_score(similarity=0.95, top_similarity=1.0,
                         category="idea", access_count=0)
    log = explore_score(similarity=0.95, top_similarity=1.0,
                        category="session_log", access_count=0)
    assert idea > log


def test_explore_takes_no_recency_input():
    # The lens ignores age BY CONSTRUCTION — the signature has no age param.
    import inspect
    assert "age_days" not in inspect.signature(explore_score).parameters


# ── endpoint: mode plumbing ────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic-embed-text")),
        cache=CacheConfig(), server=ServerConfig(host="127.0.0.1", port=50098),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=False),
        # ranking DISABLED on purpose: explore must still work (no silent no-op)
        ranking=RankingConfig(enabled=False),
        personas=dict(DEFAULT_PERSONAS),
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as c:
            yield c


def test_explore_mode_reorders_and_focus_is_default(tmp_path, client):
    # bullseye: exact query vector (cosine 1.0). adjacent: a small step
    # sideways (L2 0.1 → cosine 0.995 → 0.975 on the pool-normalised scale,
    # inside the explore band and closer to its target than the bullseye).
    # far: L2 0.7 → cosine 0.755, a full SIMILARITY_SPAN below the top — the
    # noise band. All same category so ordering is pure lens geometry.
    adjacent = list(VEC_A); adjacent[1] = 0.1
    far = list(VEC_A); far[1] = 0.7
    _seed_vec_memory(tmp_path, "bullseye", "the exact thing you asked about",
                     "decision", distance_vec=list(VEC_A))
    _seed_vec_memory(tmp_path, "adjacent", "the thing this reminds you of",
                     "decision", distance_vec=adjacent)
    _seed_vec_memory(tmp_path, "far", "an unrelated memory", "decision",
                     distance_vec=far)

    focus = client.post("/context", json={"prompt": "the exact thing", "max_results": 3})
    assert focus.status_code == 200, focus.text
    focus_ids = [c["memory_id"] for c in focus.json()["chunks"]]
    assert focus_ids and focus_ids[0] == "bullseye", "default mode must stay best-match-first"

    explore = client.post("/context", json={"prompt": "the exact thing",
                                            "max_results": 3, "mode": "explore"})
    assert explore.status_code == 200, explore.text
    explore_ids = [c["memory_id"] for c in explore.json()["chunks"]]
    assert explore_ids, "explore must return results even with ranking disabled"
    assert explore_ids[0] == "adjacent", "explore must surface the adjacent memory first"
    assert "far" not in explore_ids, "the noise band must not pad explore results"


def test_invalid_mode_is_rejected():
    from agentb.server import ContextRequest
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ContextRequest(prompt="x", mode="wander")


# ── v4.17: the persona picks the lens when the caller names none ──────────

def _seed_lens_world(tmp_path):
    adjacent = list(VEC_A); adjacent[1] = 0.1
    _seed_vec_memory(tmp_path, "bullseye", "the exact thing you asked about",
                     "decision", distance_vec=list(VEC_A))
    _seed_vec_memory(tmp_path, "adjacent", "the thing this reminds you of",
                     "decision", distance_vec=adjacent)


def test_creative_persona_defaults_to_explore(tmp_path, client):
    _seed_lens_world(tmp_path)
    r = client.post("/context", json={"prompt": "the exact thing", "max_results": 2,
                                      "persona": "creative"})
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "explore"
    assert [c["memory_id"] for c in r.json()["chunks"]][0] == "adjacent"


def test_strict_and_default_personas_default_to_focus(tmp_path, client):
    _seed_lens_world(tmp_path)
    for persona in ("strict", "default", None):
        body = {"prompt": "the exact thing", "max_results": 2}
        if persona:
            body["persona"] = persona
        r = client.post("/context", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["mode"] == "focus", persona
        assert [c["memory_id"] for c in r.json()["chunks"]][0] == "bullseye", persona


def test_explicit_mode_beats_persona(tmp_path, client):
    _seed_lens_world(tmp_path)
    r = client.post("/context", json={"prompt": "the exact thing", "max_results": 2,
                                      "persona": "creative", "mode": "focus"})
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "focus"
    assert [c["memory_id"] for c in r.json()["chunks"]][0] == "bullseye"


def test_default_persona_config_is_the_server_wide_switch():
    from agentb.config import _parse_config, get_persona, persona_recall_mode
    cfg = _parse_config({"default_persona": "creative"})
    assert get_persona(cfg).name == "creative"
    assert persona_recall_mode(get_persona(cfg)) == "explore"
    assert persona_recall_mode(get_persona(_parse_config({}))) == "focus"
    with pytest.raises(ValueError, match="default_persona"):
        _parse_config({"default_persona": "artist"})


def test_named_agent_without_persona_follows_the_switch():
    # Review finding (S297): an agents: entry used to be pinned to the literal
    # "default" persona, which masked default_persona for every named tenant.
    from agentb.config import _parse_config, get_persona
    cfg = _parse_config({"default_persona": "creative",
                         "agents": {"cc": {"data_dir": "~/x"},
                                    "biz": {"data_dir": "~/y", "persona": "strict"}}})
    assert get_persona(cfg, None, "cc").name == "creative"      # follows the switch
    assert get_persona(cfg, None, "biz").name == "strict"       # pinned itself
    assert get_persona(cfg, "default", "cc").name == "default"  # explicit call wins


def test_health_reports_the_configured_default_persona(tmp_path):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic-embed-text")),
        cache=CacheConfig(), server=ServerConfig(host="127.0.0.1", port=50098),
        data_dir=str(tmp_path), classification=ClassificationConfig(enabled=False),
        personas=dict(DEFAULT_PERSONAS), default_persona="creative",
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as c:
            assert c.get("/health").json()["default_persona"] == "creative"
            r = c.post("/context", json={"prompt": "anything", "max_results": 2})
            assert r.status_code == 200, r.text
            assert r.json()["mode"] == "explore"


def test_chatgpt_gate_forwards_mode_only_when_given():
    # Review finding (S297): the gate defaulted mode to "focus", so every
    # forwarded recall carried an explicit lens and the persona never applied.
    import importlib.util
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "integrations" / "chatgpt" / "server.py"
    spec = importlib.util.spec_from_file_location("chatgpt_gate_for_lens_test", src)
    gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
    RecallRequest = gate.RecallRequest
    omitted = RecallRequest(prompt="x").model_dump(exclude={"agent_id"}, exclude_none=True)
    assert "mode" not in omitted
    given = RecallRequest(prompt="x", mode="explore").model_dump(exclude={"agent_id"}, exclude_none=True)
    assert given["mode"] == "explore"
