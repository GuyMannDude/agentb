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


def test_beyond_the_pool_span_is_noise_whatever_the_floor():
    # E4: a hit a full SIMILARITY_SPAN below the pool's best normalises to
    # 0.0 — off-topic by the pool's own geometry, so it is zeroed even if a
    # future floor were loosened past 1.0.
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
