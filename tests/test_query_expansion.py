"""v4.2 — the Thesaurus Loop (query expansion on a whiff).

Covers the four moving parts:
  - merge_passes        — max-relevance-per-memory_id fusion (the crux); a single
                          pass is byte-identical to the pre-expansion handler.
  - should_expand       — escalation trigger (short query / zero / weak / strong).
  - expand_query        — the isolated Flash call: no-key no-op, timeout = [] (no
                          regression), parsing/cap, drop-original, LRU cache.
  - /context end-to-end — escalation fires on a whiff, and never fires for a batch
                          call, an expand=False call, or a strong first pass.
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, ResilientProviderConfig, ProviderConfig,
    CacheConfig, ServerConfig, ClassificationConfig, ExpansionConfig, DEFAULT_PERSONAS,
)
from agentb.cache import ContextChunk
from agentb.server import merge_passes, should_expand, top_relevance, expand_query, _EXPAND_CACHE


def _chunk(mid, rel, tier="VEC", content="c", source=None):
    return ContextChunk(content=content, source=source or f"memory:{mid}",
                        relevance=rel, cache_tier=tier, memory_id=mid)


# ── merge_passes: max-relevance fusion ──────────────────────────────────────

def test_merge_single_pass_is_identity():
    p = [_chunk("a", 0.9), _chunk("b", 0.5), _chunk("c", 0.7)]
    out = merge_passes([p])
    assert [c.memory_id for c in out] == ["a", "b", "c"]
    assert out[0] is p[0]  # same objects, untouched order


def test_merge_keeps_max_relevance_and_original_position():
    p1 = [_chunk("a", 0.30), _chunk("b", 0.60)]
    p2 = [_chunk("a", 0.80), _chunk("d", 0.40)]  # 'a' collides, stronger in p2
    out = merge_passes([p1, p2])
    assert [c.memory_id for c in out] == ["a", "b", "d"]  # 'a' holds first slot
    a = next(c for c in out if c.memory_id == "a")
    assert a.relevance == 0.80  # the better dart wins (NOT first-wins)


def test_merge_lower_relevance_does_not_replace():
    out = merge_passes([[_chunk("a", 0.90)], [_chunk("a", 0.10)]])
    assert len(out) == 1 and out[0].relevance == 0.90


def test_merge_hot_chunks_dedup_by_source_and_content():
    h1 = ContextChunk("same", "hot-session:1", 0.75, "HOT")
    h2 = ContextChunk("same", "hot-session:1", 0.75, "HOT")  # dup of h1
    h3 = ContextChunk("diff", "hot-session:1", 0.75, "HOT")  # distinct content
    out = merge_passes([[h1], [h2, h3]])
    assert len(out) == 2


# ── should_expand / top_relevance: the escalation trigger ───────────────────

def test_top_relevance_empty_is_zero():
    assert top_relevance([]) == 0.0


def test_short_query_never_expands_even_with_zero_results():
    # 2 words < min_query_words(3): a likely single-entity lookup, skip it.
    assert should_expand("solr index", [], ExpansionConfig()) is False


def test_zero_results_expands():
    assert should_expand("how do we deploy", [], ExpansionConfig()) is True


def test_weak_top_relevance_expands():
    cfg = ExpansionConfig(relevance_floor=0.5)
    assert should_expand("how do we deploy", [_chunk("a", 0.3)], cfg) is True


def test_strong_top_relevance_does_not_expand():
    cfg = ExpansionConfig(relevance_floor=0.5)
    assert should_expand("how do we deploy", [_chunk("a", 0.8)], cfg) is False


# ── expand_query: the isolated Flash call ───────────────────────────────────

class _FakeResp:
    def __init__(self, content):
        self._c = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


class _FakeClient:
    """Stands in for httpx.AsyncClient as an async context manager."""
    def __init__(self, content=None, exc=None):
        self._content, self._exc = content, exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, *a, **k):
        if self._exc:
            raise self._exc
        return _FakeResp(self._content)


def test_no_api_key_is_a_silent_noop():
    out = asyncio.run(expand_query("a long enough query", ExpansionConfig(), "", "base"))
    assert out == []


def test_happy_path_strips_numbering_and_caps():
    _EXPAND_CACHE.clear()
    cfg = ExpansionConfig(max_variants=2)
    content = "1. first rephrasing here\n- second rephrasing\nthird rephrasing\nfourth one"
    with patch("agentb.server.httpx.AsyncClient", lambda *a, **k: _FakeClient(content=content)):
        out = asyncio.run(expand_query("original query text", cfg, "key", "base"))
    assert out == ["first rephrasing here", "second rephrasing"]  # numbering/bullets stripped, capped to 2


def test_timeout_returns_empty_no_regression():
    _EXPAND_CACHE.clear()
    with patch("agentb.server.httpx.AsyncClient",
               lambda *a, **k: _FakeClient(exc=httpx.TimeoutException("slow"))):
        out = asyncio.run(expand_query("original query text", ExpansionConfig(), "key", "base"))
    assert out == []  # graceful → handler behaves exactly like today


def test_null_content_does_not_crash():
    # A refused / tool-only completion returns content: null. The parse must not
    # AttributeError past the guard (that would 500 the live recall path).
    _EXPAND_CACHE.clear()
    with patch("agentb.server.httpx.AsyncClient", lambda *a, **k: _FakeClient(content=None)):
        out = asyncio.run(expand_query("a long enough query", ExpansionConfig(), "key", "base"))
    assert out == []


def test_digit_leading_phrasing_is_preserved():
    # Char-set lstrip would eat the leading digits ("3D printing" -> "D printing").
    # Only genuine list markers (1. / - / *) should be stripped.
    _EXPAND_CACHE.clear()
    cfg = ExpansionConfig(max_variants=4)
    content = "1. 3D printing workflow\n- 2024 roadmap notes"
    with patch("agentb.server.httpx.AsyncClient", lambda *a, **k: _FakeClient(content=content)):
        out = asyncio.run(expand_query("original query", cfg, "key", "base"))
    assert out == ["3D printing workflow", "2024 roadmap notes"]


def test_original_phrasing_is_dropped():
    _EXPAND_CACHE.clear()
    cfg = ExpansionConfig(max_variants=4)
    content = "original query\na different phrasing"
    with patch("agentb.server.httpx.AsyncClient", lambda *a, **k: _FakeClient(content=content)):
        out = asyncio.run(expand_query("original query", cfg, "key", "base"))
    assert out == ["a different phrasing"]


def test_nonempty_result_is_cached():
    _EXPAND_CACHE.clear()
    cfg = ExpansionConfig(max_variants=3)
    calls = {"n": 0}

    def factory(*a, **k):
        calls["n"] += 1
        return _FakeClient(content="alpha rephrase\nbeta rephrase")

    with patch("agentb.server.httpx.AsyncClient", factory):
        out1 = asyncio.run(expand_query("same query here", cfg, "key", "base"))
        out2 = asyncio.run(expand_query("same query here", cfg, "key", "base"))
    assert out1 == out2 == ["alpha rephrase", "beta rephrase"]
    assert calls["n"] == 1  # second call served from the LRU


def test_failure_is_not_cached():
    _EXPAND_CACHE.clear()
    cfg = ExpansionConfig()
    with patch("agentb.server.httpx.AsyncClient",
               lambda *a, **k: _FakeClient(exc=httpx.ConnectError("down"))):
        asyncio.run(expand_query("a long enough query", cfg, "key", "base"))
    assert len(_EXPAND_CACHE) == 0  # transient failure must not poison the cache


# ── /context end-to-end: escalation wiring ──────────────────────────────────

_STATUS = {"primary": "fake", "active": "fake", "failed_over": False,
           "circuit_open": False, "primary_retry_in": None, "fallback_count": 0}
VEC = [0.0] * 768
VEC[0] = 1.0


class _RecordingEmbedding:
    active_label = "fake/embed"

    def __init__(self):
        self.embedded: list[str] = []

    @property
    def status(self):
        return _STATUS

    async def embed(self, text, *, use_breaker=True):
        self.embedded.append(text)
        return list(VEC)

    async def health_check(self):
        return True


class _FakeReasoning:
    active_label = "fake/reason"

    @property
    def status(self):
        return _STATUS

    async def generate(self, prompt, system="", max_tokens=2048, *, use_breaker=True):
        return "topology"

    async def health_check(self):
        return True


def _make_client(tmp_path, embedder):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic")),
        cache=CacheConfig(), server=ServerConfig(port=50097),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=False),
        expansion=ExpansionConfig(),  # default-ON
        personas=dict(DEFAULT_PERSONAS),
    )
    patch_embed = patch("agentb.server.create_resilient_embedding", return_value=embedder)
    patch_reason = patch("agentb.server.create_resilient_reasoning", return_value=_FakeReasoning())
    patch_embed.start()
    patch_reason.start()
    from agentb.server import create_app
    client = TestClient(create_app(cfg))
    client.__enter__()
    return client, (patch_embed, patch_reason)


def _seed_memory(tmp_path, memory_id, summary, category="doctrine"):
    """Seed one memory into VEC's store + its disk JSON so a normal recall finds
    it at high relevance (FakeEmbedding maps everything to the same vector)."""
    base = tmp_path / "agents" / "default"
    mem_dir = base / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    (mem_dir / f"{memory_id}.json").write_text(json.dumps({
        "id": memory_id, "summary": summary, "key_facts": [],
        "category": category, "source": "tool", "created_at": now,
    }))


def test_escalation_fires_on_empty_store():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        tmp = Path(td)
        emb = _RecordingEmbedding()
        client, patches = _make_client(tmp, emb)
        try:
            mock_expand = AsyncMock(return_value=["an alternate phrasing of it"])
            with patch("agentb.server.expand_query", mock_expand):
                r = client.post("/context", json={"prompt": "how do we deploy this", "max_results": 5})
            assert r.status_code == 200, r.text
            mock_expand.assert_awaited_once()                 # whiff → escalated
            assert "an alternate phrasing of it" in emb.embedded  # variant got searched
        finally:
            client.__exit__(None, None, None)
            for p in patches:
                p.stop()


def test_batch_call_never_expands():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        client, patches = _make_client(tmp, _RecordingEmbedding())
        try:
            mock_expand = AsyncMock(return_value=["x"])
            with patch("agentb.server.expand_query", mock_expand):
                r = client.post("/context", json={"prompt": "how do we deploy this",
                                                  "max_results": 5, "batch": True})
            assert r.status_code == 200, r.text
            mock_expand.assert_not_awaited()  # batch is live-path-exempt
        finally:
            client.__exit__(None, None, None)
            for p in patches:
                p.stop()


def test_expand_false_never_expands():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        client, patches = _make_client(tmp, _RecordingEmbedding())
        try:
            mock_expand = AsyncMock(return_value=["x"])
            with patch("agentb.server.expand_query", mock_expand):
                r = client.post("/context", json={"prompt": "how do we deploy this",
                                                  "max_results": 5, "expand": False})
            assert r.status_code == 200, r.text
            mock_expand.assert_not_awaited()
        finally:
            client.__exit__(None, None, None)
            for p in patches:
                p.stop()


def test_strong_first_pass_skips_expansion():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _seed_memory(tmp, "m-strong", "deployment runbook for the service")
        client, patches = _make_client(tmp, _RecordingEmbedding())
        try:
            mock_expand = AsyncMock(return_value=["x"])
            with patch("agentb.server.expand_query", mock_expand):
                r = client.post("/context", json={"prompt": "how do we deploy this", "max_results": 5})
            assert r.status_code == 200, r.text
            # A strong VEC hit (relevance ~1.0 ≥ floor) means no whiff → no expansion.
            assert r.json()["total_found"] >= 1
            mock_expand.assert_not_awaited()
        finally:
            client.__exit__(None, None, None)
            for p in patches:
                p.stop()
