"""H4/H5 embedding-integrity regression tests (clean-room review).

H4: a malformed-but-200 provider response used to become an empty [] vector;
the first one locked the resilient wrapper's dim to 0 and every later valid
vector was rejected until restart. Providers now raise on empty, and an empty
vector can never lock or pass the dim check.

H5: a same-dimension fallback from a DIFFERENT model embeds in a different
vector space; storing it beside primary-space vectors makes those memories
silent unrecallable noise. The store path (task_type="document") now refuses
foreign-model fallbacks; the query path still serves them (degraded recall
during an outage, nothing durable written).
"""
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from agentb.config import ProviderConfig, ResilientProviderConfig
from agentb.providers import (
    EmbeddingRefused, GoogleEmbedding, OllamaEmbedding, create_resilient_embedding,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, *args, **kwargs):
        return _FakeResp(self._payload)


def _resilient(fallback_model=None, fallback_provider="google"):
    fallbacks = []
    if fallback_model:
        fallbacks = [ProviderConfig(provider=fallback_provider, model=fallback_model,
                                    api_key="k", api_base="http://localhost:11434")]
    return create_resilient_embedding(ResilientProviderConfig(
        primary=ProviderConfig(provider="ollama", model="nomic-embed-text",
                               api_base="http://localhost:11434"),
        fallbacks=fallbacks,
    ))


# ── H4: empty embeddings raise at the provider ──

@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"embeddings": []}, {"embeddings": [[]]}])
async def test_ollama_empty_embedding_raises(monkeypatch, payload):
    provider = OllamaEmbedding(ProviderConfig(provider="ollama", model="nomic-embed-text",
                                              api_base="http://localhost:11434"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeClient(payload))
    with pytest.raises(RuntimeError):
        await provider.embed("text")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{}, {"embedding": {}}, {"embedding": {"values": []}}])
async def test_google_empty_embedding_raises(monkeypatch, payload):
    provider = GoogleEmbedding(ProviderConfig(provider="google", model="gemini-embedding-001",
                                              api_key="k"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: _FakeClient(payload))
    with pytest.raises(RuntimeError):
        await provider.embed("text")


# ── H4: an empty vector can never lock the dim / brick the pipeline ──

@pytest.mark.asyncio
async def test_empty_primary_vector_does_not_lock_dim_zero(monkeypatch):
    resilient = _resilient()
    resilient.primary.embed = AsyncMock(return_value=[])  # a provider that slipped through

    with patch("agentb.providers._alerter.scream", new=AsyncMock()):
        with pytest.raises(EmbeddingRefused):
            await resilient.embed("boom")
    assert resilient._locked_dim is None  # the old code locked 0 here

    # Provider recovers → embeds work again WITHOUT a process restart.
    resilient.primary.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    assert await resilient.embed("ok") == [0.1, 0.2, 0.3]
    assert resilient._locked_dim == 3


# ── H5: store path refuses foreign-model fallbacks; query path serves ──

@pytest.mark.asyncio
async def test_store_path_refuses_foreign_model_fallback(monkeypatch):
    """The live config: nomic primary + Gemini fallback truncated to the same
    dim. Same dimension, different space — must never be WRITTEN."""
    resilient = _resilient(fallback_model="gemini-embedding-001")
    resilient.primary.embed = AsyncMock(return_value=[1.0, 2.0, 3.0])
    assert await resilient.embed("lock the dim") == [1.0, 2.0, 3.0]

    resilient.primary.embed = AsyncMock(side_effect=Exception("ollama down"))
    resilient.fallbacks[0].embed = AsyncMock(return_value=[9.0, 9.0, 9.0])  # dim matches!

    with patch("agentb.providers._alerter.scream", new=AsyncMock()) as scream:
        with pytest.raises(EmbeddingRefused):
            await resilient.embed("must not store", task_type="document")
        scream.assert_awaited_once()
    assert resilient.fallbacks[0].embed.await_count == 0  # never even attempted


@pytest.mark.asyncio
async def test_query_path_still_serves_foreign_model_fallback():
    resilient = _resilient(fallback_model="gemini-embedding-001")
    resilient.primary.embed = AsyncMock(return_value=[1.0, 2.0, 3.0])
    await resilient.embed("lock the dim")

    resilient.primary.embed = AsyncMock(side_effect=Exception("ollama down"))
    resilient.fallbacks[0].embed = AsyncMock(return_value=[9.0, 9.0, 9.0])

    result = await resilient.embed("recall query", task_type="query")
    assert result == [9.0, 9.0, 9.0]  # degraded recall beats hard failure


@pytest.mark.asyncio
async def test_store_path_serves_same_model_fallback():
    """A mirror of the primary (same model, different host) IS the same
    embedding space — store-path fallback to it stays allowed."""
    resilient = _resilient(fallback_model="nomic-embed-text", fallback_provider="ollama")
    resilient.primary.embed = AsyncMock(return_value=[1.0, 2.0, 3.0])
    await resilient.embed("lock the dim")

    resilient.primary.embed = AsyncMock(side_effect=Exception("ollama down"))
    resilient.fallbacks[0].embed = AsyncMock(return_value=[4.0, 5.0, 6.0])

    result = await resilient.embed("store me", task_type="document")
    assert result == [4.0, 5.0, 6.0]
    assert resilient.failed_over


@pytest.mark.asyncio
async def test_store_path_mirror_with_latest_tag_serves():
    """Ollama treats "m" and "m:latest" as the same model — a mirror tagged
    :latest must not be refused on the store path."""
    resilient = _resilient(fallback_model="nomic-embed-text:latest",
                           fallback_provider="ollama")
    resilient.primary.embed = AsyncMock(return_value=[1.0, 2.0, 3.0])
    await resilient.embed("lock the dim")

    resilient.primary.embed = AsyncMock(side_effect=Exception("ollama down"))
    resilient.fallbacks[0].embed = AsyncMock(return_value=[4.0, 5.0, 6.0])

    assert await resilient.embed("store me", task_type="document") == [4.0, 5.0, 6.0]


@pytest.mark.asyncio
async def test_store_path_refuses_different_explicit_tag():
    """Two different explicit tags may be different weights → different space.
    Only the implicit :latest is normalized; anything else stays distinct."""
    resilient = _resilient(fallback_model="nomic-embed-text:v2",
                           fallback_provider="ollama")
    resilient.primary.embed = AsyncMock(return_value=[1.0, 2.0, 3.0])
    await resilient.embed("lock the dim")

    resilient.primary.embed = AsyncMock(side_effect=Exception("ollama down"))
    resilient.fallbacks[0].embed = AsyncMock(return_value=[4.0, 5.0, 6.0])

    with patch("agentb.providers._alerter.scream", new=AsyncMock()):
        with pytest.raises(EmbeddingRefused):
            await resilient.embed("store me", task_type="document")
