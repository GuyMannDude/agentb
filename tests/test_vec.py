"""Tests for the sqlite-vec backed vector index (Mnemo v4 Phase 2)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Awaitable, Callable

import pytest

import httpx

from agentb.vec import (
    EMBED_DIM,
    MAX_EMBED_INPUT_CHARS,
    VecDimMismatch,
    VecStore,
    backfill,
    detect_mode,
    embed_with_adaptive_truncation,
    iter_memory_entries,
)


def _vec_along(axis: int, magnitude: float = 1.0) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[axis] = magnitude
    return v


def test_store_init_and_count(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    assert store.count() == 0
    assert (tmp_path / "vec.sqlite").exists()


def test_upsert_and_search_returns_nearest(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    store.upsert("m1", "hotdogs make me fart", _vec_along(0))
    store.upsert("m2", "completely different topic", _vec_along(1))
    store.upsert("m3", "another unrelated thing", _vec_along(2))

    hits = store.search(_vec_along(0), top_k=2)
    assert hits[0].memory_id == "m1"
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    assert len(hits) == 2


def test_upsert_replaces_existing(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    store.upsert("m1", "old text", _vec_along(0))
    store.upsert("m1", "new text", _vec_along(1))
    assert store.count() == 1
    hits = store.search(_vec_along(1), top_k=1)
    assert hits[0].text == "new text"


def test_dim_mismatch_rejected_loudly(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    with pytest.raises(VecDimMismatch):
        store.upsert("bad", "x", [0.1, 0.2, 0.3])
    with pytest.raises(VecDimMismatch):
        store.search([0.1, 0.2, 0.3])
    # Failed write must not leave a partial source row behind.
    assert store.count() == 0
    assert not store.has("bad")


def test_delete_removes_both_tables(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    store.upsert("m1", "text", _vec_along(0))
    assert store.has("m1")
    store.delete("m1")
    assert not store.has("m1")
    assert store.count() == 0


def test_missing_ids_returns_unindexed(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    store.upsert("m1", "t", _vec_along(0))
    missing = store.missing_ids(["m1", "m2", "m3"])
    assert set(missing) == {"m2", "m3"}


def test_detect_mode_clean_when_no_json(tmp_path: Path):
    (tmp_path / "memory").mkdir()
    assert detect_mode(tmp_path / "memory") == "clean"


def test_detect_mode_migration_when_json_present(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "abc.json").write_text("{}")
    assert detect_mode(mem) == "migration"


def test_detect_mode_clean_when_dir_missing(tmp_path: Path):
    assert detect_mode(tmp_path / "no-such-dir") == "clean"


def test_iter_memory_entries_uses_summary_plus_key_facts(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "a.json").write_text(json.dumps({
        "id": "a",
        "summary": "core summary",
        "key_facts": ["fact one", "fact two"],
        "created_at": 123.0,
    }))
    entries = list(iter_memory_entries(mem))
    assert len(entries) == 1
    mid, text, path, created_at = entries[0]
    assert mid == "a"
    assert text == "core summary\nfact one\nfact two"
    assert created_at == 123.0


def test_iter_memory_entries_skips_empty(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "empty.json").write_text(json.dumps({"id": "e", "summary": "", "key_facts": []}))
    (mem / "good.json").write_text(json.dumps({"id": "g", "summary": "real"}))
    entries = list(iter_memory_entries(mem))
    assert [e[0] for e in entries] == ["g"]


def test_iter_memory_entries_truncates_oversize(tmp_path: Path, caplog):
    """Oversize entries (e.g. wiki FILE INDEX batches) must NOT 400 the embedder
    and trip the circuit breaker. Truncation keeps the run alive."""
    mem = tmp_path / "memory"
    mem.mkdir()
    huge_summary = "x" * (MAX_EMBED_INPUT_CHARS + 5000)
    (mem / "huge.json").write_text(json.dumps({"id": "h", "summary": huge_summary}))
    with caplog.at_level("WARNING", logger="agentb.vec"):
        entries = list(iter_memory_entries(mem))
    assert len(entries) == 1
    _, text, _, _ = entries[0]
    assert len(text) == MAX_EMBED_INPUT_CHARS
    assert any("Truncating oversize memory h" in r.message for r in caplog.records)


def test_iter_memory_entries_tolerates_corrupt(tmp_path: Path):
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "broken.json").write_text("not json {")
    (mem / "good.json").write_text(json.dumps({"id": "g", "summary": "real"}))
    entries = list(iter_memory_entries(mem))
    assert [e[0] for e in entries] == ["g"]


def _make_embedder(axis_for: Callable[[str], int]) -> Callable[[str], Awaitable[list[float]]]:
    async def _embed(text: str) -> list[float]:
        return _vec_along(axis_for(text))
    return _embed


@pytest.mark.asyncio
async def test_backfill_embeds_each_entry_once(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    mem = tmp_path / "memory"
    mem.mkdir()
    for i in range(3):
        (mem / f"m{i}.json").write_text(json.dumps({
            "id": f"m{i}",
            "summary": f"entry {i}",
            "created_at": time.time(),
        }))

    embed = _make_embedder(lambda t: int(t.split()[-1]))
    stats = await backfill(store, mem, embed)
    assert stats["total"] == 3
    assert stats["embedded"] == 3
    assert stats["skipped"] == 0
    assert stats["failed"] == 0
    assert stats["truncated"] == 0
    assert store.count() == 3


@pytest.mark.asyncio
async def test_backfill_is_idempotent(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "a.json").write_text(json.dumps({"id": "a", "summary": "one"}))

    embed = _make_embedder(lambda t: 0)
    first = await backfill(store, mem, embed)
    second = await backfill(store, mem, embed)
    assert first["embedded"] == 1
    assert second["embedded"] == 0
    assert second["skipped"] == 1


@pytest.mark.asyncio
async def test_backfill_continues_past_failures(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "a.json").write_text(json.dumps({"id": "a", "summary": "one"}))
    (mem / "b.json").write_text(json.dumps({"id": "b", "summary": "two"}))

    async def flaky(text: str) -> list[float]:
        if "one" in text:
            raise RuntimeError("simulated embed failure")
        return _vec_along(0)

    stats = await backfill(store, mem, flaky)
    assert stats["embedded"] == 1
    assert stats["failed"] == 1
    assert store.count() == 1


def _http_400() -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://localhost/api/embed")
    resp = httpx.Response(400, request=req, text='{"error":"context length"}')
    return httpx.HTTPStatusError("400", request=req, response=resp)


@pytest.mark.asyncio
async def test_adaptive_truncation_halves_on_400():
    calls: list[int] = []

    async def embed(text: str) -> list[float]:
        calls.append(len(text))
        if len(text) > 1000:
            raise _http_400()
        return _vec_along(0)

    vec, used = await embed_with_adaptive_truncation(embed, "x" * 8000, min_chars=200)
    assert vec == _vec_along(0)
    assert len(used) <= 1000
    # 8000 -> 4000 -> 2000 -> 1000 -> succeeds
    assert calls == [8000, 4000, 2000, 1000]


@pytest.mark.asyncio
async def test_adaptive_truncation_gives_up_at_min_chars():
    async def embed(text: str) -> list[float]:
        raise _http_400()

    with pytest.raises(httpx.HTTPStatusError):
        await embed_with_adaptive_truncation(embed, "x" * 8000, min_chars=500)


@pytest.mark.asyncio
async def test_adaptive_truncation_propagates_non_400(tmp_path: Path):
    async def embed(text: str) -> list[float]:
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError, match="network down"):
        await embed_with_adaptive_truncation(embed, "hello world")


@pytest.mark.asyncio
async def test_backfill_counts_adaptive_truncations(tmp_path: Path):
    store = VecStore(tmp_path / "vec.sqlite")
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "huge.json").write_text(json.dumps({"id": "h", "summary": "x" * 2000}))
    (mem / "fine.json").write_text(json.dumps({"id": "f", "summary": "y" * 100}))

    async def embed(text: str) -> list[float]:
        # 400 on long "x" content; succeeds on short content or "y" content.
        if "x" in text and len(text) > 500:
            raise _http_400()
        return _vec_along(0)

    stats = await backfill(store, mem, embed)
    assert stats["embedded"] == 2
    assert stats["truncated"] == 1
    assert stats["failed"] == 0


def test_semantic_hit_where_keywords_miss(tmp_path: Path):
    """Canonical scenario from mnemo-v4-research.md Addition 1.

    With FTS5, the search 'hotdogs art' does NOT find 'hotdogs make me fart'.
    Vector similarity (over real embeddings) should — but in this test we
    simulate that by giving the related sentences close vectors and the
    unrelated sentences far ones. The store proves it returns the related
    memory by vector proximity even though keywords don't overlap.
    """
    store = VecStore(tmp_path / "vec.sqlite")

    near = [0.0] * EMBED_DIM
    near[:3] = [0.9, 0.1, 0.05]
    far1 = [0.0] * EMBED_DIM
    far1[:3] = [-0.9, 0.1, 0.05]
    far2 = [0.0] * EMBED_DIM
    far2[:3] = [0.05, -0.9, 0.1]

    store.upsert("m_related", "hotdogs make me fart", near)
    store.upsert("m_other_1", "completely unrelated phrase", far1)
    store.upsert("m_other_2", "another unrelated phrase", far2)

    query = [0.0] * EMBED_DIM
    query[:3] = [0.88, 0.12, 0.06]  # close to 'near', not overlapping any keywords
    hits = store.search(query, top_k=3)
    assert hits[0].memory_id == "m_related"
    assert hits[0].distance < hits[1].distance
