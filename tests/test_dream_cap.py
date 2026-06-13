"""Dreamer overflow guards — the per-agent section cap + adaptive-halving backstop.

These are the two defenses v4.2.1 added so a single high-volume agent (opie's
auto-capture once hit ~19MB / ~4.9M tokens) can't 400 the whole nightly run.
They are *insurance*: in normal nightly operation each window is ~24h and stays
well under the cap, so the cap never fires — which is exactly why it needs a
test rather than waiting for the next stuck-window incident to exercise it.

  - _build_agent_section: bounds one agent's brief to MAX_AGENT_SECTION_CHARS,
    recency-first (drop oldest), announcing the drop (never silent truncation).
  - _call_openrouter_adaptive: belt-and-suspenders for token-density spikes —
    halve the input and retry on a context-length 400 (incl. the provider-side
    400 OpenRouter wraps in a 200), keeping the most-recent tail.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The dreamer is a top-level script with a hyphen in its name — load it by path.
_DREAM_PATH = Path(__file__).resolve().parent.parent / "mnemo-dream.py"
_spec = importlib.util.spec_from_file_location("mnemo_dream", _DREAM_PATH)
dream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dream)


def _mem(i: int, summary: str) -> dict:
    """One AgentB-shaped memory entry, timestamp increasing with i (newest = highest i)."""
    return {
        "timestamp": f"2026-06-{i + 1:02d}T03:00:00+00:00",
        "session_id": f"sess-{i}",
        "summary": summary,
        "key_facts": [],
    }


# ── _build_agent_section: the cap ──

def test_section_caps_oversized_input(monkeypatch):
    """10 entries × ~2KB each = ~20KB; a 5KB cap must bound the section."""
    monkeypatch.setattr(dream, "MAX_AGENT_SECTION_CHARS", 5_000)
    mems = [_mem(i, f"<<E{i}>>" + "z" * 2_000) for i in range(10)]

    section = dream._build_agent_section("opie", mems)

    # Body (everything the cap governs) stays within budget; header + newlines
    # are the only slack, and they're tiny.
    assert len(section) <= 5_000 + 500, f"section not capped: {len(section):,} chars"
    assert "omitted to fit" in section, "the drop must be announced, never silent"


def test_section_keeps_most_recent_drops_oldest(monkeypatch):
    """Recency-first: 'since last dream' cares about the newest entries."""
    monkeypatch.setattr(dream, "MAX_AGENT_SECTION_CHARS", 5_000)
    mems = [_mem(i, f"<<E{i}>>" + "z" * 2_000) for i in range(10)]

    section = dream._build_agent_section("opie", mems)

    assert "<<E9>>" in section, "newest entry must survive the cap"
    assert "<<E8>>" in section, "second-newest entry must survive the cap"
    assert "<<E0>>" not in section, "oldest entry must be dropped first"


def test_section_no_cap_when_under_budget(monkeypatch):
    """Under budget → every entry kept, no 'omitted' notice."""
    monkeypatch.setattr(dream, "MAX_AGENT_SECTION_CHARS", 1_000_000)
    mems = [_mem(i, f"<<E{i}>> small entry") for i in range(5)]

    section = dream._build_agent_section("cc", mems)

    assert "omitted" not in section
    for i in range(5):
        assert f"<<E{i}>>" in section
    assert "# Agent: cc (5 entries)" in section


# ── _call_openrouter_adaptive: the halving backstop ──

def _big_content() -> str:
    """200KB with distinct head/tail markers so we can prove the tail is kept."""
    return "HEAD-MARKER" + "q" * (200_000 - 22) + "TAIL-MARKER"


def test_adaptive_halves_until_under_limit(monkeypatch):
    """Oversize 400 → halve + retry; succeed once small enough, keeping the tail."""
    seen: list[int] = []
    final = {}

    def fake_call(system, content, max_tokens=4096):
        seen.append(len(content))
        if len(content) > 60_000:
            raise RuntimeError(
                "OpenRouter 400: This endpoint's maximum context length is "
                "1048576 tokens. However, you requested about 4926022 tokens."
            )
        final["content"] = content
        return "synthesized brief", {"prompt_tokens": 100}

    monkeypatch.setattr(dream, "_call_openrouter", fake_call)
    out, usage = dream._call_openrouter_adaptive("sys", _big_content(), max_tokens=2048)

    assert out == "synthesized brief"
    assert seen == [200_000, 100_000, 50_000], f"unexpected halving path: {seen}"
    assert final["content"].endswith("TAIL-MARKER"), "must keep the most-recent tail"
    assert "HEAD-MARKER" not in final["content"], "oldest head should be dropped on halving"


def test_adaptive_retries_on_200_wrapped_400(monkeypatch):
    """OpenRouter's provider-side 400 wrapped in a 200 ('no choices') is oversize too."""
    calls = {"n": 0}

    def fake_call(system, content, max_tokens=4096):
        calls["n"] += 1
        if len(content) > 60_000:
            raise RuntimeError('OpenRouter 200 but no choices: {"error": {"code": 400}}')
        return "ok", {}

    monkeypatch.setattr(dream, "_call_openrouter", fake_call)
    out, _ = dream._call_openrouter_adaptive("sys", _big_content())

    assert out == "ok"
    assert calls["n"] > 1, "the 200-wrapped-400 must trigger a smaller retry"


def test_adaptive_reraises_non_size_error(monkeypatch):
    """A non-size failure must propagate immediately — no pointless shrinking."""
    calls = {"n": 0}

    def fake_call(system, content, max_tokens=4096):
        calls["n"] += 1
        raise RuntimeError("network exploded")

    monkeypatch.setattr(dream, "_call_openrouter", fake_call)
    with pytest.raises(RuntimeError, match="network exploded"):
        dream._call_openrouter_adaptive("sys", _big_content())
    assert calls["n"] == 1, "must not retry on a non-size error"


def test_adaptive_gives_up_at_min_chars(monkeypatch):
    """Already at/under min_chars and still oversize → raise, don't loop forever."""
    calls = {"n": 0}

    def fake_call(system, content, max_tokens=4096):
        calls["n"] += 1
        raise RuntimeError("maximum context length exceeded")

    monkeypatch.setattr(dream, "_call_openrouter", fake_call)
    with pytest.raises(RuntimeError, match="maximum context"):
        dream._call_openrouter_adaptive("sys", "x" * 10_000, min_chars=20_000)
    assert calls["n"] == 1, "content below min_chars must not be halved again"


# ── Stage 0.5 fact extraction: chunking (the 2026-06-13 fix) ──
#
# The bug: one big batch (cc's 165-entry / 64K-char day) was sent in a single
# call capped at max_tokens=4096 output. The fact array overran the output cap,
# truncated mid-string, json.loads failed, and the WHOLE agent's facts were lost.
# Fix: chunk by input chars so each call's output fits, and isolate a parse
# failure to one chunk instead of the whole agent.

def test_chunk_splits_over_budget():
    """Input above the chunk budget is split into >1 chunk; nothing is dropped."""
    mems = [_mem(i, "z" * 1_000) for i in range(10)]
    chunks = dream._chunk_memories_by_chars(mems, budget=3_000)
    assert len(chunks) > 1, "oversized input must be chunked"
    assert sum(len(c) for c in chunks) == 10, "chunking must not drop entries"


def test_chunk_single_when_under_budget():
    mems = [_mem(i, "small") for i in range(5)]
    chunks = dream._chunk_memories_by_chars(mems, budget=1_000_000)
    assert len(chunks) == 1 and len(chunks[0]) == 5


def test_chunk_oversized_single_memory_becomes_own_chunk():
    """A lone memory bigger than the budget is its own chunk — never silently dropped."""
    mems = [_mem(0, "z" * 50_000)]
    chunks = dream._chunk_memories_by_chars(mems, budget=10_000)
    assert len(chunks) == 1 and len(chunks[0]) == 1


def test_chunk_preserves_chronological_order():
    mems = [_mem(i, f"<<E{i}>>" + "z" * 1_000) for i in range(6)]
    chunks = dream._chunk_memories_by_chars(mems, budget=2_500)
    flat = [m for c in chunks for m in c]
    ts = [m["timestamp"] for m in flat]
    assert ts == sorted(ts), "chronological order must survive chunking"


def test_extract_chunks_big_input(monkeypatch):
    """Big input → multiple extraction calls (the fix). Mutation: disable chunking
    (huge budget) and this drops to 1 call, failing the assertion — a real guard."""
    monkeypatch.setattr(dream, "FACT_EXTRACTION_CHUNK_CHARS", 3_000)
    calls: list[str] = []

    def fake(agent_id, section, label=""):
        calls.append(label)
        return [{"entity": "e", "attribute": "a", "value": "v", "evidence_source": "t"}]

    monkeypatch.setattr(dream, "_extract_facts_from_section", fake)
    mems = [_mem(i, "z" * 1_000) for i in range(10)]

    facts = dream.extract_facts_for_agent("cc", mems)

    assert len(calls) > 1, "a large day must be split into multiple LLM calls"
    assert len(facts) == len(calls), "one fact per successful chunk, accumulated"


def test_extract_one_bad_chunk_does_not_drop_agent(monkeypatch):
    """A parse failure in one chunk keeps every other chunk's facts — the exact
    regression from 2026-06-13 where one truncation lost all of cc's facts."""
    monkeypatch.setattr(dream, "FACT_EXTRACTION_CHUNK_CHARS", 3_000)
    seen: list[str] = []

    def fake(agent_id, section, label=""):
        seen.append(label)
        if len(seen) == 2:  # second chunk fails to parse
            return None
        return [{"entity": "e", "attribute": "a", "value": "v", "evidence_source": "t"}]

    monkeypatch.setattr(dream, "_extract_facts_from_section", fake)
    mems = [_mem(i, "z" * 1_000) for i in range(10)]

    facts = dream.extract_facts_for_agent("cc", mems)

    assert len(seen) >= 3, "expected several chunks"
    assert len(facts) == len(seen) - 1, "one bad chunk must not zero out the agent"
