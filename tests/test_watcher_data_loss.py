"""H2/H3 regression tests — the watcher's offset is a commit record.

H2: a poll landing mid-append must not consume the torn final line.
H3: a failed /ingest must not advance the offset past the failed exchange.
Plus the adjacent same-file hardening: a trailing unpaired user message is
held back until its assistant reply lands, a truncated/rotated file resets
its offset, and the positions file is written atomically.
"""
import json

import pytest

import agentb.watcher as watcher


def user_line(text):
    return json.dumps({"type": "message", "message": {
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-07-06T20:00:00Z",
    }})


def assistant_line(text):
    return json.dumps({"type": "message", "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "timestamp": "2026-07-06T20:00:01Z",
    }})


def header_line():
    return json.dumps({"type": "session", "id": "abc123"})


@pytest.fixture
def ingest_log(monkeypatch):
    """Replace ingest_exchange with a recorder; tests set .fail_prompts to
    simulate per-exchange server failures."""
    class Recorder:
        def __init__(self):
            self.calls = []
            self.fail_prompts = set()

        def __call__(self, prompt, response, metadata=None):
            self.calls.append(prompt)
            return prompt not in self.fail_prompts

    rec = Recorder()
    monkeypatch.setattr(watcher, "ingest_exchange", rec)
    return rec


# ── H2: torn final line stays unread ──

def test_torn_line_not_consumed(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    complete = user_line("first question") + "\n" + assistant_line("first answer") + "\n"
    torn = user_line("second question")[:20]  # mid-append, no newline
    f.write_bytes((complete + torn).encode())

    pos, ingested = watcher.process_session_file(f, 0)

    assert ingested == 1
    assert pos == len(complete.encode())  # stopped at the last newline

    # The append finishes; the once-torn exchange is picked up whole.
    with open(f, "ab") as fh:
        fh.write((user_line("second question")[20:] + "\n" +
                  assistant_line("second answer") + "\n").encode())
    pos, ingested = watcher.process_session_file(f, pos)
    assert ingested == 1
    assert ingest_log.calls == ["first question", "second question"]


def test_chunk_with_no_newline_leaves_position(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    f.write_text(user_line("partial")[:15])
    pos, ingested = watcher.process_session_file(f, 0)
    assert (pos, ingested) == (0, 0)


# ── H3: failed ingest does not advance ──

def test_failed_ingest_does_not_advance(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    pair1 = user_line("q1") + "\n" + assistant_line("a1") + "\n"
    pair2 = user_line("q2") + "\n" + assistant_line("a2") + "\n"
    f.write_bytes((pair1 + pair2).encode())

    ingest_log.fail_prompts = {"q2"}
    pos, ingested = watcher.process_session_file(f, 0)
    assert ingested == 1
    assert pos == len(pair1.encode())  # parked at the failed exchange's user line

    # Server recovers — the retry picks up exactly the lost exchange.
    ingest_log.fail_prompts = set()
    pos, ingested = watcher.process_session_file(f, pos)
    assert ingested == 1
    assert pos == len((pair1 + pair2).encode())
    assert ingest_log.calls == ["q1", "q2", "q2"]


def test_first_pair_failure_keeps_original_position(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    f.write_text(user_line("q1") + "\n" + assistant_line("a1") + "\n")
    ingest_log.fail_prompts = {"q1"}
    pos, ingested = watcher.process_session_file(f, 0)
    assert (pos, ingested) == (0, 0)


# ── Trailing unpaired user message is held back ──

def test_trailing_unpaired_user_held_back(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    head = header_line() + "\n"
    f.write_bytes((head + user_line("still thinking") + "\n").encode())

    pos, ingested = watcher.process_session_file(f, 0)
    assert ingested == 0
    assert pos == len(head.encode())  # consumed the header, parked at the user line

    with open(f, "ab") as fh:
        fh.write((assistant_line("the answer") + "\n").encode())
    pos, ingested = watcher.process_session_file(f, pos)
    assert ingested == 1
    assert ingest_log.calls == ["still thinking"]


def test_non_message_lines_alone_are_consumed(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    f.write_text(header_line() + "\n")
    pos, ingested = watcher.process_session_file(f, 0)
    assert ingested == 0
    assert pos == f.stat().st_size  # no wedge on housekeeping lines


# ── Truncated / rotated file resets ──

def test_truncated_file_resets_offset(tmp_path, ingest_log):
    f = tmp_path / "s.jsonl"
    f.write_text(user_line("q1") + "\n" + assistant_line("a1") + "\n")
    stale_position = f.stat().st_size + 500

    pos, ingested = watcher.process_session_file(f, stale_position)
    assert ingested == 1  # re-read from the top instead of wedging
    assert pos == f.stat().st_size


# ── Atomic positions file ──

def test_save_positions_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher, "STATE_DIR", tmp_path)
    monkeypatch.setattr(watcher, "STATE_FILE", tmp_path / "positions.json")

    watcher.save_positions({"s.jsonl": 42})

    assert json.loads((tmp_path / "positions.json").read_text()) == {"s.jsonl": 42}
    assert list(tmp_path.iterdir()) == [tmp_path / "positions.json"]  # no tmp leftover
