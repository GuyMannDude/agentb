"""Ingest retry-dedup tests (H3's server half).

Clients re-send a whole chunk when any ingest in it failed, so /ingest must
treat an identical exchange inside the dedup window as the same exchange —
otherwise crash-recovery retries write duplicate memories.
"""
import json
import time

from agentb.sessions import SessionManager, SessionConfig


def exchanges_on_disk(data_dir):
    entries = []
    for f in (data_dir / "sessions" / "hot").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("_type") == "exchange":
                entries.append(entry)
    return entries


def test_duplicate_within_window_is_skipped(tmp_path):
    mgr = SessionManager(tmp_path)
    first = mgr.ingest("what changed?", "the watcher")
    retry = mgr.ingest("what changed?", "the watcher")

    assert first["status"] == "captured"
    assert retry["status"] == "duplicate"
    assert len(exchanges_on_disk(tmp_path)) == 1


def test_distinct_exchanges_both_captured(tmp_path):
    mgr = SessionManager(tmp_path)
    assert mgr.ingest("q1", "a1")["status"] == "captured"
    assert mgr.ingest("q2", "a2")["status"] == "captured"
    assert len(exchanges_on_disk(tmp_path)) == 2


def test_repeat_after_window_is_captured(tmp_path):
    mgr = SessionManager(tmp_path)
    mgr.ingest("ok", "done")
    # Age the hash past the window — a genuine repeat later must still save.
    key = SessionManager._hash_exchange("ok", "done")
    mgr._dedup[key] = time.time() - (mgr.config.dedup_window_seconds + 1)

    assert mgr.ingest("ok", "done")["status"] == "captured"
    assert len(exchanges_on_disk(tmp_path)) == 2


def test_restart_still_dedups_recent_exchanges(tmp_path):
    # Crash-mid-batch case: exchange landed, server restarted, client retries.
    SessionManager(tmp_path).ingest("survived the crash?", "yes")

    reborn = SessionManager(tmp_path)
    assert reborn.ingest("survived the crash?", "yes")["status"] == "duplicate"
    assert len(exchanges_on_disk(tmp_path)) == 1


def test_dedup_cache_is_bounded(tmp_path):
    mgr = SessionManager(tmp_path, SessionConfig(dedup_cache_size=2))
    mgr.ingest("q1", "a1")
    mgr.ingest("q2", "a2")
    mgr.ingest("q3", "a3")  # evicts q1's hash

    assert len(mgr._dedup) == 2
    # q1 aged out of the cache — a re-send now writes again (bounded memory
    # traded for a shorter dedup horizon).
    assert mgr.ingest("q1", "a1")["status"] == "captured"
