"""Cortex Stick v1 — courier sync engine contract.

Two simulated full Mnemo installs (host A, host B) and one stick directory.
The whole product is the merge matrix:

  - new/changed/deleted on either side propagates through the courier
  - deletes are detected against the per-host base inventory (no tombstones)
  - both-edit file conflict: deterministic winner, loser preserved on stick
  - edit-vs-delete: edit wins (under-delete is the only permitted failure)
  - trajectory JSONLs union-merge by record id — append-only truth never loses
  - torn generation (manifest lies) refuses to sync
  - mass-delete guard refuses to carry a massacre without --force
  - re-sync with no changes is a no-op (idempotent)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentb.stick import (
    StickError,
    init_stick,
    sha256_file,
    sync,
    verify_manifest,
)


# ── fixtures ───────────────────────────────────────────────────────────────

def mem_path(host: Path, tenant: str, mem_id: str) -> Path:
    return host / tenant / "memory" / f"{mem_id}.json"


def write_mem(host: Path, tenant: str, mem_id: str, summary: str) -> Path:
    p = mem_path(host, tenant, mem_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"id": mem_id, "summary": summary}))
    return p


def read_mem(host: Path, tenant: str, mem_id: str) -> dict:
    return json.loads(mem_path(host, tenant, mem_id).read_text())


def traj_path(host: Path, tenant: str, task: str) -> Path:
    return host / tenant / "trajectories" / f"{task}.jsonl"


def append_traj(host: Path, tenant: str, task: str, rec_id: str) -> None:
    p = traj_path(host, tenant, task)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({"id": rec_id, "task_type": task}) + "\n")


@pytest.fixture
def world(tmp_path):
    """host_a, host_b (data dirs) + a provisioned stick."""
    a, b, mount = tmp_path / "host_a", tmp_path / "host_b", tmp_path / "usb"
    a.mkdir(); b.mkdir(); mount.mkdir()
    stick = init_stick(mount)
    return a, b, stick


def courier(data_dir: Path, stick: Path, host_id: str, **kw):
    return sync(data_dir, stick, host_id=host_id, pad=False, **kw)


# ── the matrix ─────────────────────────────────────────────────────────────

def test_new_memory_travels_a_to_b(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "born on A")
    r1 = courier(a, stick, "host-a")
    assert "memories/cc/memory/m1.json" in r1.to_stick
    r2 = courier(b, stick, "host-b")
    assert "memories/cc/memory/m1.json" in r2.to_host
    assert read_mem(b, "cc", "m1")["summary"] == "born on A"


def test_edit_propagates_back(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "v1")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    write_mem(b, "cc", "m1", "v2 from B")
    courier(b, stick, "host-b")
    r = courier(a, stick, "host-a")
    assert "memories/cc/memory/m1.json" in r.to_host
    assert read_mem(a, "cc", "m1")["summary"] == "v2 from B"


def test_delete_propagates_without_tombstones(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "doomed")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    assert mem_path(b, "cc", "m1").exists()
    mem_path(a, "cc", "m1").unlink()
    r_a = courier(a, stick, "host-a")
    assert "memories/cc/memory/m1.json" in r_a.deleted_on_stick
    r_b = courier(b, stick, "host-b")
    assert "memories/cc/memory/m1.json" in r_b.deleted_on_host
    assert not mem_path(b, "cc", "m1").exists()


def test_delete_does_not_resurrect(world):
    """After a delete round-trips, re-syncing the deleting host must not
    bring the file back (the classic missing-tombstone failure)."""
    a, b, stick = world
    write_mem(a, "cc", "m1", "doomed")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    mem_path(a, "cc", "m1").unlink()
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    r = courier(a, stick, "host-a")
    assert not mem_path(a, "cc", "m1").exists()
    assert not r.changed


def test_both_edit_conflict_loser_preserved(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "base")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    write_mem(a, "cc", "m1", "A's edit")
    write_mem(b, "cc", "m1", "B's edit")
    courier(b, stick, "host-b")           # B's edit reaches the stick
    r = courier(a, stick, "host-a")       # A discovers the conflict
    assert any("both edited" in c for c in r.conflicts)
    # Deterministic outcome: both sides converge on ONE winner...
    winner = read_mem(a, "cc", "m1")["summary"]
    assert winner in ("A's edit", "B's edit")
    stick_copy = json.loads(
        (stick / "memories/cc/memory/m1.json").read_text())["summary"]
    assert stick_copy == winner
    # ...and the loser is preserved under state/conflicts/, not destroyed.
    saved = list((stick / "state" / "conflicts").rglob("*"))
    assert any(p.is_file() for p in saved)
    loser = "A's edit" if winner == "B's edit" else "B's edit"
    assert any(loser in p.read_text() for p in saved if p.is_file())


def test_edit_beats_delete(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "base")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    mem_path(a, "cc", "m1").unlink()      # A deletes...
    write_mem(b, "cc", "m1", "B improved it")   # ...B edits
    courier(b, stick, "host-b")
    r = courier(a, stick, "host-a")
    assert any("edit wins" in c for c in r.conflicts)
    assert read_mem(a, "cc", "m1")["summary"] == "B improved it"


def test_trajectory_jsonl_union(world):
    a, b, stick = world
    append_traj(a, "cc", "deploy", "r1")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    append_traj(a, "cc", "deploy", "r2-from-a")   # concurrent appends
    append_traj(b, "cc", "deploy", "r3-from-b")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    r = courier(a, stick, "host-a")
    assert "memories/cc/memory" not in str(r.conflicts)
    ids_a = [json.loads(l)["id"] for l in
             traj_path(a, "cc", "deploy").read_text().splitlines()]
    ids_b = [json.loads(l)["id"] for l in
             traj_path(b, "cc", "deploy").read_text().splitlines()]
    assert set(ids_a) == set(ids_b) == {"r1", "r2-from-a", "r3-from-b"}


def test_derived_sidecars_never_cross(world):
    """traj_index.sqlite / recall_stats.json are geometry, not facts (P2)."""
    a, b, stick = world
    append_traj(a, "cc", "deploy", "r1")
    (a / "cc" / "trajectories" / "traj_index.sqlite").write_bytes(b"sqlite fake")
    (a / "cc" / "trajectories" / "recall_stats.json").write_text("{}")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    assert not (b / "cc" / "trajectories" / "traj_index.sqlite").exists()
    assert not (b / "cc" / "trajectories" / "recall_stats.json").exists()
    assert traj_path(b, "cc", "deploy").exists()


def test_torn_generation_refuses(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "v1")
    courier(a, stick, "host-a")
    # Corrupt a synced file behind the manifest's back (yank mid-write).
    (stick / "memories/cc/memory/m1.json").write_text("garbage")
    with pytest.raises(StickError, match="TORN GENERATION"):
        courier(b, stick, "host-b")


def test_mass_delete_guard(world):
    a, b, stick = world
    for i in range(10):
        write_mem(a, "cc", f"m{i}", f"memory {i}")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    for i in range(9):                      # A loses 90% of its store
        mem_path(a, "cc", f"m{i}").unlink()
    with pytest.raises(StickError, match="MASS-DELETE GUARD"):
        courier(a, stick, "host-a")
    # stick untouched by the refused sync
    assert len(list((stick / "memories/cc/memory").glob("*.json"))) == 10
    # --force carries it when truly intended
    r = courier(a, stick, "host-a", force=True)
    assert len(r.deleted_on_stick) == 9


def test_idempotent_resync(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "v1")
    append_traj(a, "cc", "deploy", "r1")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    for hid, host in (("host-a", a), ("host-b", b)):
        r = courier(host, stick, hid)
        assert not r.changed, f"{hid} re-sync was not a no-op: {r}"


def test_dry_run_touches_nothing(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "v1")
    before = verify_manifest(stick)["generation"]
    r = courier(a, stick, "host-a", dry_run=True)
    assert "memories/cc/memory/m1.json" in r.to_stick
    assert not (stick / "memories/cc/memory/m1.json").exists()
    assert verify_manifest(stick)["generation"] == before


def test_stick_adopts_unknown_tenant(world):
    """A tenant that exists only on the stick is created on the new host —
    carrying a new agent to the second machine is the point of a courier."""
    a, b, stick = world
    write_mem(a, "newagent", "m1", "hello")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    assert read_mem(b, "newagent", "m1")["summary"] == "hello"


def test_manifest_hashes_match_disk(world):
    a, _, stick = world
    write_mem(a, "cc", "m1", "v1")
    courier(a, stick, "host-a")
    manifest = verify_manifest(stick)
    rel = "memories/cc/memory/m1.json"
    assert manifest["files"][rel]["sha256"] == sha256_file(stick / rel)


def _git_repo(path: Path) -> Path:
    import subprocess
    path.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-b", "main"],
                 ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
        subprocess.run(["git", "-C", str(path)] + args, capture_output=True)
    return path


def _git_commit(repo: Path, fname: str, text: str) -> None:
    import subprocess
    (repo / fname).write_text(text)
    subprocess.run(["git", "-C", str(repo), "add", fname], capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", f"edit {fname}"],
                   capture_output=True)


def test_brain_travels_via_bare_repo(world, tmp_path):
    a, b, stick = world
    repo_a = _git_repo(tmp_path / "brain_a")
    _git_commit(repo_a, "active.md", "task list v1")
    r = sync(a, stick, host_id="host-a", pad=False, brain_repo=repo_a)
    assert r.brain == "pushed"
    # Second machine clones from the stick, then couriers normally.
    import subprocess
    repo_b = tmp_path / "brain_b"
    subprocess.run(["git", "clone", str(stick / "brain" / "brain.git"),
                    str(repo_b)], capture_output=True)
    _git_commit(repo_b, "active.md", "task list v2 from B")
    r = sync(b, stick, host_id="host-b", pad=False, brain_repo=repo_b)
    assert r.brain == "pushed"
    r = sync(a, stick, host_id="host-a", pad=False, brain_repo=repo_a)
    assert r.brain == "merged"
    assert (repo_a / "active.md").read_text() == "task list v2 from B"


def test_brain_conflict_aborts_loudly(world, tmp_path):
    import subprocess
    a, b, stick = world
    repo_a = _git_repo(tmp_path / "brain_a")
    _git_commit(repo_a, "active.md", "base")
    sync(a, stick, host_id="host-a", pad=False, brain_repo=repo_a)
    repo_b = tmp_path / "brain_b"
    subprocess.run(["git", "clone", str(stick / "brain" / "brain.git"),
                    str(repo_b)], capture_output=True)
    subprocess.run(["git", "-C", str(repo_b), "config", "user.email", "t@t"],
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo_b), "config", "user.name", "t"],
                   capture_output=True)
    _git_commit(repo_a, "active.md", "A's version")
    _git_commit(repo_b, "active.md", "B's version")
    sync(b, stick, host_id="host-b", pad=False, brain_repo=repo_b)
    r = sync(a, stick, host_id="host-a", pad=False, brain_repo=repo_a)
    assert r.brain == "CONFLICT"
    assert any("brain: git merge conflict" in c for c in r.conflicts)
    # repo left clean (merge aborted), A's version intact
    status = subprocess.run(["git", "-C", str(repo_a), "status", "--porcelain"],
                            capture_output=True, text=True).stdout
    assert status.strip() == ""
    assert (repo_a / "active.md").read_text() == "A's version"


def test_dirty_brain_skipped_not_touched(world, tmp_path):
    a, _, stick = world
    repo_a = _git_repo(tmp_path / "brain_a")
    _git_commit(repo_a, "active.md", "committed")
    (repo_a / "active.md").write_text("uncommitted edits")
    r = sync(a, stick, host_id="host-a", pad=False, brain_repo=repo_a)
    assert r.brain == "skipped-dirty"
    assert (repo_a / "active.md").read_text() == "uncommitted edits"


def test_pad_files_travel(world):
    a, b, stick = world
    pad_file = a / "pad" / "notes" / "wip.md"
    pad_file.parent.mkdir(parents=True)
    pad_file.write_text("dragging this to the other desk")
    sync(a, stick, host_id="host-a")
    sync(b, stick, host_id="host-b")
    assert (b / "pad" / "notes" / "wip.md").read_text() == \
        "dragging this to the other desk"


def test_host_id_is_store_scoped_not_hostname(tmp_path):
    """Two data stores on same-named machines must be distinct sync peers.

    Regression: with host_id = bare hostname, the second store loaded the
    first store's base inventory, read its own emptiness as deletions, and
    deleted the first store's memories off the stick."""
    from agentb.stick import load_host_config
    id_a = load_host_config(tmp_path / "a")["host_id"]
    id_b = load_host_config(tmp_path / "b")["host_id"]
    assert id_a != id_b
    # ...and each store's identity is stable across reloads.
    assert load_host_config(tmp_path / "a")["host_id"] == id_a


def test_schema_forward_version_refuses(world):
    a, _, stick = world
    manifest = json.loads((stick / "manifest.json").read_text())
    manifest["schema_version"] = 99
    (stick / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(StickError, match="newer than this host"):
        courier(a, stick, "host-a")


# ── transactional envelope (review findings, 2026-07-12) ──────────────────

def test_refused_sync_changes_nothing_and_stick_stays_syncable(world):
    """A guard tripping on a LATER channel must not have already mutated an
    earlier one — 'Nothing was changed' has to be literally true, and the
    stick must not be left torn (regression: channel writes preceded the
    guard evaluation of subsequent channels)."""
    a, b, stick = world
    write_mem(a, "aaa", "keep", "original")       # sorts before zzz
    for i in range(10):
        write_mem(a, "zzz", f"m{i}", f"mem {i}")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    # aaa gets a legitimate edit; zzz trips the mass-delete guard.
    write_mem(a, "aaa", "keep", "EDITED on A")
    for i in range(9):
        mem_path(a, "zzz", f"m{i}").unlink()
    with pytest.raises(StickError, match="MASS-DELETE GUARD"):
        courier(a, stick, "host-a")
    # The earlier channel was NOT partially applied...
    stick_copy = json.loads(
        (stick / "memories/aaa/memory/keep.json").read_text())
    assert stick_copy["summary"] == "original"
    # ...the stick is not torn (a later sync still works)...
    verify_manifest(stick)
    r = courier(b, stick, "host-b")
    assert not r.changed
    # ...and --force carries the whole thing through afterwards.
    r = courier(a, stick, "host-a", force=True)
    assert "memories/aaa/memory/keep.json" in r.to_stick
    assert len(r.deleted_on_stick) == 9


def test_torn_stick_repair_recovers(world):
    """TORN GENERATION must have an in-tool escape: repair accepts the
    stick's contents as truth and the next sync merges from there."""
    from agentb.stick import repair_manifest
    a, b, stick = world
    write_mem(a, "cc", "m1", "v1")
    write_mem(a, "cc", "m2", "v2")
    courier(a, stick, "host-a")
    # Simulate a yank mid-write: one stick file replaced by a torn write.
    (stick / "memories/cc/memory/m1.json").write_text(
        '{"id": "m1", "summary": "half-carried edit"}')
    with pytest.raises(StickError, match="TORN GENERATION"):
        courier(b, stick, "host-b")
    manifest = repair_manifest(stick)
    assert "memories/cc/memory/m1.json" in manifest["files"]
    # Post-repair: B syncs cleanly and receives the stick's surviving truth.
    courier(b, stick, "host-b")
    assert read_mem(b, "cc", "m1")["summary"] == "half-carried edit"
    assert read_mem(b, "cc", "m2")["summary"] == "v2"
    # A also converges (its m1 differs from repaired stick → conflict path,
    # deterministic, nothing silently lost).
    courier(a, stick, "host-a")
    verify_manifest(stick)


def test_conflict_converges_on_both_hosts(world):
    """After a both-edit conflict resolves, BOTH machines end on the winner."""
    a, b, stick = world
    write_mem(a, "cc", "m1", "base")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    write_mem(a, "cc", "m1", "A's edit")
    write_mem(b, "cc", "m1", "B's edit")
    courier(b, stick, "host-b")
    courier(a, stick, "host-a")     # conflict resolves here
    courier(b, stick, "host-b")     # B picks up the outcome
    assert read_mem(a, "cc", "m1") == read_mem(b, "cc", "m1")


def test_multi_tenant_sync(world):
    a, b, stick = world
    write_mem(a, "cc", "m1", "cc's memory")
    write_mem(a, "opie", "m1", "opie's memory")
    append_traj(a, "cc", "deploy", "r1")
    courier(a, stick, "host-a")
    courier(b, stick, "host-b")
    assert read_mem(b, "cc", "m1")["summary"] == "cc's memory"
    assert read_mem(b, "opie", "m1")["summary"] == "opie's memory"
    assert traj_path(b, "cc", "deploy").exists()


def test_engine_requires_host_id(world):
    """The engine must refuse a missing host_id rather than fall back to the
    hostname (the shared-base-inventory massacre footgun)."""
    a, _, stick = world
    with pytest.raises(StickError, match="host_id"):
        sync(a, stick, host_id="", pad=False)
