"""Cortex Stick — USB courier sync between two full Mnemo installations.

The stick is NOT a server. Both machines run full Mnemo; the stick carries
the delta between them: memory JSONs, trajectory JSONLs, the brain git repo,
and a free-form project pad. Plug in → sync; pull out → carry → plug in →
the other machine catches up. No cloud, no VPN.

Constitutional principles (brain/cortex-stick-spec.md, carried over):
  P2 — sync the facts, not the geometry. Only truth files cross the stick.
       Vec indexes, caches, and sidecars are derived and rebuilt per-host.
  F-4 — data files → manifest LAST. The manifest write is the commit point;
        a torn generation is detected on mount and refused loudly.
  F-2/F-5 — per-host base inventories make every sync a 3-way merge; no
        silent overwrite, and "safe to remove" fires only after readback-
        verified hashes.
  #1121 — conflicts resolve deterministically, but the LOSER is preserved
        under state/conflicts/ on the stick, never silently destroyed.

Design note: this is a STATE-BASED sync. The stick keeps, per host, the
inventory both sides agreed on at last sync (relpath → sha256 + version).
Deletes are detected against that base — no tombstones, no record-schema
changes, works against any existing install.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agentb.fsutil import atomic_write_text

log = logging.getLogger("agentb.stick")

STICK_DIRNAME = "cortex"          # <mount>/cortex/ — human-findable by eye
STICK_SCHEMA_VERSION = 1
MASS_DELETE_FRACTION = 0.25       # refuse if a sync would delete more than this
MASS_DELETE_MIN_FILES = 8         # ...but only guard channels at least this big
FREE_SPACE_MARGIN = 4 * 1024 * 1024  # keep 4MB headroom on the stick

# Truth patterns per channel. Everything else in those dirs (traj_index.sqlite,
# recall_stats.json, vec dbs, caches) is derived geometry and never crosses.
MEMORY_GLOB = "*.json"
TRAJECTORY_GLOB = "*.jsonl"

DEFAULT_MOUNT_ROOTS = [
    "/media/{user}", "/run/media/{user}",   # Linux automounts
    "/Volumes",                              # macOS
]


# ── small helpers ──────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        raise StickError(f"Unreadable JSON at {path}: {e}") from e


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


def default_host_id() -> str:
    """Lowercase hostname, exFAT-safe (used in on-stick filenames)."""
    raw = socket.gethostname().split(".")[0].lower()
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in raw) or "host"


class StickError(RuntimeError):
    """Loud failure — sync refuses rather than guessing."""


# ── stick discovery / provisioning ─────────────────────────────────────────

def candidate_mount_roots(extra: Optional[list[str]] = None) -> list[Path]:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    roots = [Path(r.format(user=user)) for r in DEFAULT_MOUNT_ROOTS]
    # Windows drive letters — cheap probe, no psutil dependency.
    if os.name == "nt":
        roots += [Path(f"{c}:/") for c in "DEFGHIJKLMNOPQRSTUVWXYZ"]
    for r in extra or []:
        roots.insert(0, Path(r).expanduser())
    return roots


def find_stick(extra_roots: Optional[list[str]] = None) -> Optional[Path]:
    """Locate a provisioned stick: a cortex/passport.json under a mount root
    (or the root itself being the stick dir). Returns the cortex/ dir."""
    for root in candidate_mount_roots(extra_roots):
        if (root / "passport.json").is_file() and root.name == STICK_DIRNAME:
            return root
        if not root.is_dir():
            continue
        direct = root / STICK_DIRNAME
        if (direct / "passport.json").is_file():
            return direct
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for mount in children:
            cand = mount / STICK_DIRNAME
            if (cand / "passport.json").is_file():
                return cand
    return None


def init_stick(mount: Path, name: str = "cortex-stick") -> Path:
    """Provision <mount>/cortex/. Idempotent-hostile on purpose: refuses to
    re-init an existing stick (that would orphan host inventories)."""
    stick = mount / STICK_DIRNAME if mount.name != STICK_DIRNAME else mount
    if (stick / "passport.json").exists():
        raise StickError(f"Already a Cortex Stick: {stick}")
    for sub in ("memories", "brain", "pad", "state/conflicts"):
        (stick / sub).mkdir(parents=True, exist_ok=True)
    passport = {
        "stick_id": hashlib.sha256(
            f"{name}:{time.time_ns()}".encode()
        ).hexdigest()[:16],
        "name": name,
        "schema_version": STICK_SCHEMA_VERSION,
        "created_at": time.time(),
        "hosts": {},
    }
    _write_json(stick / "passport.json", passport)
    _write_json(stick / "manifest.json",
                {"schema_version": STICK_SCHEMA_VERSION, "generation": 0,
                 "files": {}})
    return stick


# ── manifest (the commit point) ────────────────────────────────────────────

def verify_manifest(stick: Path) -> dict:
    """Check every manifest hash against the stick's actual files.

    A mismatch means a torn generation (yank mid-sync, bit rot, tampering) —
    refuse to sync rather than merge from a lying base. The manifest is
    written LAST during sync, so a clean manifest = a complete generation.
    """
    manifest = _load_json(stick / "manifest.json", None)
    if manifest is None:
        raise StickError(f"No manifest.json on stick {stick} — not provisioned?")
    if manifest.get("schema_version", 0) > STICK_SCHEMA_VERSION:
        raise StickError(
            "Stick schema is newer than this host's Cortex Stick tool — "
            "update mnemo-cortex on this machine before syncing."
        )
    bad = []
    for rel, meta in manifest.get("files", {}).items():
        sha = meta.get("sha256") if isinstance(meta, dict) else None
        p = stick / rel
        if sha is None:
            bad.append(f"malformed manifest entry: {rel}")
        elif not p.is_file():
            bad.append(f"missing: {rel}")
        elif sha256_file(p) != sha:
            bad.append(f"hash mismatch: {rel}")
    if bad:
        detail = "\n  ".join(bad[:10])
        more = f"\n  … and {len(bad) - 10} more" if len(bad) > 10 else ""
        raise StickError(
            f"TORN GENERATION on stick — manifest disagrees with contents:\n"
            f"  {detail}{more}\n"
            f"Sync refused. Run `mnemo-cortex stick repair` to accept the "
            f"stick's current contents as truth and rebuild the manifest — "
            f"the next sync then 3-way-merges from the repaired state."
        )
    return manifest


def repair_manifest(stick: Path) -> dict:
    """Rebuild manifest.json from what is actually on the stick.

    The escape hatch after a mid-write yank or bit rot: accepts the stick's
    current contents as truth and re-hashes everything. Per-host base
    inventories are kept — they still describe the last state each host
    agreed on, so the next sync 3-way-merges from the repaired state
    (partially-carried files simply look like "changed on the stick").
    Also clears a stale lock left by a killed sync."""
    manifest = _load_json(stick / "manifest.json", None)
    if manifest is None:
        raise StickError(f"No manifest.json at {stick} — not a Cortex Stick.")
    if manifest.get("schema_version", 0) > STICK_SCHEMA_VERSION:
        raise StickError(
            "Stick schema is newer than this host's Cortex Stick tool — "
            "update mnemo-cortex on this machine before repairing."
        )
    old_files = manifest.get("files", {})
    files: dict[str, dict] = {}

    def add(rel: str, sha: str) -> None:
        prev = old_files.get(rel)
        prev = prev if isinstance(prev, dict) else None
        if prev and prev.get("sha256") == sha:
            files[rel] = prev
        else:
            files[rel] = {"sha256": sha,
                          "version": (prev or {}).get("version", 0) + 1}

    mems = stick / "memories"
    if mems.is_dir():
        for tenant in sorted(d for d in mems.iterdir() if d.is_dir()):
            for sub, pat in (("memory", MEMORY_GLOB),
                             ("trajectories", TRAJECTORY_GLOB)):
                for rel, sha in _scan(tenant / sub, pat).items():
                    add(f"memories/{tenant.name}/{sub}/{rel}", sha)
    for rel, sha in _scan(stick / "pad", "**/*").items():
        add(f"pad/{rel}", sha)

    manifest["files"] = files
    manifest["generation"] = manifest.get("generation", 0) + 1
    _write_json(stick / "manifest.json", manifest)
    (stick / "state" / "lock").unlink(missing_ok=True)
    return manifest


# ── channels ───────────────────────────────────────────────────────────────

@dataclass
class Channel:
    """One synced surface. file-unit 3-way merge; 'jsonl' additionally
    union-merges concurrent edits line-by-line instead of conflicting."""
    name: str            # inventory namespace, e.g. "memories/cc/memory"
    host_dir: Path
    stick_dir: Path      # absolute, under the stick
    glob: str
    unit: str = "file"   # "file" | "jsonl"


@dataclass
class SyncReport:
    to_stick: list[str] = field(default_factory=list)
    to_host: list[str] = field(default_factory=list)
    deleted_on_stick: list[str] = field(default_factory=list)
    deleted_on_host: list[str] = field(default_factory=list)
    merged_jsonl: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    brain: str = "skipped"

    @property
    def changed(self) -> bool:
        return bool(self.to_stick or self.to_host or self.deleted_on_stick
                    or self.deleted_on_host or self.merged_jsonl
                    or self.conflicts or self.brain in ("pushed", "merged"))


def _scan(root: Path, pattern: str) -> dict[str, str]:
    """relpath → sha256 for truth files directly under root (non-recursive
    for globs; pad uses rglob via pattern '**/*')."""
    if not root.is_dir():
        return {}
    out = {}
    it = root.rglob(pattern[3:]) if pattern.startswith("**/") else root.glob(pattern)
    for p in sorted(it):
        if p.is_file() and not p.name.endswith(".tmp"):
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def _copy_verified(src: Path, dst: Path, expect_sha: str) -> None:
    """Copy + readback-verify. tmp+rename so a yank never leaves a torn file
    where a truth file used to be."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    if sha256_file(tmp) != expect_sha:
        tmp.unlink(missing_ok=True)
        raise StickError(f"Readback verify FAILED copying {src} → {dst}")
    os.replace(tmp, dst)


def _jsonl_union(host_file: Path, stick_file: Path) -> str:
    """Union-merge two versions of an append-only JSONL by record id
    (content-hash for id-less lines). Order: stick lines first (they're the
    older courier state), then host-only appends. Returns merged text."""
    def parse(path: Path) -> list[tuple[str, str]]:
        pairs = []
        if not path.is_file():
            return pairs
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                key = json.loads(line).get("id") or hashlib.sha256(
                    line.encode()).hexdigest()
            except (json.JSONDecodeError, AttributeError):
                key = hashlib.sha256(line.encode()).hexdigest()
            pairs.append((str(key), line))
        return pairs

    seen: dict[str, str] = {}
    for key, line in parse(stick_file) + parse(host_file):
        seen.setdefault(key, line)
    return "\n".join(seen.values()) + ("\n" if seen else "")


def sync_channel(
    ch: Channel,
    base: dict[str, dict],
    manifest_files: dict[str, dict],
    report: SyncReport,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, dict]:
    """3-way merge of one channel. Returns the channel's new base inventory
    and updates manifest_files (keyed by stick-relative path) in place."""
    host = _scan(ch.host_dir, ch.glob)
    stick_root = ch.stick_dir
    stick = _scan(stick_root, ch.glob)

    # Mass-delete guard: count deletions this sync WOULD apply on each side.
    host_dels = [r for r in base if r not in host and r in stick
                 and stick[r] == base[r].get("sha256")]
    stick_dels = [r for r in base if r not in stick and r in host
                  and host[r] == base[r].get("sha256")]
    for side, dels, total in (("host", stick_dels, len(host)),
                              ("stick", host_dels, len(stick))):
        if (total >= MASS_DELETE_MIN_FILES and dels
                and len(dels) / max(total, 1) > MASS_DELETE_FRACTION
                and not force):
            raise StickError(
                f"MASS-DELETE GUARD [{ch.name}]: this sync would delete "
                f"{len(dels)}/{total} files on the {side}. If that is truly "
                f"intended, re-run with --force. Nothing was changed."
            )

    new_base: dict[str, dict] = {}
    for rel in sorted(set(host) | set(stick) | set(base)):
        h, s = host.get(rel), stick.get(rel)
        b = base.get(rel, {}).get("sha256")
        ver = base.get(rel, {}).get("version", 0)
        hp, sp = ch.host_dir / rel, stick_root / rel
        entry = None

        if h and s and h == s:                              # agree
            entry = {"sha256": h,
                     "version": manifest_files.get(_mkey(ch, rel), {}).get(
                         "version", max(ver, 1))}
        elif h == b and s and s != b:                       # stick changed → host
            if not dry_run:
                _copy_verified(sp, hp, s)
            report.to_host.append(f"{ch.name}/{rel}")
            entry = {"sha256": s,
                     "version": manifest_files.get(_mkey(ch, rel), {}).get("version", ver + 1)}
        elif s == b and h and h != b:                       # host changed → stick
            if not dry_run:
                _copy_verified(hp, sp, h)
            report.to_stick.append(f"{ch.name}/{rel}")
            entry = {"sha256": h, "version": ver + 1}
        elif h is None and s == b and b is not None:        # host deleted → stick
            if not dry_run:
                sp.unlink(missing_ok=True)
            report.deleted_on_stick.append(f"{ch.name}/{rel}")
        elif s is None and h == b and b is not None:        # stick deleted → host
            if not dry_run:
                hp.unlink(missing_ok=True)
            report.deleted_on_host.append(f"{ch.name}/{rel}")
        elif h is None and s is None:                       # gone both sides
            pass
        else:                                               # CONFLICT
            entry = _resolve_conflict(ch, rel, h, s, ver, manifest_files,
                                      report, dry_run=dry_run)

        if entry:
            new_base[rel] = entry
            manifest_files[_mkey(ch, rel)] = entry
        else:
            manifest_files.pop(_mkey(ch, rel), None)
    return new_base


def _mkey(ch: Channel, rel: str) -> str:
    """Manifest key: channel's stick dir relative to the cortex root."""
    return f"{ch.name}/{rel}"


def _resolve_conflict(
    ch: Channel, rel: str, h: Optional[str], s: Optional[str], base_ver: int,
    manifest_files: dict, report: SyncReport, *, dry_run: bool,
) -> Optional[dict]:
    """Deterministic conflict resolution, loser preserved on the stick.

    Edit beats delete (under-delete is the only permitted failure mode).
    Both-edit: JSONL channels union-merge; file channels pick a winner by
    stick generation, then mtime, then hash — and the loser is copied to
    state/conflicts/ before being overwritten."""
    hp, sp = ch.host_dir / rel, ch.stick_dir / rel
    tag = f"{ch.name}/{rel}"

    if h and s is None:            # host edited, stick (other machine) deleted
        if not dry_run:
            _copy_verified(hp, sp, h)
        report.conflicts.append(f"{tag}: edit-vs-delete — edit wins, restored")
        return {"sha256": h, "version": base_ver + 1}
    if s and h is None:            # other machine edited, this host deleted
        if not dry_run:
            _copy_verified(sp, hp, s)
        report.conflicts.append(f"{tag}: delete-vs-edit — edit wins, restored")
        return {"sha256": s, "version": base_ver + 1}

    # From here on it's both-edit: neither side is a delete.
    assert h is not None and s is not None

    if ch.unit == "jsonl":         # append-only truth: union, nobody loses
        merged = _jsonl_union(hp, sp)
        sha = hashlib.sha256(merged.encode()).hexdigest()
        if not dry_run:
            hp.parent.mkdir(parents=True, exist_ok=True)
            sp.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(hp, merged)
            atomic_write_text(sp, merged)
            if sha256_file(sp) != sha:
                raise StickError(f"Readback verify FAILED on merged {sp}")
        report.merged_jsonl.append(tag)
        return {"sha256": sha, "version": base_ver + 1}

    # Both-edit on a file channel.
    stick_ver = manifest_files.get(_mkey(ch, rel), {}).get("version", base_ver)
    if stick_ver > base_ver + 1:
        winner, w_sha, loser_path, loser_sha = "stick", s, hp, h
    elif stick_ver < base_ver + 1:
        winner, w_sha, loser_path, loser_sha = "host", h, sp, s
    else:  # tie → newer mtime, then lexicographic hash (fully deterministic)
        hm = hp.stat().st_mtime if hp.exists() else 0
        sm = sp.stat().st_mtime if sp.exists() else 0
        if hm != sm:
            winner = "host" if hm > sm else "stick"
        else:
            winner = "host" if (h or "") >= (s or "") else "stick"
        w_sha, loser_path, loser_sha = \
            (h, sp, s) if winner == "host" else (s, hp, h)

    if not dry_run:
        # The loser backup is the last copy of a losing edit — it gets the
        # same tmp+rename+readback guard as every truth write, and a hash
        # uniquifier so two conflicts in the same second can't clobber it.
        conflicts_dir = _conflicts_dir(ch)
        dst = conflicts_dir / (f"{int(time.time())}-{loser_sha[:8]}-"
                               f"{rel.replace('/', '__')}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_verified(loser_path, dst, loser_sha)
        if winner == "host":
            _copy_verified(hp, sp, h)
        else:
            _copy_verified(sp, hp, s)
    report.conflicts.append(
        f"{tag}: both edited — {winner} wins, loser saved to state/conflicts/"
    )
    return {"sha256": w_sha, "version": base_ver + 2}


def _conflicts_dir(ch: Channel) -> Path:
    """state/conflicts/ at the stick root (cortex/)."""
    p = ch.stick_dir
    while p.name != STICK_DIRNAME and p.parent != p:
        p = p.parent
    return p / "state" / "conflicts"


# ── brain channel (git courier remote) ─────────────────────────────────────

def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=120,
    )


def sync_brain(repo: Path, stick: Path, report: SyncReport, *,
               dry_run: bool = False) -> None:
    """Sync the brain git repo through a bare repo on the stick.

    fetch → ff/merge → push. A merge that conflicts is aborted and reported;
    the human resolves — the courier never writes conflict markers into a
    brain file behind anyone's back. A dirty working tree skips the channel."""
    if not (repo / ".git").exists():
        report.warnings.append(f"brain: {repo} is not a git repo — skipped")
        report.brain = "skipped"
        return
    if _git(repo, "status", "--porcelain").stdout.strip():
        report.warnings.append(
            "brain: working tree dirty — commit first, brain channel skipped")
        report.brain = "skipped-dirty"
        return
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    bare = stick / "brain" / "brain.git"
    if dry_run:
        report.brain = "dry-run"
        return
    if not bare.exists():
        r = subprocess.run(["git", "init", "--bare", str(bare)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise StickError(f"git init --bare failed on stick: {r.stderr.strip()}")
        # git init --bare defaults HEAD to master; if the couriered branch is
        # named anything else, a later clone from the stick checks out nothing
        # ("remote HEAD refers to nonexistent ref"). Point HEAD at the branch
        # we actually carry.
        subprocess.run(
            ["git", "-C", str(bare), "symbolic-ref", "HEAD",
             f"refs/heads/{branch}"],
            capture_output=True, text=True,
        )

    fetch = _git(repo, "fetch", str(bare), branch)
    have_remote = fetch.returncode == 0
    if have_remote:
        counts = _git(repo, "rev-list", "--left-right", "--count",
                      f"HEAD...FETCH_HEAD").stdout.split()
        ahead, behind = (int(counts[0]), int(counts[1])) if len(counts) == 2 else (1, 0)
    else:
        ahead, behind = 1, 0  # empty bare repo: nothing to merge, just push

    if behind:
        merge = _git(repo, "merge", "--no-edit", "FETCH_HEAD")
        if merge.returncode != 0:
            _git(repo, "merge", "--abort")
            report.brain = "CONFLICT"
            report.conflicts.append(
                "brain: git merge conflict — resolve by hand "
                f"(git fetch {bare} {branch} && git merge FETCH_HEAD)")
            return
        report.brain = "merged"
    push = _git(repo, "push", str(bare), f"{branch}:{branch}")
    if push.returncode != 0:
        raise StickError(f"brain: push to stick failed: {push.stderr.strip()}")
    if ahead and report.brain != "merged":
        report.brain = "pushed"
    elif not ahead and not behind:
        report.brain = "clean"


# ── host config + the full sync ────────────────────────────────────────────

def load_host_config(data_dir: Path) -> dict:
    """{data_dir}/stick.json: host_id, brain_repo, tenants, mount_roots, pad.

    host_id is generated once and PERSISTED — identity belongs to the DATA
    STORE, not the machine name. A bare hostname is a footgun: two desks named
    "ubuntu" (or one reinstalled machine) would share a base inventory, and
    the second store's missing files would read as deletions — the courier
    would carry a massacre. The random suffix makes every store a distinct
    sync peer; a fresh install gets a fresh id and merges instead of deleting."""
    path = data_dir / "stick.json"
    cfg = _load_json(path, {})
    if not cfg.get("host_id"):
        cfg["host_id"] = f"{default_host_id()}-{os.urandom(3).hex()}"
        _write_json(path, cfg)
    cfg.setdefault("brain_repo", None)
    cfg.setdefault("tenants", None)       # None = every agent dir with memory/
    cfg.setdefault("mount_roots", [])
    cfg.setdefault("pad", True)
    return cfg


def discover_tenants(data_dir: Path, explicit: Optional[list[str]]) -> list[str]:
    """Tenant stores live at {data_dir}/agents/<id> (get_agent_data_dir's
    layout) — NOT directly under data_dir. Archived agents (agents kept as
    <id>.archived-YYYYMMDD) are skipped: validate_agent_id would reject the
    dot anyway, and a courier must not resurrect an archived store."""
    if explicit:
        return list(explicit)
    agents_root = data_dir / "agents"
    if not agents_root.is_dir():
        return []
    return sorted(
        d.name for d in agents_root.iterdir()
        if d.is_dir() and "." not in d.name
        and ((d / "memory").is_dir() or (d / "trajectories").is_dir())
    )


def build_channels(data_dir: Path, stick: Path, tenants: list[str],
                   pad: bool) -> list[Channel]:
    chans = []
    for t in tenants:
        chans.append(Channel(
            name=f"memories/{t}/memory",
            host_dir=data_dir / "agents" / t / "memory",
            stick_dir=stick / "memories" / t / "memory",
            glob=MEMORY_GLOB,
        ))
        chans.append(Channel(
            name=f"memories/{t}/trajectories",
            host_dir=data_dir / "agents" / t / "trajectories",
            stick_dir=stick / "memories" / t / "trajectories",
            glob=TRAJECTORY_GLOB,
            unit="jsonl",
        ))
    if pad:
        chans.append(Channel(
            name="pad",
            host_dir=data_dir / "pad",
            stick_dir=stick / "pad",
            glob="**/*",
        ))
    return chans


def _plan_need_bytes(channels: list[Channel], plan: SyncReport) -> int:
    """Bytes the apply pass will write to the stick, from the plan report:
    host→stick copies, union-merged JSONLs, and conflict-loser backups
    (counted at host-file size — conservative)."""
    by_name = sorted(channels, key=lambda c: len(c.name), reverse=True)
    need = 0
    entries = (plan.to_stick + plan.merged_jsonl
               + [c.split(":", 1)[0] for c in plan.conflicts])
    for entry in entries:
        for ch in by_name:   # longest-prefix match: names can nest
            if entry.startswith(ch.name + "/"):
                p = ch.host_dir / entry[len(ch.name) + 1:]
                try:
                    need += p.stat().st_size
                except OSError:
                    pass
                break
    return need


def sync(
    data_dir: Path,
    stick: Path,
    *,
    host_id: str,
    tenants: Optional[list[str]] = None,
    brain_repo: Optional[Path] = None,
    pad: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> SyncReport:
    """The courier sync.

    Order is the whole design:
      1. verify manifest (torn-generation gate)
      2. PLAN PASS — every channel runs dry against a scratch manifest, so
         every guard (mass-delete, conflicts) fires BEFORE a single byte
         moves. A refusal really does mean "nothing was changed"; without
         this, a guard tripping on the second tenant would abandon the first
         tenant's already-written files and tear the generation.
      3. brain git sync — brain.git is NOT manifest-covered, so its failures
         (push rejected, disk full from pack objects) can't tear the
         generation; running it before the channels also lets the free-space
         check see the space it consumed.
      4. free-space preflight, sized from the plan
      5. APPLY PASS — the channels, for real
      6. commit: fsync → inventory → manifest LAST → passport

    host_id must be the store-scoped id from load_host_config — a bare
    hostname is how two same-named machines end up sharing a base inventory
    and fabricating deletions.

    A yank during step 5 still tears the generation (USB media offers no
    transactions) — that's what `repair_manifest` is for; the next mount
    detects it and says so."""
    if not host_id:
        raise StickError("sync() requires a store-scoped host_id "
                         "(see load_host_config).")
    report = SyncReport()
    hid = host_id
    manifest = verify_manifest(stick)

    lock = stick / "state" / "lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.exists() and time.time() - lock.stat().st_mtime < 3600:
        raise StickError(f"Another sync holds the stick lock ({lock}). "
                         "If that sync is dead, delete the lock file.")
    if not dry_run:
        lock.write_text(f"{hid} {int(time.time())}\n")

    try:
        tenant_list = discover_tenants(data_dir, tenants)
        # A stick may know tenants this host doesn't have yet — adopt them,
        # that's the point of a courier.
        stick_tenants = sorted(
            d.name for d in (stick / "memories").iterdir() if d.is_dir()
        ) if (stick / "memories").is_dir() else []
        tenant_list = sorted(set(tenant_list) | set(stick_tenants))

        channels = build_channels(data_dir, stick, tenant_list, pad)
        inv_path = stick / "state" / f"inventory-{hid}.json"
        base_all: dict = _load_json(inv_path, {})
        manifest_files = manifest.get("files", {})

        # ── 2. PLAN PASS — mutates nothing, fires every guard ──
        plan = SyncReport()
        for ch in channels:
            sync_channel(ch, base_all.get(ch.name, {}), dict(manifest_files),
                         plan, force=force, dry_run=True)
        if dry_run:
            if brain_repo:
                sync_brain(Path(brain_repo), stick, plan, dry_run=True)
            return plan

        # ── 3. brain (outside the manifest's failure domain) ──
        if brain_repo:
            sync_brain(Path(brain_repo), stick, report)

        # ── 4. free-space, sized from the plan ──
        need = _plan_need_bytes(channels, plan)
        free = shutil.disk_usage(stick).free
        if need + FREE_SPACE_MARGIN > free:
            raise StickError(
                f"Stick too full: sync needs ~{need // 1024}KB, "
                f"only {free // 1024}KB free. No memory files were changed."
            )

        # ── 5. APPLY PASS ──
        new_base_all = {}
        for ch in channels:
            new_base_all[ch.name] = sync_channel(
                ch, base_all.get(ch.name, {}), manifest_files, report,
                force=force,
            )

        # ── 6. commit: fsync data to media, then inventory, manifest LAST ──
        if hasattr(os, "sync"):
            os.sync()
        _write_json(inv_path, new_base_all)
        manifest["files"] = manifest_files
        manifest["generation"] = manifest.get("generation", 0) + 1
        _write_json(stick / "manifest.json", manifest)

        passport = _load_json(stick / "passport.json", {})
        passport.setdefault("hosts", {})[hid] = {
            "last_sync": time.time(),
            "generation": manifest["generation"],
        }
        _write_json(stick / "passport.json", passport)
        if hasattr(os, "sync"):
            os.sync()
    finally:
        if not dry_run:
            lock.unlink(missing_ok=True)
    return report
