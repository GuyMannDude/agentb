"""passport/git_helper.py against a real tmp_path repo (clean-room review H10).

The helper auto-commits every passport mutation; a regression here silently
drops the audit trail's git anchor (commit_sha) on every observe/promote.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def passport_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MNEMO_PASSPORT_DIR", str(tmp_path))
    from passport import config as config_mod
    config_mod.reload()
    yield tmp_path
    config_mod.reload()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_ensure_repo_initializes_once_with_identity(passport_dir):
    from passport import git_helper

    root = git_helper.ensure_repo()
    assert (root / ".git").is_dir()
    assert _git(root, "config", "--local", "--get", "user.name") == git_helper.FALLBACK_NAME
    assert _git(root, "config", "--local", "--get", "user.email") == git_helper.FALLBACK_EMAIL
    # No auto-commit of the bootstrap layout — the first real commit owns it.
    assert _git(root, "status", "--porcelain")

    # Idempotent: a second call must not re-init or touch the repo.
    head_dir_before = sorted((root / ".git").iterdir())
    assert git_helper.ensure_repo() == root
    assert sorted((root / ".git").iterdir()) == head_dir_before


def test_commit_returns_sha_and_records_message(passport_dir):
    from passport import git_helper

    sha = git_helper.commit("observe", "obs_000001", "Prefers terse comments")
    assert sha and len(sha) == 40
    root = git_helper.ensure_repo()
    assert _git(root, "rev-parse", "HEAD") == sha
    msg = _git(root, "log", "-1", "--format=%s")
    assert msg == "passport: observe obs_000001 — Prefers terse comments"
    # Everything staged and committed — clean tree.
    assert _git(root, "status", "--porcelain") == ""


def test_commit_with_no_changes_returns_none(passport_dir):
    from passport import git_helper

    first = git_helper.commit("observe", "obs_000001", "first")
    assert first
    assert git_helper.commit("observe", "obs_000001", "nothing changed") is None
    root = git_helper.ensure_repo()
    assert _git(root, "rev-parse", "HEAD") == first


def test_commit_message_truncates_and_flattens_description(passport_dir):
    from passport import git_helper

    desc = ("line one\nline two " + "x" * 100)
    sha = git_helper.commit("promote", "clm_000001", desc)
    assert sha
    root = git_helper.ensure_repo()
    msg = _git(root, "log", "-1", "--format=%s")
    assert "\n" not in msg
    flattened = desc[:60].replace("\n", " ").strip()
    assert msg == f"passport: promote clm_000001 — {flattened}"


def test_commit_never_configures_a_remote(passport_dir):
    # "Never pushes" starts with having nowhere to push.
    from passport import git_helper

    git_helper.commit("observe", "obs_000001", "claim")
    root = git_helper.ensure_repo()
    remotes = subprocess.run(
        ["git", "-C", str(root), "remote"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert remotes == ""
