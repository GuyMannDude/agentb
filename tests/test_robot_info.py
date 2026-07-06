"""robot.info drift guard.

A user asked (2026-07-05, via Guy) how the robot.info manifest stays current
with releases. Honest answer at the time: manually, and it was three versions
behind. This test IS the new answer: the suite fails the moment the manifest's
version drifts from pyproject.toml, so a release can't ship without touching
robot.info. Parsing follows ROBOT-INFO-SPEC.md: strip // line comments, then
standard JSON.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_robot_info() -> dict:
    raw = (ROOT / "robot.info").read_text(encoding="utf-8")
    stripped = re.sub(r"^\s*//.*$", "", raw, flags=re.M)
    return json.loads(stripped)


def _pyproject_version() -> str:
    raw = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', raw, flags=re.M)
    assert m, "pyproject.toml has no version line"
    return m.group(1)


def test_robot_info_parses_per_spec():
    info = _load_robot_info()
    for field in ("robot_info_version", "name", "version", "license", "source"):
        assert isinstance(info.get(field), str) and info[field], f"missing/empty {field}"


def test_robot_info_version_matches_release():
    info = _load_robot_info()
    assert info["version"] == _pyproject_version(), (
        f'robot.info says {info["version"]} but pyproject.toml says '
        f"{_pyproject_version()} — update robot.info as part of the version bump "
        "(and consider whether capabilities/endpoints need the new feature listed)."
    )
