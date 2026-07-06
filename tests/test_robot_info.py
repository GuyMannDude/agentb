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


def test_served_versions_match_release():
    # v4.9.1 and v4.9.2 both shipped follow-up commits solely to fix these two
    # hardcoded strings; v4.9.4's review caught the same miss a third time.
    # This guard makes the suite fail instead: every version="X.Y.Z" literal in
    # agentb/server.py and agentb/cli.py must match pyproject.toml.
    release = _pyproject_version()
    for rel_path in ("agentb/server.py", "agentb/cli.py"):
        src = (ROOT / rel_path).read_text(encoding="utf-8")
        found = re.findall(r'version="(\d+\.\d+\.\d+)"', src)
        assert found, f"{rel_path}: expected at least one hardcoded version string"
        for v in found:
            assert v == release, (
                f'{rel_path} serves version "{v}" but pyproject.toml says '
                f'"{release}" — bump every hardcoded version string in the bump '
                "commit (server.py app+/health, cli.py, robot.info, CHANGELOG)."
            )
