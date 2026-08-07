"""Regression: a literal "~" in YAML path fields must expand at parse time.

YAML never expands "~", and Path("~/x") is RELATIVE — before this fix a
configured data_dir of "~/.config/agentb/..." was created under the process
CWD as a directory literally named "~" (the stray-tilde bug: four specimens
on IGOR, 2026-03..2026-08, planted by test runs and module import of
agentb.server from assorted working directories).
"""
from pathlib import Path

from agentb.config import _parse_config, get_agent_data_dir


def test_agent_data_dir_tilde_expands():
    cfg = _parse_config({"agents": {"a1": {"data_dir": "~/agentb-test/a1"}}})
    p = get_agent_data_dir(cfg, "a1")
    assert p.is_absolute()
    assert p == Path.home() / "agentb-test" / "a1"


def test_top_level_data_dir_tilde_expands():
    cfg = _parse_config({"data_dir": "~/agentb-test"})
    assert Path(cfg.data_dir).is_absolute()
    assert Path(cfg.data_dir) == Path.home() / "agentb-test"


def test_storage_path_tilde_expands():
    cfg = _parse_config({"storage": {"path": "~/agentb-test/store"}})
    assert Path(cfg.storage.path) == Path.home() / "agentb-test" / "store"


def test_empty_agent_data_dir_stays_empty():
    # Path("").expanduser() would become "." — empty must stay falsy so
    # get_agent_data_dir falls through to <data_dir>/agents/<id>.
    cfg = _parse_config({"agents": {"a1": {"persona": "default"}}})
    assert cfg.agents["a1"].data_dir == ""


def test_absolute_data_dir_passes_through_unchanged():
    # The deployed invariant: existing configs with absolute paths must be
    # byte-identical after parse.
    cfg = _parse_config({"data_dir": "/srv/agentb-data"})
    assert cfg.data_dir == "/srv/agentb-data"


def test_missing_data_dir_still_defaults_to_home_agentb():
    cfg = _parse_config({})
    assert Path(cfg.data_dir) == Path.home() / ".agentb"


def test_empty_storage_path_inherits_expanded_data_dir():
    cfg = _parse_config({"data_dir": "~/agentb-test"})
    assert Path(cfg.storage.path) == Path.home() / "agentb-test"


def test_agentb_config_env_var_tilde_expands(monkeypatch, tmp_path):
    # systemd Environment= lines don't expand "~"; a tilde in AGENTB_CONFIG
    # must not silently fall through to the all-defaults config.
    from agentb.config import load_config
    cfg_file = tmp_path / "agentb.yaml"
    cfg_file.write_text("data_dir: /srv/from-env-config\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTB_CONFIG", "~/agentb.yaml")
    cfg = load_config()
    assert cfg.data_dir == "/srv/from-env-config"
