"""v4.20.0 Memory Ledger — a tamper-evident chain over the testimony.

Presence: a writeback seals; an edited summary reports `altered`; a removed
file reports `missing`; a dropped or edited ledger line reports `broken`.
Absence: a filing change (category, superseded_by) does NOT alter the seal;
a fresh store verifies `empty` and ok. Control: `seal` adopts unsealed
records but never launders an altered one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, ResilientProviderConfig, ProviderConfig,
    CacheConfig, ServerConfig, ClassificationConfig, DEFAULT_PERSONAS,
    ScopedToken,
)
from agentb.ledger import (
    Ledger, LedgerBroken, GENESIS, content_hash, entry_hash, get_ledger, ledger_path_for,
)

VEC = [0.0] * 768
VEC[0] = 1.0
_STATUS = {"primary": "fake", "active": "fake", "failed_over": False,
           "circuit_open": False, "primary_retry_in": None, "fallback_count": 0}
MASTER = "master-secret"
AL_TOKEN = "al-scoped-secret"


class FakeEmbedding:
    active_label = "fake/embed"
    @property
    def status(self): return _STATUS
    async def embed(self, text, *, use_breaker=True, task_type="document"):
        # Distinct vectors per text so the dedup gate never holds a test write.
        import hashlib
        d = hashlib.sha256(text.encode()).digest()
        v = [0.0] * 768
        v[d[0] * 3 % 768] = 1.0
        v[(d[1] * 3 + 1) % 768] = 1.0
        return v
    async def health_check(self): return True


class FakeReasoning:
    active_label = "fake/reason"
    @property
    def status(self): return _STATUS
    async def generate(self, prompt, system="", max_tokens=2048, *, use_breaker=True): return "topology"
    async def health_check(self): return True


def _cfg(tmp_path, **server_kw):
    return AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic-embed-text")),
        cache=CacheConfig(), server=ServerConfig(host="127.0.0.1", port=50098, **server_kw),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=False),
        personas=dict(DEFAULT_PERSONAS),
    )


@pytest.fixture
def client(tmp_path):
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(_cfg(tmp_path))) as c:
            yield c


def _save(client, agent, summary, session="ledger-test-1", **extra):
    r = client.post("/writeback", json={
        "agent_id": agent, "session_id": session, "summary": summary,
        "key_facts": [f"fact about {summary}"], **extra})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "archived", body
    return body["memory_id"]


def _memory_dir(tmp_path, agent) -> Path:
    return tmp_path / "agents" / agent / "memory"


def _rewrite(path: Path, mutate):
    rec = json.loads(path.read_text(encoding="utf-8"))
    mutate(rec)
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")


# ── pure chain mechanics ──

def test_fresh_store_is_empty_and_ok(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    rep = Ledger(tmp_path / "ledger.jsonl").verify(mem)
    assert rep.chain == "empty" and rep.ok and rep.entries == 0


def test_seal_chains_from_genesis(tmp_path):
    led = Ledger(tmp_path / "ledger.jsonl")
    e1 = led.seal({"id": "a", "summary": "one"})
    e2 = led.seal({"id": "b", "summary": "two"})
    assert e1["prev"] == GENESIS and e1["seq"] == 1
    assert e2["prev"] == e1["hash"] and e2["seq"] == 2
    assert e1["hash"] == entry_hash(e1)


def test_reopened_ledger_continues_the_chain(tmp_path):
    path = tmp_path / "ledger.jsonl"
    e1 = Ledger(path).seal({"id": "a", "summary": "one"})
    e2 = Ledger(path).seal({"id": "b", "summary": "two"})
    assert e2["prev"] == e1["hash"] and e2["seq"] == 2


def test_content_hash_ignores_filing_fields():
    rec = {"id": "a", "summary": "s", "key_facts": ["k"], "category": "decision"}
    h = content_hash(rec)
    rec["category"] = "topology"
    rec["superseded_by"] = "zzz"
    rec["additional_tags"] = ["x"]
    assert content_hash(rec) == h
    rec["summary"] = "s2"
    assert content_hash(rec) != h


def test_edited_ledger_line_breaks_the_chain(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        rec = {"id": f"m{i}", "summary": f"s{i}"}
        (mem / f"m{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        led.seal(rec)
    lines = path.read_text(encoding="utf-8").splitlines()
    doctored = json.loads(lines[1]); doctored["memory_id"] = "m9"
    lines[1] = json.dumps(doctored)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = Ledger(path).verify(mem)
    assert rep.chain == "broken" and rep.broken_at == 2 and not rep.ok
    assert rep.reason and "entry hash" in rep.reason


def test_dropped_ledger_line_breaks_the_chain(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        led.seal({"id": f"m{i}", "summary": f"s{i}"})
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = Ledger(path).verify(mem)
    assert rep.chain == "broken" and rep.broken_at == 2
    assert rep.reason and "sequence gap" in rep.reason


def test_torn_final_line_is_quarantined_by_the_next_seal(tmp_path):
    """A crash mid-append leaves an unterminated fragment. Until the next
    write, verify names it (broken). The next seal moves the fragment to
    ledger.torn, truncates back to the last complete line, and continues
    the chain from the last GOOD entry — ON DISK, not just in the returned
    dict. The record the fragment was for reads unsealed."""
    path = tmp_path / "ledger.jsonl"
    mem = tmp_path / "memory"; mem.mkdir()
    a = {"id": "a", "summary": "one"}; b = {"id": "b", "summary": "two"}; c = {"id": "c", "summary": "three"}
    for r in (a, b, c):
        (mem / f"{r['id']}.json").write_text(json.dumps(r), encoding="utf-8")
    e1 = Ledger(path).seal(a)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "op": "save", "memory_id": "b"')   # crash mid-write
    led = Ledger(path)
    before = led.verify(mem)
    assert before.chain == "broken" and before.reason == "unparseable line"
    e3 = led.seal(c)
    assert e3["prev"] == e1["hash"] and e3["seq"] == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1]) == e3
    torn = (path.with_suffix(".torn")).read_text(encoding="utf-8")
    assert '"memory_id": "b"' in torn and torn.startswith("# quarantined")
    rep = Ledger(path).verify(mem)
    assert rep.chain == "intact" and rep.entries == 2
    assert rep.sealed == 2 and rep.unsealed == ["b"] and rep.altered == []


def test_crash_fragment_does_not_brick_adoption(tmp_path):
    """One crash must not leave a tenant unable to adopt forever: seal
    adopts through it (quarantine first), then the chain is intact."""
    path = tmp_path / "ledger.jsonl"
    mem = tmp_path / "memory"; mem.mkdir()
    a = {"id": "a", "summary": "one"}
    (mem / "a.json").write_text(json.dumps(a), encoding="utf-8")
    Ledger(path).seal(a)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"seq": 2, "op": "sa')                          # crash
    (mem / "carried.json").write_text(json.dumps({"id": "carried", "summary": "off a stick"}), encoding="utf-8")
    led = Ledger(path)
    assert led.seal_unsealed(mem) == ["carried"]
    rep = led.verify(mem)
    assert rep.chain == "intact" and rep.sealed == 2 and rep.ok


def test_bad_line_mid_file_stays_broken_and_is_not_quarantined(tmp_path):
    """Only an UNTERMINATED tail is a crash. Junk in the middle is
    tampering or truncation: it stays, the chain stays broken, adoption
    is refused."""
    path = tmp_path / "ledger.jsonl"
    mem = tmp_path / "memory"; mem.mkdir()
    led = Ledger(path)
    for i in range(3):
        led.seal({"id": f"m{i}", "summary": f"s{i}"})
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[1] = "not json at all"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    led = Ledger(path)
    with pytest.raises(LedgerBroken):
        led.seal_unsealed(mem)
    assert not path.with_suffix(".torn").exists()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_invalid_entry_cannot_vouch_for_a_record(tmp_path):
    """Attack A: edit a record AND rewrite its ledger line in place to match.
    Attack B: append junk (bad prev/hash/seq) re-claiming an EARLIER id.
    Neither may make the record read `sealed`."""
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        rec = {"id": f"m{i}", "summary": f"s{i}"}
        (mem / f"m{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        led.seal(rec)
    # A: forge m1 and re-seal it in place with a self-consistent hash.
    _rewrite(mem / "m1.json", lambda r: r.__setitem__("summary", "forged"))
    lines = path.read_text(encoding="utf-8").splitlines()
    e = json.loads(lines[1])
    e["content_sha256"] = content_hash(json.loads((mem / "m1.json").read_text()))
    e["hash"] = entry_hash(e)
    lines[1] = json.dumps(e)
    # B: append junk claiming m0 with a forged content hash.
    _rewrite(mem / "m0.json", lambda r: r.__setitem__("summary", "also forged"))
    junk = {"seq": 999, "ts": 0, "op": "save", "memory_id": "m0",
            "content_sha256": content_hash(json.loads((mem / "m0.json").read_text())),
            "prev": "deadbeef", "hash": "nope"}
    lines.append(json.dumps(junk))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep = Ledger(path).verify(mem)
    # The doctored m1 line is self-consistent; the break shows at m2 (seq 3,
    # prev mismatch) — so BOTH seq 2 and seq 3 are untrusted.
    assert rep.chain == "broken" and rep.broken_at == 3
    assert rep.disputed == ["m1", "m2"]      # matching an entry the chain cannot vouch for ≠ sealed
    assert rep.altered == ["m0"]             # the trusted seq-1 seal still names the forgery;
    assert rep.sealed == 0                   # the junk claiming m0 could not overrule it
    assert rep.unsealed == []


def test_forgery_after_a_break_is_still_named_altered(tmp_path):
    """Delete one record's ledger line and forge two records. The chain is
    broken; the record whose seal is gone reads `unsealed` (only the break
    tells), the record whose seal survives still reads `altered`, and
    adoption is refused — otherwise "adopt the unsealed" seals the forgery."""
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(4):
        rec = {"id": f"m{i}", "summary": f"s{i}"}
        (mem / f"m{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        led.seal(rec)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]                                             # m1's seal gone
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _rewrite(mem / "m1.json", lambda r: r.__setitem__("summary", "forged"))
    _rewrite(mem / "m3.json", lambda r: r.__setitem__("summary", "also forged"))
    led = Ledger(path)
    rep = led.verify(mem)
    assert rep.chain == "broken" and rep.broken_at == 2
    assert rep.altered == ["m3"], rep            # the later forgery is STILL named
    assert rep.unsealed == ["m1"]                # its seal is gone; only the chain break tells
    assert rep.disputed == ["m2"]                # untouched, but past the break: evidence, not proof
    assert rep.sealed == 1 and rep.entries == 3  # m0, on the trusted side
    with pytest.raises(LedgerBroken):
        led.seal_unsealed(mem)
    assert "m1" not in led.latest_by_memory()    # nothing was adopted


def test_seal_unsealed_adopts_but_never_launders(tmp_path):
    mem = tmp_path / "memory"; mem.mkdir()
    led = Ledger(tmp_path / "ledger.jsonl")
    good = {"id": "good", "summary": "as written"}
    (mem / "good.json").write_text(json.dumps(good), encoding="utf-8")
    led.seal(good)
    _rewrite(mem / "good.json", lambda r: r.__setitem__("summary", "rewritten"))
    (mem / "new.json").write_text(json.dumps({"id": "new", "summary": "carried in"}), encoding="utf-8")
    assert led.seal_unsealed(mem) == ["new"]
    rep = led.verify(mem)
    assert rep.altered == ["good"] and rep.unsealed == [] and rep.sealed == 1
    # Sealing again changes nothing: the altered record stays named.
    assert led.seal_unsealed(mem) == []
    assert led.verify(mem).altered == ["good"]


def test_get_ledger_is_one_instance_per_path(tmp_path):
    mem = tmp_path / "agents" / "x" / "memory"; mem.mkdir(parents=True)
    assert get_ledger(mem) is get_ledger(mem)
    assert ledger_path_for(mem) == (tmp_path / "agents" / "x" / "ledger.jsonl").resolve()


# ── through the server ──

def test_writeback_seals_and_verify_reports_sealed(client, tmp_path):
    mid = _save(client, "cc", "the thing we decided")
    r = client.get("/ledger/verify", params={"agent_id": "cc"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["ok"] and rep["chain"] == "intact"
    assert rep["sealed"] == 1 and rep["entries"] == 1
    assert rep["altered"] == [] and rep["unsealed"] == []
    line = json.loads((tmp_path / "agents" / "cc" / "ledger.jsonl").read_text().splitlines()[0])
    assert line["memory_id"] == mid and line["op"] == "save"


def test_altered_summary_is_named(client, tmp_path):
    mid = _save(client, "cc", "what Guy actually said")
    _rewrite(_memory_dir(tmp_path, "cc") / f"{mid}.json",
             lambda r: r.__setitem__("summary", "what somebody wishes he said"))
    rep = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    assert rep["altered"] == [mid] and not rep["ok"]


def test_filing_change_keeps_the_seal(client, tmp_path):
    mid = _save(client, "cc", "reclassify me later")
    _rewrite(_memory_dir(tmp_path, "cc") / f"{mid}.json",
             lambda r: (r.__setitem__("category", "doctrine"),
                        r.__setitem__("needs_reclassification", True)))
    rep = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    assert rep["ok"] and rep["sealed"] == 1 and rep["altered"] == []


def test_supersede_keeps_both_seals(client, tmp_path):
    old = _save(client, "cc", "the old truth", session="s-old")
    new = _save(client, "cc", "the new truth", session="s-new", supersedes=[old])
    rec = json.loads((_memory_dir(tmp_path, "cc") / f"{old}.json").read_text())
    assert rec["superseded_by"] == new
    rep = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    assert rep["ok"] and rep["sealed"] == 2


def test_missing_sealed_file_is_named(client, tmp_path):
    mid = _save(client, "cc", "soon to vanish")
    (_memory_dir(tmp_path, "cc") / f"{mid}.json").unlink()
    rep = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    assert rep["missing"] == [mid] and not rep["ok"]


def test_seal_endpoint_adopts_carried_records(client, tmp_path):
    _save(client, "cc", "born here")
    mem = _memory_dir(tmp_path, "cc")
    carried = {"id": "abcdef0123456789", "session_id": "stick-carry", "agent_id": "cc",
               "summary": "arrived on a stick", "key_facts": [], "timestamp": "2026-09-01T00:00:00+00:00"}
    (mem / f"{carried['id']}.json").write_text(json.dumps(carried), encoding="utf-8")
    before = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    assert before["unsealed"] == [carried["id"]] and before["ok"]   # unsealed is not a failure
    r = client.post("/ledger/seal", json={"agent_id": "cc"})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["sealed_now"] == 1 and rep["sealed_ids"] == [carried["id"]]
    assert rep["sealed"] == 2 and rep["unsealed"] == [] and rep["ok"]


def test_tenants_have_separate_chains(client, tmp_path):
    _save(client, "cc", "cc's memory")
    _save(client, "rocky", "rocky's memory")
    cc = client.get("/ledger/verify", params={"agent_id": "cc"}).json()
    rk = client.get("/ledger/verify", params={"agent_id": "rocky"}).json()
    assert cc["entries"] == 1 and rk["entries"] == 1
    assert (tmp_path / "agents" / "cc" / "ledger.jsonl").exists()
    assert (tmp_path / "agents" / "rocky" / "ledger.jsonl").exists()


def test_multi_tenant_requires_agent_id(client):
    _save(client, "cc", "named tenant exists now")
    r = client.get("/ledger/verify")
    assert r.status_code == 400


def test_scoped_token_pins_the_ledger(tmp_path):
    cfg = _cfg(tmp_path, auth_token=MASTER, scoped_tokens=[
        ScopedToken(token=AL_TOKEN, agent_id="al",
                    endpoints=["/writeback", "/ledger/verify", "/ledger/seal"])])
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=FakeReasoning()):
        from agentb.server import create_app
        with TestClient(create_app(cfg)) as c:
            al = {"Authorization": f"Bearer {AL_TOKEN}"}
            r = c.post("/writeback", json={"agent_id": "al", "session_id": "s", "summary": "al's words",
                                           "key_facts": ["k"]}, headers=al)
            assert r.status_code == 200, r.text
            # Omitted agent_id resolves to the pin — and the report names it.
            r = c.get("/ledger/verify", headers=al)
            assert r.status_code == 200 and r.json()["sealed"] == 1
            assert r.json()["agent_id"] == "al"
            r = c.post("/ledger/seal", json={}, headers=al)
            assert r.status_code == 200 and r.json()["agent_id"] == "al"
            # Another tenant is refused.
            assert c.get("/ledger/verify", params={"agent_id": "cc"}, headers=al).status_code == 403
            assert c.post("/ledger/seal", json={"agent_id": "cc"}, headers=al).status_code == 403
            # No token at all is refused.
            assert c.get("/ledger/verify", params={"agent_id": "al"}).status_code == 401


def test_seal_endpoint_refuses_a_broken_chain(client, tmp_path):
    mid = _save(client, "cc", "sealed then attacked")
    path = tmp_path / "agents" / "cc" / "ledger.jsonl"
    path.write_text("", encoding="utf-8")                    # every seal deleted
    # Server still holds its in-memory tail; the on-disk chain now has a
    # gap the moment it writes again. Add an attacker-forged record too.
    _rewrite(_memory_dir(tmp_path, "cc") / f"{mid}.json",
             lambda r: r.__setitem__("summary", "forged"))
    _save(client, "cc", "a later honest save")
    r = client.post("/ledger/seal", json={"agent_id": "cc"})
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["chain"] == "broken" and detail["agent_id"] == "cc"
    assert mid in detail["unsealed"]                         # not adopted
    assert (path.read_text().count("\n")) == 1              # only the honest save's line


# ── CLI ──

def test_cli_all_includes_config_relocated_tenant(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from agentb.config import AgentConfig
    import agentb.config as config_mod
    from agentb.cli import main
    near = tmp_path / "agents" / "near" / "memory"; near.mkdir(parents=True)
    far = tmp_path / "elsewhere" / "far-store"; (far / "memory").mkdir(parents=True)
    (far / "memory" / "x.json").write_text(json.dumps({"id": "x", "summary": "far away"}), encoding="utf-8")
    cfg = AgentBConfig(data_dir=str(tmp_path), agents={"far": AgentConfig(data_dir=str(far))})
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)
    res = CliRunner().invoke(main, ["ledger", "verify", "--all", "--json"])
    assert res.exit_code == 0, res.output
    reports = json.loads(res.output)
    assert set(reports) == {"near", "far"}
    assert reports["far"]["unsealed"] == ["x"]


def test_cli_all_with_no_tenants_is_an_error(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import agentb.config as config_mod
    from agentb.cli import main
    cfg = AgentBConfig(data_dir=str(tmp_path / "empty"))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)
    res = CliRunner().invoke(main, ["ledger", "verify", "--all"])
    assert res.exit_code == 1 and "No tenants" in res.output


def test_cli_seal_refuses_disk_write_when_server_state_is_ambiguous(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import httpx
    import agentb.config as config_mod
    from agentb.cli import main
    mem = tmp_path / "agents" / "cc" / "memory"; mem.mkdir(parents=True)
    (mem / "r.json").write_text(json.dumps({"id": "r", "summary": "unsealed"}), encoding="utf-8")
    cfg = AgentBConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)

    def timeout(*a, **k): raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(httpx, "get", timeout)
    res = CliRunner().invoke(main, ["ledger", "seal", "--agent", "cc"])
    assert res.exit_code == 1 and "Refusing" in res.output
    assert not (tmp_path / "agents" / "cc" / "ledger.jsonl").exists()

    # Connection refused = down: the disk branch runs and seals.
    def refused(*a, **k): raise httpx.ConnectError("refused")
    monkeypatch.setattr(httpx, "get", refused)
    res = CliRunner().invoke(main, ["ledger", "seal", "--agent", "cc"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "agents" / "cc" / "ledger.jsonl").read_text().count("\n") == 1


def test_replayed_low_seq_after_a_break_cannot_enter_the_prefix(tmp_path):
    """The cheapest attack: append one line re-claiming an earlier id with a
    LOW seq (inside the prefix) and junk prev/hash. Trust is positional, so
    it lands on the untrusted side and the trusted verdict stands."""
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(5):
        rec = {"id": f"m{i}", "summary": f"s{i}"}
        (mem / f"m{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        led.seal(rec)
    _rewrite(mem / "m0.json", lambda r: r.__setitem__("summary", "forged"))
    junk = {"seq": 2, "op": "save", "memory_id": "m0",
            "content_sha256": content_hash(json.loads((mem / "m0.json").read_text())),
            "prev": "junk", "hash": "junk"}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(junk) + "\n")
    rep = Ledger(path).verify(mem)
    assert rep.chain == "broken" and rep.broken_at == 6
    assert rep.altered == ["m0"] and rep.sealed == 4 and rep.disputed == []


def test_unterminated_but_complete_final_line_is_not_a_crash(tmp_path):
    """An editor or a join-rewrite strips the last newline. That is a
    complete entry, not a fragment: nothing is quarantined, the next seal
    starts on its own line, and the chain stays intact."""
    mem = tmp_path / "memory"; mem.mkdir()
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    for i in range(3):
        rec = {"id": f"m{i}", "summary": f"s{i}"}
        (mem / f"m{i}.json").write_text(json.dumps(rec), encoding="utf-8")
        led.seal(rec)
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    m3 = {"id": "m3", "summary": "s3"}
    (mem / "m3.json").write_text(json.dumps(m3), encoding="utf-8")
    e4 = Ledger(path).seal(m3)
    assert e4["seq"] == 4
    assert not path.with_suffix(".torn").exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and json.loads(lines[3]) == e4
    rep = Ledger(path).verify(mem)
    assert rep.chain == "intact" and rep.sealed == 4 and rep.ok


def test_cli_all_skips_archived_tenant_dirs_out_loud(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import agentb.config as config_mod
    from agentb.cli import main
    for name in ("rocky", "rocky.archived-20260516", "_doctor_test.archived-20260516"):
        (tmp_path / "agents" / name / "memory").mkdir(parents=True)
    (tmp_path / "agents" / "rocky.archived-20260516" / "memory" / "r.json").write_text(
        json.dumps({"id": "r", "summary": "old"}), encoding="utf-8")
    cfg = AgentBConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)
    res = CliRunner().invoke(main, ["ledger", "verify", "--all"])
    assert res.exit_code == 0, res.output
    assert "skipped rocky.archived-20260516: not a valid agent_id" in res.output
    assert "skipped _doctor_test.archived-20260516" in res.output
    assert "\n  rocky:" in res.output or "rocky: chain" in res.output


def test_cli_seal_all_continues_past_one_refused_tenant(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import httpx
    import agentb.config as config_mod
    from agentb.cli import main
    for name in ("aa", "bb"):
        (tmp_path / "agents" / name / "memory").mkdir(parents=True)
    cfg = AgentBConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)

    class R:
        def __init__(self, code, body): self.status_code, self._b = code, body
        @property
        def text(self): return json.dumps(self._b)
        def json(self): return self._b
    posted = []
    def fake_get(*a, **k): return R(200, {"status": "ok"})
    def fake_post(url, json=None, headers=None, timeout=None):
        posted.append(json["agent_id"])
        if json["agent_id"] == "aa":
            return R(400, {"detail": "Invalid agent_id"})
        return R(200, {"agent_id": "bb", "sealed_now": 0, "sealed_ids": [], "ok": True,
                       "chain": "empty", "entries": 0, "broken_at": None, "reason": None,
                       "sealed": 0, "altered": [], "missing": [], "unsealed": [], "disputed": []})
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    res = CliRunner().invoke(main, ["ledger", "seal", "--all"])
    assert posted == ["aa", "bb"]          # bb still ran after aa's 400
    assert res.exit_code == 1 and "server said 400" in res.output


def test_cli_seal_all_continues_past_a_transport_error(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import httpx
    import agentb.config as config_mod
    from agentb.cli import main
    for name in ("aa", "bb"):
        (tmp_path / "agents" / name / "memory").mkdir(parents=True)
    cfg = AgentBConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)

    class R:
        status_code = 200
        text = "{}"
        def json(self): return {"agent_id": "bb", "sealed_now": 0, "sealed_ids": [], "ok": True,
                                "chain": "empty", "entries": 0, "broken_at": None, "reason": None,
                                "sealed": 0, "altered": [], "missing": [], "unsealed": [], "disputed": []}
    posted, timeouts = [], []
    def fake_get(*a, **k): return R()
    def fake_post(url, json=None, headers=None, timeout=None):
        posted.append(json["agent_id"]); timeouts.append(timeout)
        if json["agent_id"] == "aa":
            raise httpx.ReadTimeout("timed out")
        return R()
    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)
    res = CliRunner().invoke(main, ["ledger", "seal", "--all"])
    assert posted == ["aa", "bb"]
    assert res.exit_code == 1 and "ReadTimeout" in res.output and "rerun" in res.output
    assert all(t.read is None for t in timeouts)      # no read cap on the adopt


def test_cli_verify_json_is_pure_with_archived_dirs_present(tmp_path, monkeypatch):
    from click.testing import CliRunner
    import agentb.config as config_mod
    from agentb.cli import main
    for name in ("rocky", "rocky.archived-20260516"):
        (tmp_path / "agents" / name / "memory").mkdir(parents=True)
    cfg = AgentBConfig(data_dir=str(tmp_path))
    monkeypatch.setattr(config_mod, "load_config", lambda path=None: cfg)
    res = CliRunner().invoke(main, ["ledger", "verify", "--all", "--json"])
    assert res.exit_code == 0, res.output
    reports = json.loads(res.stdout)                  # would raise if a notice leaked
    assert set(reports) == {"rocky"}
    assert "skipped rocky.archived-20260516" in res.stderr
