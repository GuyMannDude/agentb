"""v4.19 Pocket Mnemo: POST /chat (recall -> reason -> save in one tenant) and
the /app/ shell the phone opens.

End-to-end through create_app + TestClient with fake providers. The lid test
is the whole product: tell it something, open a new session, it remembers.
"""
import hashlib
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agentb.config import (
    AgentBConfig, AgentConfig, CacheConfig, ClassificationConfig, ProviderConfig,
    ResilientProviderConfig, ServerConfig, ScopedToken, DEFAULT_PERSONAS,
    SCOPABLE_ENDPOINTS,
)

MASTER = "master-secret"
POCKET_TOKEN = "pocket-scoped-secret"
_STATUS = {"primary": "fake", "active": "fake", "failed_over": False,
           "circuit_open": False, "primary_retry_in": None, "fallback_count": 0}


class FakeEmbedding:
    """Deterministic, content-sensitive: the same words land near each other."""
    active_label = "fake/embed"

    @property
    def status(self):
        return _STATUS

    async def embed(self, text, *, use_breaker=True, task_type="document"):
        vec = [0.0] * 768
        for word in text.lower().split():
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 768] += 1.0
        return vec

    async def health_check(self):
        return True


class FakeReasoning:
    active_label = "fake/reason"

    def __init__(self, reply="I'll remember that.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    @property
    def status(self):
        return _STATUS

    async def generate(self, prompt, system="", max_tokens=2048, *, use_breaker=True):
        self.calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        if self.fail:
            raise RuntimeError("All reasoning providers failed")
        return self.reply

    async def health_check(self):
        return True


def _make(tmp_path, reasoner, scoped=None, agents=None, classify=False):
    cfg = AgentBConfig(
        reasoning=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="x")),
        embedding=ResilientProviderConfig(primary=ProviderConfig(provider="ollama", model="nomic-embed-text")),
        cache=CacheConfig(),
        server=ServerConfig(port=50097, auth_token=MASTER, scoped_tokens=scoped or []),
        data_dir=str(tmp_path),
        classification=ClassificationConfig(enabled=classify),
        personas=dict(DEFAULT_PERSONAS),
        agents=agents or {},
    )
    with patch("agentb.server.create_resilient_embedding", return_value=FakeEmbedding()), \
         patch("agentb.server.create_resilient_reasoning", return_value=reasoner):
        from agentb.server import create_app
        return create_app(cfg)


def _auth(token):
    return {"X-API-KEY": token}


@pytest.fixture
def reasoner():
    return FakeReasoning()


@pytest.fixture
def client(tmp_path, reasoner):
    with TestClient(_make(tmp_path, reasoner)) as c:
        yield c


def _memory_files(tmp_path, agent):
    return sorted((tmp_path / "agents" / agent / "memory").glob("*.json"))


# ── the lid test ─────────────────────────────────────────────────────────

def test_chat_saves_the_turn_and_remembers_it_next_session(client, reasoner, tmp_path):
    r = client.post("/chat", json={"agent_id": "pocket", "message": "my locker code is 4471"},
                    headers=_auth(MASTER))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["reply"] == "I'll remember that."
    assert j["agent_id"] == "pocket"
    assert j["save_status"] == "archived" and j["memory_id"]
    assert j["session_id"].startswith("pocket-")
    assert j["memories"] == []                      # nothing to remember yet
    files = _memory_files(tmp_path, "pocket")
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["source"] == "user" and "pocket" in saved["additional_tags"]
    assert saved["summary"] == "my locker code is 4471"          # the person's words, only
    assert saved["key_facts"] == ["Pocket replied: I'll remember that."]
    assert "no memories of this person yet" in reasoner.calls[0]["prompt"]

    # lid closed: a NEW session, no history — the engine still has it
    r = client.post("/chat", json={"agent_id": "pocket", "message": "what is my locker code?",
                                   "save": False},
                    headers=_auth(MASTER))
    assert r.status_code == 200, r.text
    j = r.json()
    assert any("4471" in m["content"] for m in j["memories"])
    assert "4471" in reasoner.calls[1]["prompt"]
    assert "What you remember about this person" in reasoner.calls[1]["prompt"]
    assert j["save_status"] == "skipped" and j["memory_id"] == ""
    assert len(_memory_files(tmp_path, "pocket")) == 1


def test_chat_history_rides_into_the_prompt(client, reasoner):
    r = client.post("/chat", json={
        "agent_id": "pocket", "message": "and the second one?", "save": False,
        "history": [{"role": "user", "content": "I have two bikes"},
                    {"role": "assistant", "content": "Noted. What are they?"}],
    }, headers=_auth(MASTER))
    assert r.status_code == 200, r.text
    p = reasoner.calls[-1]["prompt"]
    assert "Conversation so far:" in p
    assert "User: I have two bikes" in p and "You: Noted. What are they?" in p
    assert p.rstrip().endswith("User: and the second one?\nYou:")
    assert reasoner.calls[-1]["system"].startswith("You are Pocket Mnemo")


def test_chat_keeps_the_callers_session_id(client):
    r = client.post("/chat", json={"agent_id": "pocket", "message": "hi", "save": False,
                                   "session_id": "pocket-lid-7"}, headers=_auth(MASTER))
    assert r.status_code == 200 and r.json()["session_id"] == "pocket-lid-7"
    r = client.post("/chat", json={"agent_id": "pocket", "message": "hi", "save": False,
                                   "session_id": "../escape"}, headers=_auth(MASTER))
    assert r.status_code == 400


def test_chat_refuses_blank_message(client):
    r = client.post("/chat", json={"agent_id": "pocket", "message": "   "}, headers=_auth(MASTER))
    assert r.status_code == 422


def test_chat_reasoning_failure_is_503_not_a_saved_turn(tmp_path):
    with TestClient(_make(tmp_path, FakeReasoning(fail=True))) as c:
        r = c.post("/chat", json={"agent_id": "pocket", "message": "remember my dog is Rex"},
                   headers=_auth(MASTER))
        assert r.status_code == 503
        assert _memory_files(tmp_path, "pocket") == []


def test_chat_read_only_tenant_refused_before_reasoning(tmp_path):
    reasoner = FakeReasoning()
    agents = {"shared": AgentConfig(read_only=True)}
    with TestClient(_make(tmp_path, reasoner, agents=agents)) as c:
        r = c.post("/chat", json={"agent_id": "shared", "message": "x"}, headers=_auth(MASTER))
        assert r.status_code == 403
        assert reasoner.calls == []
        r = c.post("/chat", json={"agent_id": "shared", "message": "x", "save": False},
                   headers=_auth(MASTER))
        assert r.status_code == 200          # reading a read-only lane is fine


def test_chat_no_agent_on_multitenant_install_is_400(tmp_path):
    agents = {"cc": AgentConfig(), "rocky": AgentConfig()}
    with TestClient(_make(tmp_path, FakeReasoning(), agents=agents)) as c:
        r = c.post("/chat", json={"message": "hi"}, headers=_auth(MASTER))
        assert r.status_code == 400


def test_chat_save_status_held_on_a_repeat(client, tmp_path):
    body = {"agent_id": "pocket", "message": "my sister's birthday is the 14th of March"}
    assert client.post("/chat", json=body, headers=_auth(MASTER)).json()["save_status"] == "archived"
    j = client.post("/chat", json=body, headers=_auth(MASTER)).json()
    assert j["save_status"] == "held" and j["memory_id"] == ""
    assert j["reply"]                                    # the reply still comes back
    assert len(_memory_files(tmp_path, "pocket")) == 1


def test_chat_honours_the_capture_pause(client, tmp_path):
    assert client.post("/capture/pause", json={"minutes": 5, "reason": "test"},
                       headers=_auth(MASTER)).status_code == 200
    r = client.post("/chat", json={"agent_id": "pocket", "message": "private thing"},
                    headers=_auth(MASTER))
    assert r.status_code == 200 and r.json()["save_status"] == "paused"
    assert r.json()["reply"] == "I'll remember that."
    assert _memory_files(tmp_path, "pocket") == []
    client.post("/capture/resume", headers=_auth(MASTER))


def test_chat_redacts_before_the_reasoner_sees_it(client, reasoner, tmp_path):
    secret = "abcdef1234567890XYZ"        # matches redact.py generic-assignment
    r = client.post("/chat", json={
        "agent_id": "pocket", "message": f'my api_key = "{secret}" for the weather app',
        "history": [{"role": "user", "content": f"auth_token: {secret}"},
                    {"role": "assistant", "content": "ok"}],
    }, headers=_auth(MASTER))
    assert r.status_code == 200, r.text
    assert secret not in reasoner.calls[-1]["prompt"]
    assert "[REDACTED:" in reasoner.calls[-1]["prompt"]
    saved = json.loads(_memory_files(tmp_path, "pocket")[0].read_text())
    assert secret not in json.dumps(saved)


def test_chat_still_remembers_a_turn_the_classifier_files_as_session_log(tmp_path):
    # Production runs the LLM classifier; a chat-shaped turn can come back as
    # session_log, which /context hides by default. The door must not forget.
    reasoner = FakeReasoning(reply="session_log")
    with TestClient(_make(tmp_path, reasoner, classify=True)) as c:
        r = c.post("/chat", json={"agent_id": "pocket", "message": "my bus pass number is 88213"},
                   headers=_auth(MASTER))
        assert r.status_code == 200, r.text
        saved = json.loads(_memory_files(tmp_path, "pocket")[0].read_text())
        assert saved["category"] == "session_log"            # the trap is real
        r = c.post("/chat", json={"agent_id": "pocket", "message": "what is my bus pass number?",
                                  "save": False}, headers=_auth(MASTER))
        assert any("88213" in m["content"] for m in r.json()["memories"])


def test_chat_validates_agent_id_even_when_nothing_touches_the_store(client):
    r = client.post("/chat", json={"agent_id": "../etc", "message": "hi", "save": False,
                                   "max_memories": 0}, headers=_auth(MASTER))
    assert r.status_code == 400


# ── the phone's token: scoped, pinned, kill-switchable ───────────────────

def test_chat_is_scopable():
    assert "/chat" in SCOPABLE_ENDPOINTS


@pytest.fixture
def scoped_client(tmp_path):
    scoped = [ScopedToken(token=POCKET_TOKEN, agent_id="pocket", endpoints=["/chat"])]
    with TestClient(_make(tmp_path, FakeReasoning(), scoped=scoped)) as c:
        yield c


def test_scoped_token_fills_its_own_tenant(scoped_client, tmp_path):
    r = scoped_client.post("/chat", json={"message": "I'm learning guitar"},
                           headers=_auth(POCKET_TOKEN))
    assert r.status_code == 200, r.text
    assert r.json()["agent_id"] == "pocket"
    assert len(_memory_files(tmp_path, "pocket")) == 1
    assert not (tmp_path / "agents" / "default").exists()


def test_scoped_token_cannot_name_another_tenant(scoped_client):
    r = scoped_client.post("/chat", json={"agent_id": "cc", "message": "hi"},
                           headers=_auth(POCKET_TOKEN))
    assert r.status_code == 403


def test_scoped_token_cannot_call_context_or_writeback_directly(scoped_client):
    r = scoped_client.post("/context", json={"agent_id": "pocket", "prompt": "hi"},
                           headers=_auth(POCKET_TOKEN))
    assert r.status_code == 403
    r = scoped_client.post("/writeback", json={"agent_id": "pocket", "summary": "x",
                                               "session_id": "s"},
                           headers=_auth(POCKET_TOKEN))
    assert r.status_code == 403


def test_chat_without_token_is_401(client):
    assert client.post("/chat", json={"agent_id": "pocket", "message": "hi"}).status_code == 401


# ── the door: /app/ is public shell, nothing else ─────────────────────────

def test_app_shell_needs_no_token(client):
    r = client.get("/app/")
    assert r.status_code == 200
    assert "Pocket Mnemo" in r.text and 'fetch(\'/chat\'' in r.text
    assert r.headers["content-type"].startswith("text/html")
    assert client.get("/app", follow_redirects=False).status_code == 307
    for asset, ctype in (("manifest.webmanifest", "application/manifest+json"),
                         ("sw.js", "text/javascript"), ("icon.svg", "image/svg+xml")):
        r = client.get(f"/app/{asset}")
        assert r.status_code == 200, asset
        assert r.headers["content-type"].startswith(ctype), asset


def test_app_serves_only_the_allowlist(client):
    assert client.get("/app/index.html").status_code == 404      # only via /app/
    assert client.get("/app/..%2Fserver.py").status_code == 404
    assert client.get("/app/nope.js").status_code == 404
    # the bypass is /app/* only — the API next door still needs a token
    assert client.get("/sessions").status_code == 401
    assert client.get("/apple").status_code == 401


def test_app_shell_holds_no_secret(client):
    text = client.get("/app/").text
    assert MASTER not in text and "auth_token" not in text
