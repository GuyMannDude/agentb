import importlib.util
import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

MODULE_PATH = Path(__file__).parents[1] / "integrations" / "chatgpt" / "server.py"
SPEC = importlib.util.spec_from_file_location("chatgpt_gate_server", MODULE_PATH)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

GATE = "g" * 48
UPSTREAM = "u" * 48


def _mock_upstream(calls, status=200, response=None):
    def handler(request):
        calls.append({"path": request.url.path, "headers": request.headers,
                      "json": json.loads(request.content)})
        return httpx.Response(status, json=response or {"status": "ok"})
    return httpx.MockTransport(handler)


def _client(tmp_path, calls=None, rate=10, status=200, **kwargs):
    calls = calls if calls is not None else []
    app = gate.create_app(
        gate_token=GATE, upstream_token=UPSTREAM,
        audit_file=tmp_path / "audit.jsonl", rate_limit=rate,
        transport=_mock_upstream(calls, status=status), **kwargs,
    )
    return TestClient(app), calls


def _auth(token=GATE):
    return {"Authorization": f"Bearer {token}"}


def test_auth_unknown_route_and_audit(tmp_path):
    client, _ = _client(tmp_path)
    assert client.post("/recall", json={"prompt": "x"}).status_code == 401
    assert client.post("/recall", json={"prompt": "x"}, headers=_auth(UPSTREAM)).status_code == 401
    assert client.post("/facts", json={}, headers=_auth()).status_code == 404
    assert client.post("/recall/", json={"prompt": "x"}, headers=_auth()).status_code == 404
    rows = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [row["status"] for row in rows] == [401, 401, 404, 404]


def test_recall_forces_tenant(tmp_path):
    client, calls = _client(tmp_path)
    result = client.post("/recall", headers=_auth(), json={
        "prompt": "genesis", "agent_id": "other-agent", "category": "identity",
    })
    assert result.status_code == 200
    assert calls[0]["path"] == "/context"
    assert calls[0]["json"]["agent_id"] == "chatgpt"
    assert calls[0]["headers"]["x-api-key"] == UPSTREAM


def test_pinned_tenant_is_configurable(tmp_path):
    client, calls = _client(tmp_path, agent_id="assistant-two")
    assert client.post("/recall", headers=_auth(), json={"prompt": "x"}).status_code == 200
    assert calls[0]["json"]["agent_id"] == "assistant-two"


def test_save_forces_provenance_and_rejects_category(tmp_path):
    client, calls = _client(tmp_path)
    payload = {"session_id": "gpt-test-1", "summary": "remember this",
               "category": "decision", "agent_id": "other-agent",
               "additional_tags": ["one", "chatgpt-gate"]}
    assert client.post("/save", headers=_auth(), json=payload).status_code == 200
    saved = calls[0]["json"]
    assert saved["agent_id"] == "chatgpt"
    assert saved["source"] == "user"
    assert saved["additional_tags"] == ["one", "chatgpt-gate"]
    payload["category"] = "topology"
    assert client.post("/save", headers=_auth(), json=payload).status_code == 422
    assert len(calls) == 1


def test_rate_limit_is_ten_per_hour(tmp_path):
    client, calls = _client(tmp_path, rate=10)
    for n in range(10):
        assert client.post("/recall", headers=_auth(), json={"prompt": str(n)}).status_code == 200
    response = client.post("/recall", headers=_auth(), json={"prompt": "blocked"})
    assert response.status_code == 429
    assert len(calls) == 10


def test_save_body_cap_header_and_stream(tmp_path):
    client, calls = _client(tmp_path)
    huge = b'{' + b'"summary":"' + b'x' * 9000 + b'"}'
    assert client.post("/save", headers={**_auth(), "Content-Type": "application/json"},
                       content=huge).status_code == 413
    assert not calls


def test_recall_body_cap_header_and_stream(tmp_path):
    client, calls = _client(tmp_path)
    huge = b'{' + b'"prompt":"' + b'x' * 9000 + b'"}'
    response = client.post("/recall", headers={**_auth(), "Content-Type": "application/json"},
                           content=huge)
    assert response.status_code == 413
    assert not calls


def test_audit_rotates_and_keeps_current_request(tmp_path):
    path = tmp_path / "audit.jsonl"
    audit = gate.AuditLog(path, rotate_bytes=150)
    audit.path.write_text("x" * 149, encoding="utf-8")
    request = type("AuditRequest", (), {
        "url": type("Url", (), {"path": "/recall"})(),
        "state": type("State", (), {"body_size": 1, "snippet": "x"})(),
    })()
    audit.append(request, 401)
    assert path.with_name("audit.jsonl.1").stat().st_size == 149
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == 401


def test_upstream_error_is_generic(tmp_path):
    client, _ = _client(tmp_path, status=500)
    response = client.post("/recall", headers=_auth(), json={"prompt": "x"})
    assert response.status_code == 502
    assert response.json() == {"detail": "Memory service rejected the request"}


def test_recall_rejects_disallowed_category(tmp_path):
    client, calls = _client(tmp_path)
    response = client.post("/recall", headers=_auth(),
                           json={"prompt": "x", "category": "doctrine"})
    assert response.status_code == 422
    assert not calls
