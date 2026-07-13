"""AL Gate: two-route, tenant-pinned facade for Mnemo Cortex."""

import asyncio
import hmac
import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

UPSTREAM_URL = os.environ.get("AL_GATE_UPSTREAM_URL", "http://127.0.0.1:50001").rstrip("/")
GATE_TOKEN_FILE = Path(os.environ.get("AL_GATE_TOKEN_FILE", "~/.al-gate/token")).expanduser()
UPSTREAM_TOKEN_FILE = Path(os.environ.get("AL_GATE_UPSTREAM_TOKEN_FILE", "~/.mnemo-auth-token")).expanduser()
AUDIT_FILE = Path(os.environ.get("AL_GATE_AUDIT_FILE", "~/.al-gate/audit.jsonl")).expanduser()
RATE_LIMIT = int(os.environ.get("AL_GATE_RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = 3600
SAVE_BODY_LIMIT = 8 * 1024
ALLOWED_CATEGORIES = {"session_log", "idea", "decision", "identity", "relationship"}
_SAFE_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _read_secret(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError) as exc:
        raise RuntimeError(f"{label} file unavailable: {path}") from exc
    if len(value) < 32:
        raise RuntimeError(f"{label} is missing or too short")
    return value


class RecallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    agent_id: str | None = None  # accepted but always ignored
    max_results: int = Field(default=5, ge=1, le=10)
    category: str | None = None
    mode: Literal["focus", "explore"] = "focus"


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=6000)
    key_facts: list[str] = Field(default_factory=list, max_length=30)
    category: str
    additional_tags: list[str] = Field(default_factory=list, max_length=20)
    agent_id: str | None = None  # accepted but always ignored


class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT, window: int = RATE_WINDOW_SECONDS):
        self.limit = limit
        self.window = window
        self._hits: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        now = time.monotonic()
        async with self._lock:
            while self._hits and now - self._hits[0] >= self.window:
                self._hits.popleft()
            if len(self._hits) >= self.limit:
                return False
            self._hits.append(now)
            return True


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def append(self, request: Request, status: int) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "op": request.url.path,
            "size": getattr(request.state, "body_size", 0),
            "snippet": getattr(request.state, "snippet", ""),
            "status": status,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=True, separators=(",", ":")) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def create_app(
    *, gate_token: str | None = None, upstream_token: str | None = None,
    upstream_url: str = UPSTREAM_URL, audit_file: Path = AUDIT_FILE,
    rate_limit: int = RATE_LIMIT, transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    gate_token = gate_token or _read_secret(GATE_TOKEN_FILE, "gate token")
    upstream_token = upstream_token or _read_secret(UPSTREAM_TOKEN_FILE, "upstream token")
    limiter = RateLimiter(rate_limit)
    audit = AuditLog(audit_file)
    app = FastAPI(title="AL Mnemo Gate", version="1.0.0", docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.router.redirect_slashes = False

    @app.middleware("http")
    async def audit_every_request(request: Request, call_next):
        try:
            request.state.body_size = max(
                0, int(request.headers.get("content-length", "0") or 0)
            )
        except ValueError:
            request.state.body_size = 0
        request.state.snippet = ""
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            audit.append(request, status)

    async def authorize_and_limit(request: Request) -> None:
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or not hmac.compare_digest(
                header[7:].encode(), gate_token.encode()):
            raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        if not await limiter.acquire():
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "3600"})

    async def upstream_post(path: str, payload: dict) -> Response:
        try:
            async with httpx.AsyncClient(transport=transport, timeout=20.0) as client:
                result = await client.post(
                    f"{upstream_url}{path}", json=payload,
                    headers={"X-API-KEY": upstream_token},
                )
        except httpx.TimeoutException as exc:
            raise HTTPException(504, "Memory service timed out") from exc
        except httpx.HTTPError as exc:
            raise HTTPException(502, "Memory service unavailable") from exc
        if result.status_code >= 400:
            raise HTTPException(502, "Memory service rejected the request")
        return Response(result.content, status_code=result.status_code,
                        media_type=result.headers.get("content-type", "application/json"))

    @app.post("/recall")
    async def recall(request: Request):
        await authorize_and_limit(request)
        raw = await request.body()
        request.state.body_size = len(raw)
        try:
            body = RecallRequest.model_validate_json(raw)
        except Exception as exc:
            raise HTTPException(422, "Invalid recall request") from exc
        if body.category is not None and body.category not in ALLOWED_CATEGORIES:
            raise HTTPException(422, "Category is not allowed")
        request.state.snippet = body.prompt[:160].replace("\n", " ")
        payload = body.model_dump(exclude={"agent_id"}, exclude_none=True)
        payload["agent_id"] = "al"
        return await upstream_post("/context", payload)

    @app.post("/save")
    async def save(request: Request):
        await authorize_and_limit(request)
        declared = request.headers.get("content-length")
        if declared:
            try:
                if int(declared) > SAVE_BODY_LIMIT:
                    raise HTTPException(413, "Save request exceeds 8KB")
            except ValueError as exc:
                raise HTTPException(400, "Invalid Content-Length") from exc
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > SAVE_BODY_LIMIT:
                raise HTTPException(413, "Save request exceeds 8KB")
        raw = bytes(chunks)
        request.state.body_size = len(raw)
        try:
            body = SaveRequest.model_validate_json(raw)
        except Exception as exc:
            raise HTTPException(422, "Invalid save request") from exc
        if body.category not in ALLOWED_CATEGORIES:
            raise HTTPException(422, "Category is not allowed")
        if not _SAFE_SESSION_ID.fullmatch(body.session_id):
            raise HTTPException(422, "Invalid session_id")
        request.state.snippet = body.summary[:160].replace("\n", " ")
        payload = body.model_dump(exclude={"agent_id"})
        payload.update(agent_id="al", source="user")
        payload["additional_tags"] = list(dict.fromkeys(
            [*body.additional_tags, "al-bridge"]
        ))
        return await upstream_post("/writeback", payload)

    return app
