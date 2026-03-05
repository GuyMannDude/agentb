"""
AgentB — Drop-in Memory Coprocessor for AI Agents
==================================================
Four endpoints. Any LLM. Any embedding model. Zero cloud lock-in.

  /health      → Heartbeat
  /context     → L1/L2/L3 memory retrieval ("tap on the shoulder")
  /preflight   → PASS / ENRICH / WARN / BLOCK validation
  /writeback   → Session archiving to long-term memory

Created by Guy Hoffman, Rocky Moltman, and Opie (Claude).
https://github.com/GuyMannDude/agentb
"""

import json
import time
import hashlib
import logging
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agentb.config import load_config, AgentBConfig
from agentb.providers import create_reasoning_provider, create_embedding_provider
from agentb.cache import L1Cache, L2Index, l3_scan, ContextChunk

# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("agentb")


# ─────────────────────────────────────────────
#  Request/Response Models
# ─────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    reasoning_provider: str
    reasoning_model: str
    embedding_provider: str
    embedding_model: str
    reasoning_healthy: bool
    embedding_healthy: bool
    l1_cache_size: int
    l2_index_size: int
    memory_files: int


class ContextRequest(BaseModel):
    prompt: str = Field(..., description="The incoming prompt to search context for")
    agent_id: Optional[str] = Field(None, description="Agent identifier for multi-agent isolation")
    max_results: int = Field(5, ge=1, le=20, description="Max context chunks to return")


class ContextChunkResponse(BaseModel):
    content: str
    source: str
    relevance: float
    cache_tier: str


class ContextResponse(BaseModel):
    chunks: list[ContextChunkResponse]
    total_found: int
    latency_ms: float
    cache_hits: dict


class PreflightRequest(BaseModel):
    prompt: str = Field(..., description="The user's original prompt")
    draft_response: str = Field(..., description="The agent's draft response to validate")
    agent_id: Optional[str] = Field(None, description="Agent identifier")


class PreflightResponse(BaseModel):
    verdict: str = Field(..., description="PASS | ENRICH | WARN | BLOCK")
    confidence: float
    reason: str
    enrichment: Optional[str] = None
    latency_ms: float


class WritebackRequest(BaseModel):
    session_id: str
    summary: str
    key_facts: list[str] = []
    projects_referenced: list[str] = []
    decisions_made: list[str] = []
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None


class WritebackResponse(BaseModel):
    status: str
    memory_id: str
    l1_bundles_updated: int
    message: str


# ─────────────────────────────────────────────
#  Application Factory
# ─────────────────────────────────────────────

def create_app(config: Optional[AgentBConfig] = None) -> FastAPI:
    """Create and configure the AgentB FastAPI application."""

    if config is None:
        config = load_config()

    log.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Initialize providers
    reasoner = create_reasoning_provider(config.reasoning)
    embedder = create_embedding_provider(config.embedding)

    # Initialize storage paths
    data_dir = Path(config.data_dir)
    memory_dir = data_dir / "memory"
    l1_dir = data_dir / "cache" / "l1"
    l2_dir = data_dir / "cache" / "l2"
    log_dir = data_dir / "logs"

    for d in [memory_dir, l1_dir, l2_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initialize cache layers
    l1 = L1Cache(l1_dir, config.cache)
    l2 = L2Index(l2_dir, config.cache)

    # Build app
    app = FastAPI(
        title="AgentB",
        description="Drop-in memory coprocessor for AI agents",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth middleware ──
    if config.server.auth_token:
        @app.middleware("http")
        async def check_auth(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            token = request.headers.get("X-API-KEY") or request.headers.get("Authorization", "").replace("Bearer ", "")
            if token != config.server.auth_token:
                return Response("Unauthorized", status_code=401)
            return await call_next(request)

    # ── Health ──
    @app.get("/health", response_model=HealthResponse)
    async def health():
        reasoning_ok = await reasoner.health_check()
        embedding_ok = await embedder.health_check()
        mem_count = len(list(memory_dir.glob("*.json")))
        return HealthResponse(
            status="ok" if (reasoning_ok and embedding_ok) else "degraded",
            timestamp=datetime.now(timezone.utc).isoformat(),
            reasoning_provider=config.reasoning.provider,
            reasoning_model=config.reasoning.model,
            embedding_provider=config.embedding.provider,
            embedding_model=config.embedding.model,
            reasoning_healthy=reasoning_ok,
            embedding_healthy=embedding_ok,
            l1_cache_size=l1.size,
            l2_index_size=l2.size,
            memory_files=mem_count,
        )

    # ── Context Retrieval ──
    @app.post("/context", response_model=ContextResponse)
    async def context(req: ContextRequest):
        start = time.time()
        all_chunks: list[ContextChunk] = []
        cache_hits = {"L1": 0, "L2": 0, "L3": 0}

        try:
            query_embedding = await embedder.embed(req.prompt)
        except Exception as e:
            raise HTTPException(503, f"Embedding provider unavailable: {e}")

        # L1
        l1_results = l1.search(query_embedding, top_k=req.max_results)
        all_chunks.extend(l1_results)
        cache_hits["L1"] = len(l1_results)

        # L2
        remaining = req.max_results - len(all_chunks)
        if remaining > 0:
            l2_results = l2.search(query_embedding, top_k=remaining)
            all_chunks.extend(l2_results)
            cache_hits["L2"] = len(l2_results)

        # L3
        remaining = req.max_results - len(all_chunks)
        if remaining > 0:
            l3_results = await l3_scan(
                memory_dir, query_embedding,
                embed_fn=embedder.embed,
                threshold=config.cache.l3_similarity_threshold,
                top_k=remaining,
            )
            all_chunks.extend(l3_results)
            cache_hits["L3"] = len(l3_results)

        latency = (time.time() - start) * 1000
        return ContextResponse(
            chunks=[ContextChunkResponse(**c.to_dict()) for c in all_chunks],
            total_found=len(all_chunks),
            latency_ms=round(latency, 1),
            cache_hits=cache_hits,
        )

    # ── Preflight Check ──
    @app.post("/preflight", response_model=PreflightResponse)
    async def preflight(req: PreflightRequest):
        start = time.time()

        system_prompt = """You are AgentB, a memory coprocessor for AI agents.
Review the agent's draft response against the user's prompt and any memory context.

Respond with EXACTLY this JSON format (no markdown, no backticks):
{
    "verdict": "PASS|ENRICH|WARN|BLOCK",
    "confidence": 0.0-1.0,
    "reason": "brief explanation",
    "enrichment": "additional context if ENRICH, otherwise null"
}

Verdicts:
- PASS: Accurate and complete.
- ENRICH: Correct but could be improved with context you have.
- WARN: May contain inaccuracies. Flag for review.
- BLOCK: Contains a clear factual error."""

        user_prompt = f"""USER'S PROMPT:
{req.prompt}

AGENT'S DRAFT RESPONSE:
{req.draft_response}

Review and provide your preflight verdict as JSON."""

        # Cross-reference with memory
        try:
            query_embedding = await embedder.embed(req.prompt)
            l1_hits = l1.search(query_embedding, top_k=2)
            l2_hits = l2.search(query_embedding, top_k=2)
            context_chunks = l1_hits + l2_hits
            if context_chunks:
                context_text = "\n\n".join(
                    f"[{c.cache_tier}] {c.content}" for c in context_chunks
                )
                user_prompt = f"MEMORY CONTEXT:\n{context_text}\n\n{user_prompt}"
        except Exception as e:
            log.warning(f"Preflight context retrieval failed: {e}")

        try:
            raw = await reasoner.generate(user_prompt, system=system_prompt)
            cleaned = raw.strip().strip("`").strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            result = json.loads(cleaned)
            latency = (time.time() - start) * 1000

            return PreflightResponse(
                verdict=result.get("verdict", "PASS").upper(),
                confidence=float(result.get("confidence", 0.5)),
                reason=result.get("reason", ""),
                enrichment=result.get("enrichment"),
                latency_ms=round(latency, 1),
            )
        except (json.JSONDecodeError, Exception) as e:
            latency = (time.time() - start) * 1000
            log.warning(f"Preflight parse/error: {e}")
            return PreflightResponse(
                verdict="PASS",
                confidence=0.2,
                reason=f"AgentB couldn't validate — defaulting to PASS ({str(e)[:80]})",
                latency_ms=round(latency, 1),
            )

    # ── Writeback ──
    @app.post("/writeback", response_model=WritebackResponse)
    async def writeback(req: WritebackRequest):
        ts = req.timestamp or datetime.now(timezone.utc).isoformat()
        memory_id = hashlib.sha256(f"{req.session_id}:{ts}".encode()).hexdigest()[:16]

        memory_entry = {
            "id": memory_id,
            "session_id": req.session_id,
            "agent_id": req.agent_id,
            "summary": req.summary,
            "key_facts": req.key_facts,
            "projects_referenced": req.projects_referenced,
            "decisions_made": req.decisions_made,
            "timestamp": ts,
            "created_at": time.time(),
        }

        mem_path = memory_dir / f"{memory_id}.json"
        mem_path.write_text(json.dumps(memory_entry, indent=2, default=str))
        log.info(f"Writeback: {req.session_id} → {memory_id}")

        # Index in L2 and build L1 bundles
        l1_updated = 0
        try:
            full_text = req.summary + "\n" + "\n".join(req.key_facts)
            embedding = await embedder.embed(full_text)

            await l2.add(
                content=full_text,
                source=f"session:{req.session_id}",
                embedding=embedding,
                metadata={
                    "projects": req.projects_referenced,
                    "decisions": req.decisions_made,
                    "agent_id": req.agent_id,
                },
            )

            for project in req.projects_referenced:
                project_context = f"Project: {project}\n"
                project_context += f"Session: {req.session_id}\n"
                project_context += f"Summary: {req.summary}\n"
                relevant_facts = [f for f in req.key_facts if project.lower() in f.lower()]
                if relevant_facts:
                    project_context += "Facts:\n" + "\n".join(f"- {f}" for f in relevant_facts)

                proj_embedding = await embedder.embed(project_context)
                await l1.add(project_context, f"project:{project}", proj_embedding)
                l1_updated += 1

        except Exception as e:
            log.error(f"Writeback indexing failed: {e}")

        return WritebackResponse(
            status="archived",
            memory_id=memory_id,
            l1_bundles_updated=l1_updated,
            message=f"Session {req.session_id} archived. {l1_updated} L1 bundles updated.",
        )

    # ── Background: Idle pre-caching ──
    async def precache_loop():
        while True:
            await asyncio.sleep(300)
            try:
                recent = sorted(memory_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]
                for mem_file in recent:
                    mem = json.loads(mem_file.read_text())
                    content = mem.get("summary", "")
                    if not content:
                        continue
                    bundle_id = hashlib.sha256(content.encode()).hexdigest()[:12]
                    existing = {b.get("id") for b in l1.bundles}
                    if bundle_id in existing:
                        continue
                    embedding = await embedder.embed(content)
                    await l1.add(content, f"precache:{mem.get('id', mem_file.stem)}", embedding)
            except Exception as e:
                log.warning(f"Precache error: {e}")

    @app.on_event("startup")
    async def startup():
        log.info(f"AgentB v0.2.0 starting")
        log.info(f"  Reasoning: {config.reasoning.provider}/{config.reasoning.model}")
        log.info(f"  Embedding: {config.embedding.provider}/{config.embedding.model}")
        log.info(f"  Data dir:  {config.data_dir}")
        log.info(f"  L1 cache:  {l1.size} bundles")
        log.info(f"  L2 index:  {l2.size} entries")
        asyncio.create_task(precache_loop())

    return app


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    import uvicorn
    cfg = load_config()
    uvicorn.run(
        "agentb.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_level=cfg.log_level,
    )
