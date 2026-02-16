"""FastAPI server for the doc-qa tool."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from doc_qa.config import (
    AppConfig,
    UNSAFE_SECTIONS,
    _OPTIONAL_FACTORIES,
    _apply_dict,
    config_to_dict,
    load_config,
    resolve_cody_endpoint,
    resolve_db_path,
    save_config,
)
from doc_qa.indexing.indexer import DocIndex
from doc_qa.indexing.manager import IndexingManager
from doc_qa.retrieval.query_pipeline import QueryPipeline
from doc_qa.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ── Request / Response models ────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    session_id: str | None = None


class SourceInfo(BaseModel):
    file_path: str
    section_title: str
    score: float


class AttributionInfo(BaseModel):
    sentence: str
    source_index: int
    similarity: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    chunks_retrieved: int
    model: str
    session_id: str  # Returned so frontend can send follow-ups
    error: str | None = None
    attributions: list[AttributionInfo] | None = None
    # Intelligence fields
    intent: str | None = None
    confidence: float = 0.0
    is_abstained: bool = False
    diagrams: list[str] | None = None
    detected_formats: dict | None = None


class StatsResponse(BaseModel):
    total_chunks: int
    total_files: int
    db_path: str
    embedding_model: str


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    top_k: int = 5


class RetrievalChunkResponse(BaseModel):
    text: str
    score: float
    file_path: str
    section_title: str
    chunk_id: str


class RetrievalResponse(BaseModel):
    chunks: list[RetrievalChunkResponse]


# ── Session store ────────────────────────────────────────────────────


class _Session:
    """In-memory session with conversation history."""

    __slots__ = ("history", "last_active", "conversation_id")

    def __init__(self, conversation_id: str | None = None) -> None:
        self.history: list[dict] = []
        self.last_active: float = time.time()
        self.conversation_id: str | None = conversation_id

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_active) > ttl


class _SessionStore:
    """In-memory session store with optional DB-backed persistence."""

    def __init__(self, ttl: int = 1800, db_enabled: bool = False) -> None:
        self._sessions: dict[str, _Session] = {}
        self._ttl = ttl
        self.db_enabled = db_enabled

    def get_or_create(self, session_id: str | None) -> tuple[str, _Session]:
        self._cleanup_expired()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_expired(self._ttl):
                session.touch()
                return session_id, session
            del self._sessions[session_id]

        new_id = uuid.uuid4().hex[:12]
        session = _Session()
        self._sessions[new_id] = session
        return new_id, session

    async def get_or_create_async(self, session_id: str | None) -> tuple[str, _Session]:
        """DB-aware session creation. Uses conversation UUID as session ID."""
        self._cleanup_expired()

        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            if not session.is_expired(self._ttl):
                session.touch()
                return session_id, session
            del self._sessions[session_id]

        # If resuming a DB conversation, load history
        if session_id and self.db_enabled:
            try:
                from doc_qa.db.repository import get_conversation_with_messages

                conv = await get_conversation_with_messages(session_id)
                if conv is not None:
                    session = _Session(conversation_id=session_id)
                    for msg in conv.get("messages", []):
                        session.history.append({
                            "role": msg["role"],
                            "text": msg["content"],
                        })
                    self._sessions[session_id] = session
                    return session_id, session
            except Exception as exc:
                logger.warning("Failed to load conversation %s from DB: %s", session_id, exc)

        # Create new session, optionally backed by DB
        conversation_id = None
        if self.db_enabled:
            try:
                from doc_qa.db.repository import create_conversation

                conv = await create_conversation()
                conversation_id = conv["id"]
            except Exception as exc:
                logger.warning("Failed to create DB conversation: %s", exc)

        new_id = conversation_id or uuid.uuid4().hex[:12]
        session = _Session(conversation_id=conversation_id)
        self._sessions[new_id] = session
        return new_id, session

    def _cleanup_expired(self) -> None:
        expired = [
            sid for sid, s in self._sessions.items() if s.is_expired(self._ttl)
        ]
        for sid in expired:
            del self._sessions[sid]


# ── App factory ──────────────────────────────────────────────────────


def create_app(
    repo_path: str,
    config: AppConfig | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        repo_path: Path to the documentation repository (must be indexed).
        config: Optional AppConfig (loaded from config.yaml if None).

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = load_config()

    if repo_path:
        config.doc_repo.path = repo_path

    # Resolve db_path (uses CWD-relative default when repo_path is empty)
    effective_repo = config.doc_repo.path or "."
    db_path = resolve_db_path(config, effective_repo)

    # Open index
    index = DocIndex(db_path=db_path, embedding_model=config.indexing.embedding_model)
    if index.count_rows() == 0:
        logger.warning(
            "Index at '%s' is empty. You can index from the Settings UI or run 'doc-qa index <repo>'.",
            db_path,
        )

    # Retriever (for retrieval-only endpoint)
    retriever = HybridRetriever(
        table=index._table,
        embedding_model=config.indexing.embedding_model,
        mode=config.retrieval.search_mode,
    )

    # Resolve database URL
    db_url: str | None = None
    if config.database and config.database.url:
        from doc_qa.db.engine import get_database_url

        db_url = get_database_url(config.database.url)

    # Session store for multi-turn conversations
    sessions = _SessionStore(ttl=config.api.session_ttl, db_enabled=db_url is not None)

    # LLM backend — lazy-initialized
    llm_backend: Any = None
    llm_lock = asyncio.Lock()

    # SSE concurrency limiter
    _max_streams = config.streaming.max_concurrent_streams if config.streaming else 20
    _sse_semaphore = asyncio.Semaphore(_max_streams)

    # Indexing manager (singleton for background indexing jobs)
    indexing_manager = IndexingManager()

    async def _do_swap(swap_result) -> None:
        """Hot-swap index + retriever after successful indexing."""
        app.state.index = swap_result.index
        app.state.retriever = swap_result.retriever
        logger.info("Index hot-swapped to new build")

    _NO_REPO_MSG = (
        "No documentation repository is configured yet. "
        "Go to Settings \u2192 Indexing to set a repository path and start indexing."
    )

    def _check_repo_ready() -> str | None:
        """Return an error message if the repo is not configured or index is empty, else None."""
        cfg = app.state.config
        if not cfg.doc_repo.path:
            return _NO_REPO_MSG
        try:
            if app.state.index.count_rows() == 0:
                return (
                    "The documentation index is empty. "
                    "Go to Settings \u2192 Indexing and click Start Indexing, "
                    "or run: doc-qa index <repo-path>"
                )
        except Exception:
            return _NO_REPO_MSG
        return None

    def _create_pipeline(history: list[dict]) -> QueryPipeline:
        """Create a QueryPipeline with given history.

        Reads config from ``app.state.config`` so hot-reloaded values
        (retrieval, intelligence, etc.) take effect immediately.
        Uses ``app.state.index`` so a post-swap index is picked up.
        """
        cfg = app.state.config
        return QueryPipeline(
            table=app.state.index._table,
            llm_backend=llm_backend,
            embedding_model=cfg.indexing.embedding_model,
            search_mode=cfg.retrieval.search_mode,
            candidate_pool=cfg.retrieval.candidate_pool,
            top_k=cfg.retrieval.top_k,
            min_score=cfg.retrieval.min_score,
            max_chunks_per_file=cfg.retrieval.max_chunks_per_file,
            rerank=cfg.retrieval.rerank,
            max_history_turns=cfg.retrieval.max_history_turns,
            intelligence_config=cfg.intelligence,
            generation_config=cfg.generation,
            verification_config=cfg.verification,
        )

    async def _ensure_llm() -> None:
        nonlocal llm_backend
        if llm_backend is not None:
            return
        async with llm_lock:
            if llm_backend is not None:
                return
            cfg = app.state.config
            from doc_qa.llm.backend import create_backend
            llm_backend = create_backend(
                primary=cfg.llm.primary,
                fallback=cfg.llm.fallback,
                cody_endpoint=resolve_cody_endpoint(cfg),
                cody_model=cfg.cody.model,
                cody_binary=cfg.cody.agent_binary,
                workspace_root=repo_path,
                ollama_host=cfg.ollama.host,
                ollama_model=cfg.ollama.model,
                cody_access_token_env=cfg.cody.access_token_env,
            )

    # ── Lifespan ─────────────────────────────────────────────────────

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        # Startup: initialise DB engine if configured
        if db_url:
            from doc_qa.db.engine import init_engine

            init_engine(db_url)
            logger.info("Database persistence enabled")

        yield

        # Shutdown
        if llm_backend is not None:
            await llm_backend.close()
        if db_url:
            from doc_qa.db.engine import close_engine

            await close_engine()

    # ── Build FastAPI app ────────────────────────────────────────────

    app = FastAPI(
        title="Doc QA",
        description="Documentation Q&A powered by RAG + Cody AI",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict:
        index_ok = False
        index_count = 0
        try:
            index_count = app.state.index.count_rows()
            index_ok = True
        except Exception:
            pass

        return {
            "status": "ok",
            "components": {
                "index": {"ok": index_ok, "chunks": index_count},
            },
        }

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        s = app.state.index.stats()
        return StatsResponse(
            total_chunks=s["total_chunks"],
            total_files=s["total_files"],
            db_path=s["db_path"],
            embedding_model=s["embedding_model"],
        )

    @app.post("/api/retrieve", response_model=RetrievalResponse)
    async def retrieve(req: RetrievalRequest) -> RetrievalResponse:
        """Retrieve relevant chunks without LLM (for testing/debugging)."""
        cfg = app.state.config
        if len(req.question) > cfg.retrieval.max_query_length:
            raise HTTPException(
                status_code=400,
                detail=f"Query too long. Maximum length is {cfg.retrieval.max_query_length} characters.",
            )
        chunks = app.state.retriever.search(
            query=req.question,
            top_k=req.top_k,
            min_score=cfg.retrieval.min_score,
        )
        return RetrievalResponse(
            chunks=[
                RetrievalChunkResponse(
                    text=c.text,
                    score=c.score,
                    file_path=c.file_path,
                    section_title=c.section_title,
                    chunk_id=c.chunk_id,
                )
                for c in chunks
            ]
        )

    @app.post("/api/query", response_model=QueryResponse)
    async def query(req: QueryRequest) -> QueryResponse:
        """Full Q&A: retrieve + rerank + LLM answer.

        Supports multi-turn: pass session_id from a previous response
        to continue the conversation with history context.
        """
        repo_err = _check_repo_ready()
        if repo_err:
            raise HTTPException(status_code=422, detail=repo_err)

        cfg = app.state.config
        if len(req.question) > cfg.retrieval.max_query_length:
            raise HTTPException(
                status_code=400,
                detail=f"Query too long. Maximum length is {cfg.retrieval.max_query_length} characters.",
            )
        await _ensure_llm()

        # Get or create session (DB-aware when persistence is enabled)
        if sessions.db_enabled:
            session_id, session = await sessions.get_or_create_async(req.session_id)
        else:
            session_id, session = sessions.get_or_create(req.session_id)

        # Create pipeline with session history
        pipeline = _create_pipeline(session.history)
        pipeline._history = session.history  # share history reference

        try:
            result = await pipeline.query(req.question)
        except Exception as e:
            logger.error("Query failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while processing your query.",
            )

        return QueryResponse(
            answer=result.answer,
            sources=[
                SourceInfo(
                    file_path=s.file_path,
                    section_title=s.section_title,
                    score=s.score,
                )
                for s in result.sources
            ],
            chunks_retrieved=result.chunks_retrieved,
            model=result.model,
            session_id=session_id,
            error=result.error,
            attributions=result.attributions,
            intent=result.intent,
            confidence=result.confidence_score,
            is_abstained=result.is_abstained,
            diagrams=result.diagrams,
            detected_formats=result.detected_formats,
        )

    # ── Conversation persistence endpoints ──────────────────────────────

    class ConversationUpdate(BaseModel):
        title: str = Field(min_length=1, max_length=200)

    @app.get("/api/conversations")
    async def list_conversations(limit: int = 50, offset: int = 0):
        if not sessions.db_enabled:
            raise HTTPException(status_code=501, detail="Database not configured")
        from doc_qa.db.repository import list_conversations as db_list

        return await db_list(limit=limit, offset=offset)

    @app.get("/api/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str):
        if not sessions.db_enabled:
            raise HTTPException(status_code=501, detail="Database not configured")
        from doc_qa.db.repository import get_conversation_with_messages

        conv = await get_conversation_with_messages(conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conv

    @app.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(conversation_id: str, body: ConversationUpdate):
        if not sessions.db_enabled:
            raise HTTPException(status_code=501, detail="Database not configured")
        from doc_qa.db.repository import update_conversation_title

        ok = await update_conversation_title(conversation_id, body.title)
        if not ok:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"ok": True}

    @app.delete("/api/conversations/{conversation_id}")
    async def delete_conversation_endpoint(conversation_id: str):
        if not sessions.db_enabled:
            raise HTTPException(status_code=501, detail="Database not configured")
        from doc_qa.db.repository import delete_conversation

        ok = await delete_conversation(conversation_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Conversation not found")
        # Also evict from in-memory cache
        self_sessions = sessions._sessions
        if conversation_id in self_sessions:
            del self_sessions[conversation_id]
        return {"ok": True}

    # ── SSE streaming endpoint ────────────────────────────────────────

    @app.get("/api/query/stream")
    async def query_stream(q: str, request: Request, session_id: str | None = None):
        """SSE streaming variant of /api/query.

        Emits phased events: status, intent, sources, answer, verified, done.
        """
        repo_err = _check_repo_ready()
        if repo_err:
            from sse_starlette.sse import EventSourceResponse, ServerSentEvent

            async def _repo_err():
                yield ServerSentEvent(
                    data=json.dumps({"error": repo_err, "type": "SetupRequired"}),
                    event="error",
                )
            return EventSourceResponse(_repo_err())

        cfg = app.state.config
        if len(q) > cfg.retrieval.max_query_length:
            from sse_starlette.sse import EventSourceResponse, ServerSentEvent

            async def _err():
                yield ServerSentEvent(
                    data=json.dumps({"error": "Query too long", "type": "ValidationError"}),
                    event="error",
                )
            return EventSourceResponse(_err())

        if _sse_semaphore.locked():
            raise HTTPException(status_code=503, detail="Too many concurrent queries")

        await _ensure_llm()
        if sessions.db_enabled:
            sid, session = await sessions.get_or_create_async(session_id)
        else:
            sid, session = sessions.get_or_create(session_id)

        pipeline = _create_pipeline(session.history)
        pipeline._history = session.history

        from sse_starlette.sse import EventSourceResponse, ServerSentEvent

        async def _guarded():
            async with _sse_semaphore:
                try:
                    from doc_qa.streaming.sse import streaming_query
                    async for event in streaming_query(
                        question=q,
                        pipeline=pipeline,
                        request=request,
                        session_history=session.history,
                        config=cfg,
                        session_id=sid,
                        conversation_id=session.conversation_id,
                    ):
                        yield event
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("SSE stream error")
                    yield ServerSentEvent(
                        data=json.dumps({"error": str(exc), "type": type(exc).__name__}),
                        event="error",
                    )

        return EventSourceResponse(
            _guarded(),
            ping=cfg.streaming.sse_ping_interval if cfg.streaming else 15,
            ping_message_factory=lambda: ServerSentEvent(comment="keepalive"),
            headers={"X-Accel-Buffering": "no"},
        )

    # ── Indexing endpoints ─────────────────────────────────────────

    @app.get("/api/index/stream")
    async def index_stream(
        request: Request,
        action: str | None = None,
        repo_path: str | None = None,
        force_reindex: bool = False,
    ):
        """SSE stream for real-time indexing progress.

        - ``?action=start&repo_path=/path`` — start a new indexing job
        - ``?action=start&repo_path=/path&force_reindex=true`` — full re-index
        - No params — reconnect to an existing job (replay + live)
        """
        from sse_starlette.sse import EventSourceResponse, ServerSentEvent

        cfg = app.state.config

        if action == "start":
            # Validate repo_path
            if not repo_path:
                raise HTTPException(status_code=400, detail="repo_path is required")

            rp = Path(repo_path)
            if not rp.is_dir():
                raise HTTPException(status_code=400, detail=f"Path not found: {repo_path}")

            # Let the manager decide — it handles stale/dead job cleanup
            # internally.  A pre-check here would bypass that logic and
            # return 409 even for jobs that should be replaced.
            try:
                cfg.doc_repo.path = repo_path
                save_config(cfg)

                current_db_path = resolve_db_path(cfg, repo_path)
                job = await indexing_manager.start(
                    repo_path=repo_path,
                    config=cfg,
                    db_path=current_db_path,
                    on_swap=_do_swap,
                    force_reindex=force_reindex,
                )
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
        else:
            # Reconnect to existing job
            job = indexing_manager.current_job
            if job is None:
                # No job at all — return SSE with idle status so the
                # frontend gets a proper text/event-stream response.
                async def _no_job():
                    yield ServerSentEvent(
                        data=json.dumps({"state": "idle"}),
                        event="status",
                    )
                return EventSourceResponse(
                    _no_job(),
                    headers={"X-Accel-Buffering": "no"},
                )

            if job.is_terminal:
                # Job finished — replay buffered events (including the
                # terminal done/error/cancelled event) so the frontend
                # can display the correct final state.
                past_events = list(job._event_buffer)

                async def _replay_terminal():
                    for evt in past_events:
                        yield ServerSentEvent(
                            data=json.dumps(evt.data),
                            event=evt.event,
                        )

                return EventSourceResponse(
                    _replay_terminal(),
                    headers={"X-Accel-Buffering": "no"},
                )

        # Subscribe and stream events
        past_events, queue = job.subscribe()

        async def _stream():
            try:
                # Replay past events
                for evt in past_events:
                    if await request.is_disconnected():
                        return
                    yield ServerSentEvent(
                        data=json.dumps(evt.data),
                        event=evt.event,
                    )

                # Stream live events
                while True:
                    if await request.is_disconnected():
                        return

                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue

                    if evt is None:
                        # Sentinel — job finished
                        return
                    yield ServerSentEvent(
                        data=json.dumps(evt.data),
                        event=evt.event,
                    )
            finally:
                job.unsubscribe(queue)

        return EventSourceResponse(
            _stream(),
            ping=cfg.streaming.sse_ping_interval if cfg.streaming else 15,
            ping_message_factory=lambda: ServerSentEvent(comment="keepalive"),
            headers={"X-Accel-Buffering": "no"},
        )

    @app.post("/api/index/cancel")
    async def cancel_indexing():
        """Cancel the running indexing job."""
        if not indexing_manager.is_running:
            raise HTTPException(status_code=409, detail="No indexing job is running")
        indexing_manager.cancel()
        return {"ok": True}

    @app.get("/api/index/status")
    async def index_status():
        """Poll-based status of the current/last indexing job."""
        return indexing_manager.get_status()

    # ── Config management endpoints ─────────────────────────────────

    @app.get("/api/config")
    async def get_config() -> dict:
        """Return current configuration with resolved env vars."""
        cfg = app.state.config
        data = config_to_dict(cfg)

        # Resolve Cody env vars so the Settings UI shows actual values
        if "cody" in data:
            token_env = cfg.cody.access_token_env or "SRC_ACCESS_TOKEN"
            token = os.environ.get(token_env, "")
            data["cody"]["_token_is_set"] = bool(token)
            # Masked preview: first 4 + *** + last 4  (or just *** if short)
            if token:
                if len(token) > 10:
                    data["cody"]["_token_masked"] = f"{token[:4]}{'*' * (len(token) - 8)}{token[-4:]}"
                else:
                    data["cody"]["_token_masked"] = "*" * len(token)
            else:
                data["cody"]["_token_masked"] = ""
            # Always show the resolved endpoint (config → SRC_ENDPOINT → default)
            data["cody"]["endpoint"] = resolve_cody_endpoint(cfg)

        return data

    @app.patch("/api/config")
    async def update_config(request: Request) -> dict:
        """Partial config update.  Safe sections are hot-reloaded;
        unsafe sections are saved to disk with a restart-required flag.
        """
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Expected JSON object")

        cfg = app.state.config
        restart_sections: list[str] = []

        for section_name, section_data in body.items():
            if not isinstance(section_data, dict):
                continue

            # Ensure optional section exists
            if section_name in _OPTIONAL_FACTORIES:
                if getattr(cfg, section_name, None) is None:
                    setattr(cfg, section_name, _OPTIONAL_FACTORIES[section_name]())

            target = getattr(cfg, section_name, None)
            if target is None:
                continue

            _apply_dict(target, section_data)

            if section_name in UNSAFE_SECTIONS:
                restart_sections.append(section_name)

        # LLM hot-reload: if any LLM section changed, close old backend
        # so _ensure_llm() recreates it on next query with new config
        _LLM_SECTIONS = {"llm", "cody", "ollama"}
        if _LLM_SECTIONS & body.keys():
            nonlocal llm_backend
            old_backend = llm_backend
            llm_backend = None
            if old_backend is not None:
                try:
                    await old_backend.close()
                except Exception:
                    pass
                logger.info("LLM backend reset due to config change")

        # Database special case: if url changed, reinitialise engine
        if "database" in body and cfg.database and cfg.database.url:
            from doc_qa.db.engine import close_engine, get_database_url, init_engine

            new_url = get_database_url(cfg.database.url)
            if new_url:
                try:
                    await close_engine()
                except Exception:
                    pass
                init_engine(new_url)
                sessions.db_enabled = True
                app.state.db_enabled = True
                logger.info("Database engine re-initialised from config update")

        # Persist to disk
        save_config(cfg)

        return {
            "saved": True,
            "restart_required": len(restart_sections) > 0,
            "restart_sections": restart_sections,
        }

    @app.post("/api/config/db/test")
    async def test_db_connection(request: Request) -> dict:
        """Test a database connection URL without persisting it."""
        body = await request.json()
        url = body.get("url", "")
        if not url:
            return {"ok": False, "error": "No URL provided"}

        from doc_qa.db.engine import get_database_url

        resolved = get_database_url(url)
        if not resolved:
            return {"ok": False, "error": "Could not resolve database URL"}

        try:
            from sqlalchemy.ext.asyncio import create_async_engine
            from sqlalchemy import text

            engine = create_async_engine(resolved, pool_pre_ping=True)
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            await engine.dispose()
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/config/db/migrate")
    async def run_db_migrations() -> dict:
        """Run Alembic migrations (upgrade to head)."""
        cfg = app.state.config
        if not cfg.database or not cfg.database.url:
            return {"ok": False, "error": "No database URL configured"}

        from doc_qa.db.engine import get_database_url

        db_url_resolved = get_database_url(cfg.database.url)
        if not db_url_resolved:
            return {"ok": False, "error": "Could not resolve database URL"}

        try:
            import functools
            from alembic.config import Config as AlembicConfig
            from alembic import command as alembic_command

            alembic_ini = Path(__file__).resolve().parent.parent.parent / "alembic.ini"
            alembic_cfg = AlembicConfig(str(alembic_ini))
            # Override the URL so Alembic uses the resolved async URL
            alembic_cfg.set_main_option("sqlalchemy.url", db_url_resolved)

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                functools.partial(alembic_command.upgrade, alembic_cfg, "head"),
            )

            # Read current revision from alembic_version table
            current_rev: str | None = None
            try:
                from sqlalchemy import create_engine, text as sa_text

                sync_url = db_url_resolved.replace("+asyncpg", "")
                eng = create_engine(sync_url)
                with eng.connect() as conn:
                    row = conn.execute(sa_text("SELECT version_num FROM alembic_version LIMIT 1")).fetchone()
                    if row:
                        current_rev = row[0]
                eng.dispose()
            except Exception:
                pass  # Non-critical — revision display is best-effort

            return {"ok": True, "revision": current_rev}
        except Exception as exc:
            logger.exception("Migration failed")
            return {"ok": False, "error": str(exc)}

    # ── Directory browse endpoint ────────────────────────────────────

    @app.get("/api/browse")
    async def browse_directory(path: str = "") -> dict:
        """List directories at a given path for the folder picker."""
        import platform

        if not path:
            # Return filesystem roots
            if platform.system() == "Windows":
                import string
                drives = []
                for letter in string.ascii_uppercase:
                    dp = Path(f"{letter}:\\")
                    if dp.exists():
                        drives.append({"name": f"{letter}:\\", "path": f"{letter}:\\"})
                return {"path": "", "parent": "", "dirs": drives}
            else:
                path = str(Path.home())

        p = Path(path)
        if not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Not a valid directory: {path}")

        dirs = []
        try:
            for entry in sorted(p.iterdir()):
                if entry.is_dir() and not entry.name.startswith("."):
                    dirs.append({"name": entry.name, "path": str(entry)})
        except PermissionError:
            pass  # Return empty list for inaccessible dirs

        parent = str(p.parent) if p.parent != p else ""
        return {"path": str(p), "parent": parent, "dirs": dirs}

    # ── LLM test endpoints ──────────────────────────────────────────

    @app.post("/api/llm/cody/test")
    async def test_cody_connection(request: Request) -> dict:
        """Test Cody connection: spawn agent, authenticate, list models."""
        body = await request.json()
        cfg = app.state.config
        endpoint = body.get("endpoint") or resolve_cody_endpoint(cfg)
        access_token_env = body.get("access_token_env") or cfg.cody.access_token_env or "SRC_ACCESS_TOKEN"

        # Read token from the specified env var
        token = os.environ.get(access_token_env, "")
        if not token:
            return {"ok": False, "error": f"Environment variable {access_token_env} is not set or empty"}

        from doc_qa.llm.backend import CodyBackend

        result = await CodyBackend.test_connection(
            endpoint=endpoint,
            access_token=token,
            workspace_root=repo_path,
        )
        return result

    @app.post("/api/llm/ollama/test")
    async def test_ollama_connection(request: Request) -> dict:
        """Test Ollama connection and list available models."""
        import httpx

        body = await request.json()
        host = body.get("host", "http://localhost:11434").rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{host}/api/tags")
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError:
            return {"ok": False, "error": f"Cannot connect to Ollama at {host}. Is it running?"}
        except httpx.HTTPStatusError as e:
            return {"ok": False, "error": f"Ollama HTTP error: {e.response.status_code}"}
        except Exception as e:
            return {"ok": False, "error": f"Cannot connect to Ollama at {host}: {e}"}

        from doc_qa.llm.models import format_ollama_model

        raw_models = data.get("models", [])
        models = [format_ollama_model(m) for m in raw_models]

        return {"ok": True, "models": models}

    # ── File open endpoint ─────────────────────────────────────────────

    @app.post("/api/files/open")
    async def open_file(request: Request) -> dict:
        """Open a file with the system's default application."""
        import platform
        import subprocess

        body = await request.json()
        file_path = body.get("file_path", "")

        if not file_path:
            return {"ok": False, "error": "No file_path provided"}

        path = Path(file_path)
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}

        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.Popen(["open", str(path)])
            elif system == "Linux":
                subprocess.Popen(["xdg-open", str(path)])
            elif system == "Windows":
                os.startfile(str(path))
            else:
                return {"ok": False, "error": f"Unsupported platform: {system}"}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Static files (frontend SPA) ──────────────────────────────────
    # NOTE: SPA mount MUST be last — it catches all unmatched paths.

    class SPAStaticFiles(StaticFiles):
        """Serve index.html for unknown paths (client-side routing)."""

        async def get_response(self, path: str, scope):
            try:
                return await super().get_response(path, scope)
            except HTTPException as ex:
                if ex.status_code == 404:
                    return await super().get_response("index.html", scope)
                raise

    spa_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if spa_dir.is_dir():
        app.mount("/", SPAStaticFiles(directory=str(spa_dir), html=True), name="spa")

    # Store references for testing
    app.state.config = config
    app.state.config_path = Path("config.yaml")
    app.state.index = index
    app.state.retriever = retriever
    app.state.sessions = sessions
    app.state.db_enabled = sessions.db_enabled
    app.state.indexing_manager = indexing_manager

    return app
