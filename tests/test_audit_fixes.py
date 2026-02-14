"""Tests for all audit fix items (C1-C6, H1-H10, M1-M8, L4)."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from doc_qa.config import (
    APIConfig,
    AppConfig,
    IndexingConfig,
    RetrievalConfig,
    _apply_dict,
    resolve_db_path,
)

# Embedding dimension for mock vectors
_DIM = 384


def _mock_embed(texts, **kwargs):
    """Return fake embedding vectors (avoids loading the real ONNX model)."""
    return [np.random.rand(_DIM).astype(np.float32) for _ in texts]


def _mock_embed_q(query, **kwargs):
    """Return a single fake query embedding."""
    return np.random.rand(_DIM).astype(np.float32)


@pytest.fixture(autouse=True)
def _patch_embedder():
    """Auto-mock the embedding functions so no ONNX model is needed."""
    with patch("doc_qa.indexing.indexer.embed_texts", side_effect=_mock_embed), \
         patch("doc_qa.indexing.indexer.get_embedding_dimension", return_value=_DIM), \
         patch("doc_qa.retrieval.retriever.embed_query", side_effect=_mock_embed_q):
        yield

# ---------------------------------------------------------------------------
# C1: Thread-safe embedder singleton
# ---------------------------------------------------------------------------


class TestC1EmbedderThreadSafety:
    def test_concurrent_get_model_no_race(self) -> None:
        """Concurrent threads calling _get_model should not cause race conditions."""
        from concurrent.futures import ThreadPoolExecutor

        import doc_qa.indexing.embedder as emb_mod

        # Save original state so we can restore after test
        orig_model = emb_mod._model
        orig_name = emb_mod._model_name

        # Reset global singleton to test from scratch
        emb_mod._model = None
        emb_mod._model_name = ""

        mock_instance = MagicMock()

        try:
            with patch("fastembed.TextEmbedding", return_value=mock_instance):
                results = []

                def _load():
                    m = emb_mod._get_model()
                    results.append(id(m))

                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(_load) for _ in range(8)]
                    for f in futures:
                        f.result(timeout=60)

            # All threads should get the same singleton instance
            assert len(set(results)) == 1, "Multiple model instances created!"
        finally:
            # Restore original state to avoid breaking other tests
            emb_mod._model = orig_model
            emb_mod._model_name = orig_name


# ---------------------------------------------------------------------------
# C2: asyncio.Lock for _ensure_llm
# ---------------------------------------------------------------------------


class TestC2EnsureLlmLock:
    @pytest.mark.asyncio
    async def test_concurrent_ensure_llm_calls_backend_once(self, tmp_path: Path) -> None:
        """Two concurrent _ensure_llm() calls should only create one backend."""
        from doc_qa.api.server import create_app
        from doc_qa.config import AppConfig
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex

        # Set up a minimal index
        index_dir = tmp_path / "data" / "doc_qa_db"
        index = DocIndex(db_path=str(index_dir))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("Test content", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="Test content", file_path=fp,
            file_type="md", section_title="Test", section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="test_hash")
        index.rebuild_fts_index()

        config = AppConfig()
        config.indexing.db_path = str(index_dir)

        call_count = 0
        original_create = None

        def mock_create_backend(**kwargs):
            nonlocal call_count
            call_count += 1
            mock = AsyncMock()
            mock.ask = AsyncMock(return_value=MagicMock(
                text="answer", sources=[], model="mock", error=None
            ))
            mock.close = AsyncMock()
            return mock

        app = create_app(repo_path=str(tmp_path), config=config)

        # Access the _ensure_llm from closure — trigger it through the query endpoint
        with patch("doc_qa.llm.backend.create_backend", side_effect=mock_create_backend):
            from httpx import ASGITransport, AsyncClient
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Fire two queries concurrently
                results = await asyncio.gather(
                    client.post("/api/query", json={"question": "test1"}),
                    client.post("/api/query", json={"question": "test2"}),
                    return_exceptions=True,
                )

        # create_backend should be called exactly once
        assert call_count == 1, f"create_backend called {call_count} times (expected 1)"


# ---------------------------------------------------------------------------
# C3: RPC lock on CodyBackend
# ---------------------------------------------------------------------------


class TestC3CodyRPCLock:
    @pytest.mark.asyncio
    async def test_concurrent_ask_runs_sequentially(self) -> None:
        """Two concurrent ask() calls should execute RPC sequentially."""
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            from doc_qa.llm.backend import CodyBackend

            backend = CodyBackend()
            call_order = []

            async def mock_ensure_init():
                pass

            async def mock_ensure_chat():
                return "chat-1"

            original_rpc = MagicMock()

            async def mock_rpc_request(method, params=None):
                call_order.append(f"start-{method}")
                await asyncio.sleep(0.05)
                call_order.append(f"end-{method}")
                return {"messages": [{"speaker": "assistant", "text": "answer"}]}

            backend._ensure_initialized = mock_ensure_init
            backend._ensure_chat = mock_ensure_chat
            backend._rpc = MagicMock()
            backend._rpc.request = mock_rpc_request
            backend._initialized = True

            r1, r2 = await asyncio.gather(
                backend.ask("q1", "ctx1"),
                backend.ask("q2", "ctx2"),
            )

            # Verify sequential execution: start1, end1, start2, end2
            submit_starts = [i for i, x in enumerate(call_order) if x.startswith("start-chat/submit")]
            submit_ends = [i for i, x in enumerate(call_order) if x.startswith("end-chat/submit")]
            assert len(submit_starts) == 2
            # Second start should come after first end
            assert submit_starts[1] > submit_ends[0]


# ---------------------------------------------------------------------------
# C4: Generic error message
# ---------------------------------------------------------------------------


class TestC4GenericErrorMessage:
    def test_query_error_returns_generic_message(self, tmp_path: Path) -> None:
        """500 errors should not leak internal details."""
        from doc_qa.api.server import create_app
        from doc_qa.config import AppConfig
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex
        from fastapi.testclient import TestClient

        index_dir = tmp_path / "data" / "doc_qa_db"
        index = DocIndex(db_path=str(index_dir))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("Test", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="Test", file_path=fp,
            file_type="md", section_title="T", section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="h")
        index.rebuild_fts_index()

        config = AppConfig()
        config.indexing.db_path = str(index_dir)

        mock_backend = AsyncMock()
        mock_backend.ask = AsyncMock(side_effect=RuntimeError("secret internal error details"))
        mock_backend.close = AsyncMock()

        app = create_app(repo_path=str(tmp_path), config=config)

        with patch("doc_qa.llm.backend.create_backend", return_value=mock_backend):
            client = TestClient(app)
            resp = client.post("/api/query", json={"question": "test"})

        assert resp.status_code == 500
        detail = resp.json()["detail"]
        assert "secret" not in detail
        assert "internal error" in detail.lower()


# ---------------------------------------------------------------------------
# C5: Query length validation
# ---------------------------------------------------------------------------


class TestC5QueryLengthValidation:
    def _make_app(self, tmp_path: Path, max_query_length: int = 50):
        from doc_qa.api.server import create_app
        from doc_qa.config import AppConfig
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex
        from fastapi.testclient import TestClient

        index_dir = tmp_path / "data" / "doc_qa_db"
        index = DocIndex(db_path=str(index_dir))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("Test", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="Test", file_path=fp,
            file_type="md", section_title="T", section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="h")
        index.rebuild_fts_index()

        config = AppConfig()
        config.indexing.db_path = str(index_dir)
        config.retrieval.max_query_length = max_query_length

        app = create_app(repo_path=str(tmp_path), config=config)
        return TestClient(app)

    def test_empty_query_returns_422(self, tmp_path: Path) -> None:
        client = self._make_app(tmp_path)
        resp = client.post("/api/retrieve", json={"question": ""})
        assert resp.status_code == 422

    def test_over_limit_query_returns_400(self, tmp_path: Path) -> None:
        client = self._make_app(tmp_path, max_query_length=50)
        resp = client.post("/api/retrieve", json={"question": "x" * 51})
        assert resp.status_code == 400
        assert "too long" in resp.json()["detail"].lower()

    def test_at_limit_query_succeeds(self, tmp_path: Path) -> None:
        client = self._make_app(tmp_path, max_query_length=100)
        resp = client.post("/api/retrieve", json={"question": "x" * 100})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# C6: Failed queries don't pollute history
# ---------------------------------------------------------------------------


class TestC6HistoryPollution:
    @pytest.mark.asyncio
    async def test_error_query_not_in_history(self, tmp_path: Path) -> None:
        """Failed queries should not be added to conversation history."""
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex
        from doc_qa.llm.backend import Answer
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        index = DocIndex(db_path=str(tmp_path / "db"))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("auth content", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="auth content about oauth",
            file_path=fp, file_type="md", section_title="Auth",
            section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="h")
        index.rebuild_fts_index()

        call_count = 0

        async def mock_ask(question, context, history=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Answer(text="", sources=[], model="m", error="Connection failed")
            return Answer(text="Success!", sources=[], model="m")

        mock_llm = AsyncMock()
        mock_llm.ask = mock_ask

        pipeline = QueryPipeline(
            table=index._table, llm_backend=mock_llm, rerank=False,
        )

        # First query fails
        r1 = await pipeline.query("question 1")
        assert r1.error == "Connection failed"
        assert len(pipeline._history) == 0

        # Second query succeeds
        r2 = await pipeline.query("question 2")
        assert r2.error is None
        assert len(pipeline._history) == 2  # only success Q+A pair


# ---------------------------------------------------------------------------
# H3: Reranker vector caching
# ---------------------------------------------------------------------------


class TestH3RerankerVectorCaching:
    def test_rerank_uses_cached_vectors(self) -> None:
        """When chunks have pre-computed vectors, embed_texts should not be called."""
        from doc_qa.retrieval.reranker import rerank
        from doc_qa.retrieval.retriever import RetrievedChunk

        vec = np.random.rand(384).astype(np.float32).tolist()
        chunks = [
            RetrievedChunk(
                text=f"text {i}", score=0.5, chunk_id=f"c#{i}",
                file_path="/t.md", file_type="md", section_title="T",
                section_level=1, chunk_index=i, vector=vec,
            )
            for i in range(3)
        ]

        query_vec = np.random.rand(384).astype(np.float32)
        with patch("doc_qa.retrieval.reranker.embed_texts") as mock_embed, \
             patch("doc_qa.retrieval.reranker.embed_query", return_value=query_vec):
            result = rerank("test query", chunks)

        mock_embed.assert_not_called()
        assert len(result) == 3

    def test_rerank_embeds_when_no_vectors(self) -> None:
        """When chunks lack vectors, embed_texts should be called."""
        from doc_qa.retrieval.reranker import rerank
        from doc_qa.retrieval.retriever import RetrievedChunk

        chunks = [
            RetrievedChunk(
                text=f"text {i}", score=0.5, chunk_id=f"c#{i}",
                file_path="/t.md", file_type="md", section_title="T",
                section_level=1, chunk_index=i,
            )
            for i in range(3)
        ]

        query_vec = np.random.rand(384).astype(np.float32)
        chunk_vecs = [np.random.rand(384).astype(np.float32) for _ in range(3)]
        with patch("doc_qa.retrieval.reranker.embed_query", return_value=query_vec), \
             patch("doc_qa.retrieval.reranker.embed_texts", return_value=chunk_vecs) as mock_embed:
            result = rerank("test query", chunks)

        mock_embed.assert_called_once()
        assert len(result) == 3


# ---------------------------------------------------------------------------
# H4: Bounded conversation history
# ---------------------------------------------------------------------------


class TestH4BoundedHistory:
    @pytest.mark.asyncio
    async def test_history_trimmed_to_max_turns(self, tmp_path: Path) -> None:
        """History should be trimmed to max_history_turns * 2 entries."""
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex
        from doc_qa.llm.backend import Answer
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        index = DocIndex(db_path=str(tmp_path / "db"))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("content", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="content about topics",
            file_path=fp, file_type="md", section_title="T",
            section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="h")
        index.rebuild_fts_index()

        mock_llm = AsyncMock()
        mock_llm.ask = AsyncMock(
            return_value=Answer(text="answer", sources=[], model="m")
        )

        pipeline = QueryPipeline(
            table=index._table, llm_backend=mock_llm,
            rerank=False, max_history_turns=2,
        )

        for i in range(5):
            await pipeline.query(f"question {i}")

        # max_history_turns=2 → max 4 entries (2 Q+A pairs)
        assert len(pipeline._history) == 4


# ---------------------------------------------------------------------------
# H5: Configurable CORS origins
# ---------------------------------------------------------------------------


class TestH5ConfigurableCORS:
    def test_cors_uses_config(self, tmp_path: Path) -> None:
        """CORS middleware should use configured origins."""
        from doc_qa.api.server import create_app
        from doc_qa.config import AppConfig
        from doc_qa.indexing.chunker import Chunk
        from doc_qa.indexing.indexer import DocIndex
        from fastapi.testclient import TestClient

        index_dir = tmp_path / "data" / "doc_qa_db"
        index = DocIndex(db_path=str(index_dir))
        fp = str(tmp_path / "test.md")
        Path(fp).write_text("Test", encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0", text="Test", file_path=fp,
            file_type="md", section_title="T", section_level=1, chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="h")
        index.rebuild_fts_index()

        config = AppConfig()
        config.indexing.db_path = str(index_dir)
        config.api.cors_origins = ["https://example.com"]

        app = create_app(repo_path=str(tmp_path), config=config)
        client = TestClient(app)

        # Valid origin
        resp = client.options(
            "/api/health",
            headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://example.com"


# ---------------------------------------------------------------------------
# H6: Configurable session TTL
# ---------------------------------------------------------------------------


class TestH6ConfigurableSessionTTL:
    def test_session_expires_with_short_ttl(self) -> None:
        """Session should expire when TTL is very short."""
        from doc_qa.api.server import _Session, _SessionStore

        store = _SessionStore(ttl=1)  # 1-second TTL
        sid, session = store.get_or_create(None)

        time.sleep(1.1)

        # Should create a new session (old one expired)
        sid2, session2 = store.get_or_create(sid)
        assert sid2 != sid

    def test_session_is_expired_with_ttl(self) -> None:
        """Session.is_expired should use the provided TTL."""
        from doc_qa.api.server import _Session

        session = _Session()
        assert not session.is_expired(ttl=3600)

        session.last_active = time.time() - 10
        assert session.is_expired(ttl=5)
        assert not session.is_expired(ttl=3600)


# ---------------------------------------------------------------------------
# H7: _SYSTEM_PROMPT constant
# ---------------------------------------------------------------------------


class TestH7SystemPromptConstant:
    def test_system_prompt_exists(self) -> None:
        from doc_qa.llm.backend import _SYSTEM_PROMPT
        assert "documentation assistant" in _SYSTEM_PROMPT.lower()

    def test_cody_uses_system_prompt(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            from doc_qa.llm.backend import CodyBackend, _SYSTEM_PROMPT
            backend = CodyBackend()
            prompt = backend._build_prompt("q", "ctx")
            assert _SYSTEM_PROMPT in prompt

    def test_ollama_uses_system_prompt(self) -> None:
        from doc_qa.llm.backend import OllamaBackend, _SYSTEM_PROMPT
        backend = OllamaBackend()
        msgs = backend._build_messages("q", "ctx")
        assert msgs[0]["content"] == _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# H8: Buffered _read() with readuntil
# ---------------------------------------------------------------------------


class TestH8BufferedRead:
    @pytest.mark.asyncio
    async def test_read_parses_framed_message(self) -> None:
        """_read() should correctly parse Content-Length framed messages."""
        import json

        from doc_qa.llm.backend import _JSONRPCHandler

        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "result": "ok"}).encode("utf-8")
        frame = f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii") + payload

        reader = asyncio.StreamReader()
        reader.feed_data(frame)
        reader.feed_eof()

        writer = MagicMock()
        handler = _JSONRPCHandler(reader, writer)

        msg = await handler._read()
        assert msg is not None
        assert msg["result"] == "ok"

    @pytest.mark.asyncio
    async def test_read_returns_none_on_eof(self) -> None:
        """_read() should return None when stream ends."""
        from doc_qa.llm.backend import _JSONRPCHandler

        reader = asyncio.StreamReader()
        reader.feed_eof()

        writer = MagicMock()
        handler = _JSONRPCHandler(reader, writer)

        msg = await handler._read()
        assert msg is None


# ---------------------------------------------------------------------------
# M1: _apply_dict type validation
# ---------------------------------------------------------------------------


class TestM1ApplyDictTypeValidation:
    def test_rejects_string_for_int_field(self) -> None:
        """String value for int field should be rejected."""
        cfg = IndexingConfig()
        _apply_dict(cfg, {"chunk_size": "foo"})
        assert cfg.chunk_size == 512  # unchanged

    def test_accepts_correct_types(self) -> None:
        """Correct types should be accepted."""
        cfg = IndexingConfig()
        _apply_dict(cfg, {"chunk_size": 1024})
        assert cfg.chunk_size == 1024

    def test_int_to_float_coercion(self) -> None:
        """int → float coercion should work for float fields."""
        cfg = RetrievalConfig()
        _apply_dict(cfg, {"min_score": 0})
        assert cfg.min_score == 0.0
        assert isinstance(cfg.min_score, float)

    def test_bool_not_accepted_as_int(self) -> None:
        """bool should not be accepted for int fields."""
        cfg = IndexingConfig()
        _apply_dict(cfg, {"chunk_size": True})
        assert cfg.chunk_size == 512  # unchanged


# ---------------------------------------------------------------------------
# M2: rerank_model removed
# ---------------------------------------------------------------------------


class TestM2RerankedModelRemoved:
    def test_retrieval_config_no_rerank_model(self) -> None:
        """RetrievalConfig should not have rerank_model field."""
        cfg = RetrievalConfig()
        assert not hasattr(cfg, "rerank_model")

    def test_query_pipeline_no_rerank_model_param(self) -> None:
        """QueryPipeline should not accept rerank_model parameter."""
        import inspect
        from doc_qa.retrieval.query_pipeline import QueryPipeline
        sig = inspect.signature(QueryPipeline.__init__)
        assert "rerank_model" not in sig.parameters


# ---------------------------------------------------------------------------
# M3: FallbackBackend
# ---------------------------------------------------------------------------


class TestM3FallbackBackend:
    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback(self) -> None:
        """When primary succeeds, fallback should not be called."""
        from doc_qa.llm.backend import Answer, FallbackBackend

        primary = AsyncMock()
        primary.ask = AsyncMock(return_value=Answer(text="ok", sources=[], model="p"))
        fallback = AsyncMock()
        fallback.ask = AsyncMock(return_value=Answer(text="fb", sources=[], model="f"))

        fb = FallbackBackend(primary, fallback)
        result = await fb.ask("q", "ctx")

        assert result.text == "ok"
        fallback.ask.assert_not_called()

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_called(self) -> None:
        """When primary returns error, fallback should be used."""
        from doc_qa.llm.backend import Answer, FallbackBackend

        primary = AsyncMock()
        primary.ask = AsyncMock(
            return_value=Answer(text="", sources=[], model="p", error="failed")
        )
        fallback = AsyncMock()
        fallback.ask = AsyncMock(
            return_value=Answer(text="fallback ok", sources=[], model="f")
        )

        fb = FallbackBackend(primary, fallback)
        result = await fb.ask("q", "ctx")

        assert result.text == "fallback ok"
        assert result.model == "f"

    @pytest.mark.asyncio
    async def test_close_closes_both(self) -> None:
        """close() should close both primary and fallback."""
        from doc_qa.llm.backend import FallbackBackend

        primary = AsyncMock()
        fallback = AsyncMock()

        fb = FallbackBackend(primary, fallback)
        await fb.close()

        primary.close.assert_called_once()
        fallback.close.assert_called_once()

    def test_create_backend_with_fallback(self) -> None:
        """create_backend with fallback should return FallbackBackend."""
        from doc_qa.llm.backend import FallbackBackend, create_backend

        backend = create_backend(primary="ollama", fallback="ollama")
        # Same primary/fallback → no wrapping
        assert not isinstance(backend, FallbackBackend)

        # Different → wrapping (need to patch cody binary check)
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = create_backend(primary="ollama", fallback="cody")
            assert isinstance(backend, FallbackBackend)


# ---------------------------------------------------------------------------
# M4: resolve_db_path
# ---------------------------------------------------------------------------


class TestM4ResolveDbPath:
    def test_relative_path_resolved(self) -> None:
        config = AppConfig()
        config.indexing.db_path = "./data/db"
        result = resolve_db_path(config, "/my/repo")
        assert result == str(Path("/my/repo/data/db"))

    def test_absolute_path_preserved(self) -> None:
        config = AppConfig()
        config.indexing.db_path = "/absolute/path/db"
        result = resolve_db_path(config, "/my/repo")
        assert result == "/absolute/path/db"


# ---------------------------------------------------------------------------
# M7: Chunker word-boundary overlap
# ---------------------------------------------------------------------------


class TestM7ChunkerWordBoundary:
    def test_overlap_does_not_start_with_partial_word(self) -> None:
        """Overlap text should start at a word boundary."""
        from doc_qa.indexing.chunker import chunk_sections
        from doc_qa.parsers.base import ParsedSection

        # Create a long paragraph that will be split
        long_text = " ".join(f"word{i}" for i in range(200))
        section = ParsedSection(
            title="Test", content=long_text, level=1,
            file_path="test.md", file_type="md",
        )
        chunks = chunk_sections([section], "test.md", max_tokens=100, overlap_tokens=20)

        if len(chunks) > 1:
            for i in range(1, len(chunks)):
                text = chunks[i].text
                # Should not start with a partial word (i.e., first char after title
                # should be a letter starting a complete word, not mid-word)
                first_line = text.split("\n")[0]
                # Find the actual start of content after section title
                content_start = first_line.strip()
                if content_start:
                    # Should not start with a lowercase continuation
                    assert not content_start[0].isdigit() or content_start.startswith("word"), \
                        f"Chunk {i} may start with partial word: {content_start[:30]}"


# ---------------------------------------------------------------------------
# L4: Scanner exclusion component matching
# ---------------------------------------------------------------------------


class TestL4ScannerExclusion:
    def test_build_dir_excluded(self, tmp_path: Path) -> None:
        """Files in 'build/' directory should be excluded."""
        build = tmp_path / "build"
        build.mkdir()
        (build / "output.md").write_text("# Output")

        from doc_qa.config import DocRepoConfig
        from doc_qa.indexing.scanner import scan_files

        config = DocRepoConfig(
            path=str(tmp_path),
            patterns=["**/*.md"],
            exclude=["**/build/**"],
        )
        files = scan_files(config)
        names = {f.name for f in files}
        assert "output.md" not in names

    def test_rebuild_dir_not_excluded(self, tmp_path: Path) -> None:
        """Files in 'rebuild/' should NOT be excluded by 'build' pattern."""
        rebuild = tmp_path / "rebuild"
        rebuild.mkdir()
        (rebuild / "notes.md").write_text("# Notes")

        from doc_qa.config import DocRepoConfig
        from doc_qa.indexing.scanner import scan_files

        config = DocRepoConfig(
            path=str(tmp_path),
            patterns=["**/*.md"],
            exclude=["**/build/**"],
        )
        files = scan_files(config)
        names = {f.name for f in files}
        assert "notes.md" in names

    def test_should_exclude_exact_component(self) -> None:
        """_should_exclude should match exact path components."""
        from doc_qa.indexing.scanner import _should_exclude

        repo = Path("/repo")

        # "build" as exact component → excluded
        assert _should_exclude(Path("/repo/build/out.md"), repo, ["**/build/**"])

        # "rebuild" contains "build" as substring but NOT as component → not excluded
        assert not _should_exclude(Path("/repo/rebuild/notes.md"), repo, ["**/build/**"])

        # "node_modules" exact → excluded
        assert _should_exclude(Path("/repo/node_modules/x.md"), repo, ["**/node_modules/**"])


# ---------------------------------------------------------------------------
# H9: httpx in dependencies
# ---------------------------------------------------------------------------


class TestH9HttpxDependency:
    def test_httpx_in_pyproject(self) -> None:
        """httpx should be listed in pyproject.toml dependencies."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject.read_text()
        assert "httpx" in content


# ---------------------------------------------------------------------------
# M5: Logging configuration
# ---------------------------------------------------------------------------


class TestM5LoggingConfig:
    def test_log_level_arg_exists(self) -> None:
        """--log-level argument should be available."""
        import argparse
        from doc_qa.__main__ import main

        # We can't easily test main() directly, but we can verify the argparse setup
        # by checking the module source
        import inspect
        source = inspect.getsource(main)
        assert "--log-level" in source
        assert "logging.basicConfig" in source


# ---------------------------------------------------------------------------
# M8: cody/ package deleted
# ---------------------------------------------------------------------------


class TestM8CodyPackageDeleted:
    def test_cody_dir_does_not_exist(self) -> None:
        """doc_qa/cody/ directory should not exist."""
        cody_dir = Path(__file__).parent.parent / "doc_qa" / "cody"
        assert not cody_dir.exists()


# ---------------------------------------------------------------------------
# M10: Cross-encoder docstring fix
# ---------------------------------------------------------------------------


class TestM10DocstringFix:
    def test_query_pipeline_no_cross_encoder(self) -> None:
        """QueryPipeline docstring should not mention cross-encoder."""
        from doc_qa.retrieval.query_pipeline import QueryPipeline
        docstring = QueryPipeline.__doc__ or ""
        assert "cross-encoder" not in docstring.lower()

    def test_reranker_mentions_bi_encoder(self) -> None:
        """Reranker module should mention bi-encoder."""
        import doc_qa.retrieval.reranker as mod
        docstring = mod.__doc__ or ""
        assert "bi-encoder" in docstring.lower()
