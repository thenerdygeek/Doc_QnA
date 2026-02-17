"""Tests for the query pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from doc_qa.indexing.chunker import Chunk
from doc_qa.indexing.indexer import DocIndex
from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.retrieval.query_pipeline import QueryPipeline, QueryResult


class MockLLMBackend(LLMBackend):
    """Mock LLM backend that echoes the question."""

    def __init__(self) -> None:
        self.last_question: str = ""
        self.last_context: str = ""

    async def ask(self, question: str, context: str, history: list[dict] | None = None) -> Answer:
        self.last_question = question
        self.last_context = context
        return Answer(
            text=f"Mock answer for: {question}",
            sources=[],
            model="mock-model",
        )

    async def close(self) -> None:
        pass


@pytest.fixture
def indexed_data(tmp_path: Path) -> tuple[DocIndex, list[str]]:
    """Create a populated index for pipeline testing."""
    index = DocIndex(db_path=str(tmp_path / "pipeline_db"))

    docs = [
        ("auth.md", "Authentication", "OAuth 2.0 tokens are used for user authentication. The login endpoint validates credentials."),
        ("auth.md", "Sessions", "User sessions are stored in Redis with a 30-minute TTL expiration policy."),
        ("deploy.md", "Docker", "Docker containers run the application. Images are built with multi-stage Dockerfiles."),
    ]

    file_paths = []
    for file_name, section, content in docs:
        fp = str(tmp_path / file_name)
        if fp not in file_paths:
            Path(fp).write_text(content, encoding="utf-8")
            file_paths.append(fp)

        chunk = Chunk(
            chunk_id=f"{fp}#{section}",
            text=content,
            file_path=fp,
            file_type="md",
            section_title=section,
            section_level=1,
            chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="hash")

    index.rebuild_fts_index()
    return index, file_paths


class TestQueryPipeline:
    @pytest.mark.asyncio
    async def test_basic_query(self, indexed_data: tuple) -> None:
        """Pipeline should return a QueryResult with answer and sources."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,  # Skip reranking in unit tests for speed
        )

        result = await pipeline.query("How does authentication work?")

        assert isinstance(result, QueryResult)
        assert "Mock answer" in result.answer
        assert result.chunks_retrieved > 0
        assert result.model == "mock-model"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_query_passes_context_to_llm(self, indexed_data: tuple) -> None:
        """LLM should receive relevant context from retrieved chunks."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
        )

        await pipeline.query("How does authentication work?")

        # Context should contain auth-related content
        assert "OAuth" in mock_llm.last_context or "auth" in mock_llm.last_context.lower()

    @pytest.mark.asyncio
    async def test_empty_index_returns_no_results(self, tmp_path: Path) -> None:
        """Pipeline with empty index should return helpful message."""
        index = DocIndex(db_path=str(tmp_path / "empty_db"))
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
        )

        result = await pipeline.query("anything")

        assert "couldn't find" in result.answer.lower() or "no" in result.answer.lower()
        assert result.chunks_retrieved == 0

    @pytest.mark.asyncio
    async def test_file_diversity_cap(self, indexed_data: tuple) -> None:
        """Should limit chunks from the same file."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
            max_chunks_per_file=1,
            top_k=3,
        )

        result = await pipeline.query("authentication sessions")

        # With max_chunks_per_file=1, each source should be from a different file
        file_paths = [s.file_path for s in result.sources]
        assert len(file_paths) == len(set(file_paths))

    @pytest.mark.asyncio
    async def test_history_accumulates(self, indexed_data: tuple) -> None:
        """Conversation history should accumulate across queries."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
        )

        await pipeline.query("What is OAuth?")
        await pipeline.query("How are sessions managed?")

        # History should have both Q&A pairs
        assert len(pipeline._history) == 4  # 2 questions + 2 answers

    @pytest.mark.asyncio
    async def test_reset_history(self, indexed_data: tuple) -> None:
        """reset_history should clear conversation state."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()
        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
        )

        await pipeline.query("What is OAuth?")
        assert len(pipeline._history) == 2

        pipeline.reset_history()
        assert len(pipeline._history) == 0

    @pytest.mark.asyncio
    async def test_llm_error_propagated(self, indexed_data: tuple) -> None:
        """LLM errors should be captured in QueryResult."""
        index, _ = indexed_data
        mock_llm = MockLLMBackend()

        # Override ask to return an error
        async def ask_with_error(question, context, history=None):
            return Answer(text="", sources=[], model="mock", error="Connection failed")

        mock_llm.ask = ask_with_error

        pipeline = QueryPipeline(
            table=index._table,
            llm_backend=mock_llm,
            rerank=False,
        )

        result = await pipeline.query("test question")
        assert result.error == "Connection failed"


class TestContextBuilding:
    def test_build_context_format(self, indexed_data: tuple) -> None:
        """Built context should include source headers and text."""
        from doc_qa.retrieval.retriever import RetrievedChunk

        chunks = [
            RetrievedChunk(
                text="OAuth authentication flow.",
                score=0.9,
                chunk_id="auth.md#0",
                file_path="/docs/auth.md",
                file_type="md",
                section_title="Authentication",
                section_level=1,
                chunk_index=0,
            ),
        ]
        context = QueryPipeline._build_context(chunks)
        assert "[Source 1: auth.md" in context
        assert "Authentication" in context
        assert "0.900" in context
        assert "OAuth authentication flow." in context

    def test_build_context_empty(self) -> None:
        """Empty chunk list should produce empty context."""
        assert QueryPipeline._build_context([]) == ""


class TestContextReordering:
    def test_reorder_empty(self) -> None:
        """Empty list should return empty."""
        assert QueryPipeline._reorder_chunks([]) == []

    def test_reorder_single(self) -> None:
        """Single chunk should return as-is."""
        from doc_qa.retrieval.retriever import RetrievedChunk
        chunk = RetrievedChunk(
            text="a", score=1.0, chunk_id="a#0", file_path="a.md",
            file_type="md", section_title="A", section_level=1, chunk_index=0,
        )
        result = QueryPipeline._reorder_chunks([chunk])
        assert len(result) == 1

    def test_reorder_two(self) -> None:
        """Two chunks should return as-is."""
        from doc_qa.retrieval.retriever import RetrievedChunk
        chunks = [
            RetrievedChunk(text="a", score=1.0, chunk_id="a#0", file_path="a.md",
                          file_type="md", section_title="A", section_level=1, chunk_index=0),
            RetrievedChunk(text="b", score=0.9, chunk_id="b#0", file_path="b.md",
                          file_type="md", section_title="B", section_level=1, chunk_index=0),
        ]
        result = QueryPipeline._reorder_chunks(chunks)
        assert len(result) == 2

    def test_reorder_places_best_at_edges(self) -> None:
        """Best-scored chunks should end up at beginning and end."""
        from doc_qa.retrieval.retriever import RetrievedChunk
        # Input is assumed already sorted by score descending
        chunks = [
            RetrievedChunk(text=f"chunk_{i}", score=1.0 - i * 0.1, chunk_id=f"c#{i}",
                          file_path="f.md", file_type="md", section_title="S",
                          section_level=1, chunk_index=i)
            for i in range(5)
        ]
        result = QueryPipeline._reorder_chunks(chunks)
        assert len(result) == 5
        # First chunk should be the highest scored (index 0)
        assert result[0].chunk_id == "c#0"
        # Last chunk should be the second highest (index 1)
        assert result[-1].chunk_id == "c#1"
        # All chunks should be present
        result_ids = {c.chunk_id for c in result}
        assert result_ids == {f"c#{i}" for i in range(5)}
