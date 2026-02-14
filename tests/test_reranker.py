"""Tests for the reranker."""

from __future__ import annotations

import pytest

from doc_qa.retrieval.reranker import rerank
from doc_qa.retrieval.retriever import RetrievedChunk


def _make_chunk(text: str, chunk_id: str = "test#0", score: float = 0.5) -> RetrievedChunk:
    """Create a test RetrievedChunk."""
    return RetrievedChunk(
        text=text,
        score=score,
        chunk_id=chunk_id,
        file_path="/tmp/test.md",
        file_type="md",
        section_title="Test",
        section_level=1,
        chunk_index=0,
    )


class TestReranker:
    def test_rerank_empty_list(self) -> None:
        """Reranking empty list should return empty list."""
        result = rerank("test query", [])
        assert result == []

    def test_rerank_single_chunk(self) -> None:
        """Reranking single chunk should return it as-is."""
        chunk = _make_chunk("Authentication uses OAuth tokens.")
        result = rerank("How does auth work?", [chunk])
        assert len(result) == 1
        assert result[0].text == chunk.text

    def test_rerank_orders_by_relevance(self) -> None:
        """Relevant chunk should be ranked higher than irrelevant one."""
        relevant = _make_chunk(
            "OAuth 2.0 authentication flow requires a valid access token from the identity provider.",
            chunk_id="auth#0",
        )
        irrelevant = _make_chunk(
            "PostgreSQL database migrations are managed with Flyway version scripts.",
            chunk_id="db#0",
        )
        # Put irrelevant first to test reordering
        result = rerank("How does authentication work?", [irrelevant, relevant])
        assert len(result) == 2
        assert result[0].chunk_id == "auth#0"

    def test_rerank_respects_top_k(self) -> None:
        """Should return at most top_k results."""
        chunks = [
            _make_chunk(f"Content about topic {i} with enough text for embedding.", chunk_id=f"test#{i}")
            for i in range(5)
        ]
        result = rerank("test query", chunks, top_k=2)
        assert len(result) == 2

    def test_rerank_updates_scores(self) -> None:
        """Reranked chunks should have updated cosine similarity scores."""
        chunks = [
            _make_chunk("Authentication with OAuth and access tokens.", chunk_id="a#0", score=0.3),
            _make_chunk("Database migration scripts using Flyway.", chunk_id="b#0", score=0.8),
        ]
        result = rerank("How does OAuth work?", chunks)
        # Scores should be cosine similarities (between -1 and 1, typically 0-1)
        for r in result:
            assert -1.0 <= r.score <= 1.0
        # Auth chunk should score higher for OAuth query
        assert result[0].chunk_id == "a#0"
