"""Tests for the cross-encoder reranker."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
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


@pytest.fixture(autouse=True)
def mock_cross_encoder():
    """Mock the cross-encoder to avoid model downloads in tests."""
    mock_ce = MagicMock()

    def predict_scores(pairs):
        """Simple mock: score higher if query words appear in document."""
        scores = []
        for query, doc in pairs:
            query_words = set(query.lower().split())
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            scores.append(overlap / max(len(query_words), 1))
        return np.array(scores)

    mock_ce.predict = predict_scores

    with patch("doc_qa.retrieval.reranker._get_cross_encoder", return_value=mock_ce):
        yield mock_ce


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
            "OAuth 2.0 authentication flow requires a valid access token.",
            chunk_id="auth#0",
        )
        irrelevant = _make_chunk(
            "PostgreSQL database migrations are managed with Flyway.",
            chunk_id="db#0",
        )
        result = rerank("How does authentication work?", [irrelevant, relevant])
        assert len(result) == 2
        # The auth chunk should score higher because it has more word overlap
        assert result[0].chunk_id == "auth#0"

    def test_rerank_respects_top_k(self) -> None:
        """Should return at most top_k results."""
        chunks = [
            _make_chunk(f"Content about topic {i} with enough text.", chunk_id=f"test#{i}")
            for i in range(5)
        ]
        result = rerank("test query", chunks, top_k=2)
        assert len(result) == 2

    def test_rerank_updates_scores(self) -> None:
        """Reranked chunks should have updated cross-encoder scores."""
        chunks = [
            _make_chunk("Authentication with OAuth and access tokens.", chunk_id="a#0", score=0.3),
            _make_chunk("Database migration scripts using Flyway.", chunk_id="b#0", score=0.8),
        ]
        result = rerank("How does OAuth authentication work?", chunks)
        # Scores should be updated by the cross-encoder
        for r in result:
            assert isinstance(r.score, float)
        # Auth chunk should score higher for OAuth query
        assert result[0].chunk_id == "a#0"

    def test_rerank_accepts_model_name(self) -> None:
        """Should accept a custom model_name parameter."""
        chunks = [
            _make_chunk("Some content.", chunk_id="a#0"),
            _make_chunk("Other content.", chunk_id="b#0"),
        ]
        result = rerank("query", chunks, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert len(result) == 2
