"""Tests for the hybrid retriever."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doc_qa.indexing.chunker import Chunk
from doc_qa.indexing.indexer import DocIndex
from doc_qa.retrieval.retriever import HybridRetriever, RetrievedChunk


@pytest.fixture
def populated_index(tmp_path: Path) -> DocIndex:
    """Create a DocIndex with embedded test chunks about different topics."""
    index = DocIndex(db_path=str(tmp_path / "test_db"))

    topics = [
        ("auth.md", "Authentication", "Users authenticate using OAuth 2.0 tokens. The login flow requires a valid access token from the identity provider."),
        ("auth.md", "Authorization", "Role-based access control (RBAC) determines which API endpoints a user can access. Admin roles have full permissions."),
        ("deploy.md", "Deployment", "The application is deployed using Docker containers on Kubernetes. Helm charts manage the deployment configuration."),
        ("deploy.md", "Scaling", "Horizontal pod autoscaling adjusts replica count based on CPU and memory utilization metrics."),
        ("api.md", "REST API", "The REST API follows OpenAPI 3.0 specification. All endpoints require Bearer token authentication."),
        ("api.md", "Rate Limiting", "API rate limiting uses a token bucket algorithm. Default limit is 100 requests per minute per user."),
        ("db.md", "Database", "PostgreSQL is the primary database. Connection pooling uses PgBouncer with a max pool size of 50."),
        ("db.md", "Migrations", "Database migrations are managed with Flyway. Each migration has a version number and SQL script."),
    ]

    for file_name, section, content in topics:
        fp = str(tmp_path / file_name)
        Path(fp).write_text(content, encoding="utf-8")

        chunk = Chunk(
            chunk_id=f"{fp}#0",
            text=content,
            file_path=fp,
            file_type="md",
            section_title=section,
            section_level=1,
            chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="test_hash")

    # Build FTS index for hybrid search
    index.rebuild_fts_index()
    return index


class TestHybridRetriever:
    def test_vector_search_returns_results(self, populated_index: DocIndex) -> None:
        """Vector search should return relevant chunks."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="vector",
        )
        results = retriever.search("How does user login work?", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_vector_search_ranks_by_relevance(self, populated_index: DocIndex) -> None:
        """Authentication query should rank auth chunks higher than database chunks."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="vector",
        )
        results = retriever.search("How does OAuth authentication work?", top_k=8)
        assert len(results) > 0
        # Top result should be about authentication
        assert "auth" in results[0].section_title.lower() or "auth" in results[0].text.lower()

    def test_vector_search_respects_top_k(self, populated_index: DocIndex) -> None:
        """Should return at most top_k results."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="vector",
        )
        results = retriever.search("deployment", top_k=3)
        assert len(results) <= 3

    def test_vector_search_min_score_filter(self, populated_index: DocIndex) -> None:
        """Should filter out results below min_score."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="vector",
        )
        # Very high threshold should return fewer or no results
        results = retriever.search("random unrelated query xyz", top_k=8, min_score=0.99)
        assert len(results) < 8

    def test_fts_search_returns_results(self, populated_index: DocIndex) -> None:
        """Full-text search should find chunks by keyword match."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="fts",
        )
        results = retriever.search("Kubernetes Docker", top_k=3)
        assert len(results) > 0

    def test_hybrid_search_returns_results(self, populated_index: DocIndex) -> None:
        """Hybrid search should return results combining both methods."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="hybrid",
        )
        results = retriever.search("How to deploy the application?", top_k=5)
        assert len(results) > 0

    def test_chunk_metadata_preserved(self, populated_index: DocIndex) -> None:
        """Retrieved chunks should have all metadata fields."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="vector",
        )
        results = retriever.search("database", top_k=1)
        assert len(results) == 1
        chunk = results[0]
        assert chunk.file_path  # non-empty
        assert chunk.file_type == "md"
        assert chunk.section_title  # non-empty
        assert isinstance(chunk.score, float)
        assert isinstance(chunk.chunk_index, int)

    def test_empty_index_returns_empty(self, tmp_path: Path) -> None:
        """Searching an empty index should return no results."""
        index = DocIndex(db_path=str(tmp_path / "empty_db"))
        retriever = HybridRetriever(table=index._table, mode="vector")
        results = retriever.search("anything", top_k=5)
        assert results == []


class TestRRFScoring:
    def test_rrf_merge_produces_valid_scores(self, populated_index: DocIndex) -> None:
        """Manual RRF should produce positive scores."""
        retriever = HybridRetriever(
            table=populated_index._table,
            mode="hybrid",
        )
        # Use manual RRF directly
        from doc_qa.indexing.embedder import embed_query
        query_vec = embed_query("authentication login")
        results = retriever._manual_rrf("authentication login", query_vec, top_k=5, min_score=0.0)
        assert len(results) > 0
        for r in results:
            assert r.score > 0.0
