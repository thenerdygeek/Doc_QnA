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
        """Manual RRF should produce non-negative normalized scores."""
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
            # After min-max normalization, lowest score maps to 0.0
            assert r.score >= 0.0


class TestMetadataBoost:
    """Tests for _apply_metadata_boost static method."""

    def test_section_level_boost(self) -> None:
        """Level 1-2 sections should get boosted more than level 3+."""
        chunks = [
            RetrievedChunk(
                text="t1", score=0.5, chunk_id="a#0", file_path="a.md",
                file_type="md", section_title="Top", section_level=1,
                chunk_index=0,
            ),
            RetrievedChunk(
                text="t2", score=0.5, chunk_id="a#1", file_path="a.md",
                file_type="md", section_title="Sub", section_level=3,
                chunk_index=1,
            ),
            RetrievedChunk(
                text="t3", score=0.5, chunk_id="a#2", file_path="a.md",
                file_type="md", section_title="Deep", section_level=5,
                chunk_index=2,
            ),
        ]

        HybridRetriever._apply_metadata_boost(
            chunks, section_level_boost=0.10, recency_boost=0.0,
        )

        # Level 1 gets full boost (0.5 + 0.10 = 0.60)
        assert chunks[0].score == pytest.approx(0.60, abs=1e-6)
        # Level 3 gets half boost (0.5 + 0.05 = 0.55)
        level3 = [c for c in chunks if c.section_title == "Sub"][0]
        assert level3.score == pytest.approx(0.55, abs=1e-6)
        # Level 5 gets no boost (stays 0.5)
        level5 = [c for c in chunks if c.section_title == "Deep"][0]
        assert level5.score == pytest.approx(0.50, abs=1e-6)

    def test_recency_boost(self) -> None:
        """Most recent doc should get full recency boost; older docs get less."""
        import time
        now = time.time()
        one_year_ago = now - 365 * 86400
        half_year_ago = now - 182.5 * 86400

        chunks = [
            RetrievedChunk(
                text="old", score=0.5, chunk_id="a#0", file_path="a.md",
                file_type="md", section_title="A", section_level=1,
                chunk_index=0, doc_date=one_year_ago,
            ),
            RetrievedChunk(
                text="mid", score=0.5, chunk_id="b#0", file_path="b.md",
                file_type="md", section_title="B", section_level=1,
                chunk_index=0, doc_date=half_year_ago,
            ),
            RetrievedChunk(
                text="new", score=0.5, chunk_id="c#0", file_path="c.md",
                file_type="md", section_title="C", section_level=1,
                chunk_index=0, doc_date=now,
            ),
        ]

        HybridRetriever._apply_metadata_boost(
            chunks, section_level_boost=0.0, recency_boost=0.10,
        )

        # Newest should have highest score
        newest = [c for c in chunks if c.text == "new"][0]
        oldest = [c for c in chunks if c.text == "old"][0]
        assert newest.score > oldest.score

        # Newest gets full boost (0.5 + 0.10 = 0.60)
        assert newest.score == pytest.approx(0.60, abs=1e-6)

    def test_scores_clamped_to_unit(self) -> None:
        """Scores should be clamped to [0, 1] after boosting."""
        chunks = [
            RetrievedChunk(
                text="high", score=0.98, chunk_id="a#0", file_path="a.md",
                file_type="md", section_title="A", section_level=1,
                chunk_index=0,
            ),
        ]
        HybridRetriever._apply_metadata_boost(
            chunks, section_level_boost=0.10, recency_boost=0.0,
        )
        assert chunks[0].score <= 1.0

    def test_re_sorted_by_boosted_score(self) -> None:
        """After boost, chunks should be re-sorted by score descending."""
        chunks = [
            RetrievedChunk(
                text="low_level", score=0.80, chunk_id="a#0", file_path="a.md",
                file_type="md", section_title="A", section_level=5,
                chunk_index=0,
            ),
            RetrievedChunk(
                text="high_level", score=0.75, chunk_id="b#0", file_path="b.md",
                file_type="md", section_title="B", section_level=1,
                chunk_index=0,
            ),
        ]

        HybridRetriever._apply_metadata_boost(
            chunks, section_level_boost=0.10, recency_boost=0.0,
        )

        # high_level started at 0.75 but gets +0.10 = 0.85
        # low_level started at 0.80 but gets 0 = 0.80
        # So high_level should be first now
        assert chunks[0].text == "high_level"
        assert chunks[1].text == "low_level"

    def test_empty_chunks_no_error(self) -> None:
        """Applying boost to empty list should not raise."""
        result = HybridRetriever._apply_metadata_boost(
            [], section_level_boost=0.05, recency_boost=0.03,
        )
        assert result == []

    def test_no_dated_chunks_recency_skipped(self) -> None:
        """If no chunks have doc_date > 0, recency boost should be skipped."""
        chunks = [
            RetrievedChunk(
                text="t", score=0.5, chunk_id="a#0", file_path="a.md",
                file_type="md", section_title="A", section_level=1,
                chunk_index=0, doc_date=0.0,
            ),
        ]
        HybridRetriever._apply_metadata_boost(
            chunks, section_level_boost=0.0, recency_boost=0.10,
        )
        # No recency boost applied, section_level_boost=0, so score unchanged
        assert chunks[0].score == pytest.approx(0.5, abs=1e-6)

    def test_content_type_field_on_chunk(self) -> None:
        """RetrievedChunk should have a content_type field."""
        chunk = RetrievedChunk(
            text="t", score=0.5, chunk_id="a#0", file_path="a.md",
            file_type="md", section_title="A", section_level=1,
            chunk_index=0, content_type="table",
        )
        assert chunk.content_type == "table"

    def test_content_type_default_is_prose(self) -> None:
        """Default content_type should be 'prose'."""
        chunk = RetrievedChunk(
            text="t", score=0.5, chunk_id="a#0", file_path="a.md",
            file_type="md", section_title="A", section_level=1,
            chunk_index=0,
        )
        assert chunk.content_type == "prose"
