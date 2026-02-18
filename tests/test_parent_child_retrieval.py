"""Tests for parent-child retrieval (context building + migration)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from doc_qa.retrieval.retriever import RetrievedChunk


def _make_chunk(
    chunk_id: str = "file.md#0",
    text: str = "child text",
    parent_chunk_id: str = "",
    parent_text: str = "",
    score: float = 0.9,
    file_path: str = "docs/test.md",
    section_title: str = "Intro",
    doc_date: float = 0.0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        chunk_id=chunk_id,
        file_path=file_path,
        file_type="md",
        section_title=section_title,
        section_level=1,
        chunk_index=0,
        parent_chunk_id=parent_chunk_id,
        parent_text=parent_text,
        doc_date=doc_date,
    )


class TestBuildContextParentDedup:
    """Tests for _build_context with parent retrieval enabled."""

    def _make_pipeline(self, enable_parent: bool = True):
        """Create a minimal QueryPipeline with parent retrieval config."""
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        # Create pipeline with mocked dependencies
        pipeline = QueryPipeline.__new__(QueryPipeline)
        pipeline._enable_parent_retrieval = enable_parent
        return pipeline

    def test_parent_text_used_when_enabled(self) -> None:
        """When parent retrieval is on, parent_text appears in context."""
        pipeline = self._make_pipeline(enable_parent=True)
        chunks = [
            _make_chunk(
                chunk_id="docs/test.md#0",
                text="child snippet",
                parent_chunk_id="docs/test.md#parent_0",
                parent_text="full parent paragraph with more context",
            ),
        ]
        ctx = pipeline._build_context(chunks)
        assert "full parent paragraph with more context" in ctx
        assert "child snippet" not in ctx

    def test_child_text_used_when_disabled(self) -> None:
        """When parent retrieval is off, child text is used directly."""
        pipeline = self._make_pipeline(enable_parent=False)
        chunks = [
            _make_chunk(
                chunk_id="docs/test.md#0",
                text="child snippet",
                parent_chunk_id="docs/test.md#parent_0",
                parent_text="full parent paragraph",
            ),
        ]
        ctx = pipeline._build_context(chunks)
        assert "child snippet" in ctx
        assert "full parent paragraph" not in ctx

    def test_dedup_multiple_children_same_parent(self) -> None:
        """Multiple children from same parent appear only once in context."""
        pipeline = self._make_pipeline(enable_parent=True)
        parent_id = "docs/test.md#parent_0"
        parent_text = "The complete parent text block"
        chunks = [
            _make_chunk(
                chunk_id="docs/test.md#0",
                text="child A",
                parent_chunk_id=parent_id,
                parent_text=parent_text,
                score=0.95,
            ),
            _make_chunk(
                chunk_id="docs/test.md#1",
                text="child B",
                parent_chunk_id=parent_id,
                parent_text=parent_text,
                score=0.90,
            ),
            _make_chunk(
                chunk_id="docs/test.md#2",
                text="child C from different parent",
                parent_chunk_id="docs/test.md#parent_1",
                parent_text="A different parent text",
                score=0.85,
            ),
        ]
        ctx = pipeline._build_context(chunks)
        # Parent text should appear exactly once
        assert ctx.count(parent_text) == 1
        # The other parent also appears once
        assert "A different parent text" in ctx

    def test_fallback_to_child_when_no_parent(self) -> None:
        """Chunks without parent info use child text even when parent retrieval is on."""
        pipeline = self._make_pipeline(enable_parent=True)
        chunks = [
            _make_chunk(
                chunk_id="docs/test.md#0",
                text="standalone child text",
                parent_chunk_id="",
                parent_text="",
            ),
        ]
        ctx = pipeline._build_context(chunks)
        assert "standalone child text" in ctx

    def test_empty_chunks(self) -> None:
        """Empty chunk list produces empty context."""
        pipeline = self._make_pipeline(enable_parent=True)
        assert pipeline._build_context([]) == ""

    def test_mixed_parent_and_standalone_chunks(self) -> None:
        """Mix of parent-child and standalone chunks both appear in context."""
        pipeline = self._make_pipeline(enable_parent=True)
        chunks = [
            _make_chunk(
                chunk_id="docs/a.md#0",
                text="child with parent",
                parent_chunk_id="docs/a.md#parent_0",
                parent_text="parent A text",
                score=0.9,
            ),
            _make_chunk(
                chunk_id="docs/b.md#0",
                text="standalone text",
                parent_chunk_id="",
                parent_text="",
                score=0.8,
            ),
        ]
        ctx = pipeline._build_context(chunks)
        assert "parent A text" in ctx
        assert "standalone text" in ctx


class TestRetrievedChunkParentFields:
    """Verify RetrievedChunk dataclass has parent fields with defaults."""

    def test_default_parent_fields(self) -> None:
        """Parent fields default to empty string."""
        chunk = RetrievedChunk(
            text="test",
            score=0.5,
            chunk_id="t#0",
            file_path="t.md",
            file_type="md",
            section_title="",
            section_level=1,
            chunk_index=0,
        )
        assert chunk.parent_chunk_id == ""
        assert chunk.parent_text == ""

    def test_parent_fields_populated(self) -> None:
        """Parent fields can be set explicitly."""
        chunk = _make_chunk(
            parent_chunk_id="file.md#parent_0",
            parent_text="parent context",
        )
        assert chunk.parent_chunk_id == "file.md#parent_0"
        assert chunk.parent_text == "parent context"


class TestMigrationSelfParenting:
    """Tests that schema migration creates self-parenting chunks."""

    def test_chunk_dataclass_parent_defaults(self) -> None:
        """Chunk dataclass defaults for parent_chunk_id and parent_text."""
        from doc_qa.indexing.chunker import Chunk

        chunk = Chunk(
            chunk_id="file.md#0",
            text="some text",
            file_path="file.md",
            file_type="md",
            section_title="Title",
            section_level=1,
            chunk_index=0,
        )
        assert chunk.parent_chunk_id == ""
        assert chunk.parent_text == ""
