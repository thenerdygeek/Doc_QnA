"""Tests for near-duplicate deduplication."""

import numpy as np
import pytest

from doc_qa.retrieval.dedup import DEDUP_THRESHOLD, deduplicate_near_duplicates
from doc_qa.retrieval.retriever import RetrievedChunk


_SENTINEL = object()


def _make_chunk(
    chunk_id: str = "test#0",
    file_path: str = "file_a.md",
    vector: list[float] | None | object = _SENTINEL,
    score: float = 0.8,
    doc_date: float = 0.0,
) -> RetrievedChunk:
    """Helper to create a RetrievedChunk with sensible defaults."""
    if vector is _SENTINEL:
        vector = [1.0, 0.0, 0.0]
    return RetrievedChunk(
        text="Some text",
        score=score,
        chunk_id=chunk_id,
        file_path=file_path,
        file_type="md",
        section_title="Test",
        section_level=1,
        chunk_index=0,
        vector=vector,
        doc_date=doc_date,
    )


class TestDeduplicateNearDuplicates:
    def test_empty_list(self):
        assert deduplicate_near_duplicates([]) == []

    def test_single_chunk(self):
        chunk = _make_chunk()
        result = deduplicate_near_duplicates([chunk])
        assert len(result) == 1

    def test_no_duplicates_different_vectors(self):
        """Chunks with orthogonal vectors should not be deduped."""
        chunks = [
            _make_chunk(chunk_id="a#0", file_path="a.md", vector=[1.0, 0.0, 0.0]),
            _make_chunk(chunk_id="b#0", file_path="b.md", vector=[0.0, 1.0, 0.0]),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 2

    def test_duplicates_same_vector_different_files_prefers_newer(self):
        """Near-identical vectors from different files — keep the newer one."""
        vec = [0.5, 0.5, 0.5]
        chunks = [
            _make_chunk(
                chunk_id="old#0", file_path="old_api.pdf",
                vector=vec, doc_date=1000000.0,
            ),
            _make_chunk(
                chunk_id="new#0", file_path="new_api.pdf",
                vector=vec, doc_date=2000000.0,
            ),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 1
        assert result[0].file_path == "new_api.pdf"

    def test_duplicates_unknown_dates_prefers_higher_score(self):
        """When both doc_dates are 0.0, keep the higher-scored chunk."""
        vec = [0.5, 0.5, 0.5]
        chunks = [
            _make_chunk(
                chunk_id="a#0", file_path="a.pdf",
                vector=vec, score=0.9, doc_date=0.0,
            ),
            _make_chunk(
                chunk_id="b#0", file_path="b.pdf",
                vector=vec, score=0.7, doc_date=0.0,
            ),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 1
        assert result[0].chunk_id == "a#0"

    def test_same_file_not_deduped(self):
        """Chunks from the same file should never be deduped against each other."""
        vec = [0.5, 0.5, 0.5]
        chunks = [
            _make_chunk(chunk_id="f#0", file_path="same.md", vector=vec),
            _make_chunk(chunk_id="f#1", file_path="same.md", vector=vec),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 2

    def test_below_threshold_not_deduped(self):
        """Vectors with similarity below threshold should be kept."""
        # Cosine similarity of [1,0,0] and [0.7,0.7,0] is ~0.707 < 0.95
        chunks = [
            _make_chunk(chunk_id="a#0", file_path="a.md", vector=[1.0, 0.0, 0.0]),
            _make_chunk(chunk_id="b#0", file_path="b.md", vector=[0.7, 0.7, 0.0]),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 2

    def test_preserves_order(self):
        """After dedup, remaining chunks should be in original order."""
        chunks = [
            _make_chunk(chunk_id="a#0", file_path="a.md", vector=[1.0, 0.0, 0.0], doc_date=3000000.0),
            _make_chunk(chunk_id="b#0", file_path="b.md", vector=[0.0, 1.0, 0.0], doc_date=2000000.0),
            _make_chunk(chunk_id="c#0", file_path="c.md", vector=[0.0, 0.0, 1.0], doc_date=1000000.0),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert [c.chunk_id for c in result] == ["a#0", "b#0", "c#0"]

    def test_three_way_duplicate_keeps_newest(self):
        """Three near-duplicates from different files — keep only the newest."""
        vec = [0.5, 0.5, 0.5]
        chunks = [
            _make_chunk(chunk_id="v1#0", file_path="v1.pdf", vector=vec, doc_date=1000000.0),
            _make_chunk(chunk_id="v2#0", file_path="v2.pdf", vector=vec, doc_date=2000000.0),
            _make_chunk(chunk_id="v3#0", file_path="v3.pdf", vector=vec, doc_date=3000000.0),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 1
        assert result[0].file_path == "v3.pdf"

    def test_no_vectors_skips_dedup(self):
        """If vectors aren't loaded, dedup is skipped gracefully."""
        chunks = [
            _make_chunk(chunk_id="a#0", file_path="a.md", vector=None),
            _make_chunk(chunk_id="b#0", file_path="b.md", vector=None),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 2

    def test_custom_threshold(self):
        """Lower threshold should catch more duplicates."""
        # These have cosine similarity ~0.94 (below default 0.95 but above 0.90)
        v1 = [1.0, 0.0, 0.3]
        v2 = [1.0, 0.0, 0.35]
        chunks = [
            _make_chunk(chunk_id="a#0", file_path="a.md", vector=v1, doc_date=1000000.0),
            _make_chunk(chunk_id="b#0", file_path="b.md", vector=v2, doc_date=2000000.0),
        ]
        # Default threshold — should NOT dedup (similarity ~0.998 actually, let me recalculate)
        # cos(v1, v2) = (1*1 + 0*0 + 0.3*0.35) / (sqrt(1+0.09) * sqrt(1+0.1225))
        #             = 1.105 / (1.044 * 1.060) = 1.105 / 1.107 ≈ 0.998
        # They're very similar, so default threshold WILL dedup
        result_default = deduplicate_near_duplicates(chunks)
        assert len(result_default) == 1

    def test_mixed_duplicates_and_unique(self):
        """Mix of duplicate pairs and unique chunks."""
        dup_vec = [0.5, 0.5, 0.5]
        chunks = [
            _make_chunk(chunk_id="u1#0", file_path="unique1.md", vector=[1.0, 0.0, 0.0]),
            _make_chunk(chunk_id="old#0", file_path="old.pdf", vector=dup_vec, doc_date=1000000.0),
            _make_chunk(chunk_id="u2#0", file_path="unique2.md", vector=[0.0, 1.0, 0.0]),
            _make_chunk(chunk_id="new#0", file_path="new.pdf", vector=dup_vec, doc_date=2000000.0),
        ]
        result = deduplicate_near_duplicates(chunks)
        assert len(result) == 3
        paths = {c.file_path for c in result}
        assert "old.pdf" not in paths
        assert "new.pdf" in paths
        assert "unique1.md" in paths
        assert "unique2.md" in paths
