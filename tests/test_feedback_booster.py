"""Tests for active learning feedback-based score boosting."""
from __future__ import annotations

import pytest

from doc_qa.retrieval.feedback_booster import apply_chunk_feedback_boost
from doc_qa.retrieval.retriever import RetrievedChunk


def _make_chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        text="test",
        score=score,
        chunk_id=chunk_id,
        file_path="test.md",
        file_type="md",
        section_title="",
        section_level=1,
        chunk_index=0,
    )


class TestApplyChunkFeedbackBoost:

    def test_positive_boost_increases_score(self) -> None:
        """Positive feedback should increase chunk score."""
        chunks = [_make_chunk("a#0", 0.8)]
        fb = {"a#0": 1.0}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == pytest.approx(0.9, abs=1e-6)

    def test_negative_feedback_decreases_score(self) -> None:
        """Negative feedback should decrease chunk score."""
        chunks = [_make_chunk("a#0", 0.8)]
        fb = {"a#0": -1.0}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == pytest.approx(0.7, abs=1e-6)

    def test_clamping_upper_bound(self) -> None:
        """Score should not exceed 1.0."""
        chunks = [_make_chunk("a#0", 0.95)]
        fb = {"a#0": 1.0}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == 1.0

    def test_clamping_lower_bound(self) -> None:
        """Score should not go below 0.0."""
        chunks = [_make_chunk("a#0", 0.05)]
        fb = {"a#0": -1.0}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == 0.0

    def test_empty_feedback_is_noop(self) -> None:
        """Empty feedback dict should not change anything."""
        chunks = [_make_chunk("a#0", 0.5)]
        original_score = chunks[0].score
        apply_chunk_feedback_boost(chunks, {}, max_boost=0.10)
        assert chunks[0].score == original_score

    def test_empty_chunks_is_noop(self) -> None:
        """Empty chunk list returns immediately."""
        result = apply_chunk_feedback_boost([], {"a#0": 1.0}, max_boost=0.10)
        assert result == []

    def test_max_boost_cap_respected(self) -> None:
        """Score shift should not exceed max_boost."""
        chunks = [_make_chunk("a#0", 0.5)]
        fb = {"a#0": 1.0}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.05)
        assert chunks[0].score == pytest.approx(0.55, abs=1e-6)

    def test_partial_feedback_score(self) -> None:
        """Feedback score of 0.5 should apply half the max_boost."""
        chunks = [_make_chunk("a#0", 0.5)]
        fb = {"a#0": 0.5}
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == pytest.approx(0.55, abs=1e-6)

    def test_resort_order_after_boost(self) -> None:
        """Chunks should be re-sorted by score after boost."""
        chunks = [
            _make_chunk("a#0", 0.7),
            _make_chunk("b#0", 0.8),
        ]
        fb = {"a#0": 1.0, "b#0": -1.0}  # boost a, penalize b
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].chunk_id == "a#0"
        assert chunks[0].score == pytest.approx(0.8, abs=1e-6)
        assert chunks[1].chunk_id == "b#0"
        assert chunks[1].score == pytest.approx(0.7, abs=1e-6)

    def test_unmatched_chunks_unchanged(self) -> None:
        """Chunks not in feedback dict should retain original score."""
        chunks = [
            _make_chunk("a#0", 0.5),
            _make_chunk("b#0", 0.6),
        ]
        fb = {"a#0": 1.0}  # only a has feedback
        apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert chunks[0].score == pytest.approx(0.6, abs=1e-6)  # a boosted
        # b unchanged at 0.6 — but may be reordered
        b_chunk = next(c for c in chunks if c.chunk_id == "b#0")
        assert b_chunk.score == pytest.approx(0.6, abs=1e-6)

    def test_returns_same_list(self) -> None:
        """Function modifies in-place and returns the same list object."""
        chunks = [_make_chunk("a#0", 0.5)]
        fb = {"a#0": 0.5}
        result = apply_chunk_feedback_boost(chunks, fb, max_boost=0.10)
        assert result is chunks
