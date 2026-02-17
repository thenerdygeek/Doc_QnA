"""Tests for score normalization functions."""
from __future__ import annotations

import math

import pytest

from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.retrieval.score_normalizer import (
    filter_by_score,
    normalize_min_max,
    normalize_sigmoid,
)


def _make_chunk(score: float, chunk_id: str = "c0") -> RetrievedChunk:
    return RetrievedChunk(
        text="test",
        score=score,
        chunk_id=chunk_id,
        file_path="/tmp/test.md",
        file_type="md",
        section_title="Test",
        section_level=1,
        chunk_index=0,
    )


# ── normalize_min_max ────────────────────────────────────────────


class TestNormalizeMinMax:
    def test_empty_list(self):
        assert normalize_min_max([]) == []

    def test_single_chunk(self):
        chunks = [_make_chunk(0.015)]
        normalize_min_max(chunks)
        assert chunks[0].score == 1.0

    def test_identical_scores(self):
        chunks = [_make_chunk(0.02, f"c{i}") for i in range(3)]
        normalize_min_max(chunks)
        for c in chunks:
            assert c.score == 1.0

    def test_two_chunks(self):
        chunks = [_make_chunk(0.03, "c0"), _make_chunk(0.01, "c1")]
        normalize_min_max(chunks)
        assert chunks[0].score == pytest.approx(1.0)
        assert chunks[1].score == pytest.approx(0.0)

    def test_multiple_chunks_rrf_range(self):
        """Typical RRF scores (~0.01 to ~0.03) should map to [0, 1]."""
        chunks = [
            _make_chunk(0.033, "c0"),
            _make_chunk(0.020, "c1"),
            _make_chunk(0.015, "c2"),
            _make_chunk(0.010, "c3"),
        ]
        normalize_min_max(chunks)

        assert chunks[0].score == pytest.approx(1.0)
        assert chunks[-1].score == pytest.approx(0.0)
        # Middle scores should be between 0 and 1
        for c in chunks[1:-1]:
            assert 0.0 < c.score < 1.0

    def test_preserves_order(self):
        chunks = [_make_chunk(0.01, "c0"), _make_chunk(0.03, "c1")]
        normalize_min_max(chunks)
        assert chunks[0].chunk_id == "c0"
        assert chunks[1].chunk_id == "c1"

    def test_returns_same_list(self):
        chunks = [_make_chunk(0.5)]
        result = normalize_min_max(chunks)
        assert result is chunks


# ── normalize_sigmoid ────────────────────────────────────────────


class TestNormalizeSigmoid:
    def test_empty_list(self):
        assert normalize_sigmoid([]) == []

    def test_zero_score_maps_to_half(self):
        chunks = [_make_chunk(0.0)]
        normalize_sigmoid(chunks)
        assert chunks[0].score == pytest.approx(0.5)

    def test_positive_score_above_half(self):
        chunks = [_make_chunk(5.0)]
        normalize_sigmoid(chunks)
        assert chunks[0].score > 0.5

    def test_negative_score_below_half(self):
        chunks = [_make_chunk(-3.0)]
        normalize_sigmoid(chunks)
        assert chunks[0].score < 0.5

    def test_large_positive_near_one(self):
        chunks = [_make_chunk(10.0)]
        normalize_sigmoid(chunks)
        assert chunks[0].score > 0.99

    def test_large_negative_near_zero(self):
        chunks = [_make_chunk(-10.0)]
        normalize_sigmoid(chunks)
        assert chunks[0].score < 0.01

    def test_shift_parameter(self):
        """With shift=3, score=3 should map to 0.5."""
        chunks = [_make_chunk(3.0)]
        normalize_sigmoid(chunks, shift=3.0)
        assert chunks[0].score == pytest.approx(0.5)

    def test_scale_parameter(self):
        """Higher scale makes the transition sharper."""
        chunk_soft = _make_chunk(1.0, "soft")
        chunk_sharp = _make_chunk(1.0, "sharp")
        normalize_sigmoid([chunk_soft], scale=0.5)
        normalize_sigmoid([chunk_sharp], scale=2.0)
        # Both should be > 0.5, but sharp should be closer to 1.0
        assert chunk_soft.score > 0.5
        assert chunk_sharp.score > chunk_soft.score

    def test_output_bounded(self):
        """Outputs should be in [0, 1].  Extreme values may hit 0.0/1.0
        exactly due to float64 precision limits, which is acceptable."""
        chunks = [_make_chunk(s, f"c{i}") for i, s in enumerate([-10, -5, 0, 5, 10])]
        normalize_sigmoid(chunks)
        for c in chunks:
            assert 0.0 < c.score < 1.0

    def test_preserves_ranking(self):
        chunks = [_make_chunk(8.0, "c0"), _make_chunk(2.0, "c1"), _make_chunk(-1.0, "c2")]
        normalize_sigmoid(chunks)
        assert chunks[0].score > chunks[1].score > chunks[2].score

    def test_returns_same_list(self):
        chunks = [_make_chunk(0.0)]
        result = normalize_sigmoid(chunks)
        assert result is chunks


# ── filter_by_score ──────────────────────────────────────────────


class TestFilterByScore:
    def test_empty_list(self):
        assert filter_by_score([], 0.5) == []

    def test_all_pass(self):
        chunks = [_make_chunk(0.8, "c0"), _make_chunk(0.6, "c1")]
        assert filter_by_score(chunks, 0.5) == chunks

    def test_all_fail(self):
        chunks = [_make_chunk(0.1, "c0"), _make_chunk(0.2, "c1")]
        assert filter_by_score(chunks, 0.5) == []

    def test_partial_filter(self):
        chunks = [
            _make_chunk(0.9, "c0"),
            _make_chunk(0.3, "c1"),
            _make_chunk(0.7, "c2"),
        ]
        result = filter_by_score(chunks, 0.5)
        assert len(result) == 2
        assert result[0].chunk_id == "c0"
        assert result[1].chunk_id == "c2"

    def test_boundary_exact_match(self):
        """Chunks at exactly min_score should be kept."""
        chunks = [_make_chunk(0.5, "c0")]
        assert len(filter_by_score(chunks, 0.5)) == 1

    def test_zero_threshold(self):
        chunks = [_make_chunk(0.0, "c0"), _make_chunk(0.001, "c1")]
        assert len(filter_by_score(chunks, 0.0)) == 2
