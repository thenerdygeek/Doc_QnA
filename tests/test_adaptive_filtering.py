"""Tests for adaptive score filtering and dynamic top_k."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from doc_qa.retrieval.adaptive_filtering import (
    compute_adaptive_min_score,
    compute_dynamic_top_k,
)


# ── Stub chunk for testing ───────────────────────────────────────────


@dataclass
class _Chunk:
    chunk_id: str = ""
    score: float = 0.0


def _make_chunks(scores: list[float]) -> list:
    """Create stub chunks with given scores (descending order expected)."""
    return [_Chunk(chunk_id=f"c{i}", score=s) for i, s in enumerate(scores)]


# ── compute_adaptive_min_score ───────────────────────────────────────


class TestAdaptiveMinScore:
    def test_empty_chunks_returns_floor(self) -> None:
        assert compute_adaptive_min_score([]) == 0.15

    def test_single_chunk_returns_floor(self) -> None:
        chunks = _make_chunks([0.9])
        assert compute_adaptive_min_score(chunks) == 0.15

    def test_two_chunks_returns_floor(self) -> None:
        chunks = _make_chunks([0.9, 0.5])
        assert compute_adaptive_min_score(chunks) == 0.15

    def test_high_tight_scores_produce_high_threshold(self) -> None:
        """Easy query: all scores 0.8-0.9 → threshold near 0.8."""
        chunks = _make_chunks([0.90, 0.88, 0.85, 0.82, 0.80])
        threshold = compute_adaptive_min_score(chunks)
        assert threshold > 0.7, f"Expected >0.7 for tight high scores, got {threshold}"

    def test_spread_scores_produce_low_threshold(self) -> None:
        """Hard query: scores from 0.2-0.9 → threshold drops."""
        chunks = _make_chunks([0.9, 0.7, 0.5, 0.3, 0.2])
        threshold = compute_adaptive_min_score(chunks)
        assert threshold < 0.3, f"Expected <0.3 for spread scores, got {threshold}"

    def test_all_identical_scores(self) -> None:
        """All scores identical → std=0, threshold=mean, but at least floor."""
        chunks = _make_chunks([0.5, 0.5, 0.5, 0.5])
        threshold = compute_adaptive_min_score(chunks)
        # mean=0.5, std=0 → threshold=0.5
        assert threshold == pytest.approx(0.5, abs=0.01)

    def test_floor_enforced(self) -> None:
        """Very low scores shouldn't produce threshold below floor."""
        chunks = _make_chunks([0.10, 0.08, 0.05, 0.03, 0.01])
        threshold = compute_adaptive_min_score(chunks)
        assert threshold >= 0.15

    def test_custom_floor(self) -> None:
        chunks = _make_chunks([0.10, 0.08, 0.05])
        threshold = compute_adaptive_min_score(chunks, floor=0.02)
        assert threshold >= 0.02

    def test_custom_std_factor(self) -> None:
        """Higher std_factor → more aggressive filtering."""
        chunks = _make_chunks([0.9, 0.7, 0.5, 0.3, 0.2])
        t_default = compute_adaptive_min_score(chunks, std_factor=1.0)
        t_aggressive = compute_adaptive_min_score(chunks, std_factor=0.5)
        assert t_aggressive > t_default


# ── compute_dynamic_top_k ────────────────────────────────────────────


class TestDynamicTopK:
    def test_empty_chunks(self) -> None:
        assert compute_dynamic_top_k([]) == 0

    def test_single_chunk(self) -> None:
        assert compute_dynamic_top_k(_make_chunks([0.9])) == 1

    def test_two_chunks(self) -> None:
        assert compute_dynamic_top_k(_make_chunks([0.9, 0.3])) == 2

    def test_clear_gap_cuts_at_boundary(self) -> None:
        """Top 3 scores clustered, then big drop → returns 3."""
        chunks = _make_chunks([0.90, 0.85, 0.82, 0.50, 0.45, 0.40])
        k = compute_dynamic_top_k(chunks, base_k=5, gap_threshold=0.10)
        assert k == 3

    def test_no_gap_uses_base_k(self) -> None:
        """All scores evenly spaced, no big gap → use base_k."""
        chunks = _make_chunks([0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60])
        k = compute_dynamic_top_k(chunks, base_k=5, gap_threshold=0.10)
        assert k == 5

    def test_respects_max_k(self) -> None:
        """Even with no gap, never exceed max_k."""
        chunks = _make_chunks([0.9, 0.88, 0.86, 0.84, 0.82, 0.80])
        k = compute_dynamic_top_k(chunks, base_k=20, max_k=4)
        assert k <= 4

    def test_gap_at_min_results_boundary(self) -> None:
        """Gap right after min_results (2) → returns 2."""
        chunks = _make_chunks([0.90, 0.85, 0.40, 0.35])
        k = compute_dynamic_top_k(chunks, gap_threshold=0.10)
        assert k == 2

    def test_multiple_gaps_picks_first(self) -> None:
        """Multiple gaps → cuts at the first one (most relevant boundary)."""
        chunks = _make_chunks([0.90, 0.88, 0.60, 0.58, 0.30, 0.28])
        k = compute_dynamic_top_k(chunks, gap_threshold=0.10)
        assert k == 2

    def test_small_gap_not_triggered(self) -> None:
        """Gap below threshold doesn't trigger a cut."""
        chunks = _make_chunks([0.90, 0.85, 0.82, 0.76, 0.70])
        k = compute_dynamic_top_k(chunks, base_k=5, gap_threshold=0.10)
        assert k == 5

    def test_custom_gap_threshold(self) -> None:
        """Lower gap threshold triggers cut on smaller drops."""
        chunks = _make_chunks([0.90, 0.85, 0.82, 0.76, 0.70])
        k = compute_dynamic_top_k(chunks, base_k=5, gap_threshold=0.05)
        # 0.82→0.76 = gap of 0.06 > 0.05 → cut at 3
        assert k == 3
