"""Tests for the retrieval evaluation metrics and runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_qa.eval.evaluator import (
    CaseResult,
    EvalSummary,
    TestCase,
    evaluate,
    format_report,
    hit_at_k,
    load_test_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        """All retrieved items are relevant → precision = 1.0."""
        assert precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0

    def test_none_relevant(self) -> None:
        """No retrieved items are relevant → precision = 0.0."""
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_half_relevant(self) -> None:
        """Half of retrieved items are relevant → precision = 0.5."""
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_respects_k(self) -> None:
        """Should only consider top-k results."""
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=1) == 1.0

    def test_empty_retrieval(self) -> None:
        """Empty retrieval → precision = 0.0."""
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_deduplicates_files(self) -> None:
        """Multiple chunks from same relevant file should count as one hit."""
        # 5 chunks, 2 unique files: "a" (3x) and "x" (2x), "a" is relevant
        assert precision_at_k(["a", "a", "a", "x", "x"], {"a"}, k=5) == 0.5


class TestRecallAtK:
    def test_all_found(self) -> None:
        """All relevant items found → recall = 1.0."""
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_found(self) -> None:
        """No relevant items found → recall = 0.0."""
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_found(self) -> None:
        """Half of relevant items found → recall = 0.5."""
        assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant(self) -> None:
        """No relevant items (nothing to recall) → recall = 1.0."""
        assert recall_at_k(["a", "b"], set(), k=2) == 1.0

    def test_duplicates_counted_once(self) -> None:
        """Same file appearing multiple times should count as one found."""
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


class TestReciprocalRank:
    def test_first_is_relevant(self) -> None:
        """First result is relevant → RR = 1.0."""
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self) -> None:
        """Second result is relevant → RR = 0.5."""
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_third_is_relevant(self) -> None:
        """Third result is relevant → RR = 1/3."""
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_none_relevant(self) -> None:
        """No relevant results → RR = 0.0."""
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieval(self) -> None:
        """No results → RR = 0.0."""
        assert reciprocal_rank([], {"a"}) == 0.0


class TestHitAtK:
    def test_hit_present(self) -> None:
        assert hit_at_k(["a", "x"], {"a"}, k=2) is True

    def test_no_hit(self) -> None:
        assert hit_at_k(["x", "y"], {"a"}, k=2) is False

    def test_hit_beyond_k(self) -> None:
        """Relevant item exists but beyond k → miss."""
        assert hit_at_k(["x", "y", "a"], {"a"}, k=2) is False


class TestLoadTestCases:
    def test_load_valid_json(self, tmp_path: Path) -> None:
        """Should load test cases from valid JSON."""
        data = {
            "test_cases": [
                {
                    "question": "How does auth work?",
                    "relevant_files": ["auth.md"],
                    "relevant_keywords": ["OAuth"],
                    "difficulty": "easy",
                    "description": "Auth question",
                },
                {
                    "question": "What is the API?",
                    "relevant_files": ["api.md", "rest.md"],
                },
            ]
        }
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        cases = load_test_cases(str(p))
        assert len(cases) == 2
        assert cases[0].question == "How does auth work?"
        assert cases[0].relevant_files == ["auth.md"]
        assert cases[0].difficulty == "easy"
        assert cases[1].difficulty == "medium"  # default

    def test_empty_test_cases(self, tmp_path: Path) -> None:
        """Empty test_cases array should return empty list."""
        p = tmp_path / "empty.json"
        p.write_text('{"test_cases": []}', encoding="utf-8")
        cases = load_test_cases(str(p))
        assert cases == []


class TestEvalSummary:
    def test_passed_both_thresholds(self) -> None:
        """Summary passes when both thresholds are met."""
        s = EvalSummary(num_cases=1, avg_precision=0.7, avg_recall=0.8, mrr=0.6, hit_rate=1.0, results=[])
        assert s.passed() is True

    def test_fails_precision(self) -> None:
        """Summary fails when precision is below threshold."""
        s = EvalSummary(num_cases=1, avg_precision=0.4, avg_recall=0.8, mrr=0.6, hit_rate=1.0, results=[])
        assert s.passed() is False

    def test_fails_mrr(self) -> None:
        """Summary fails when MRR is below threshold."""
        s = EvalSummary(num_cases=1, avg_precision=0.7, avg_recall=0.8, mrr=0.3, hit_rate=1.0, results=[])
        assert s.passed() is False

    def test_custom_thresholds(self) -> None:
        """Should respect custom thresholds."""
        s = EvalSummary(num_cases=1, avg_precision=0.3, avg_recall=0.5, mrr=0.3, hit_rate=0.5, results=[])
        assert s.passed(precision_threshold=0.2, mrr_threshold=0.2) is True


class TestFormatReport:
    def test_report_contains_metrics(self) -> None:
        """Report should contain all metric labels."""
        case = CaseResult(
            question="Test?",
            precision=0.8,
            recall=0.6,
            reciprocal_rank=1.0,
            hit=True,
            retrieved_files=["a.md"],
            relevant_files=["a.md"],
            difficulty="easy",
        )
        summary = EvalSummary(
            num_cases=1,
            avg_precision=0.8,
            avg_recall=0.6,
            mrr=1.0,
            hit_rate=1.0,
            results=[case],
        )
        report = format_report(summary, k=5)
        assert "Precision@5" in report
        assert "Recall@5" in report
        assert "MRR" in report
        assert "Hit Rate" in report
        assert "PASS" in report
