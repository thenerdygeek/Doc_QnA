"""Tests for the retrieval evaluation metrics and runner."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from doc_qa.eval.evaluator import (
    CaseResult,
    ComparisonResult,
    DifficultyBreakdown,
    EvalSummary,
    TestCase,
    _build_relevance_map,
    _compute_difficulty_breakdown,
    compare_evaluations,
    evaluate,
    f1_at_k,
    format_comparison,
    format_report,
    hit_at_k,
    load_test_cases,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from doc_qa.retrieval.retriever import RetrievedChunk


# ── Helpers ──────────────────────────────────────────────────────────

def _make_chunk(file_path: str, score: float = 0.5) -> RetrievedChunk:
    """Create a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        text="dummy",
        score=score,
        chunk_id="c1",
        file_path=file_path,
        file_type="md",
        section_title="Section",
        section_level=1,
        chunk_index=0,
    )


def _make_retriever(file_paths: list[str]) -> MagicMock:
    """Create a mock HybridRetriever that returns chunks for the given file paths."""
    retriever = MagicMock()
    chunks = [_make_chunk(fp, score=1.0 - 0.1 * i) for i, fp in enumerate(file_paths)]
    retriever.search.return_value = chunks
    return retriever


# ── Precision@k ──────────────────────────────────────────────────────

class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        """All retrieved items are relevant -> precision = 1.0."""
        assert precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0

    def test_none_relevant(self) -> None:
        """No retrieved items are relevant -> precision = 0.0."""
        assert precision_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_half_relevant(self) -> None:
        """Half of retrieved items are relevant -> precision = 0.5."""
        assert precision_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_respects_k(self) -> None:
        """Should only consider top-k results."""
        assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=1) == 1.0

    def test_empty_retrieval(self) -> None:
        """Empty retrieval -> precision = 0.0."""
        assert precision_at_k([], {"a"}, k=5) == 0.0

    def test_deduplicates_files(self) -> None:
        """Multiple chunks from same relevant file should count as one hit."""
        # 5 chunks, 2 unique files: "a" (3x) and "x" (2x), "a" is relevant
        assert precision_at_k(["a", "a", "a", "x", "x"], {"a"}, k=5) == 0.5


# ── Recall@k ─────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_all_found(self) -> None:
        """All relevant items found -> recall = 1.0."""
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_found(self) -> None:
        """No relevant items found -> recall = 0.0."""
        assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_partial_found(self) -> None:
        """Half of relevant items found -> recall = 0.5."""
        assert recall_at_k(["a", "x"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant(self) -> None:
        """No relevant items (nothing to recall) -> recall = 1.0."""
        assert recall_at_k(["a", "b"], set(), k=2) == 1.0

    def test_duplicates_counted_once(self) -> None:
        """Same file appearing multiple times should count as one found."""
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


# ── Reciprocal Rank ──────────────────────────────────────────────────

class TestReciprocalRank:
    def test_first_is_relevant(self) -> None:
        """First result is relevant -> RR = 1.0."""
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self) -> None:
        """Second result is relevant -> RR = 0.5."""
        assert reciprocal_rank(["x", "a", "c"], {"a"}) == 0.5

    def test_third_is_relevant(self) -> None:
        """Third result is relevant -> RR = 1/3."""
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_none_relevant(self) -> None:
        """No relevant results -> RR = 0.0."""
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieval(self) -> None:
        """No results -> RR = 0.0."""
        assert reciprocal_rank([], {"a"}) == 0.0


# ── Hit@k ────────────────────────────────────────────────────────────

class TestHitAtK:
    def test_hit_present(self) -> None:
        assert hit_at_k(["a", "x"], {"a"}, k=2) is True

    def test_no_hit(self) -> None:
        assert hit_at_k(["x", "y"], {"a"}, k=2) is False

    def test_hit_beyond_k(self) -> None:
        """Relevant item exists but beyond k -> miss."""
        assert hit_at_k(["x", "y", "a"], {"a"}, k=2) is False


# ── NDCG@k ───────────────────────────────────────────────────────────

class TestNdcgAtK:
    def test_perfect_ranking(self) -> None:
        """Perfect ranking: all relevant items at top in correct order."""
        relevance_map = {"a": 2, "b": 1}
        # Retrieved: ["a", "b"] -> DCG = 2/log2(2) + 1/log2(3)
        # IDCG = 2/log2(2) + 1/log2(3) (same order is ideal)
        assert ndcg_at_k(["a", "b"], relevance_map, k=2) == pytest.approx(1.0)

    def test_reversed_ranking(self) -> None:
        """Reversed ranking: lower-grade item first."""
        relevance_map = {"a": 2, "b": 1}
        # Retrieved: ["b", "a"] -> DCG = 1/log2(2) + 2/log2(3)
        # IDCG = 2/log2(2) + 1/log2(3)
        dcg = 1.0 / math.log2(2) + 2.0 / math.log2(3)
        idcg = 2.0 / math.log2(2) + 1.0 / math.log2(3)
        expected = dcg / idcg
        assert ndcg_at_k(["b", "a"], relevance_map, k=2) == pytest.approx(expected)

    def test_no_relevant(self) -> None:
        """No relevant items -> NDCG = 0.0."""
        relevance_map = {"a": 2}
        assert ndcg_at_k(["x", "y"], relevance_map, k=2) == 0.0

    def test_empty_retrieval(self) -> None:
        """Empty retrieval -> NDCG = 0.0."""
        assert ndcg_at_k([], {"a": 2}, k=5) == 0.0

    def test_empty_relevance_map(self) -> None:
        """Empty relevance map -> NDCG = 0.0."""
        assert ndcg_at_k(["a", "b"], {}, k=2) == 0.0

    def test_k_zero(self) -> None:
        """k=0 -> NDCG = 0.0."""
        assert ndcg_at_k(["a"], {"a": 2}, k=0) == 0.0

    def test_all_irrelevant_grades(self) -> None:
        """All grades are 0 -> NDCG = 0.0."""
        assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, k=2) == 0.0

    def test_partial_relevance(self) -> None:
        """Mix of relevant and irrelevant items."""
        relevance_map = {"a": 2, "b": 1, "c": 0}
        # Retrieved: ["c", "a", "b"] (worst first)
        # DCG = 0/log2(2) + 2/log2(3) + 1/log2(4)
        dcg = 0.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        # IDCG = 2/log2(2) + 1/log2(3) + 0/log2(4)
        idcg = 2.0 / math.log2(2) + 1.0 / math.log2(3) + 0.0 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(["c", "a", "b"], relevance_map, k=3) == pytest.approx(expected)

    def test_deduplicates_retrieved(self) -> None:
        """Duplicate files in retrieval are deduplicated."""
        relevance_map = {"a": 2}
        # ["a", "a"] -> unique: ["a"] -> DCG = 2/log2(2), IDCG = 2/log2(2)
        assert ndcg_at_k(["a", "a"], relevance_map, k=2) == pytest.approx(1.0)

    def test_single_item_perfect(self) -> None:
        """Single relevant item at position 1."""
        assert ndcg_at_k(["a"], {"a": 2}, k=1) == pytest.approx(1.0)

    def test_known_value(self) -> None:
        """Verify NDCG with a manually computed value."""
        # 3 docs: a=3, b=2, c=1. Retrieved: ["b", "c", "a"] at k=3
        relevance_map = {"a": 3, "b": 2, "c": 1}
        # DCG = 2/log2(2) + 1/log2(3) + 3/log2(4)
        dcg = 2 / math.log2(2) + 1 / math.log2(3) + 3 / math.log2(4)
        # IDCG = 3/log2(2) + 2/log2(3) + 1/log2(4)
        idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        expected = dcg / idcg
        assert ndcg_at_k(["b", "c", "a"], relevance_map, k=3) == pytest.approx(expected)


# ── F1@k ─────────────────────────────────────────────────────────────

class TestF1AtK:
    def test_perfect_scores(self) -> None:
        """Both precision and recall are 1.0 -> F1 = 1.0."""
        assert f1_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_zero_precision_and_recall(self) -> None:
        """Both precision and recall are 0.0 -> F1 = 0.0."""
        assert f1_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0

    def test_empty_retrieval(self) -> None:
        """Empty retrieval -> F1 = 0.0."""
        assert f1_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        """No relevant items -> recall = 1.0, precision = 0.0 when k>0 retrieval has items.

        Actually: recall_at_k returns 1.0 for empty relevant set,
        but precision_at_k returns 0.0 since no retrieved item is in empty set.
        F1 = 2 * 0.0 * 1.0 / (0.0 + 1.0) = 0.0.
        """
        assert f1_at_k(["x"], set(), k=1) == 0.0

    def test_known_value(self) -> None:
        """F1 with known precision and recall.

        Retrieved: ["a", "x"], relevant: {"a", "b"}, k=2
        Precision = 0.5, Recall = 0.5
        F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        """
        assert f1_at_k(["a", "x"], {"a", "b"}, k=2) == pytest.approx(0.5)

    def test_high_precision_low_recall(self) -> None:
        """Precision = 1.0, Recall = 0.25 -> F1 = 0.4."""
        # Retrieved: ["a"], relevant: {"a", "b", "c", "d"}, k=1
        # P = 1.0, R = 0.25, F1 = 2 * 1 * 0.25 / (1 + 0.25) = 0.4
        assert f1_at_k(["a"], {"a", "b", "c", "d"}, k=1) == pytest.approx(0.4)

    def test_with_duplicates(self) -> None:
        """Duplicates in retrieval handled correctly."""
        # ["a", "a", "b"], relevant={"a", "b"}, k=3
        # P = 2/2 = 1.0 (unique: a, b), R = 2/2 = 1.0
        assert f1_at_k(["a", "a", "b"], {"a", "b"}, k=3) == pytest.approx(1.0)


# ── Load Test Cases ──────────────────────────────────────────────────

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

    def test_load_with_relevance_grades(self, tmp_path: Path) -> None:
        """Should load relevance_grades and relevant_sections."""
        data = {
            "test_cases": [
                {
                    "question": "How does auth work?",
                    "relevant_files": ["auth.md", "login.adoc"],
                    "relevance_grades": {"auth.md": 2, "login.adoc": 1},
                    "relevant_sections": ["Authentication Overview", "Login Flow"],
                },
            ]
        }
        p = tmp_path / "graded.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        cases = load_test_cases(str(p))
        assert len(cases) == 1
        assert cases[0].relevance_grades == {"auth.md": 2, "login.adoc": 1}
        assert cases[0].relevant_sections == ["Authentication Overview", "Login Flow"]

    def test_load_without_new_fields(self, tmp_path: Path) -> None:
        """Missing new fields should get default values (backward compat)."""
        data = {
            "test_cases": [
                {
                    "question": "Test?",
                    "relevant_files": ["a.md"],
                },
            ]
        }
        p = tmp_path / "old_format.json"
        p.write_text(json.dumps(data), encoding="utf-8")

        cases = load_test_cases(str(p))
        assert cases[0].relevance_grades == {}
        assert cases[0].relevant_sections == []


# ── EvalSummary ──────────────────────────────────────────────────────

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

    def test_new_fields_have_defaults(self) -> None:
        """New fields should have sensible defaults for backward compat."""
        s = EvalSummary(num_cases=1, avg_precision=0.5, avg_recall=0.5, mrr=0.5, hit_rate=0.5, results=[])
        assert s.avg_ndcg == 0.0
        assert s.avg_f1 == 0.0
        assert s.by_difficulty == []


# ── Format Report ────────────────────────────────────────────────────

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
            ndcg=0.9,
            f1=0.686,
        )
        summary = EvalSummary(
            num_cases=1,
            avg_precision=0.8,
            avg_recall=0.6,
            mrr=1.0,
            hit_rate=1.0,
            results=[case],
            avg_ndcg=0.9,
            avg_f1=0.686,
        )
        report = format_report(summary, k=5)
        assert "Precision@5" in report
        assert "Recall@5" in report
        assert "MRR" in report
        assert "Hit Rate" in report
        assert "NDCG@5" in report
        assert "F1@5" in report
        assert "PASS" in report

    def test_report_contains_ndcg_and_f1_columns(self) -> None:
        """Per-case lines should contain NDCG and F1 values."""
        case = CaseResult(
            question="Short Q",
            precision=1.0,
            recall=1.0,
            reciprocal_rank=1.0,
            hit=True,
            retrieved_files=["a.md"],
            relevant_files=["a.md"],
            difficulty="easy",
            ndcg=1.0,
            f1=1.0,
        )
        summary = EvalSummary(
            num_cases=1,
            avg_precision=1.0,
            avg_recall=1.0,
            mrr=1.0,
            hit_rate=1.0,
            results=[case],
            avg_ndcg=1.0,
            avg_f1=1.0,
        )
        report = format_report(summary, k=5)
        # The per-case line should have NDCG and F1 header
        assert "NDCG" in report
        assert "F1" in report

    def test_report_contains_difficulty_breakdown(self) -> None:
        """Report should contain per-difficulty breakdown table."""
        cases = [
            CaseResult(
                question="Easy Q", precision=1.0, recall=1.0, reciprocal_rank=1.0,
                hit=True, retrieved_files=["a.md"], relevant_files=["a.md"],
                difficulty="easy", ndcg=1.0, f1=1.0,
            ),
            CaseResult(
                question="Hard Q", precision=0.5, recall=0.5, reciprocal_rank=0.5,
                hit=True, retrieved_files=["a.md", "x.md"], relevant_files=["a.md", "b.md"],
                difficulty="hard", ndcg=0.6, f1=0.5,
            ),
        ]
        summary = EvalSummary(
            num_cases=2,
            avg_precision=0.75,
            avg_recall=0.75,
            mrr=0.75,
            hit_rate=1.0,
            results=cases,
            avg_ndcg=0.8,
            avg_f1=0.75,
            by_difficulty=[
                DifficultyBreakdown("easy", 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                DifficultyBreakdown("hard", 1, 0.5, 0.5, 0.6, 0.5, 0.5, 1.0),
            ],
        )
        report = format_report(summary, k=5)
        assert "Per-Difficulty Breakdown" in report
        assert "easy" in report
        assert "hard" in report


# ── build_relevance_map ──────────────────────────────────────────────

class TestBuildRelevanceMap:
    def test_uses_explicit_grades(self) -> None:
        """Should use explicit relevance_grades when provided."""
        case = TestCase(
            question="Q",
            relevant_files=["a.md", "b.md"],
            relevance_grades={"a.md": 2, "b.md": 1},
        )
        result = _build_relevance_map(case)
        assert result == {"a.md": 2, "b.md": 1}

    def test_defaults_to_grade_2(self) -> None:
        """Should default all relevant_files to grade 2 when no explicit grades."""
        case = TestCase(
            question="Q",
            relevant_files=["a.md", "b.md"],
        )
        result = _build_relevance_map(case)
        assert result == {"a.md": 2, "b.md": 2}


# ── Per-Difficulty Breakdown ─────────────────────────────────────────

class TestDifficultyBreakdown:
    def test_single_difficulty(self) -> None:
        """Single difficulty group should aggregate correctly."""
        results = [
            CaseResult("Q1", 0.8, 0.6, 1.0, True, ["a"], ["a"], "easy", 0.9, 0.7),
            CaseResult("Q2", 0.6, 0.4, 0.5, True, ["b"], ["b"], "easy", 0.7, 0.5),
        ]
        breakdowns = _compute_difficulty_breakdown(results)
        assert len(breakdowns) == 1
        bd = breakdowns[0]
        assert bd.difficulty == "easy"
        assert bd.count == 2
        assert bd.avg_precision == pytest.approx(0.7)
        assert bd.avg_recall == pytest.approx(0.5)
        assert bd.avg_ndcg == pytest.approx(0.8)
        assert bd.avg_f1 == pytest.approx(0.6)
        assert bd.mrr == pytest.approx(0.75)
        assert bd.hit_rate == pytest.approx(1.0)

    def test_multiple_difficulties(self) -> None:
        """Multiple difficulty groups should be sorted: easy, medium, hard."""
        results = [
            CaseResult("Q1", 1.0, 1.0, 1.0, True, ["a"], ["a"], "hard", 1.0, 1.0),
            CaseResult("Q2", 0.5, 0.5, 0.5, True, ["b"], ["b"], "easy", 0.5, 0.5),
            CaseResult("Q3", 0.7, 0.7, 0.7, True, ["c"], ["c"], "medium", 0.7, 0.7),
        ]
        breakdowns = _compute_difficulty_breakdown(results)
        assert len(breakdowns) == 3
        assert breakdowns[0].difficulty == "easy"
        assert breakdowns[1].difficulty == "medium"
        assert breakdowns[2].difficulty == "hard"

    def test_empty_results(self) -> None:
        """Empty results should produce no breakdowns."""
        breakdowns = _compute_difficulty_breakdown([])
        assert breakdowns == []

    def test_hit_rate_computed(self) -> None:
        """Hit rate should reflect mix of hits and misses."""
        results = [
            CaseResult("Q1", 1.0, 1.0, 1.0, True, ["a"], ["a"], "easy", 1.0, 1.0),
            CaseResult("Q2", 0.0, 0.0, 0.0, False, ["x"], ["a"], "easy", 0.0, 0.0),
        ]
        breakdowns = _compute_difficulty_breakdown(results)
        assert breakdowns[0].hit_rate == pytest.approx(0.5)


# ── evaluate() with mock retriever ──────────────────────────────────

class TestEvaluateWithMock:
    def test_evaluate_computes_ndcg_and_f1(self) -> None:
        """evaluate() should compute NDCG and F1 for each case."""
        retriever = _make_retriever(["a.md", "b.md"])
        cases = [
            TestCase("Q1", ["a.md", "b.md"]),
        ]
        summary = evaluate(cases, retriever, k=5)
        assert summary.avg_ndcg > 0.0
        assert summary.avg_f1 > 0.0
        assert len(summary.results) == 1
        assert summary.results[0].ndcg > 0.0
        assert summary.results[0].f1 > 0.0

    def test_evaluate_builds_difficulty_breakdown(self) -> None:
        """evaluate() should populate by_difficulty."""
        retriever = _make_retriever(["a.md"])
        cases = [
            TestCase("Q1", ["a.md"], difficulty="easy"),
            TestCase("Q2", ["b.md"], difficulty="hard"),
        ]
        summary = evaluate(cases, retriever, k=5)
        assert len(summary.by_difficulty) == 2
        difficulties = [bd.difficulty for bd in summary.by_difficulty]
        assert "easy" in difficulties
        assert "hard" in difficulties

    def test_evaluate_empty_cases(self) -> None:
        """evaluate() with empty cases returns zero summary."""
        retriever = _make_retriever([])
        summary = evaluate([], retriever, k=5)
        assert summary.num_cases == 0
        assert summary.avg_ndcg == 0.0
        assert summary.avg_f1 == 0.0
        assert summary.by_difficulty == []

    def test_evaluate_with_relevance_grades(self) -> None:
        """evaluate_case should use explicit relevance_grades for NDCG."""
        retriever = _make_retriever(["a.md", "b.md"])
        case = TestCase(
            "Q", ["a.md", "b.md"],
            relevance_grades={"a.md": 2, "b.md": 1},
        )
        summary = evaluate([case], retriever, k=5)
        assert summary.results[0].ndcg > 0.0


# ── Comparison Mode ──────────────────────────────────────────────────

class TestComparisonMode:
    def test_compare_evaluations_basic(self) -> None:
        """compare_evaluations should compute deltas between two retrievers."""
        # Retriever A returns relevant files
        retriever_a = _make_retriever(["a.md", "b.md"])
        # Retriever B returns only irrelevant files
        retriever_b = _make_retriever(["x.md", "y.md"])

        cases = [TestCase("Q1", ["a.md", "b.md"])]
        result = compare_evaluations(cases, retriever_a, retriever_b, k=5)

        assert isinstance(result, ComparisonResult)
        assert result.config_a_summary.avg_precision > result.config_b_summary.avg_precision
        assert result.precision_delta < 0  # B is worse
        assert result.recall_delta < 0
        assert result.f1_delta < 0

    def test_compare_evaluations_same_retriever(self) -> None:
        """Comparing a retriever to itself should yield zero deltas."""
        retriever = _make_retriever(["a.md"])
        cases = [TestCase("Q1", ["a.md"])]
        result = compare_evaluations(cases, retriever, retriever, k=5)

        assert result.precision_delta == pytest.approx(0.0)
        assert result.recall_delta == pytest.approx(0.0)
        assert result.ndcg_delta == pytest.approx(0.0)
        assert result.f1_delta == pytest.approx(0.0)
        assert result.mrr_delta == pytest.approx(0.0)

    def test_format_comparison_output(self) -> None:
        """format_comparison should produce a readable report."""
        sa = EvalSummary(1, 0.8, 0.7, 0.9, 1.0, [], avg_ndcg=0.85, avg_f1=0.75)
        sb = EvalSummary(1, 0.6, 0.5, 0.7, 0.8, [], avg_ndcg=0.65, avg_f1=0.55)
        comp = ComparisonResult(
            config_a_summary=sa,
            config_b_summary=sb,
            precision_delta=-0.2,
            recall_delta=-0.2,
            ndcg_delta=-0.2,
            f1_delta=-0.2,
            mrr_delta=-0.2,
        )
        report = format_comparison(comp, k=5)
        assert "A/B Comparison Report" in report
        assert "Config A" in report
        assert "Config B" in report
        assert "Delta" in report
        assert "Precision" in report
        assert "NDCG" in report
        assert "Verdict" in report

    def test_format_comparison_b_better(self) -> None:
        """Verdict should indicate B is better when all deltas positive."""
        sa = EvalSummary(1, 0.5, 0.5, 0.5, 0.5, [], avg_ndcg=0.5, avg_f1=0.5)
        sb = EvalSummary(1, 0.8, 0.8, 0.8, 0.8, [], avg_ndcg=0.8, avg_f1=0.8)
        comp = ComparisonResult(
            config_a_summary=sa, config_b_summary=sb,
            precision_delta=0.3, recall_delta=0.3,
            ndcg_delta=0.3, f1_delta=0.3, mrr_delta=0.3,
        )
        report = format_comparison(comp, k=5)
        assert "Config B is better" in report

    def test_format_comparison_a_better(self) -> None:
        """Verdict should indicate A is better when all deltas negative."""
        sa = EvalSummary(1, 0.8, 0.8, 0.8, 0.8, [], avg_ndcg=0.8, avg_f1=0.8)
        sb = EvalSummary(1, 0.5, 0.5, 0.5, 0.5, [], avg_ndcg=0.5, avg_f1=0.5)
        comp = ComparisonResult(
            config_a_summary=sa, config_b_summary=sb,
            precision_delta=-0.3, recall_delta=-0.3,
            ndcg_delta=-0.3, f1_delta=-0.3, mrr_delta=-0.3,
        )
        report = format_comparison(comp, k=5)
        assert "Config A is better" in report

    def test_format_comparison_mixed(self) -> None:
        """Verdict should indicate 'mixed' when some improve and some regress."""
        sa = EvalSummary(1, 0.5, 0.8, 0.5, 0.8, [], avg_ndcg=0.5, avg_f1=0.8)
        sb = EvalSummary(1, 0.8, 0.5, 0.8, 0.5, [], avg_ndcg=0.8, avg_f1=0.5)
        comp = ComparisonResult(
            config_a_summary=sa, config_b_summary=sb,
            precision_delta=0.3, recall_delta=-0.3,
            ndcg_delta=0.3, f1_delta=-0.3, mrr_delta=0.3,
        )
        report = format_comparison(comp, k=5)
        assert "mixed" in report.lower() or "Config B is better" in report


# ── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_ndcg_with_very_large_k(self) -> None:
        """NDCG with k larger than retrieved list should work."""
        relevance_map = {"a": 2}
        assert ndcg_at_k(["a"], relevance_map, k=100) == pytest.approx(1.0)

    def test_f1_empty_both(self) -> None:
        """Empty retrieval and empty relevant set -> F1 = 0.0."""
        # recall returns 1.0 for empty relevant, precision returns 0.0 for empty retrieval
        assert f1_at_k([], set(), k=5) == 0.0

    def test_case_result_new_fields_default(self) -> None:
        """CaseResult new fields should have defaults."""
        cr = CaseResult(
            question="Q", precision=0.5, recall=0.5, reciprocal_rank=0.5,
            hit=True, retrieved_files=["a"], relevant_files=["a"], difficulty="easy",
        )
        assert cr.ndcg == 0.0
        assert cr.f1 == 0.0

    def test_difficulty_breakdown_dataclass(self) -> None:
        """DifficultyBreakdown should store all fields correctly."""
        bd = DifficultyBreakdown(
            difficulty="hard", count=3, avg_precision=0.5, avg_recall=0.4,
            avg_ndcg=0.6, avg_f1=0.45, mrr=0.7, hit_rate=0.8,
        )
        assert bd.difficulty == "hard"
        assert bd.count == 3
        assert bd.avg_precision == 0.5
        assert bd.avg_ndcg == 0.6

    def test_comparison_result_dataclass(self) -> None:
        """ComparisonResult should store all fields."""
        sa = EvalSummary(1, 0.5, 0.5, 0.5, 0.5, [])
        sb = EvalSummary(1, 0.6, 0.6, 0.6, 0.6, [])
        cr = ComparisonResult(
            config_a_summary=sa, config_b_summary=sb,
            precision_delta=0.1, recall_delta=0.1,
            ndcg_delta=0.1, f1_delta=0.1, mrr_delta=0.1,
        )
        assert cr.precision_delta == 0.1
        assert cr.ndcg_delta == 0.1
