"""Retrieval evaluation metrics and runner.

Measures retrieval quality without an LLM — purely tests whether the
right chunks come back for a set of hand-crafted test cases.

Metrics:
    - Precision@k: fraction of top-k results that are relevant
    - Recall@k: fraction of all relevant items found in top-k
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant result
    - Hit Rate@k: 1 if any relevant result is in top-k, else 0
    - NDCG@k: Normalized Discounted Cumulative Gain (graded relevance)
    - F1@k: Harmonic mean of Precision@k and Recall@k
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from doc_qa.retrieval.retriever import HybridRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class TestCase:
    """A single evaluation test case."""

    question: str
    relevant_files: list[str]
    relevant_keywords: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    description: str = ""
    relevance_grades: dict[str, int] = field(default_factory=dict)
    relevant_sections: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """Evaluation result for a single test case."""

    question: str
    precision: float
    recall: float
    reciprocal_rank: float
    hit: bool
    retrieved_files: list[str]
    relevant_files: list[str]
    difficulty: str
    ndcg: float = 0.0
    f1: float = 0.0


@dataclass
class DifficultyBreakdown:
    """Aggregate metrics for a single difficulty level."""

    difficulty: str
    count: int
    avg_precision: float
    avg_recall: float
    avg_ndcg: float
    avg_f1: float
    mrr: float
    hit_rate: float


@dataclass
class EvalSummary:
    """Aggregate evaluation results across all test cases."""

    num_cases: int
    avg_precision: float
    avg_recall: float
    mrr: float
    hit_rate: float
    results: list[CaseResult]
    avg_ndcg: float = 0.0
    avg_f1: float = 0.0
    by_difficulty: list[DifficultyBreakdown] = field(default_factory=list)

    def passed(self, precision_threshold: float = 0.6, mrr_threshold: float = 0.5) -> bool:
        """Check if the evaluation meets minimum quality thresholds."""
        return self.avg_precision >= precision_threshold and self.mrr >= mrr_threshold


@dataclass
class ComparisonResult:
    """Result of comparing two retrieval configurations."""

    config_a_summary: EvalSummary
    config_b_summary: EvalSummary
    precision_delta: float
    recall_delta: float
    ndcg_delta: float
    f1_delta: float
    mrr_delta: float


# ── Metric functions ─────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant.

    Multiple chunks from the same file count as one hit for precision
    (avoids inflating the score when many chunks come from one file).

    Args:
        retrieved: Ordered list of retrieved file paths (may have duplicates).
        relevant: Set of relevant file paths.
        k: Cutoff rank.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    # Deduplicate: count each file only once
    seen: set[str] = set()
    hits = 0
    unique_count = 0
    for item in top:
        if item not in seen:
            seen.add(item)
            unique_count += 1
            if item in relevant:
                hits += 1
    return hits / unique_count if unique_count > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k results.

    Args:
        retrieved: Ordered list of retrieved file paths (may have duplicates).
        relevant: Set of relevant file paths.
        k: Cutoff rank.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not relevant:
        return 1.0  # nothing to recall
    top = retrieved[:k]
    # Count unique relevant files found
    found = relevant & set(top)
    return len(found) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result.

    Args:
        retrieved: Ordered list of retrieved file paths.
        relevant: Set of relevant file paths.

    Returns:
        Reciprocal rank between 0.0 and 1.0.
    """
    for i, item in enumerate(retrieved, 1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> bool:
    """Whether any relevant item appears in the top-k results."""
    return any(item in relevant for item in retrieved[:k])


def ndcg_at_k(retrieved: list[str], relevance_map: dict[str, int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Uses graded relevance: relevant=2, partial=1, irrelevant=0.
    Standard DCG formula: sum(rel_i / log2(i+1)) for i in 1..k.

    Args:
        retrieved: Ordered list of retrieved file paths (may have duplicates).
        relevance_map: Mapping of file path to relevance grade (0, 1, or 2).
        k: Cutoff rank.

    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if not relevance_map or k <= 0:
        return 0.0

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_retrieved: list[str] = []
    for item in retrieved[:k]:
        if item not in seen:
            seen.add(item)
            unique_retrieved.append(item)

    # Compute DCG
    dcg = 0.0
    for i, item in enumerate(unique_retrieved):
        rel = relevance_map.get(item, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed, formula uses 1-indexed + 1

    # Compute IDCG (ideal DCG: sort grades descending)
    ideal_grades = sorted(relevance_map.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_grades):
        idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def f1_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Harmonic mean of Precision@k and Recall@k.

    Args:
        retrieved: Ordered list of retrieved file paths (may have duplicates).
        relevant: Set of relevant file paths.
        k: Cutoff rank.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    prec = precision_at_k(retrieved, relevant, k)
    rec = recall_at_k(retrieved, relevant, k)
    if prec + rec == 0.0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


# ── Test case loading ────────────────────────────────────────────────

def load_test_cases(path: str | Path) -> list[TestCase]:
    """Load test cases from a JSON file.

    Expected format:
    ```json
    {
        "test_cases": [
            {
                "question": "How does authentication work?",
                "relevant_files": ["auth.md", "login.adoc"],
                "relevant_keywords": ["OAuth", "token"],
                "difficulty": "easy",
                "description": "Basic auth question",
                "relevance_grades": {"auth.md": 2, "login.adoc": 1},
                "relevant_sections": ["Authentication Overview"]
            }
        ]
    }
    ```
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    cases: list[TestCase] = []
    for item in data.get("test_cases", []):
        cases.append(
            TestCase(
                question=item["question"],
                relevant_files=item["relevant_files"],
                relevant_keywords=item.get("relevant_keywords", []),
                difficulty=item.get("difficulty", "medium"),
                description=item.get("description", ""),
                relevance_grades=item.get("relevance_grades", {}),
                relevant_sections=item.get("relevant_sections", []),
            )
        )
    return cases


# ── Evaluation runner ────────────────────────────────────────────────

def _normalize_file(fp: str) -> str:
    """Extract just the filename for matching (test cases use filenames, index uses full paths)."""
    return Path(fp).name


def _build_relevance_map(case: TestCase) -> dict[str, int]:
    """Build a relevance map from a test case.

    If the case has explicit relevance_grades, use them.
    Otherwise, default all relevant_files to grade 2 (fully relevant).
    """
    if case.relevance_grades:
        return dict(case.relevance_grades)
    return {f: 2 for f in case.relevant_files}


def _compute_difficulty_breakdown(results: list[CaseResult]) -> list[DifficultyBreakdown]:
    """Group results by difficulty and compute per-group aggregates."""
    groups: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        groups[r.difficulty].append(r)

    breakdowns: list[DifficultyBreakdown] = []
    # Sort by canonical order: easy, medium, hard, then alphabetical for others
    order = {"easy": 0, "medium": 1, "hard": 2}
    for diff in sorted(groups.keys(), key=lambda d: (order.get(d, 99), d)):
        group = groups[diff]
        n = len(group)
        breakdowns.append(
            DifficultyBreakdown(
                difficulty=diff,
                count=n,
                avg_precision=sum(r.precision for r in group) / n,
                avg_recall=sum(r.recall for r in group) / n,
                avg_ndcg=sum(r.ndcg for r in group) / n,
                avg_f1=sum(r.f1 for r in group) / n,
                mrr=sum(r.reciprocal_rank for r in group) / n,
                hit_rate=sum(1 for r in group if r.hit) / n,
            )
        )
    return breakdowns


def evaluate_case(
    case: TestCase,
    retriever: HybridRetriever,
    k: int = 5,
) -> CaseResult:
    """Run a single test case through the retriever and compute metrics."""
    chunks = retriever.search(query=case.question, top_k=k, min_score=0.0)

    # Normalize paths for comparison (test cases use filenames)
    retrieved_files = [_normalize_file(c.file_path) for c in chunks]
    relevant_set = set(case.relevant_files)

    prec = precision_at_k(retrieved_files, relevant_set, k)
    rec = recall_at_k(retrieved_files, relevant_set, k)
    rr = reciprocal_rank(retrieved_files, relevant_set)
    hit = hit_at_k(retrieved_files, relevant_set, k)

    # NDCG with graded relevance
    relevance_map = _build_relevance_map(case)
    ndcg = ndcg_at_k(retrieved_files, relevance_map, k)

    # F1
    f1 = f1_at_k(retrieved_files, relevant_set, k)

    return CaseResult(
        question=case.question,
        precision=prec,
        recall=rec,
        reciprocal_rank=rr,
        hit=hit,
        retrieved_files=retrieved_files,
        relevant_files=case.relevant_files,
        difficulty=case.difficulty,
        ndcg=ndcg,
        f1=f1,
    )


def evaluate(
    cases: list[TestCase],
    retriever: HybridRetriever,
    k: int = 5,
) -> EvalSummary:
    """Run all test cases and compute aggregate metrics."""
    results: list[CaseResult] = []
    for case in cases:
        result = evaluate_case(case, retriever, k)
        results.append(result)

    n = len(results)
    if n == 0:
        return EvalSummary(
            num_cases=0,
            avg_precision=0.0,
            avg_recall=0.0,
            mrr=0.0,
            hit_rate=0.0,
            results=[],
            avg_ndcg=0.0,
            avg_f1=0.0,
            by_difficulty=[],
        )

    return EvalSummary(
        num_cases=n,
        avg_precision=sum(r.precision for r in results) / n,
        avg_recall=sum(r.recall for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        hit_rate=sum(1 for r in results if r.hit) / n,
        results=results,
        avg_ndcg=sum(r.ndcg for r in results) / n,
        avg_f1=sum(r.f1 for r in results) / n,
        by_difficulty=_compute_difficulty_breakdown(results),
    )


def format_report(summary: EvalSummary, k: int = 5) -> str:
    """Format evaluation results as a readable report."""
    lines: list[str] = []

    lines.append(f"Retrieval Evaluation Report ({summary.num_cases} test cases, k={k})")
    lines.append("=" * 90)
    lines.append("")

    # Per-case results
    lines.append(
        f"{'Question':<40} {'P@k':>5} {'R@k':>5} {'NDCG':>5} {'F1':>5} "
        f"{'RR':>5} {'Hit':>4} {'Diff':>6}"
    )
    lines.append("-" * 90)

    for r in summary.results:
        q = r.question[:38] + ".." if len(r.question) > 40 else r.question
        hit_str = "Y" if r.hit else "N"
        lines.append(
            f"{q:<40} {r.precision:5.2f} {r.recall:5.2f} {r.ndcg:5.2f} {r.f1:5.2f} "
            f"{r.reciprocal_rank:5.2f} {hit_str:>4} {r.difficulty:>6}"
        )

    lines.append("-" * 90)
    lines.append("")

    # Aggregate
    lines.append("Aggregate Metrics:")
    lines.append(f"  Precision@{k}:  {summary.avg_precision:.3f}")
    lines.append(f"  Recall@{k}:     {summary.avg_recall:.3f}")
    lines.append(f"  NDCG@{k}:       {summary.avg_ndcg:.3f}")
    lines.append(f"  F1@{k}:         {summary.avg_f1:.3f}")
    lines.append(f"  MRR:           {summary.mrr:.3f}")
    lines.append(f"  Hit Rate@{k}:   {summary.hit_rate:.3f}")
    lines.append("")

    # Per-difficulty breakdown
    if summary.by_difficulty:
        lines.append("Per-Difficulty Breakdown:")
        lines.append(
            f"  {'Difficulty':<10} {'Count':>5} {'P@k':>6} {'R@k':>6} {'NDCG':>6} "
            f"{'F1':>6} {'MRR':>6} {'Hit%':>6}"
        )
        lines.append("  " + "-" * 58)
        for bd in summary.by_difficulty:
            lines.append(
                f"  {bd.difficulty:<10} {bd.count:>5} {bd.avg_precision:6.3f} "
                f"{bd.avg_recall:6.3f} {bd.avg_ndcg:6.3f} {bd.avg_f1:6.3f} "
                f"{bd.mrr:6.3f} {bd.hit_rate:6.3f}"
            )
        lines.append("")

    # Pass/fail
    p_pass = summary.avg_precision >= 0.6
    m_pass = summary.mrr >= 0.5
    lines.append(f"Thresholds:  Precision@{k} >= 0.60 {'PASS' if p_pass else 'FAIL'}")
    lines.append(f"             MRR >= 0.50          {'PASS' if m_pass else 'FAIL'}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if summary.passed() else 'FAIL'}")

    return "\n".join(lines)


# ── A/B Comparison ───────────────────────────────────────────────────

def compare_evaluations(
    cases: list[TestCase],
    retriever_a: HybridRetriever,
    retriever_b: HybridRetriever,
    k: int = 5,
) -> ComparisonResult:
    """Run the same test cases against two retriever configurations and compare.

    Args:
        cases: List of test cases to evaluate.
        retriever_a: First retriever configuration (baseline).
        retriever_b: Second retriever configuration (candidate).
        k: Cutoff rank for metrics.

    Returns:
        ComparisonResult with summaries and deltas (B - A).
    """
    summary_a = evaluate(cases, retriever_a, k=k)
    summary_b = evaluate(cases, retriever_b, k=k)

    return ComparisonResult(
        config_a_summary=summary_a,
        config_b_summary=summary_b,
        precision_delta=summary_b.avg_precision - summary_a.avg_precision,
        recall_delta=summary_b.avg_recall - summary_a.avg_recall,
        ndcg_delta=summary_b.avg_ndcg - summary_a.avg_ndcg,
        f1_delta=summary_b.avg_f1 - summary_a.avg_f1,
        mrr_delta=summary_b.mrr - summary_a.mrr,
    )


def format_comparison(result: ComparisonResult, k: int = 5) -> str:
    """Format an A/B comparison result as a readable report.

    Args:
        result: ComparisonResult from compare_evaluations.
        k: Cutoff rank (for labeling).

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []
    sa = result.config_a_summary
    sb = result.config_b_summary

    lines.append(f"A/B Comparison Report ({sa.num_cases} test cases, k={k})")
    lines.append("=" * 70)
    lines.append("")

    def _delta_str(val: float) -> str:
        sign = "+" if val >= 0 else ""
        return f"{sign}{val:.3f}"

    lines.append(f"{'Metric':<18} {'Config A':>10} {'Config B':>10} {'Delta':>10}")
    lines.append("-" * 50)
    lines.append(f"{'Precision@' + str(k):<18} {sa.avg_precision:10.3f} {sb.avg_precision:10.3f} {_delta_str(result.precision_delta):>10}")
    lines.append(f"{'Recall@' + str(k):<18} {sa.avg_recall:10.3f} {sb.avg_recall:10.3f} {_delta_str(result.recall_delta):>10}")
    lines.append(f"{'NDCG@' + str(k):<18} {sa.avg_ndcg:10.3f} {sb.avg_ndcg:10.3f} {_delta_str(result.ndcg_delta):>10}")
    lines.append(f"{'F1@' + str(k):<18} {sa.avg_f1:10.3f} {sb.avg_f1:10.3f} {_delta_str(result.f1_delta):>10}")
    lines.append(f"{'MRR':<18} {sa.mrr:10.3f} {sb.mrr:10.3f} {_delta_str(result.mrr_delta):>10}")
    lines.append(f"{'Hit Rate@' + str(k):<18} {sa.hit_rate:10.3f} {sb.hit_rate:10.3f} {_delta_str(sb.hit_rate - sa.hit_rate):>10}")
    lines.append("")

    # Determine winner
    improvements = sum(1 for d in [
        result.precision_delta, result.recall_delta,
        result.ndcg_delta, result.f1_delta, result.mrr_delta,
    ] if d > 0)
    regressions = sum(1 for d in [
        result.precision_delta, result.recall_delta,
        result.ndcg_delta, result.f1_delta, result.mrr_delta,
    ] if d < 0)

    if improvements > regressions:
        verdict = "Config B is better"
    elif regressions > improvements:
        verdict = "Config A is better"
    else:
        verdict = "Results are mixed"

    lines.append(f"Verdict: {verdict} ({improvements} improved, {regressions} regressed)")

    return "\n".join(lines)
