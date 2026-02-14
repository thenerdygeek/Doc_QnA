"""Retrieval evaluation metrics and runner.

Measures retrieval quality without an LLM — purely tests whether the
right chunks come back for a set of hand-crafted test cases.

Metrics:
    - Precision@k: fraction of top-k results that are relevant
    - Recall@k: fraction of all relevant items found in top-k
    - MRR (Mean Reciprocal Rank): 1 / rank of first relevant result
    - Hit Rate@k: 1 if any relevant result is in top-k, else 0
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


@dataclass
class EvalSummary:
    """Aggregate evaluation results across all test cases."""

    num_cases: int
    avg_precision: float
    avg_recall: float
    mrr: float
    hit_rate: float
    results: list[CaseResult]

    def passed(self, precision_threshold: float = 0.6, mrr_threshold: float = 0.5) -> bool:
        """Check if the evaluation meets minimum quality thresholds."""
        return self.avg_precision >= precision_threshold and self.mrr >= mrr_threshold


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
                "description": "Basic auth question"
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
            )
        )
    return cases


# ── Evaluation runner ────────────────────────────────────────────────

def _normalize_file(fp: str) -> str:
    """Extract just the filename for matching (test cases use filenames, index uses full paths)."""
    return Path(fp).name


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

    return CaseResult(
        question=case.question,
        precision=prec,
        recall=rec,
        reciprocal_rank=rr,
        hit=hit,
        retrieved_files=retrieved_files,
        relevant_files=case.relevant_files,
        difficulty=case.difficulty,
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
        )

    return EvalSummary(
        num_cases=n,
        avg_precision=sum(r.precision for r in results) / n,
        avg_recall=sum(r.recall for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        hit_rate=sum(1 for r in results if r.hit) / n,
        results=results,
    )


def format_report(summary: EvalSummary, k: int = 5) -> str:
    """Format evaluation results as a readable report."""
    lines: list[str] = []

    lines.append(f"Retrieval Evaluation Report ({summary.num_cases} test cases, k={k})")
    lines.append("=" * 70)
    lines.append("")

    # Per-case results
    lines.append(f"{'Question':<45} {'P@k':>5} {'R@k':>5} {'RR':>5} {'Hit':>4} {'Diff':>6}")
    lines.append("-" * 70)

    for r in summary.results:
        q = r.question[:43] + ".." if len(r.question) > 45 else r.question
        hit_str = "Y" if r.hit else "N"
        lines.append(
            f"{q:<45} {r.precision:5.2f} {r.recall:5.2f} "
            f"{r.reciprocal_rank:5.2f} {hit_str:>4} {r.difficulty:>6}"
        )

    lines.append("-" * 70)
    lines.append("")

    # Aggregate
    lines.append("Aggregate Metrics:")
    lines.append(f"  Precision@{k}:  {summary.avg_precision:.3f}")
    lines.append(f"  Recall@{k}:     {summary.avg_recall:.3f}")
    lines.append(f"  MRR:           {summary.mrr:.3f}")
    lines.append(f"  Hit Rate@{k}:   {summary.hit_rate:.3f}")
    lines.append("")

    # Pass/fail
    p_pass = summary.avg_precision >= 0.6
    m_pass = summary.mrr >= 0.5
    lines.append(f"Thresholds:  Precision@{k} >= 0.60 {'PASS' if p_pass else 'FAIL'}")
    lines.append(f"             MRR >= 0.50          {'PASS' if m_pass else 'FAIL'}")
    lines.append("")
    lines.append(f"Overall: {'PASS' if summary.passed() else 'FAIL'}")

    return "\n".join(lines)
