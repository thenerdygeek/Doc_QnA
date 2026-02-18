"""Convert positive user feedback into evaluation test cases.

Bridges the feedback loop: real user thumbs-up ratings become regression
test cases for the retrieval pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def feedback_to_test_cases(
    feedback_rows: list[dict],
    min_confidence: float = 0.6,
) -> list[dict]:
    """Convert positive feedback records into TestCase-compatible dicts.

    Filters:
        - rating == 1 (thumbs up only)
        - confidence >= min_confidence
        - non-empty question and chunks_used

    Args:
        feedback_rows: List of dicts from export_positive_feedback().
        min_confidence: Minimum confidence threshold.

    Returns:
        List of test case dicts ready for JSON serialization.
    """
    cases: list[dict] = []

    for row in feedback_rows:
        question = row.get("question", "")
        if not question or not question.strip():
            continue

        confidence = row.get("confidence", 0.0)
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0

        if confidence < min_confidence:
            continue

        # Parse chunks_used — may be a JSON string or already a list
        chunks_used = row.get("chunks_used", [])
        if isinstance(chunks_used, str):
            try:
                chunks_used = json.loads(chunks_used)
            except (json.JSONDecodeError, TypeError):
                chunks_used = []

        if not chunks_used:
            continue

        # Extract relevant_files from chunk_id format: "file_path#index" -> basename
        relevant_files: list[str] = []
        seen: set[str] = set()
        for chunk_id in chunks_used:
            if not isinstance(chunk_id, str):
                continue
            # Split on '#' to get file path
            parts = chunk_id.rsplit("#", 1)
            if parts:
                basename = Path(parts[0]).name
                if basename and basename not in seen:
                    relevant_files.append(basename)
                    seen.add(basename)

        if not relevant_files:
            continue

        cases.append({
            "question": question.strip(),
            "relevant_files": relevant_files,
            "difficulty": "auto",
            "description": "Generated from positive user feedback",
        })

    return cases


def export_feedback_test_cases(
    output_path: str | Path,
    existing_path: str | Path | None = None,
    min_confidence: float = 0.6,
    feedback_rows: list[dict] | None = None,
) -> int:
    """Export feedback-derived test cases to JSON, deduplicating against existing.

    Args:
        output_path: Path to write the output JSON file.
        existing_path: Optional path to an existing test_cases.json to
            merge with (deduplicates by question text).
        min_confidence: Minimum confidence threshold for inclusion.
        feedback_rows: Pre-fetched feedback rows. If None, caller must
            provide them.

    Returns:
        Number of new test cases added.
    """
    if feedback_rows is None:
        feedback_rows = []

    new_cases = feedback_to_test_cases(feedback_rows, min_confidence=min_confidence)

    # Load existing test cases for deduplication
    existing_questions: set[str] = set()
    existing_cases: list[dict] = []

    if existing_path:
        ep = Path(existing_path)
        if ep.exists():
            try:
                with open(ep, encoding="utf-8") as f:
                    data = json.load(f)
                existing_cases = data.get("test_cases", [])
                existing_questions = {
                    tc.get("question", "").strip().lower()
                    for tc in existing_cases
                }
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load existing test cases from %s: %s", ep, exc)

    # Deduplicate
    added = 0
    for case in new_cases:
        q_lower = case["question"].strip().lower()
        if q_lower not in existing_questions:
            existing_cases.append(case)
            existing_questions.add(q_lower)
            added += 1

    # Write output
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"test_cases": existing_cases}, f, indent=2, ensure_ascii=False)

    logger.info(
        "Exported %d new test cases (%d total) to %s",
        added, len(existing_cases), output,
    )
    return added
