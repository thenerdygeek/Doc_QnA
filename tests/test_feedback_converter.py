"""Tests for the feedback-to-eval converter."""
from __future__ import annotations

import json

import pytest

from doc_qa.feedback.converter import (
    export_feedback_test_cases,
    feedback_to_test_cases,
)


class TestFeedbackToTestCases:
    def test_empty_feedback(self) -> None:
        """Empty feedback list produces no test cases."""
        assert feedback_to_test_cases([]) == []

    def test_basic_conversion(self) -> None:
        """A single positive feedback row converts to a test case."""
        rows = [
            {
                "question": "How does auth work?",
                "answer": "Auth uses JWT tokens.",
                "chunks_used": '["docs/auth.md#0", "docs/auth.md#1"]',
                "confidence": 0.9,
            }
        ]
        cases = feedback_to_test_cases(rows)
        assert len(cases) == 1
        assert cases[0]["question"] == "How does auth work?"
        assert cases[0]["relevant_files"] == ["auth.md"]
        assert cases[0]["difficulty"] == "auto"
        assert "feedback" in cases[0]["description"].lower()

    def test_multiple_files_from_chunks(self) -> None:
        """Chunks from different files produce multiple relevant_files."""
        rows = [
            {
                "question": "Compare X and Y",
                "chunks_used": '["docs/x.md#0", "docs/y.md#1", "docs/x.md#2"]',
                "confidence": 0.8,
            }
        ]
        cases = feedback_to_test_cases(rows)
        assert len(cases) == 1
        # Deduplicated: x.md appears once, y.md appears once
        assert set(cases[0]["relevant_files"]) == {"x.md", "y.md"}

    def test_low_confidence_filtered(self) -> None:
        """Rows below min_confidence are excluded."""
        rows = [
            {
                "question": "Low confidence Q",
                "chunks_used": '["docs/a.md#0"]',
                "confidence": 0.3,
            }
        ]
        cases = feedback_to_test_cases(rows, min_confidence=0.6)
        assert cases == []

    def test_custom_min_confidence(self) -> None:
        """Custom min_confidence threshold is respected."""
        rows = [
            {
                "question": "Q1",
                "chunks_used": '["a.md#0"]',
                "confidence": 0.4,
            }
        ]
        # With lower threshold, it passes
        cases = feedback_to_test_cases(rows, min_confidence=0.3)
        assert len(cases) == 1

    def test_empty_question_filtered(self) -> None:
        """Rows with empty question are excluded."""
        rows = [
            {
                "question": "",
                "chunks_used": '["a.md#0"]',
                "confidence": 0.9,
            }
        ]
        assert feedback_to_test_cases(rows) == []

    def test_empty_chunks_filtered(self) -> None:
        """Rows with no chunks are excluded."""
        rows = [
            {
                "question": "Valid Q",
                "chunks_used": "[]",
                "confidence": 0.9,
            }
        ]
        assert feedback_to_test_cases(rows) == []

    def test_chunks_as_list(self) -> None:
        """chunks_used already a list (not JSON string) works."""
        rows = [
            {
                "question": "Q",
                "chunks_used": ["file.md#0"],
                "confidence": 0.8,
            }
        ]
        cases = feedback_to_test_cases(rows)
        assert len(cases) == 1
        assert cases[0]["relevant_files"] == ["file.md"]

    def test_confidence_as_string(self) -> None:
        """Confidence stored as string is parsed correctly."""
        rows = [
            {
                "question": "Q",
                "chunks_used": '["f.md#0"]',
                "confidence": "0.85",
            }
        ]
        cases = feedback_to_test_cases(rows)
        assert len(cases) == 1

    def test_chunk_id_without_hash(self) -> None:
        """chunk_id without # separator still extracts filename."""
        rows = [
            {
                "question": "Q",
                "chunks_used": '["docs/plain_file.md"]',
                "confidence": 0.9,
            }
        ]
        cases = feedback_to_test_cases(rows)
        assert len(cases) == 1
        assert cases[0]["relevant_files"] == ["plain_file.md"]


class TestExportFeedbackTestCases:
    def test_writes_valid_json(self, tmp_path) -> None:
        """Output file is valid JSON with test_cases key."""
        output = tmp_path / "out.json"
        rows = [
            {
                "question": "Q1",
                "chunks_used": '["a.md#0"]',
                "confidence": 0.8,
            }
        ]
        added = export_feedback_test_cases(output, feedback_rows=rows)
        assert added == 1

        with open(output) as f:
            data = json.load(f)
        assert "test_cases" in data
        assert len(data["test_cases"]) == 1

    def test_dedup_with_existing(self, tmp_path) -> None:
        """Duplicate questions against existing file are excluded."""
        existing = tmp_path / "existing.json"
        with open(existing, "w") as f:
            json.dump({"test_cases": [
                {"question": "Q1", "relevant_files": ["old.md"]},
            ]}, f)

        output = tmp_path / "out.json"
        rows = [
            {"question": "Q1", "chunks_used": '["new.md#0"]', "confidence": 0.9},  # duplicate
            {"question": "Q2", "chunks_used": '["b.md#0"]', "confidence": 0.8},  # new
        ]
        added = export_feedback_test_cases(
            output, existing_path=existing, feedback_rows=rows,
        )
        assert added == 1  # only Q2 is new

        with open(output) as f:
            data = json.load(f)
        assert len(data["test_cases"]) == 2  # Q1 from existing + Q2

    def test_dedup_case_insensitive(self, tmp_path) -> None:
        """Deduplication is case-insensitive on question text."""
        existing = tmp_path / "existing.json"
        with open(existing, "w") as f:
            json.dump({"test_cases": [
                {"question": "How does auth work?"},
            ]}, f)

        output = tmp_path / "out.json"
        rows = [
            {"question": "HOW DOES AUTH WORK?", "chunks_used": '["a.md#0"]', "confidence": 0.9},
        ]
        added = export_feedback_test_cases(
            output, existing_path=existing, feedback_rows=rows,
        )
        assert added == 0

    def test_no_existing_file(self, tmp_path) -> None:
        """Works when merge_with points to non-existent file."""
        output = tmp_path / "out.json"
        rows = [
            {"question": "Q1", "chunks_used": '["a.md#0"]', "confidence": 0.8},
        ]
        added = export_feedback_test_cases(
            output, existing_path=tmp_path / "nope.json", feedback_rows=rows,
        )
        assert added == 1

    def test_empty_feedback(self, tmp_path) -> None:
        """No feedback rows produces empty test_cases."""
        output = tmp_path / "out.json"
        added = export_feedback_test_cases(output, feedback_rows=[])
        assert added == 0

        with open(output) as f:
            data = json.load(f)
        assert data["test_cases"] == []
