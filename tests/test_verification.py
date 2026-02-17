"""Tests for verifier, mermaid validator, and grader."""

from __future__ import annotations

import pytest

from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.verification.grader import (
    GradedChunk,
    _parse_grading_response,
    grade_documents,
)
from doc_qa.verification.mermaid_validator import MermaidValidator
from doc_qa.config import VerificationConfig
from doc_qa.intelligence.confidence import ConfidenceAssessment, compute_confidence
from doc_qa.verification.verifier import (
    VerificationResult,
    _parse_confidence,
    _parse_verification_response,
    verify_answer,
)


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLM(LLMBackend):
    def __init__(self, text: str = "", error: str | None = None) -> None:
        self.text = text
        self.error = error

    async def ask(self, question: str, context: str, history=None) -> Answer:
        return Answer(text=self.text, sources=[], model="mock", error=self.error)

    async def close(self) -> None:
        pass


def _chunk(text="chunk text", chunk_id="c1", score=0.8):
    return RetrievedChunk(
        text=text, score=score, chunk_id=chunk_id,
        file_path="test.md", file_type="md",
        section_title="Section", section_level=1, chunk_index=0,
    )


# ── Verifier ─────────────────────────────────────────────────────────


class TestVerificationParsing:
    def test_pass_with_all_fields(self):
        resp = "Verdict: PASS\nConfidence: 0.95\nIssues: none\nSuggested fix: none"
        result = _parse_verification_response(resp)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.95)
        assert result.issues == []
        assert result.suggested_fix is None

    def test_fail_with_issues(self):
        resp = "Verdict: FAIL\nConfidence: 0.3\nIssues: hallucinated claim, missing source\nSuggested fix: remove claim"
        result = _parse_verification_response(resp)
        assert result.passed is False
        assert result.confidence == pytest.approx(0.3)
        assert len(result.issues) == 2
        assert result.suggested_fix == "remove claim"

    def test_missing_verdict_defaults_pass(self):
        resp = "Confidence: 0.7\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.passed is True

    def test_confidence_clamped(self):
        resp = "Verdict: PASS\nConfidence: 1.5\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.confidence == 1.0

    def test_unparseable_confidence_defaults(self):
        resp = "Verdict: PASS\nConfidence: abc\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.confidence == 0.5


class TestVerificationParsingExpanded:
    """Tests for expanded verdict synonyms and separator formats."""

    def test_passed_synonym(self):
        resp = "Verdict: PASSED\nConfidence: 0.9\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.passed is True

    def test_yes_synonym(self):
        resp = "Verdict: YES\nConfidence: 0.9\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.passed is True

    def test_correct_synonym(self):
        resp = "Verdict: CORRECT\nConfidence: 0.85\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.passed is True

    def test_failed_synonym(self):
        resp = "Verdict: FAILED\nConfidence: 0.2\nIssues: inaccurate"
        result = _parse_verification_response(resp)
        assert result.passed is False

    def test_no_synonym(self):
        resp = "Verdict: NO\nConfidence: 0.3\nIssues: hallucination"
        result = _parse_verification_response(resp)
        assert result.passed is False

    def test_incorrect_synonym(self):
        resp = "Verdict: INCORRECT\nConfidence: 0.1\nIssues: wrong data"
        result = _parse_verification_response(resp)
        assert result.passed is False

    def test_dash_separator(self):
        resp = "Verdict - PASS\nConfidence - 0.9\nIssues - none"
        result = _parse_verification_response(resp)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.9)

    def test_em_dash_separator(self):
        resp = "Verdict— FAIL\nConfidence— 0.3\nIssues— bad claim"
        result = _parse_verification_response(resp)
        assert result.passed is False


class TestConfidenceParsing:
    """Tests for various confidence format parsing."""

    def test_decimal(self):
        assert _parse_confidence("0.85") == pytest.approx(0.85)

    def test_fraction_8_of_10(self):
        assert _parse_confidence("8/10") == pytest.approx(0.8)

    def test_fraction_7_of_10(self):
        assert _parse_confidence("7/10") == pytest.approx(0.7)

    def test_percentage(self):
        assert _parse_confidence("85%") == pytest.approx(0.85)

    def test_integer_scale_10(self):
        # 8 interpreted as 8/10
        assert _parse_confidence("8") == pytest.approx(0.8)

    def test_zero(self):
        assert _parse_confidence("0") == pytest.approx(0.0)

    def test_one(self):
        assert _parse_confidence("1.0") == pytest.approx(1.0)

    def test_invalid_returns_default(self):
        assert _parse_confidence("abc") == pytest.approx(0.5)


class TestJsonFallbackParsing:
    """Tests for JSON-format verification parsing."""

    def test_json_pass(self):
        resp = '{"verdict": "PASS", "confidence": 0.9, "issues": [], "suggested_fix": "none"}'
        result = _parse_verification_response(resp)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.9)
        assert result.issues == []

    def test_json_fail_with_issues(self):
        resp = '{"verdict": "FAIL", "confidence": 0.3, "issues": ["bad claim", "missing source"], "suggested_fix": "remove claim"}'
        result = _parse_verification_response(resp)
        assert result.passed is False
        assert len(result.issues) == 2
        assert result.suggested_fix == "remove claim"

    def test_json_in_code_fence(self):
        resp = '```json\n{"verdict": "PASS", "confidence": 0.85, "issues": "none"}\n```'
        result = _parse_verification_response(resp)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.85)

    def test_json_with_passed_key(self):
        resp = '{"passed": true, "confidence": 0.9, "issues": []}'
        result = _parse_verification_response(resp)
        assert result.passed is True

    def test_invalid_json_falls_back_to_regex(self):
        resp = "{invalid json\nVerdict: PASS\nConfidence: 0.9\nIssues: none"
        result = _parse_verification_response(resp)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.9)


class TestCaveatMode:
    """Tests for three-zone confidence scoring (normal / caveat / abstain)."""

    def _config(self, threshold=0.6, caveat=0.4, abstain=True):
        return VerificationConfig(
            confidence_threshold=threshold,
            caveat_threshold=caveat,
            abstain_on_low_confidence=abstain,
        )

    def test_high_confidence_no_caveat(self):
        verification = VerificationResult(passed=True, confidence=0.95)
        assessment = compute_confidence([0.9, 0.8, 0.7], verification, self._config())
        assert not assessment.should_abstain
        assert not assessment.caveat_added

    def test_moderate_confidence_caveat(self):
        verification = VerificationResult(passed=True, confidence=0.5)
        assessment = compute_confidence([0.3, 0.2], verification, self._config())
        assert not assessment.should_abstain
        assert assessment.caveat_added

    def test_low_confidence_abstain(self):
        verification = VerificationResult(passed=False, confidence=0.2)
        assessment = compute_confidence([0.1], verification, self._config())
        assert assessment.should_abstain
        assert not assessment.caveat_added

    def test_abstain_disabled_no_abstain(self):
        verification = VerificationResult(passed=False, confidence=0.1)
        assessment = compute_confidence([0.1], verification, self._config(abstain=False))
        assert not assessment.should_abstain


class TestVerifyAnswer:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = MockLLM(text="Verdict: PASS\nConfidence: 0.9\nIssues: none\nSuggested fix: none")
        result = await verify_answer("q?", "The answer.", ["source text"], llm)
        assert result.passed is True
        assert result.confidence == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_error_response(self):
        llm = MockLLM(text="", error="timeout")
        result = await verify_answer("q?", "Answer.", ["src"], llm)
        assert result.passed is True
        assert result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_empty_answer(self):
        llm = MockLLM(text="should not be called")
        result = await verify_answer("q?", "", ["src"], llm)
        assert result.passed is True
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_llm_exception(self):
        class FailLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                raise ConnectionError("gone")
            async def close(self):
                pass

        result = await verify_answer("q?", "Answer.", ["src"], FailLLM())
        assert result.passed is True  # conservative fallback


# ── Mermaid Validator ────────────────────────────────────────────────


class TestMermaidValidatorRegex:
    def test_empty_diagram(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("")
        assert not result["valid"]
        assert "Empty" in result["error"]

    def test_no_type_declaration(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("this is not mermaid")
        assert not result["valid"]
        assert "recognised" in result["error"].lower()

    def test_bracket_imbalance(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph TD\n  A[label(")
        assert not result["valid"]
        assert "bracket" in result["error"].lower()

    def test_valid_flowchart(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph TD\n  A[Start] --> B[End]")
        assert result["valid"]
        assert result["diagram_type"] is not None

    def test_valid_sequence(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("sequenceDiagram\n  A->>B: Hello")
        assert result["valid"]

    def test_valid_class_diagram(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("classDiagram\n  Animal <|-- Duck")
        assert result["valid"]

    def test_valid_state_diagram(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("stateDiagram-v2\n  [*] --> Active")
        assert result["valid"]

    def test_valid_er_diagram(self):
        v = MermaidValidator(mode="regex")
        # ER diagrams use ||--o{ which has unbalanced braces; use simpler syntax
        result = v.validate("erDiagram\n  CUSTOMER ||--|| ORDER : places")
        assert result["valid"]

    def test_valid_gantt(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("gantt\n  title Schedule")
        assert result["valid"]

    def test_valid_pie(self):
        v = MermaidValidator(mode="regex")
        result = v.validate('pie\n  "A" : 60')
        assert result["valid"]


class TestMermaidValidatorBareTypes:
    """Bug 5: bare 'graph' and 'flowchart' without direction are valid Mermaid."""

    def test_bare_graph_valid(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph\n  A --> B")
        assert result["valid"], f"bare graph rejected: {result.get('error')}"

    def test_bare_flowchart_valid(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("flowchart\n  A --> B")
        assert result["valid"], f"bare flowchart rejected: {result.get('error')}"

    def test_graph_with_direction_still_valid(self):
        v = MermaidValidator(mode="regex")
        for direction in ("TD", "TB", "BT", "RL", "LR"):
            result = v.validate(f"graph {direction}\n  A --> B")
            assert result["valid"], f"graph {direction} rejected"

    def test_flowchart_with_direction_still_valid(self):
        v = MermaidValidator(mode="regex")
        for direction in ("TD", "TB", "BT", "RL", "LR"):
            result = v.validate(f"flowchart {direction}\n  A --> B")
            assert result["valid"], f"flowchart {direction} rejected"

    def test_bare_graph_detects_type(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph\n  A --> B")
        assert result["diagram_type"] == "graph"

    def test_graph_td_detects_type(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph TD\n  A --> B")
        assert result["diagram_type"] == "graph TD"


class TestMermaidValidatorModes:
    def test_none_mode_always_valid(self):
        v = MermaidValidator(mode="none")
        result = v.validate("totally invalid")
        assert result["valid"]

    def test_regex_mode_no_node(self):
        v = MermaidValidator(mode="regex")
        result = v.validate("graph TD\n  A --> B")
        assert result["valid"]

    def test_auto_mode_passes_regex(self):
        v = MermaidValidator(mode="auto")
        result = v.validate("graph TD\n  A --> B")
        assert result["valid"]


class TestDiagramFiltering:
    """Bug 2: Invalid fallback diagrams should be filtered out, not just logged."""

    def test_filter_invalid_diagrams(self):
        """Reproduce the filtering logic from query_pipeline — invalid diagrams dropped."""
        v = MermaidValidator(mode="regex")
        diagrams = [
            "graph TD\n  A --> B",          # valid
            "not_a_diagram\n  garbage",      # invalid
            "sequenceDiagram\n  A->>B: Hi",  # valid
        ]

        valid_diagrams = []
        for diagram in diagrams:
            result = v.validate(diagram)
            if result["valid"]:
                valid_diagrams.append(diagram)

        assert len(valid_diagrams) == 2
        assert "A --> B" in valid_diagrams[0]
        assert "A->>B: Hi" in valid_diagrams[1]

    def test_all_invalid_returns_none(self):
        """When every diagram is invalid, the list should become None (empty)."""
        v = MermaidValidator(mode="regex")
        diagrams = ["garbage", "also garbage("]

        valid_diagrams = [d for d in diagrams if v.validate(d)["valid"]]
        result = valid_diagrams or None
        assert result is None

    def test_all_valid_unchanged(self):
        """When all diagrams are valid, the full list is returned."""
        v = MermaidValidator(mode="regex")
        diagrams = [
            "graph TD\n  A --> B",
            "pie\n  \"Cats\" : 60\n  \"Dogs\" : 40",
        ]

        valid_diagrams = [d for d in diagrams if v.validate(d)["valid"]]
        assert len(valid_diagrams) == 2


# ── Grader ───────────────────────────────────────────────────────────


class TestGradingParsing:
    def test_valid_response(self):
        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2"), _chunk(chunk_id="c3")]
        resp = (
            "Chunk 1: RELEVANT — directly answers\n"
            "Chunk 2: PARTIAL — some info\n"
            "Chunk 3: IRRELEVANT — off topic\n"
        )
        graded = _parse_grading_response(resp, chunks)
        assert len(graded) == 3
        assert graded[0].grade == "relevant"
        assert graded[1].grade == "partial"
        assert graded[2].grade == "irrelevant"

    def test_missing_grade_defaults_relevant(self):
        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
        resp = "Chunk 1: RELEVANT — good\n"
        graded = _parse_grading_response(resp, chunks)
        assert graded[1].grade == "relevant"  # conservative default

    def test_invalid_format_all_relevant(self):
        chunks = [_chunk()]
        resp = "I can't grade these."
        graded = _parse_grading_response(resp, chunks)
        assert graded[0].grade == "relevant"


class TestGradeDocuments:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = MockLLM(text="Chunk 1: RELEVANT — good\nChunk 2: IRRELEVANT — bad")
        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
        graded = await grade_documents("query", chunks, llm)
        assert len(graded) == 2
        assert graded[0].grade == "relevant"
        assert graded[1].grade == "irrelevant"

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self):
        llm = MockLLM(text="", error="timeout")
        chunks = [_chunk()]
        graded = await grade_documents("query", chunks, llm)
        assert all(g.grade == "relevant" for g in graded)

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        llm = MockLLM(text="should not be called")
        graded = await grade_documents("query", [], llm)
        assert graded == []

    @pytest.mark.asyncio
    async def test_llm_exception_fallback(self):
        class FailLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                raise RuntimeError("boom")
            async def close(self):
                pass

        chunks = [_chunk()]
        graded = await grade_documents("query", chunks, FailLLM())
        assert all(g.grade == "relevant" for g in graded)
