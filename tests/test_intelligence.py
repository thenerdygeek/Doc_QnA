"""Tests for the intelligence layer: intent classifier, query analyzer, output detector, confidence."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from doc_qa.intelligence.intent_classifier import (
    IntentMatch,
    OutputIntent,
    _parse_llm_intent,
    classify_by_heuristic,
    classify_intent,
)
from doc_qa.intelligence.output_detector import detect_response_formats
from doc_qa.intelligence.query_analyzer import (
    _parse_sub_queries,
    assess_complexity,
    detect_multi_intent,
)
from doc_qa.intelligence.confidence import (
    ConfidenceAssessment,
    compute_confidence,
)
from doc_qa.config import VerificationConfig
from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.verification.verifier import VerificationResult


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLM(LLMBackend):
    """Mock LLM that returns a controllable response."""

    def __init__(self, text: str = "", error: str | None = None) -> None:
        self.text = text
        self.error = error

    async def ask(self, question: str, context: str, history=None) -> Answer:
        return Answer(text=self.text, sources=[], model="mock", error=self.error)

    async def close(self) -> None:
        pass


# ── Intent Classifier ────────────────────────────────────────────────


class TestHeuristicClassification:
    """Test classify_by_heuristic with positive and negative cases."""

    def test_diagram_explicit_mermaid(self):
        result = classify_by_heuristic("Show a mermaid diagram of the auth flow")
        assert result is not None
        assert result.intent == OutputIntent.DIAGRAM
        assert result.confidence >= 0.9

    def test_diagram_topic_and_verb(self):
        result = classify_by_heuristic("Draw a diagram of the authentication flow")
        assert result is not None
        assert result.intent == OutputIntent.DIAGRAM

    def test_code_example(self):
        result = classify_by_heuristic("Show me a curl API endpoint example")
        assert result is not None
        assert result.intent == OutputIntent.CODE_EXAMPLE

    def test_comparison_phrase(self):
        result = classify_by_heuristic("What are the pros and cons of JWT?")
        assert result is not None
        assert result.intent == OutputIntent.COMPARISON_TABLE

    def test_comparison_topic_verb(self):
        result = classify_by_heuristic("Compare differences between REST and GraphQL")
        assert result is not None
        assert result.intent == OutputIntent.COMPARISON_TABLE

    def test_procedural(self):
        result = classify_by_heuristic("How do I set up and configure the database?")
        assert result is not None
        assert result.intent == OutputIntent.PROCEDURAL

    def test_no_match_returns_none(self):
        result = classify_by_heuristic("What is the rate limit?")
        assert result is None


class TestDiagramSubTypeDetection:
    """Test sub-type detection for diagram intents."""

    def test_sequence_subtype(self):
        result = classify_by_heuristic("Draw a diagram of the message flow between services")
        assert result is not None
        assert result.sub_type == "sequence"

    def test_er_subtype(self):
        result = classify_by_heuristic("Visualize the entity relationship data model")
        assert result is not None
        assert result.sub_type == "erDiagram"

    def test_class_subtype(self):
        result = classify_by_heuristic("Show a mermaid class diagram with inheritance")
        assert result is not None
        assert result.sub_type == "classDiagram"

    def test_state_subtype(self):
        result = classify_by_heuristic("Draw a mermaid state machine lifecycle diagram")
        assert result is not None
        assert result.sub_type == "stateDiagram"

    def test_default_flowchart(self):
        result = classify_by_heuristic("Show a mermaid diagram of the deploy process")
        assert result is not None
        assert result.sub_type == "flowchart"


class TestCodeSubTypeDetection:
    """Test sub-type detection for code intents."""

    def test_curl_subtype(self):
        result = classify_by_heuristic("Show me a curl REST example")
        assert result is not None
        assert result.sub_type == "curl"

    def test_graphql_subtype(self):
        result = classify_by_heuristic("Show me a graphql mutation code example")
        assert result is not None
        assert result.sub_type == "graphql"

    def test_yaml_subtype(self):
        result = classify_by_heuristic("Create a kubernetes yaml example")
        assert result is not None
        assert result.sub_type == "yaml"

    def test_json_subtype(self):
        result = classify_by_heuristic("Show me a json example request body")
        assert result is not None
        assert result.sub_type == "json"


class TestLLMIntentParsing:
    """Test _parse_llm_intent with various LLM response formats."""

    def test_valid_response_with_reasoning(self):
        resp = "Reasoning: The user wants a visual flow.\nIntent: DIAGRAM\nSub-type: sequence"
        result = _parse_llm_intent(resp)
        assert result.intent == OutputIntent.DIAGRAM
        assert result.confidence == 0.85
        assert result.sub_type == "sequence"

    def test_valid_response_without_reasoning(self):
        resp = "Intent: CODE_EXAMPLE\nSub-type: curl"
        result = _parse_llm_intent(resp)
        assert result.intent == OutputIntent.CODE_EXAMPLE
        assert result.confidence == 0.70

    def test_fuzzy_match(self):
        resp = "I think this is a COMPARISON_TABLE request."
        result = _parse_llm_intent(resp)
        assert result.intent == OutputIntent.COMPARISON_TABLE
        assert result.confidence == 0.60

    def test_parse_failure_defaults_to_explanation(self):
        resp = "This is completely unparseable nonsense."
        result = _parse_llm_intent(resp)
        assert result.intent == OutputIntent.EXPLANATION
        assert result.confidence == 0.50

    def test_invalid_intent_name_falls_through(self):
        resp = "Intent: UNKNOWN_THING"
        result = _parse_llm_intent(resp)
        # Should fall through to fuzzy or default
        assert result.intent == OutputIntent.EXPLANATION


class TestClassifyIntent:
    """Test the full classify_intent entry point with mock LLM."""

    @pytest.mark.asyncio
    async def test_heuristic_match_skips_llm(self):
        llm = MockLLM(text="should not be used")
        result = await classify_intent("Show a mermaid diagram of auth", llm)
        assert result.intent == OutputIntent.DIAGRAM
        assert result.matched_pattern == "explicit_mermaid"

    @pytest.mark.asyncio
    async def test_llm_fallback(self):
        llm = MockLLM(text="Reasoning: needs code\nIntent: CODE_EXAMPLE\nSub-type: curl")
        result = await classify_intent("What is the rate limit?", llm)
        assert result.intent == OutputIntent.CODE_EXAMPLE

    @pytest.mark.asyncio
    async def test_llm_error_returns_explanation(self):
        llm = MockLLM(text="", error="connection failed")
        result = await classify_intent("What is the rate limit?", llm)
        assert result.intent == OutputIntent.EXPLANATION
        assert result.matched_pattern == "llm_error_fallback"


# ── Query Analyzer ───────────────────────────────────────────────────


class TestMultiIntentDetection:
    """Test detect_multi_intent heuristic."""

    def test_coordination_with_two_verbs(self):
        assert detect_multi_intent("Show the architecture and also explain the API")

    def test_no_coordination_marker(self):
        assert not detect_multi_intent("Show me the architecture diagram")

    def test_coordination_but_one_verb(self):
        assert not detect_multi_intent("Explain the API and also explain the flow")

    def test_plus_show(self):
        assert detect_multi_intent("Describe the flow plus show a diagram")


class TestDecompositionParsing:
    """Test _parse_sub_queries response parsing."""

    def test_sub_query_format(self):
        resp = "SUB-QUERY 1: What is OAuth?\nSUB-QUERY 2: Draw a diagram of OAuth flow."
        parts = _parse_sub_queries(resp)
        assert len(parts) == 2
        assert "OAuth" in parts[0]
        assert "diagram" in parts[1]

    def test_numbered_items_fallback(self):
        resp = "1. What is OAuth?\n2. Show OAuth diagram."
        parts = _parse_sub_queries(resp)
        assert len(parts) == 2

    def test_single_line_fallback(self):
        resp = "Just a single query here."
        parts = _parse_sub_queries(resp)
        assert len(parts) == 1
        assert "single query" in parts[0]

    def test_empty_response(self):
        parts = _parse_sub_queries("")
        assert parts == []


class TestComplexityAssessment:
    """Test assess_complexity for simple vs multi_hop."""

    def test_simple_query(self):
        assert assess_complexity("What is the API rate limit?") == "simple"

    def test_multi_hop_with_entities_and_causal(self):
        # 3+ named entities + causal chain
        result = assess_complexity(
            "How does the Auth Server interact with the Token Manager "
            "because the Session Store expires tokens, and as described "
            "in the security docs?"
        )
        assert result == "multi_hop"


# ── Output Detector ──────────────────────────────────────────────────


class TestOutputDetector:
    """Test detect_response_formats."""

    def test_mermaid_detection(self):
        text = "Here is a diagram:\n```mermaid\ngraph TD\nA-->B\n```\n"
        fmt = detect_response_formats(text)
        assert fmt.has_mermaid
        assert len(fmt.mermaid_blocks) == 1
        assert "graph TD" in fmt.mermaid_blocks[0]

    def test_code_block_detection(self):
        text = "Example:\n```python\nprint('hello')\n```\n"
        fmt = detect_response_formats(text)
        assert fmt.has_code_blocks
        assert "python" in fmt.code_languages

    def test_mermaid_excluded_from_code_blocks(self):
        text = "```mermaid\ngraph TD\nA-->B\n```\n"
        fmt = detect_response_formats(text)
        assert fmt.has_mermaid
        assert not fmt.has_code_blocks

    def test_table_detection(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |\n"
        fmt = detect_response_formats(text)
        assert fmt.has_table

    def test_numbered_list_detection(self):
        text = "Steps:\n1. First\n2. Second\n3. Third\n"
        fmt = detect_response_formats(text)
        assert fmt.has_numbered_list

    def test_empty_text(self):
        fmt = detect_response_formats("")
        assert not fmt.has_mermaid
        assert not fmt.has_code_blocks
        assert not fmt.has_table
        assert not fmt.has_numbered_list


# ── Confidence Scoring ───────────────────────────────────────────────


class TestConfidenceScoring:
    """Test compute_confidence and its components."""

    def _config(self, threshold=0.4, abstain=True):
        return VerificationConfig(
            confidence_threshold=threshold,
            abstain_on_low_confidence=abstain,
        )

    def test_high_retrieval_no_verification(self):
        result = compute_confidence(
            retrieval_scores=[0.9, 0.8, 0.7],
            verification=None,
            config=self._config(),
        )
        # retrieval avg = 0.8, no penalties; verification = 0.7 default
        # combined = 0.4 * 0.8 + 0.6 * 0.7 = 0.74
        assert result.score > 0.7
        assert not result.should_abstain

    def test_low_retrieval_penalty(self):
        result = compute_confidence(
            retrieval_scores=[0.2, 0.1, 0.1],
            verification=None,
            config=self._config(),
        )
        # All below 0.3 -> signal halved
        assert result.retrieval_signal < 0.15

    def test_single_source_reliance_penalty(self):
        result = compute_confidence(
            retrieval_scores=[0.95, 0.3, 0.2],
            verification=None,
            config=self._config(),
        )
        # Gap 0.95 - 0.3 = 0.65 > 0.3 -> 0.15 penalty on avg
        assert result.retrieval_signal < sum([0.95, 0.3, 0.2]) / 3

    def test_verification_signal_pass(self):
        vr = VerificationResult(passed=True, confidence=0.9, issues=[])
        result = compute_confidence(
            retrieval_scores=[0.8, 0.7],
            verification=vr,
            config=self._config(),
        )
        assert result.verification_signal == 0.9

    def test_verification_signal_fail_penalty(self):
        vr = VerificationResult(passed=False, confidence=0.6, issues=["hallucination"])
        result = compute_confidence(
            retrieval_scores=[0.8, 0.7],
            verification=vr,
            config=self._config(),
        )
        # 0.6 - 0.2 = 0.4
        assert result.verification_signal == pytest.approx(0.4)

    def test_abstention_below_threshold(self):
        # Three-zone logic: >= confidence → normal, >= caveat → caveat, < caveat → abstain
        # Use failed verification to push score below caveat_threshold (0.4)
        vr = VerificationResult(passed=False, confidence=0.1, issues=["wrong"])
        result = compute_confidence(
            retrieval_scores=[0.1],
            verification=vr,
            config=self._config(threshold=0.8, abstain=True),
        )
        # retrieval=0.05 (halved), verification=max(0,0.1-0.2)=0.0
        # combined = 0.4*0.05 + 0.6*0.0 = 0.02 → below caveat_threshold → abstain
        assert result.should_abstain
        assert result.abstain_reason is not None

    def test_no_abstention_when_disabled(self):
        result = compute_confidence(
            retrieval_scores=[0.1],
            verification=None,
            config=self._config(threshold=0.8, abstain=False),
        )
        assert not result.should_abstain

    def test_empty_retrieval_scores(self):
        result = compute_confidence(
            retrieval_scores=[],
            verification=None,
            config=self._config(),
        )
        assert result.retrieval_signal == 0.0
