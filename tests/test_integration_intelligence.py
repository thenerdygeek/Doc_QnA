"""Integration tests for the 8-phase intelligence pipeline.

Tests the interaction between multiple modules working together through
the full pipeline.  All assertion data is hand-crafted by analysing
each module's behaviour (regex patterns, thresholds, scoring formulae).

Phases tested:
  1. Intent Classification (heuristic + LLM fallback)
  2. Retrieval + File Diversity
  3. CRAG (grade → rewrite → re-retrieve → merge)
  4. Specialized Generation (diagram / code / comparison / procedural)
  5. Output Format Detection
  6. Mermaid Validation
  7. Verification + Confidence Scoring + Abstention
  8. Source Attribution
  + SSE Streaming (event sequence, token streaming)
  + Multi-turn History Management
  + Multi-intent Decomposition and Merging
  + Error Recovery / Graceful Degradation
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from doc_qa.config import (
    GenerationConfig,
    IntelligenceConfig,
    VerificationConfig,
)
from doc_qa.intelligence.intent_classifier import IntentMatch, OutputIntent
from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.retrieval.query_pipeline import QueryPipeline, QueryResult, SourceReference
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.streaming.sse import streaming_query
from doc_qa.verification.source_attributor import Attribution


# =====================================================================
# Test infrastructure
# =====================================================================


def _chunk(
    text: str = "Generic chunk text.",
    score: float = 0.80,
    chunk_id: str = "c1",
    file_path: str = "/docs/auth.md",
    section_title: str = "Overview",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        score=score,
        chunk_id=chunk_id,
        file_path=file_path,
        file_type="md",
        section_title=section_title,
        section_level=1,
        chunk_index=0,
    )


class RoutingMockLLM(LLMBackend):
    """Mock LLM that dispatches responses based on prompt content.

    Each key in *responses* is matched against the ``question`` parameter
    (case-insensitive substring match).  Keys are checked in order; the
    first match wins.  The special key ``"default"`` is used when nothing
    matches.
    """

    def __init__(self, responses: dict[str, str | Answer]) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    async def ask(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
    ) -> Answer:
        self.calls.append({
            "question": question[:200],
            "context": context[:200] if context else "",
        })
        q = question.lower()
        for key, resp in self._responses.items():
            if key == "default":
                continue
            if key.lower() in q:
                if isinstance(resp, Answer):
                    return resp
                return Answer(text=resp, sources=[], model="mock")
        default = self._responses.get("default", "Fallback answer.")
        if isinstance(default, Answer):
            return default
        return Answer(text=default, sources=[], model="mock")

    async def close(self) -> None:
        pass


class StreamingMockLLM(RoutingMockLLM):
    """Extends RoutingMockLLM with ``ask_streaming`` support."""

    async def ask_streaming(
        self,
        question: str,
        context: str,
        history: list[dict] | None = None,
        on_token=None,
    ) -> Answer:
        answer = await self.ask(question, context, history)
        if on_token and answer.text:
            words = answer.text.split()
            cumulative = ""
            for w in words:
                cumulative += ("" if not cumulative else " ") + w
                on_token(cumulative)
        return answer


def _make_pipeline(
    llm: LLMBackend,
    chunks: list[RetrievedChunk],
    *,
    intel: IntelligenceConfig | None = None,
    gen: GenerationConfig | None = None,
    verify: VerificationConfig | None = None,
    rerank: bool = False,
    top_k: int = 5,
    max_per_file: int = 3,
    max_history_turns: int = 10,
) -> QueryPipeline:
    """Build a QueryPipeline with a monkeypatched retriever."""
    pipeline = QueryPipeline(
        table=MagicMock(),
        llm_backend=llm,
        embedding_model="mock",
        search_mode="hybrid",
        candidate_pool=20,
        top_k=top_k,
        min_score=0.3,
        max_chunks_per_file=max_per_file,
        rerank=rerank,
        max_history_turns=max_history_turns,
        intelligence_config=intel,
        generation_config=gen,
        verification_config=verify,
    )
    pipeline._retriever.search = MagicMock(return_value=chunks)
    return pipeline


# Helper for SSE tests

class MockRequest:
    """Minimal Request stub that is never disconnected."""

    async def is_disconnected(self) -> bool:
        return False


class MockConfig:
    """Lightweight config object for SSE streaming tests."""

    def __init__(
        self,
        enable_intent: bool = False,
        enable_crag: bool = False,
        enable_verification: bool = False,
    ) -> None:
        self.intelligence = (
            type("IC", (), {
                "enable_intent_classification": True,
                "enable_multi_intent": False,
                "intent_confidence_high": 0.85,
                "intent_confidence_medium": 0.65,
            })()
            if enable_intent
            else None
        )
        self.verification = (
            type("VC", (), {
                "enable_crag": enable_crag,
                "enable_verification": enable_verification,
                "max_crag_rewrites": 2,
            })()
            if enable_crag or enable_verification
            else None
        )
        self.generation = None
        self.streaming = None


async def _collect_sse(gen) -> list[dict]:
    events = []
    async for sse in gen:
        data = json.loads(sse.data) if sse.data else {}
        events.append({"event": sse.event, "data": data})
    return events


# =====================================================================
# Phase 1 — Intent Classification through the Pipeline
# =====================================================================


class TestIntentHeuristicThroughPipeline:
    """Heuristic intent → specialized generation → correct result fields."""

    @pytest.mark.asyncio
    async def test_diagram_intent_heuristic_routes_to_diagram_generator(self):
        """Query 'Draw a diagram of the authentication flow':
        - _DIAGRAM_TOPIC matches 'flow', _DIAGRAM_VERB matches 'diagram'
        - confidence = 0.92, sub_type inferred by _detect_diagram_subtype → 'flowchart'
        - 0.92 >= 0.85 → specialized only (no explanation appended)
        - DiagramGenerator receives augmented question with DIAGRAM_GENERATION template
        """
        mermaid_response = (
            "Here is the authentication flow:\n"
            "```mermaid\n"
            "graph TD\n"
            "    A[User] --> B[Auth Server]\n"
            "    B --> C[Token]\n"
            "```\n"
            "The diagram shows the OAuth handshake."
        )
        llm = RoutingMockLLM({
            # DiagramGenerator prepends "Generate a Mermaid diagram" to the question
            "generate a mermaid diagram": mermaid_response,
            "default": "Should not be used.",
        })
        chunks = [
            _chunk(text="OAuth 2.0 uses access tokens.", score=0.85, chunk_id="c1"),
            _chunk(text="Auth server validates credentials.", score=0.72, chunk_id="c2"),
        ]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig(), gen=GenerationConfig(mermaid_validation="regex"))

        result = await pipeline.query("Draw a diagram of the authentication flow")

        assert result.intent == "DIAGRAM"
        assert result.intent_confidence == pytest.approx(0.92)
        assert result.diagrams is not None
        assert len(result.diagrams) == 1
        assert "A[User] --> B[Auth Server]" in result.diagrams[0]
        assert result.detected_formats["has_mermaid"] is True
        assert result.chunks_retrieved == 2
        assert result.chunks_after_rerank == 2

    @pytest.mark.asyncio
    async def test_procedural_intent_heuristic(self):
        """Query 'How do I set up and configure the database?':
        - _PROCEDURAL_TOPIC matches 'how do i', _PROCEDURAL_VERB matches 'set up' + 'configure'
        - confidence = 0.88, sub_type = 'none'
        - 0.88 >= 0.85 → specialized only
        - ProceduralGenerator prepends PROCEDURAL_GENERATION
        """
        procedural_response = (
            "1. Install PostgreSQL from the official site\n"
            "2. Create a new database with createdb\n"
            "3. Configure the connection string in config.yaml\n"
        )
        llm = RoutingMockLLM({
            "answer as numbered steps": procedural_response,
            "default": "Should not be used.",
        })
        chunks = [_chunk(text="Database setup guide.", score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig())

        result = await pipeline.query("How do I set up and configure the database?")

        assert result.intent == "PROCEDURAL"
        assert result.intent_confidence == pytest.approx(0.88)
        assert "1." in result.answer
        assert result.detected_formats["has_numbered_list"] is True

    @pytest.mark.asyncio
    async def test_comparison_phrase_intent(self):
        """Query 'What are the pros and cons of JWT vs sessions?':
        - _COMPARISON_PHRASE matches 'pros and cons'
        - confidence = 0.93, sub_type = 'none'
        """
        table_response = (
            "| Feature | JWT | Sessions |\n"
            "|---------|-----|----------|\n"
            "| Storage | Client | Server |\n"
            "| Scalability | High | Low |\n"
        )
        llm = RoutingMockLLM({
            "comparison as a markdown table": table_response,
            "default": "Should not be used.",
        })
        chunks = [_chunk(text="Auth comparison.", score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig())

        result = await pipeline.query("What are the pros and cons of JWT vs sessions?")

        assert result.intent == "COMPARISON_TABLE"
        assert result.intent_confidence == pytest.approx(0.93)
        assert result.detected_formats["has_table"] is True

    @pytest.mark.asyncio
    async def test_code_example_with_language_tag_injection(self):
        """Query 'Show me a curl API endpoint example':
        - _CODE_FORMAT matches 'curl', 'API', 'endpoint'; _CODE_VERB matches 'show'
        - confidence = 0.90, sub_type 'curl' detected by _SUBTYPE_CURL
        - CodeExampleGenerator injects CODE_EXAMPLE_GENERATION with code_format=curl
        - Bare ``` fences get 'curl' language tag via _ensure_language_tags
        """
        code_response = (
            "Here is the example:\n"
            "```\n"
            "curl -X POST https://api.example.com/auth\n"
            "```\n"
        )
        llm = RoutingMockLLM({
            "include a concrete, runnable code example": code_response,
            "default": "Should not be used.",
        })
        chunks = [_chunk(text="API docs.", score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig())

        result = await pipeline.query("Show me a curl API endpoint example")

        assert result.intent == "CODE_EXAMPLE"
        assert result.intent_confidence == pytest.approx(0.90)
        # Bare ``` replaced by ```curl
        assert "```curl" in result.answer
        assert result.detected_formats["has_code_blocks"] is True


class TestIntentLLMFallbackThroughPipeline:
    """When heuristics return None, the LLM fallback is used."""

    @pytest.mark.asyncio
    async def test_llm_fallback_classifies_and_routes(self):
        """Query 'What is the rate limit?' has no heuristic match.
        LLM returns 'Reasoning: factual\nIntent: EXPLANATION\nSub-type: none'
        → confidence 0.85 (has Reasoning line) → specialized (ExplanationGenerator)
        """
        llm = RoutingMockLLM({
            "query intent classifier": (
                "Reasoning: This is a factual question.\n"
                "Intent: EXPLANATION\n"
                "Sub-type: none"
            ),
            "default": "The rate limit is 100 requests per second.",
        })
        chunks = [_chunk(text="Rate limiting docs.", score=0.75, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig())

        result = await pipeline.query("What is the rate limit?")

        assert result.intent == "EXPLANATION"
        # LLM classification with Reasoning line → 0.85 confidence
        assert result.intent_confidence == pytest.approx(0.85)
        assert "100 requests per second" in result.answer


# =====================================================================
# Phase 3 — CRAG (Corrective Retrieval-Augmented Generation)
# =====================================================================


class TestCRAGIntegration:
    """Test the full CRAG flow through the pipeline."""

    @pytest.mark.asyncio
    async def test_crag_rewrites_when_mostly_irrelevant(self):
        """Initial chunks graded mostly IRRELEVANT → rewrite triggered.

        Grading response: 'Chunk 1: IRRELEVANT — off topic\\nChunk 2: IRRELEVANT — wrong'
        should_rewrite: 2/2 = 1.0 > 0.5 threshold → True

        After rewrite, retriever returns new chunks.
        Second grading: all RELEVANT → no more rewrites.
        Final result uses new chunk text.
        """
        call_count = 0

        class CRAGMockLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                nonlocal call_count
                call_count += 1
                q = question.lower()

                # Grading calls (context is "")
                if "evaluating the relevance" in q:
                    if call_count <= 2:
                        # First grading: all irrelevant
                        return Answer(
                            text=(
                                "Chunk 1: IRRELEVANT — off topic discussion\n"
                                "Chunk 2: IRRELEVANT — wrong section\n"
                            ),
                            sources=[], model="mock",
                        )
                    # Second grading after re-retrieval: all relevant
                    return Answer(
                        text="Chunk 1: RELEVANT — directly answers\n",
                        sources=[], model="mock",
                    )

                # Rewrite call
                if "initial retrieval returned" in q:
                    return Answer(
                        text="improved authentication token query",
                        sources=[], model="mock",
                    )

                # Standard generation (has context)
                return Answer(
                    text="Tokens expire after 30 minutes.",
                    sources=[], model="mock",
                )

            async def close(self):
                pass

        initial_chunks = [
            _chunk(text="Unrelated deployment docs.", score=0.60, chunk_id="c1"),
            _chunk(text="Unrelated networking docs.", score=0.55, chunk_id="c2"),
        ]
        new_chunks = [
            _chunk(text="Auth tokens expire after 30 minutes.", score=0.82, chunk_id="c3",
                   file_path="/docs/tokens.md", section_title="Token Expiry"),
        ]

        llm = CRAGMockLLM()
        verify_cfg = VerificationConfig(
            enable_crag=True,
            enable_verification=False,
            max_crag_rewrites=2,
        )
        pipeline = _make_pipeline(llm, initial_chunks, verify=verify_cfg)

        # After CRAG rewrite, retriever.search is called again with the rewritten query
        pipeline._retriever.search = MagicMock(side_effect=[initial_chunks, new_chunks])

        result = await pipeline.query("How long do tokens last?")

        assert "30 minutes" in result.answer
        # Retriever called twice: initial + re-retrieve
        assert pipeline._retriever.search.call_count == 2

    @pytest.mark.asyncio
    async def test_crag_no_rewrite_when_relevant(self):
        """All chunks graded RELEVANT → no rewrite needed."""
        llm = RoutingMockLLM({
            "evaluating the relevance": (
                "Chunk 1: RELEVANT — directly answers\n"
                "Chunk 2: RELEVANT — supporting context\n"
            ),
            "default": "Auth uses OAuth 2.0 tokens.",
        })
        chunks = [
            _chunk(text="OAuth 2.0 access tokens.", score=0.85, chunk_id="c1"),
            _chunk(text="Token validation flow.", score=0.78, chunk_id="c2"),
        ]
        verify_cfg = VerificationConfig(enable_crag=True, enable_verification=False)
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("How does authentication work?")

        assert "OAuth 2.0" in result.answer
        # Retriever only called once (initial)
        assert pipeline._retriever.search.call_count == 1


# =====================================================================
# Phase 7 — Verification + Confidence + Abstention
# =====================================================================


class TestVerificationConfidenceIntegration:
    """Test verification → confidence scoring → abstention decision."""

    @pytest.mark.asyncio
    async def test_high_confidence_no_abstention(self):
        """Retrieval scores [0.85, 0.72, 0.65], verification PASS with 0.92.

        Retrieval signal:
            avg = (0.85 + 0.72 + 0.65) / 3 = 0.74
            No all-below-0.3 penalty.
            Gap = 0.85 - 0.72 = 0.13, not > 0.3 → no single-source penalty.
            → retrieval_signal = 0.74

        Verification signal:
            passed=True, confidence=0.92 → signal = 0.92

        Combined = 0.4 * 0.74 + 0.6 * 0.92 = 0.296 + 0.552 = 0.848
        Threshold 0.4 → no abstention.
        """
        llm = RoutingMockLLM({
            "fact-checker": "Verdict: PASS\nConfidence: 0.92\nIssues: none\nSuggested fix: none",
            "default": "Auth uses OAuth 2.0.",
        })
        chunks = [
            _chunk(score=0.85, chunk_id="c1"),
            _chunk(score=0.72, chunk_id="c2"),
            _chunk(score=0.65, chunk_id="c3"),
        ]
        verify_cfg = VerificationConfig(
            enable_verification=True,
            enable_crag=False,
            confidence_threshold=0.4,
            abstain_on_low_confidence=True,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("How does auth work?")

        assert result.is_abstained is False
        assert result.confidence_score == pytest.approx(0.848, abs=0.01)
        assert result.verification_passed is True
        assert "OAuth 2.0" in result.answer

    @pytest.mark.asyncio
    async def test_low_confidence_triggers_abstention(self):
        """Retrieval scores [0.20, 0.15, 0.10], verification FAIL with 0.30.

        Retrieval signal:
            avg = (0.20 + 0.15 + 0.10) / 3 = 0.15
            All below 0.3 → halved → 0.075
            → retrieval_signal = 0.075

        Verification signal:
            passed=False → signal = max(0.30 - 0.20, 0) = 0.10

        Combined = 0.4 * 0.075 + 0.6 * 0.10 = 0.03 + 0.06 = 0.09
        Threshold 0.4 → ABSTAIN.
        abstain_reason = "Confidence 0.09 is below threshold 0.40"
        """
        llm = RoutingMockLLM({
            "fact-checker": (
                "Verdict: FAIL\n"
                "Confidence: 0.30\n"
                "Issues: hallucinated claim, unsupported detail\n"
                "Suggested fix: remove unsupported claims"
            ),
            "default": "Possibly wrong answer about auth.",
        })
        chunks = [
            _chunk(score=0.20, chunk_id="c1"),
            _chunk(score=0.15, chunk_id="c2"),
            _chunk(score=0.10, chunk_id="c3"),
        ]
        verify_cfg = VerificationConfig(
            enable_verification=True,
            enable_crag=False,
            confidence_threshold=0.4,
            abstain_on_low_confidence=True,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("How does auth work?")

        assert result.is_abstained is True
        assert result.confidence_score == pytest.approx(0.09, abs=0.01)
        assert result.verification_passed is False
        assert "don't have enough information" in result.answer
        assert "0.09" in result.answer
        assert "0.40" in result.answer

    @pytest.mark.asyncio
    async def test_single_source_reliance_penalty(self):
        """Retrieval scores [0.95, 0.40] with no verification.

        Retrieval signal:
            avg = (0.95 + 0.40) / 2 = 0.675
            Not all below 0.3.
            Gap = 0.95 - 0.40 = 0.55 > 0.3 → penalty 0.15
            → retrieval_signal = max(0, 0.675 - 0.15) = 0.525

        Verification signal: None → 0.7 (neutral default)

        Combined = 0.4 * 0.525 + 0.6 * 0.7 = 0.21 + 0.42 = 0.63
        Threshold 0.4 → no abstention.
        """
        llm = RoutingMockLLM({"default": "Answer based mostly on one source."})
        chunks = [
            _chunk(score=0.95, chunk_id="c1"),
            _chunk(score=0.40, chunk_id="c2"),
        ]
        verify_cfg = VerificationConfig(
            enable_verification=False,
            enable_crag=False,
            confidence_threshold=0.4,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("Single source question")

        assert result.is_abstained is False
        assert result.confidence_score == pytest.approx(0.63, abs=0.01)
        assert result.verification_passed is None  # no verification


# =====================================================================
# Phase 4 — Specialized Generation: medium-confidence band
# =====================================================================


class TestMediumConfidenceExplanationAppend:
    """When intent confidence is in [0.65, 0.85), both specialized and
    explanation generators run.  Result text = specialized + '---' + explanation."""

    @pytest.mark.asyncio
    async def test_medium_confidence_diagram_plus_explanation(self):
        """LLM intent → DIAGRAM confidence=0.75 (medium band).
        DiagramGenerator runs first, then ExplanationGenerator appended.
        """
        mermaid_response = "```mermaid\ngraph TD\n  A-->B\n```\n"
        explanation_response = "This shows the auth flow between services."

        call_seq = iter([mermaid_response, explanation_response])

        class SeqLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                q = question.lower()
                if "query intent classifier" in q:
                    return Answer(
                        text="Reasoning: visual\nIntent: DIAGRAM\nSub-type: flowchart",
                        sources=[], model="mock",
                    )
                # Generation calls: first specialized, then explanation
                if context:
                    return Answer(text=next(call_seq), sources=[], model="mock")
                return Answer(text="fallback", sources=[], model="mock")

            async def close(self):
                pass

        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(
            SeqLLM(), chunks,
            intel=IntelligenceConfig(intent_confidence_high=0.85, intent_confidence_medium=0.65),
        )

        # LLM intent returns 0.75 (has Reasoning → 0.85 in _parse_llm_intent)
        # Actually _parse_llm_intent with valid "Intent: DIAGRAM" + "Reasoning:" → 0.85
        # 0.85 >= 0.85 → high band, not medium
        # To get medium, we need confidence < 0.85. Use fuzzy match (0.60).
        # Let me adjust: LLM returns without "Intent:" line for fuzzy → 0.60 < 0.65 → explanation only
        # To get medium band (0.65-0.85), we need LLM to return Intent: DIAGRAM WITHOUT Reasoning → 0.70
        pipeline2 = _make_pipeline(
            RoutingMockLLM({
                "query intent classifier": "Intent: DIAGRAM\nSub-type: flowchart",
                "generate a mermaid diagram": mermaid_response,
                "default": explanation_response,
            }),
            chunks,
            intel=IntelligenceConfig(intent_confidence_high=0.85, intent_confidence_medium=0.65),
        )

        result = await pipeline2.query("What is the architecture?")

        # LLM returned "Intent: DIAGRAM" without "Reasoning:" → confidence = 0.70
        # 0.65 ≤ 0.70 < 0.85 → include_explanation = True
        assert result.intent == "DIAGRAM"
        assert result.intent_confidence == pytest.approx(0.70)
        # Specialized diagram + separator + explanation
        assert "---" in result.answer
        assert "graph TD" in result.answer
        assert "auth flow" in result.answer


# =====================================================================
# Phase 5 + 6 — Output Format Detection + Mermaid Validation
# =====================================================================


class TestOutputDetectionThroughPipeline:
    """Verify that detected_formats accurately reflects response content."""

    @pytest.mark.asyncio
    async def test_mixed_content_detection(self):
        """Response with mermaid + code + table + numbered list."""
        mixed = (
            "# Architecture\n\n"
            "```mermaid\ngraph TD\n  A-->B\n```\n\n"
            "## Setup\n"
            "1. Install deps\n"
            "2. Configure DB\n"
            "3. Start server\n\n"
            "```python\nprint('hello')\n```\n\n"
            "| Feature | Value |\n"
            "|---------|-------|\n"
            "| Port | 8080 |\n"
        )
        llm = RoutingMockLLM({"default": mixed})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)

        result = await pipeline.query("Tell me everything")

        assert result.detected_formats is not None
        assert result.detected_formats["has_mermaid"] is True
        assert result.detected_formats["has_code_blocks"] is True
        assert result.detected_formats["has_table"] is True
        assert result.detected_formats["has_numbered_list"] is True
        # Mermaid extracted into diagrams list
        assert result.diagrams is not None
        assert len(result.diagrams) == 1
        assert "A-->B" in result.diagrams[0]

    @pytest.mark.asyncio
    async def test_mermaid_validated_regex_mode(self):
        """When gen_config enables diagrams with regex validation mode,
        the pipeline validates the mermaid block.  A valid 'graph TD' passes."""
        mermaid_text = (
            "```mermaid\n"
            "graph TD\n"
            "    A[Start] --> B[End]\n"
            "```\n"
        )
        llm = RoutingMockLLM({
            "generate a mermaid diagram": mermaid_text,
            "default": "fallback",
        })
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(
            llm, chunks,
            intel=IntelligenceConfig(),
            gen=GenerationConfig(enable_diagrams=True, mermaid_validation="regex"),
        )

        result = await pipeline.query("Draw a diagram of the deployment flow")

        assert result.diagrams is not None
        assert "A[Start] --> B[End]" in result.diagrams[0]


# =====================================================================
# Phase 8 — Source Attribution in the Pipeline
# =====================================================================


class TestAttributionThroughPipeline:
    """Source attribution is wired after confidence scoring."""

    @pytest.mark.asyncio
    async def test_attribution_attached_to_result(self):
        """When answer is non-empty and not abstained, attributions are computed."""
        mock_attrs = [
            Attribution(sentence="OAuth uses tokens.", source_index=0, similarity=0.85),
            Attribution(sentence="Tokens expire.", source_index=1, similarity=0.72),
        ]
        llm = RoutingMockLLM({"default": "OAuth uses tokens. Tokens expire."})
        chunks = [
            _chunk(text="OAuth access tokens.", score=0.85, chunk_id="c1"),
            _chunk(text="Token expiry policy.", score=0.72, chunk_id="c2"),
        ]
        pipeline = _make_pipeline(llm, chunks)

        with patch("doc_qa.verification.source_attributor.attribute_sources", return_value=mock_attrs):
            result = await pipeline.query("How does auth work?")

        assert result.attributions is not None
        assert len(result.attributions) == 2
        assert result.attributions[0]["sentence"] == "OAuth uses tokens."
        assert result.attributions[0]["source_index"] == 0
        assert result.attributions[0]["similarity"] == 0.85
        assert result.attributions[1]["source_index"] == 1

    @pytest.mark.asyncio
    async def test_no_attribution_when_abstained(self):
        """When the pipeline abstains, attributions should be None."""
        llm = RoutingMockLLM({
            "fact-checker": "Verdict: FAIL\nConfidence: 0.2\nIssues: wrong\nSuggested fix: redo",
            "default": "Possibly wrong.",
        })
        chunks = [_chunk(score=0.10, chunk_id="c1")]
        verify_cfg = VerificationConfig(
            enable_verification=True, enable_crag=False,
            confidence_threshold=0.4, abstain_on_low_confidence=True,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("Unknown question")

        assert result.is_abstained is True
        assert result.attributions is None


# =====================================================================
# Multi-intent Decomposition and Merging
# =====================================================================


class TestMultiIntentIntegration:
    """Test query decomposition → parallel processing → result merging."""

    @pytest.mark.asyncio
    async def test_multi_intent_decomposition_and_merge(self):
        """Query 'Explain OAuth and also draw a diagram of the auth flow':
        - _COORDINATION_MARKERS matches 'and also'
        - _OUTPUT_VERBS: 'explain' + 'draw' → 2 distinct verbs → multi_intent=True
        - LLM decomposes into 2 sub-queries
        - Each sub-query is classified and processed separately
        - Results merged: answer = '### Part 1\\n...' + '### Part 2\\n...'
        - confidence_score = min of sub-results
        """
        llm = RoutingMockLLM({
            # Decomposition response
            "may require multiple different": (
                "SUB-QUERY 1: Explain how OAuth authentication works\n"
                "SUB-QUERY 2: Draw a diagram of the OAuth authentication flow\n"
            ),
            # Intent classification for sub-query 1 (no heuristic match → LLM)
            "query intent classifier": "Intent: EXPLANATION\nSub-type: none",
            # Sub-query 2 matches diagram heuristic (has "diagram" + "flow")
            # so LLM intent won't be called for it

            # Generation responses
            "generate a mermaid diagram": "```mermaid\ngraph TD\n  A-->B\n```\n",
            "default": "OAuth is an authorization framework using access tokens.",
        })
        chunks = [
            _chunk(text="OAuth 2.0 access tokens.", score=0.80, chunk_id="c1"),
            _chunk(text="Auth server validates.", score=0.70, chunk_id="c2",
                   file_path="/docs/auth-server.md", section_title="Validation"),
        ]
        pipeline = _make_pipeline(
            llm, chunks,
            intel=IntelligenceConfig(enable_intent_classification=True, enable_multi_intent=True),
        )

        result = await pipeline.query("Explain OAuth and also draw a diagram of the auth flow")

        assert result.sub_results is not None
        assert len(result.sub_results) == 2
        assert "### Part 1" in result.answer
        assert "### Part 2" in result.answer
        # Chunks from both sub-queries, deduplicated
        assert result.chunks_retrieved == 4  # 2 per sub-query
        # confidence_score is min of sub-results
        assert result.confidence_score == min(r.confidence_score for r in result.sub_results)


# =====================================================================
# Multi-turn History Management
# =====================================================================


class TestHistoryManagement:
    """Verify conversation history accumulation and truncation."""

    @pytest.mark.asyncio
    async def test_history_accumulates_on_success(self):
        llm = RoutingMockLLM({"default": "Answer one."})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, max_history_turns=2)

        await pipeline.query("First question")

        assert len(pipeline._history) == 2
        assert pipeline._history[0] == {"role": "user", "text": "First question"}
        assert pipeline._history[1] == {"role": "assistant", "text": "Answer one."}

    @pytest.mark.asyncio
    async def test_history_not_updated_on_abstention(self):
        llm = RoutingMockLLM({
            "fact-checker": "Verdict: FAIL\nConfidence: 0.1\nIssues: bad\nSuggested fix: redo",
            "default": "Bad answer.",
        })
        chunks = [_chunk(score=0.10, chunk_id="c1")]
        verify_cfg = VerificationConfig(
            enable_verification=True, enable_crag=False,
            confidence_threshold=0.8, abstain_on_low_confidence=True,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        await pipeline.query("Unknown?")

        assert len(pipeline._history) == 0  # abstained → no history added

    @pytest.mark.asyncio
    async def test_history_truncated_at_max(self):
        """max_history_turns=1 → max 2 entries (1 user + 1 assistant).
        After 2 queries, oldest pair is evicted.
        """
        llm = RoutingMockLLM({"default": "Answer."})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, max_history_turns=1)

        await pipeline.query("Question 1")
        await pipeline.query("Question 2")

        # max_history = 1*2 = 2 entries, so first pair evicted
        assert len(pipeline._history) == 2
        assert pipeline._history[0]["text"] == "Question 2"
        assert pipeline._history[1]["text"] == "Answer."


# =====================================================================
# File Diversity Cap
# =====================================================================


class TestFileDiversityCap:
    """Verify that max_chunks_per_file limits chunks from a single file."""

    @pytest.mark.asyncio
    async def test_diversity_cap_limits_same_file_chunks(self):
        """3 chunks from same file + 1 from another, max_per_file=1, top_k=3.
        Only 1 chunk per file → 2 chunks after diversity, then top_k=3 → 2 used.
        """
        llm = RoutingMockLLM({"default": "Answer."})
        chunks = [
            _chunk(score=0.90, chunk_id="c1", file_path="/docs/auth.md"),
            _chunk(score=0.85, chunk_id="c2", file_path="/docs/auth.md"),
            _chunk(score=0.80, chunk_id="c3", file_path="/docs/auth.md"),
            _chunk(score=0.75, chunk_id="c4", file_path="/docs/deploy.md"),
        ]
        pipeline = _make_pipeline(llm, chunks, max_per_file=1, top_k=3)

        result = await pipeline.query("Test diversity")

        # chunks_after_rerank is len(top_chunks) which is min(top_k, diverse)
        # diverse: c1 (auth, count=1), c2 (auth, count=2→skip), c3 (auth→skip), c4 (deploy, count=1) = [c1, c4]
        # top_k=3, so top_chunks = [c1, c4] (only 2 available)
        assert result.chunks_after_rerank == 2
        source_ids = {s.chunk_id for s in result.sources}
        assert source_ids == {"c1", "c4"}


# =====================================================================
# Empty Retrieval
# =====================================================================


class TestEmptyRetrieval:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_fallback_message(self):
        llm = RoutingMockLLM({"default": "Should not be called."})
        pipeline = _make_pipeline(llm, chunks=[])

        result = await pipeline.query("Anything?")

        assert result.answer == "I couldn't find any relevant information in the indexed documents."
        assert result.model == "none"
        assert result.chunks_retrieved == 0
        assert result.sources == []


# =====================================================================
# Error Recovery / Graceful Degradation
# =====================================================================


class TestGracefulDegradation:
    """Test that errors in optional phases degrade gracefully."""

    @pytest.mark.asyncio
    async def test_intent_classification_error_falls_back_to_standard(self):
        """If intent classification raises, pipeline falls back to standard LLM."""
        llm = RoutingMockLLM({
            "query intent classifier": Answer(text="", sources=[], model="mock", error="timeout"),
            "default": "Standard answer without intent.",
        })
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks, intel=IntelligenceConfig())

        result = await pipeline.query("What is the rate limit?")

        # Intent classification returned error → _parse_llm_intent gets ""
        # → defaults to EXPLANATION, confidence 0.50, matched_pattern="llm_error_fallback"
        assert result.intent == "EXPLANATION"
        assert result.intent_confidence == pytest.approx(0.50)
        # 0.50 < 0.65 (medium) → explanation only strategy
        assert "Standard answer without intent" in result.answer

    @pytest.mark.asyncio
    async def test_verification_error_assumes_pass(self):
        """When verification LLM returns an error, conservative pass is assumed."""
        llm = RoutingMockLLM({
            "fact-checker": Answer(text="", sources=[], model="mock", error="connection lost"),
            "default": "Good answer.",
        })
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        verify_cfg = VerificationConfig(
            enable_verification=True, enable_crag=False,
            confidence_threshold=0.4,
        )
        pipeline = _make_pipeline(llm, chunks, verify=verify_cfg)

        result = await pipeline.query("Test?")

        # Verification error → passed=True, confidence=0.5
        # verification_signal = 0.5, retrieval_signal = 0.80 (single score avg)
        # combined = 0.4 * 0.80 + 0.6 * 0.5 = 0.32 + 0.30 = 0.62
        assert result.is_abstained is False
        assert result.verification_passed is True
        assert result.confidence_score == pytest.approx(0.62, abs=0.01)


# =====================================================================
# Context Building Verification
# =====================================================================


class TestContextBuilding:
    """Verify the exact format of context passed to the LLM."""

    @pytest.mark.asyncio
    async def test_context_format_matches_spec(self):
        """Context is built as:
        [Source N: filename > section] (score: X.XXX)
        <text>
        <blank line>
        """
        llm = RoutingMockLLM({"default": "Answer."})
        chunks = [
            _chunk(text="OAuth tokens.", score=0.856, chunk_id="c1",
                   file_path="/docs/auth.md", section_title="OAuth"),
            _chunk(text="Redis sessions.", score=0.723, chunk_id="c2",
                   file_path="/docs/sessions.md", section_title="Sessions"),
        ]
        pipeline = _make_pipeline(llm, chunks)

        await pipeline.query("Test context?")

        # Verify the context string passed to the LLM
        gen_call = [c for c in llm.calls if c["context"]][-1]
        ctx = gen_call["context"]
        assert "[Source 1: auth.md > OAuth] (score: 0.856)" in ctx
        assert "OAuth tokens." in ctx
        assert "[Source 2: sessions.md > Sessions] (score: 0.723)" in ctx
        assert "Redis sessions." in ctx


# =====================================================================
# SSE Streaming Integration
# =====================================================================


class TestSSEFullPipelineIntegration:
    """Test SSE event sequence with all phases enabled."""

    @pytest.mark.asyncio
    async def test_full_event_sequence_with_all_phases(self):
        """With intent + CRAG + verification enabled, the event sequence is:
        status(classifying) → intent → status(retrieving) → sources →
        status(grading) → status(generating) → answer →
        attribution → status(verifying) → verified → done
        """
        mock_attr = Attribution(sentence="Test.", source_index=0, similarity=0.9)

        llm = RoutingMockLLM({
            "evaluating the relevance": "Chunk 1: RELEVANT — good\n",
            "fact-checker": "Verdict: PASS\nConfidence: 0.95\nIssues: none\nSuggested fix: none",
            "default": "The answer is here.",
        })
        chunks = [_chunk(text="Source text.", score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig(enable_intent=True, enable_crag=True, enable_verification=True)

        with patch("doc_qa.verification.source_attributor.attribute_sources", return_value=[mock_attr]):
            events = await _collect_sse(
                streaming_query("Draw a diagram of the auth flow", pipeline, MockRequest(), [], config, "s1")
            )

        types = [e["event"] for e in events]

        # Verify phase ordering
        assert types[0] == "status"
        assert events[0]["data"]["status"] == "classifying"
        assert "intent" in types
        assert "sources" in types
        assert "answer" in types
        assert "attribution" in types
        assert "verified" in types
        assert types[-1] == "done"
        assert events[-1]["data"]["status"] == "complete"

        # Verify intent event data (heuristic diagram match)
        intent_ev = next(e for e in events if e["event"] == "intent")
        assert intent_ev["data"]["intent"] == "DIAGRAM"
        assert intent_ev["data"]["confidence"] == pytest.approx(0.92, abs=0.01)

        # Verify sources event data
        src_ev = next(e for e in events if e["event"] == "sources")
        assert src_ev["data"]["chunks_retrieved"] == 1
        assert len(src_ev["data"]["sources"]) == 1
        assert src_ev["data"]["sources"][0]["score"] == 0.8

        # Verify answer event
        answer_ev = next(e for e in events if e["event"] == "answer")
        assert answer_ev["data"]["session_id"] == "s1"

        # Verify verification event
        verified_ev = next(e for e in events if e["event"] == "verified")
        assert verified_ev["data"]["passed"] is True
        assert verified_ev["data"]["confidence"] == 0.95

        # Verify attribution event
        attr_ev = next(e for e in events if e["event"] == "attribution")
        assert attr_ev["data"]["attributions"][0]["source_index"] == 0

    @pytest.mark.asyncio
    async def test_sse_skips_disabled_phases(self):
        """When intelligence and verification are not configured,
        no classifying, intent, grading, or verifying events."""
        llm = RoutingMockLLM({"default": "Simple answer."})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig(enable_intent=False, enable_crag=False, enable_verification=False)

        events = await _collect_sse(
            streaming_query("Simple?", pipeline, MockRequest(), [], config, "s2")
        )

        types = [e["event"] for e in events]
        assert "intent" not in types
        assert "verified" not in types
        # Should not have grading status
        statuses = [e["data"].get("status") for e in events if e["event"] == "status"]
        assert "classifying" not in statuses
        assert "grading" not in statuses
        assert "verifying" not in statuses
        # Core phases present
        assert "retrieving" in statuses
        assert "generating" in statuses
        # "complete" is in the "done" event, not a "status" event
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["status"] == "complete"

    @pytest.mark.asyncio
    async def test_sse_empty_retrieval_short_circuits(self):
        """No candidates → answer + done only, no generation phase."""
        llm = RoutingMockLLM({"default": "Should not be called."})
        pipeline = _make_pipeline(llm, chunks=[])
        config = MockConfig()

        events = await _collect_sse(
            streaming_query("Empty?", pipeline, MockRequest(), [], config, "s3")
        )

        types = [e["event"] for e in events]
        assert "answer" in types
        assert "done" in types
        answer_ev = next(e for e in events if e["event"] == "answer")
        assert "couldn't find" in answer_ev["data"]["answer"].lower()
        # No generating status since short-circuited
        statuses = [e["data"].get("status") for e in events if e["event"] == "status"]
        assert "generating" not in statuses


class TestSSETokenStreaming:
    """Test token-by-token streaming via ask_streaming in SSE."""

    @pytest.mark.asyncio
    async def test_streaming_emits_answer_token_events(self):
        """When LLM supports ask_streaming, answer_token events are emitted
        with incremental deltas, followed by the final answer event."""
        llm = StreamingMockLLM({"default": "Hello beautiful world"})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig()

        events = await _collect_sse(
            streaming_query("Test?", pipeline, MockRequest(), [], config, "s4")
        )

        token_events = [e for e in events if e["event"] == "answer_token"]
        answer_events = [e for e in events if e["event"] == "answer"]

        # Token events should contain incremental deltas
        assert len(token_events) >= 1
        tokens = [e["data"]["token"] for e in token_events]
        reassembled = "".join(tokens)
        assert reassembled == "Hello beautiful world"

        # Final answer event has the complete text
        assert len(answer_events) == 1
        assert answer_events[0]["data"]["answer"] == "Hello beautiful world"

    @pytest.mark.asyncio
    async def test_non_streaming_fallback_when_no_ask_streaming(self):
        """When LLM does NOT have ask_streaming, falls back to ask()."""
        llm = RoutingMockLLM({"default": "Non-streaming response."})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig()

        events = await _collect_sse(
            streaming_query("Test?", pipeline, MockRequest(), [], config, "s5")
        )

        token_events = [e for e in events if e["event"] == "answer_token"]
        answer_events = [e for e in events if e["event"] == "answer"]

        assert len(token_events) == 0  # no streaming tokens
        assert len(answer_events) == 1
        assert answer_events[0]["data"]["answer"] == "Non-streaming response."


class TestSSEHistoryUpdate:
    """Verify SSE streaming updates session history."""

    @pytest.mark.asyncio
    async def test_history_updated_after_stream(self):
        llm = RoutingMockLLM({"default": "Streamed answer."})
        chunks = [_chunk(score=0.80, chunk_id="c1")]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig()
        history: list[dict] = []

        await _collect_sse(
            streaming_query("Stream Q?", pipeline, MockRequest(), history, config, "s6")
        )

        assert len(history) == 2
        assert history[0] == {"role": "user", "text": "Stream Q?"}
        assert history[1] == {"role": "assistant", "text": "Streamed answer."}


# =====================================================================
# SSE + CRAG Integration
# =====================================================================


class TestSSEWithCRAG:
    """Verify SSE emits grading status when CRAG is active."""

    @pytest.mark.asyncio
    async def test_sse_crag_grading_phase(self):
        llm = RoutingMockLLM({
            "evaluating the relevance": "Chunk 1: RELEVANT — good\nChunk 2: RELEVANT — good\n",
            "default": "CRAG-filtered answer.",
        })
        chunks = [
            _chunk(score=0.80, chunk_id="c1"),
            _chunk(score=0.70, chunk_id="c2"),
        ]
        pipeline = _make_pipeline(llm, chunks)
        config = MockConfig(enable_crag=True)

        events = await _collect_sse(
            streaming_query("CRAG test?", pipeline, MockRequest(), [], config, "s7")
        )

        statuses = [e["data"].get("status") for e in events if e["event"] == "status"]
        assert "grading" in statuses
        # grading comes after retrieving, before generating
        grading_idx = statuses.index("grading")
        retrieving_idx = statuses.index("retrieving")
        generating_idx = statuses.index("generating")
        assert retrieving_idx < grading_idx < generating_idx


# =====================================================================
# Full Pipeline End-to-End: all phases enabled simultaneously
# =====================================================================


class TestFullPipelineEndToEnd:
    """Run the complete pipeline with every phase active and verify all
    result fields are populated with consistent, correct values."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_all_phases(self):
        """All phases enabled: intent → CRAG → specialized gen → verification
        → confidence → attribution → format detection.

        Query: 'Draw a diagram of the deployment flow'
        - Heuristic: DIAGRAM (flow + diagram) → confidence 0.92, sub_type flowchart
        - CRAG: grades relevant → no rewrite
        - DiagramGenerator → mermaid output
        - Verification: PASS, confidence 0.90
        - Confidence scoring:
            retrieval [0.85, 0.72]: avg=0.785, gap=0.13<0.3, no penalties
            retrieval_signal = 0.785
            verification_signal = 0.90 (passed)
            combined = 0.4*0.785 + 0.6*0.90 = 0.314 + 0.54 = 0.854
        - Attribution: mocked
        - Format detection: has_mermaid=True
        """
        mermaid_answer = (
            "Here is the deployment flow:\n"
            "```mermaid\n"
            "graph TD\n"
            "    A[Build] --> B[Test]\n"
            "    B --> C[Deploy]\n"
            "```\n"
            "This shows the CI/CD pipeline."
        )
        mock_attr = Attribution(sentence="Build and deploy.", source_index=0, similarity=0.88)

        llm = RoutingMockLLM({
            "evaluating the relevance": (
                "Chunk 1: RELEVANT — CI/CD context\n"
                "Chunk 2: RELEVANT — deployment docs\n"
            ),
            "fact-checker": (
                "Verdict: PASS\n"
                "Confidence: 0.90\n"
                "Issues: none\n"
                "Suggested fix: none"
            ),
            "generate a mermaid diagram": mermaid_answer,
            "default": "Should not reach default.",
        })
        chunks = [
            _chunk(text="CI/CD pipeline builds and tests code.", score=0.85, chunk_id="c1",
                   file_path="/docs/deploy.md", section_title="CI/CD"),
            _chunk(text="Deployment uses Docker containers.", score=0.72, chunk_id="c2",
                   file_path="/docs/deploy.md", section_title="Docker"),
        ]

        pipeline = _make_pipeline(
            llm, chunks,
            intel=IntelligenceConfig(),
            gen=GenerationConfig(enable_diagrams=True, mermaid_validation="regex"),
            verify=VerificationConfig(
                enable_verification=True, enable_crag=True,
                confidence_threshold=0.4, abstain_on_low_confidence=True,
            ),
        )

        with patch("doc_qa.verification.source_attributor.attribute_sources", return_value=[mock_attr]):
            result = await pipeline.query("Draw a diagram of the deployment flow")

        # Intent classification
        assert result.intent == "DIAGRAM"
        assert result.intent_confidence == pytest.approx(0.92)

        # Generation
        assert "graph TD" in result.answer
        assert result.diagrams is not None
        assert "A[Build] --> B[Test]" in result.diagrams[0]

        # Format detection
        assert result.detected_formats["has_mermaid"] is True

        # Verification
        assert result.verification_passed is True

        # Confidence scoring (computed above: 0.854)
        assert result.confidence_score == pytest.approx(0.854, abs=0.01)
        assert result.is_abstained is False

        # Attribution
        assert result.attributions is not None
        assert result.attributions[0]["sentence"] == "Build and deploy."
        assert result.attributions[0]["similarity"] == 0.88

        # Sources
        assert len(result.sources) == 2
        assert result.sources[0].file_path == "/docs/deploy.md"
        assert result.sources[0].score == 0.85

        # History updated
        assert len(pipeline._history) == 2
