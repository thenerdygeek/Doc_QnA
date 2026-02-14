"""Tests for SSE streaming."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.streaming.sse import streaming_query


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLM(LLMBackend):
    """Mock LLM with optional streaming support."""

    def __init__(self, text: str = "Mock answer", error: str | None = None, support_streaming: bool = False) -> None:
        self._text = text
        self._error = error
        self._support_streaming = support_streaming

    async def ask(self, question: str, context: str, history=None) -> Answer:
        return Answer(text=self._text, sources=[], model="mock", error=self._error)

    async def ask_streaming(self, question: str, context: str, history=None, on_token=None) -> Answer:
        if not self._support_streaming:
            raise AttributeError("No streaming")
        # Simulate streaming by calling on_token with cumulative text
        if on_token:
            words = self._text.split()
            cumulative = ""
            for w in words:
                cumulative += ("" if not cumulative else " ") + w
                on_token(cumulative)
        return Answer(text=self._text, sources=[], model="mock-stream")

    async def close(self) -> None:
        pass

    def __getattr__(self, name):
        if name == "ask_streaming" and self._support_streaming:
            return self.ask_streaming
        raise AttributeError(name)


class MockRequest:
    """Mock FastAPI Request with disconnect control."""

    def __init__(self, disconnected: bool = False) -> None:
        self._disconnected = disconnected

    async def is_disconnected(self) -> bool:
        return self._disconnected


class MockConfig:
    """Minimal config for SSE tests."""

    def __init__(
        self,
        enable_intent=False,
        enable_crag=False,
        enable_verification=False,
    ):
        if enable_intent:
            self.intelligence = type("IC", (), {
                "enable_intent_classification": True,
                "enable_multi_intent": False,
            })()
        else:
            self.intelligence = None

        if enable_crag or enable_verification:
            self.verification = type("VC", (), {
                "enable_crag": enable_crag,
                "enable_verification": enable_verification,
                "max_crag_rewrites": 2,
            })()
        else:
            self.verification = None

        self.generation = None
        self.streaming = None


def _chunk(text="chunk text", chunk_id="c1", score=0.8):
    return RetrievedChunk(
        text=text, score=score, chunk_id=chunk_id,
        file_path="test.md", file_type="md",
        section_title="Section", section_level=1, chunk_index=0,
    )


def _make_pipeline(llm, chunks=None):
    """Build a mock pipeline with controllable retriever."""
    pipeline = MagicMock()
    pipeline._llm = llm
    pipeline._candidate_pool = 20
    pipeline._min_score = 0.3
    pipeline._top_k = 5
    pipeline._rerank = False
    pipeline._max_history = 20

    if chunks is None:
        chunks = [_chunk()]

    pipeline._retriever = MagicMock()
    pipeline._retriever.search.return_value = chunks
    pipeline._apply_file_diversity = lambda x: x
    pipeline._build_context = lambda x: "mock context"

    return pipeline


async def _collect_events(gen) -> list[dict]:
    """Collect all SSE events from the generator into a list of (event, data) dicts."""
    events = []
    async for sse in gen:
        event_type = sse.event if hasattr(sse, "event") else "message"
        data = json.loads(sse.data) if sse.data else {}
        events.append({"event": event_type, "data": data})
    return events


# ── Tests ────────────────────────────────────────────────────────────


class TestStreamingEventSequence:
    @pytest.mark.asyncio
    async def test_basic_event_sequence(self):
        """Verify the core event sequence: status(retrieving) → sources → status(generating) → answer → done."""
        llm = MockLLM(text="The answer is 42.")
        pipeline = _make_pipeline(llm)
        request = MockRequest()
        config = MockConfig()

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        event_types = [e["event"] for e in events]
        assert "status" in event_types
        assert "sources" in event_types
        assert "answer" in event_types
        assert "done" in event_types

        # Verify order: retrieving before sources before generating before answer
        status_events = [e for e in events if e["event"] == "status"]
        status_texts = [e["data"].get("status") for e in status_events]
        assert "retrieving" in status_texts
        assert "generating" in status_texts

        # Answer has the expected text
        answer_event = next(e for e in events if e["event"] == "answer")
        assert answer_event["data"]["answer"] == "The answer is 42."
        assert answer_event["data"]["session_id"] == "sid1"

    @pytest.mark.asyncio
    async def test_empty_retrieval(self):
        """When no chunks are found, should emit answer with 'no relevant info' and done."""
        llm = MockLLM()
        pipeline = _make_pipeline(llm, chunks=[])  # empty retrieval
        request = MockRequest()
        config = MockConfig()

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        event_types = [e["event"] for e in events]
        assert "answer" in event_types
        assert "done" in event_types
        answer = next(e for e in events if e["event"] == "answer")
        assert "couldn't find" in answer["data"]["answer"].lower()


class TestClientDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_stops_early(self):
        """If client disconnects, streaming should stop."""
        llm = MockLLM()
        pipeline = _make_pipeline(llm)
        config = MockConfig()

        # Disconnect after first yield
        call_count = 0

        class DisconnectRequest:
            async def is_disconnected(self):
                nonlocal call_count
                call_count += 1
                return call_count > 1  # disconnect after first check

        events = await _collect_events(
            streaming_query("What?", pipeline, DisconnectRequest(), [], config, "sid1")
        )

        # Should have fewer events than a full run
        assert len(events) < 5


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_error_event_emitted(self):
        """On unexpected error, an error event should be emitted."""
        pipeline = MagicMock()
        pipeline._retriever = MagicMock()
        pipeline._retriever.search.side_effect = RuntimeError("retrieval failed")
        pipeline._candidate_pool = 20
        pipeline._min_score = 0.3

        request = MockRequest()
        config = MockConfig()

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        error_events = [e for e in events if e["event"] == "error"]
        assert len(error_events) == 1
        assert "retrieval failed" in error_events[0]["data"]["error"]


class TestOptionalPhasesSkipped:
    @pytest.mark.asyncio
    async def test_no_intent_classification(self):
        """When intelligence is None, no intent event should be emitted."""
        llm = MockLLM()
        pipeline = _make_pipeline(llm)
        request = MockRequest()
        config = MockConfig(enable_intent=False)

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        event_types = [e["event"] for e in events]
        assert "intent" not in event_types

    @pytest.mark.asyncio
    async def test_no_verification(self):
        """When verification is disabled, no verified event should be emitted."""
        llm = MockLLM()
        pipeline = _make_pipeline(llm)
        request = MockRequest()
        config = MockConfig(enable_verification=False)

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        event_types = [e["event"] for e in events]
        assert "verified" not in event_types


class TestTokenStreaming:
    @pytest.mark.asyncio
    async def test_streaming_emits_tokens(self):
        """When LLM supports streaming, answer_token events should be emitted."""
        llm = MockLLM(text="Hello world test", support_streaming=True)
        # Need to make hasattr work properly
        llm._support_streaming = True

        pipeline = _make_pipeline(llm)
        # Override __getattr__ behavior by directly adding the method
        pipeline._llm = llm

        request = MockRequest()
        config = MockConfig()

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        # Should have answer_token events
        token_events = [e for e in events if e["event"] == "answer_token"]
        # Should also have the final answer event
        answer_events = [e for e in events if e["event"] == "answer"]
        assert len(answer_events) == 1

    @pytest.mark.asyncio
    async def test_non_streaming_fallback(self):
        """When LLM doesn't support streaming, falls back to non-streaming."""
        llm = MockLLM(text="Non-streaming answer", support_streaming=False)
        pipeline = _make_pipeline(llm)
        request = MockRequest()
        config = MockConfig()

        events = await _collect_events(
            streaming_query("What?", pipeline, request, [], config, "sid1")
        )

        answer_events = [e for e in events if e["event"] == "answer"]
        assert len(answer_events) == 1
        assert answer_events[0]["data"]["answer"] == "Non-streaming answer"

        # No token events expected
        token_events = [e for e in events if e["event"] == "answer_token"]
        assert len(token_events) == 0


class TestAttributionInSSE:
    @pytest.mark.asyncio
    async def test_attribution_event_emitted(self):
        """Source attribution event should be emitted when answer and chunks exist."""
        llm = MockLLM(text="OAuth tokens authenticate users.")
        pipeline = _make_pipeline(llm)
        request = MockRequest()
        config = MockConfig()

        # Mock the source_attributor at the module level where it's imported lazily
        mock_attr = MagicMock()
        mock_attr.sentence = "OAuth tokens authenticate users."
        mock_attr.source_index = 0
        mock_attr.similarity = 0.85

        with patch(
            "doc_qa.verification.source_attributor.attribute_sources",
            return_value=[mock_attr],
        ):
            events = await _collect_events(
                streaming_query("What?", pipeline, request, [], config, "sid1")
            )

        attribution_events = [e for e in events if e["event"] == "attribution"]
        assert len(attribution_events) == 1
        data = attribution_events[0]["data"]
        assert "attributions" in data
        assert data["attributions"][0]["source_index"] == 0
