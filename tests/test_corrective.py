"""Tests for CRAG corrective retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from doc_qa.llm.backend import Answer, LLMBackend
from doc_qa.retrieval.corrective import (
    corrective_retrieve,
    merge_candidates,
    rewrite_query,
    should_rewrite,
)
from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.verification.grader import GradedChunk


# ── Helpers ──────────────────────────────────────────────────────────


class MockLLM(LLMBackend):
    def __init__(self, text: str = "", error: str | None = None) -> None:
        self.text = text
        self.error = error

    async def ask(self, question: str, context: str, history=None) -> Answer:
        return Answer(text=self.text, sources=[], model="mock", error=self.error)

    async def close(self) -> None:
        pass


def _chunk(text="chunk", chunk_id="c1", score=0.8):
    return RetrievedChunk(
        text=text, score=score, chunk_id=chunk_id,
        file_path="test.md", file_type="md",
        section_title="Section", section_level=1, chunk_index=0,
    )


def _graded(chunk, grade="relevant"):
    return GradedChunk(chunk=chunk, grade=grade, reasoning="test")


# ── should_rewrite ───────────────────────────────────────────────────


class TestShouldRewrite:
    def test_all_irrelevant(self):
        graded = [_graded(_chunk(chunk_id=f"c{i}"), "irrelevant") for i in range(5)]
        assert should_rewrite(graded, threshold=0.5) is True

    def test_all_relevant(self):
        graded = [_graded(_chunk(chunk_id=f"c{i}"), "relevant") for i in range(5)]
        assert should_rewrite(graded, threshold=0.5) is False

    def test_exactly_at_threshold(self):
        # 2 irrelevant out of 4 = 0.5 = threshold -> not above threshold
        graded = [
            _graded(_chunk(chunk_id="c1"), "irrelevant"),
            _graded(_chunk(chunk_id="c2"), "irrelevant"),
            _graded(_chunk(chunk_id="c3"), "relevant"),
            _graded(_chunk(chunk_id="c4"), "relevant"),
        ]
        assert should_rewrite(graded, threshold=0.5) is False

    def test_above_threshold(self):
        # 3 irrelevant out of 4 = 0.75 > 0.5
        graded = [
            _graded(_chunk(chunk_id="c1"), "irrelevant"),
            _graded(_chunk(chunk_id="c2"), "irrelevant"),
            _graded(_chunk(chunk_id="c3"), "irrelevant"),
            _graded(_chunk(chunk_id="c4"), "relevant"),
        ]
        assert should_rewrite(graded, threshold=0.5) is True

    def test_empty_list(self):
        assert should_rewrite([], threshold=0.5) is False


# ── rewrite_query ────────────────────────────────────────────────────


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_success(self):
        llm = MockLLM(text="improved query about authentication")
        graded = [_graded(_chunk(text="partial match"), "partial")]
        result = await rewrite_query("original query", graded, llm)
        assert result == "improved query about authentication"

    @pytest.mark.asyncio
    async def test_error_returns_original(self):
        llm = MockLLM(text="", error="timeout")
        result = await rewrite_query("original query", [], llm)
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_empty_response_returns_original(self):
        llm = MockLLM(text="   ")
        result = await rewrite_query("original query", [], llm)
        assert result == "original query"

    @pytest.mark.asyncio
    async def test_exception_returns_original(self):
        class FailLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                raise RuntimeError("boom")
            async def close(self):
                pass

        result = await rewrite_query("original query", [], FailLLM())
        assert result == "original query"


# ── merge_candidates ─────────────────────────────────────────────────


class TestMergeCandidates:
    def test_deduplication(self):
        c1 = _chunk(chunk_id="c1")
        c2 = _chunk(chunk_id="c2")
        c3 = _chunk(chunk_id="c3")
        c1_dup = _chunk(chunk_id="c1")

        graded = [_graded(c1, "relevant"), _graded(c2, "irrelevant")]
        new = [c1_dup, c3]  # c1_dup should be deduped

        merged = merge_candidates(graded, new)
        ids = [c.chunk_id for c in merged]
        assert ids == ["c1", "c3"]

    def test_relevant_first(self):
        c1 = _chunk(chunk_id="c1")
        c2 = _chunk(chunk_id="c2")
        c3 = _chunk(chunk_id="c3")

        graded = [_graded(c1, "relevant"), _graded(c2, "irrelevant")]
        merged = merge_candidates(graded, [c3])
        assert merged[0].chunk_id == "c1"
        assert merged[1].chunk_id == "c3"

    def test_empty_inputs(self):
        assert merge_candidates([], []) == []


# ── corrective_retrieve (full loop) ─────────────────────────────────


class TestCorrectiveRetrieve:
    @pytest.mark.asyncio
    async def test_no_rewrite_needed(self):
        """All chunks grade as relevant — no rewrite should happen."""
        llm = MockLLM(text="Chunk 1: RELEVANT — good\nChunk 2: RELEVANT — good")
        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]
        retriever = MagicMock()

        result, was_rewritten = await corrective_retrieve(
            query="test", initial_chunks=chunks, llm_backend=llm,
            retriever=retriever, max_rewrites=2,
        )
        assert not was_rewritten
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rewrite_triggered(self):
        """When chunks are graded irrelevant, rewrite and re-retrieve."""
        call_count = 0

        class MultiResponseLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First grading: all irrelevant
                    return Answer(
                        text="Chunk 1: IRRELEVANT — bad\nChunk 2: IRRELEVANT — bad",
                        sources=[], model="mock",
                    )
                if call_count == 2:
                    # Rewrite
                    return Answer(text="better query", sources=[], model="mock")
                # Second grading: all relevant
                return Answer(
                    text="Chunk 1: RELEVANT — good\nChunk 2: RELEVANT — good",
                    sources=[], model="mock",
                )
            async def close(self):
                pass

        chunks = [_chunk(chunk_id="c1"), _chunk(chunk_id="c2")]

        retriever = MagicMock()
        retriever.search.return_value = [
            _chunk(chunk_id="c3"), _chunk(chunk_id="c4"),
        ]

        result, was_rewritten = await corrective_retrieve(
            query="test", initial_chunks=chunks, llm_backend=MultiResponseLLM(),
            retriever=retriever, max_rewrites=2,
        )
        assert was_rewritten
        assert retriever.search.called

    @pytest.mark.asyncio
    async def test_max_rewrites_respected(self):
        """Should not exceed max_rewrites."""
        class AlwaysIrrelevantLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                if "Grade" in question or "Chunk" in question or "evaluating" in question:
                    return Answer(
                        text="Chunk 1: IRRELEVANT — bad",
                        sources=[], model="mock",
                    )
                return Answer(text="rewritten query", sources=[], model="mock")
            async def close(self):
                pass

        chunks = [_chunk(chunk_id="c1")]
        retriever = MagicMock()
        retriever.search.return_value = [_chunk(chunk_id="c2")]

        result, was_rewritten = await corrective_retrieve(
            query="test", initial_chunks=chunks, llm_backend=AlwaysIrrelevantLLM(),
            retriever=retriever, max_rewrites=1,
        )
        assert was_rewritten
        # search called once for the single rewrite
        assert retriever.search.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_initial_chunks(self):
        llm = MockLLM(text="should not be called")
        retriever = MagicMock()
        result, was_rewritten = await corrective_retrieve(
            query="test", initial_chunks=[], llm_backend=llm,
            retriever=retriever,
        )
        assert result == []
        assert not was_rewritten
