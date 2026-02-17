"""Tests for the HyDE query expander and LLM-based query expansion."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from doc_qa.retrieval.query_expander import expand_query, generate_hypothetical_document


class MockAnswer:
    def __init__(self, text: str):
        self.text = text
        self.model = "mock"
        self.error = None
        self.sources = []


class MockLLM:
    """Mock LLM backend for testing HyDE and query expansion."""

    def __init__(self, response_text: str = "This is a hypothetical answer about authentication."):
        self._response_text = response_text

    async def ask(self, question: str, context: str, history=None):
        return MockAnswer(self._response_text)


class TestHyDE:
    @pytest.mark.asyncio
    async def test_generates_embedding(self) -> None:
        """HyDE should return a numpy embedding vector."""
        mock_llm = MockLLM()
        fake_embedding = np.random.rand(768).astype(np.float32)

        with patch("doc_qa.retrieval.query_expander._embed_query", return_value=fake_embedding):
            result = await generate_hypothetical_document(
                "How does authentication work?",
                mock_llm,
                embedding_model="nomic-ai/nomic-embed-text-v1.5",
            )

        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)

    @pytest.mark.asyncio
    async def test_embeds_llm_output_not_query(self) -> None:
        """HyDE should embed the LLM's hypothetical answer, not the original query."""
        mock_llm = MockLLM("OAuth tokens are used for authenticating API requests.")
        embedded_text = None

        def capture_embed(text, model_name="nomic-ai/nomic-embed-text-v1.5"):
            nonlocal embedded_text
            embedded_text = text
            return np.random.rand(768).astype(np.float32)

        with patch("doc_qa.retrieval.query_expander._embed_query", side_effect=capture_embed):
            await generate_hypothetical_document(
                "How does auth work?",
                mock_llm,
            )

        # Should embed the LLM response, not the query
        assert embedded_text == "OAuth tokens are used for authenticating API requests."

    @pytest.mark.asyncio
    async def test_passes_formatted_prompt_to_llm(self) -> None:
        """HyDE should format the HYDE_GENERATION template with the question."""
        last_question = None

        class CaptureLLM:
            async def ask(self, question, context, history=None):
                nonlocal last_question
                last_question = question
                return MockAnswer("Some hypothesis.")

        with patch(
            "doc_qa.retrieval.query_expander._embed_query",
            return_value=np.random.rand(768).astype(np.float32),
        ):
            await generate_hypothetical_document(
                "What is the rate limit?",
                CaptureLLM(),
            )

        assert "What is the rate limit?" in last_question

    @pytest.mark.asyncio
    async def test_uses_specified_embedding_model(self) -> None:
        """HyDE should pass the embedding model name through."""
        used_model = None

        def capture_model(text, model_name="default"):
            nonlocal used_model
            used_model = model_name
            return np.random.rand(768).astype(np.float32)

        mock_llm = MockLLM()
        with patch("doc_qa.retrieval.query_expander._embed_query", side_effect=capture_model):
            await generate_hypothetical_document(
                "test query",
                mock_llm,
                embedding_model="custom-model/v1",
            )

        assert used_model == "custom-model/v1"

    @pytest.mark.asyncio
    async def test_empty_llm_response_raises(self) -> None:
        """Empty LLM response should raise RuntimeError (error hardening)."""
        mock_llm = MockLLM("")

        with pytest.raises(RuntimeError, match="empty response"):
            await generate_hypothetical_document("test", mock_llm)

    @pytest.mark.asyncio
    async def test_error_llm_response_raises(self) -> None:
        """LLM error should raise RuntimeError."""
        answer = MockAnswer("some text")
        answer.error = "timeout"

        class ErrorLLM:
            async def ask(self, question, context, history=None):
                return answer

        with pytest.raises(RuntimeError, match="HyDE LLM error"):
            await generate_hypothetical_document("test", ErrorLLM())


class TestExpandQuery:
    """Tests for the LLM-based query expansion."""

    @pytest.mark.asyncio
    async def test_returns_original_plus_variants(self) -> None:
        """expand_query should return [original, variant1, variant2, ...]."""
        llm_response = (
            "1. How do you authenticate with the API?\n"
            "2. What is the authentication mechanism?\n"
            "3. How to log in to the system?"
        )
        mock_llm = MockLLM(llm_response)

        result = await expand_query("How does auth work?", mock_llm, n_variants=3)

        assert result[0] == "How does auth work?"
        assert len(result) == 4  # original + 3 variants
        assert "How do you authenticate with the API?" in result
        assert "What is the authentication mechanism?" in result
        assert "How to log in to the system?" in result

    @pytest.mark.asyncio
    async def test_parses_numbered_variants(self) -> None:
        """Should correctly strip numbered prefixes like '1. ' or '2) '."""
        llm_response = (
            "1. What is the rate limit for API calls?\n"
            "2) How many requests per second are allowed?\n"
            "3. What are the API throttling limits?"
        )
        mock_llm = MockLLM(llm_response)

        result = await expand_query("rate limit?", mock_llm, n_variants=3)

        # Should have stripped the number prefixes
        assert "What is the rate limit for API calls?" in result
        assert "How many requests per second are allowed?" in result
        assert "What are the API throttling limits?" in result

    @pytest.mark.asyncio
    async def test_caps_at_n_variants(self) -> None:
        """Should not return more than n_variants alternative phrasings."""
        llm_response = (
            "1. Variant A\n"
            "2. Variant B\n"
            "3. Variant C\n"
            "4. Variant D\n"
            "5. Variant E\n"
        )
        mock_llm = MockLLM(llm_response)

        result = await expand_query("original", mock_llm, n_variants=2)

        # original + 2 variants max
        assert len(result) == 3
        assert result[0] == "original"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_error(self) -> None:
        """Should return [original] when the LLM raises an exception."""

        class FailingLLM:
            async def ask(self, question, context, history=None):
                raise RuntimeError("LLM is down")

        result = await expand_query("test query", FailingLLM(), n_variants=3)

        assert result == ["test query"]

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self) -> None:
        """Should return [original] when the LLM returns empty text."""
        mock_llm = MockLLM("")

        result = await expand_query("test query", mock_llm, n_variants=3)

        # No variants parsed from empty text
        assert result == ["test query"]

    @pytest.mark.asyncio
    async def test_skips_blank_lines(self) -> None:
        """Should skip blank lines in the LLM response."""
        llm_response = (
            "1. First variant\n"
            "\n"
            "   \n"
            "2. Second variant\n"
        )
        mock_llm = MockLLM(llm_response)

        result = await expand_query("original Q", mock_llm, n_variants=3)

        assert len(result) == 3  # original + 2 non-blank variants

    @pytest.mark.asyncio
    async def test_deduplicates_original(self) -> None:
        """If LLM returns the original question as a variant, it should be excluded."""
        llm_response = (
            "1. How does auth work?\n"
            "2. What is the authentication process?\n"
        )
        mock_llm = MockLLM(llm_response)

        result = await expand_query("How does auth work?", mock_llm, n_variants=3)

        # "How does auth work?" should only appear once (as the original)
        assert result.count("How does auth work?") == 1
        assert len(result) == 2  # original + 1 non-duplicate variant
