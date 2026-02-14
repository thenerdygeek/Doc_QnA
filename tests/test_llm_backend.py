"""Tests for LLM backend implementations."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_qa.llm.backend import (
    Answer,
    CodyBackend,
    LLMBackend,
    OllamaBackend,
    create_backend,
)


# --- Answer dataclass ---


class TestAnswer:
    def test_answer_fields(self) -> None:
        a = Answer(text="hello", sources=["a.md"], model="test")
        assert a.text == "hello"
        assert a.sources == ["a.md"]
        assert a.model == "test"
        assert a.error is None

    def test_answer_with_error(self) -> None:
        a = Answer(text="", sources=[], model="test", error="something broke")
        assert a.error == "something broke"


# --- Factory ---


class TestCreateBackend:
    def test_create_cody_backend(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = create_backend(primary="cody")
        assert isinstance(backend, CodyBackend)

    def test_create_ollama_backend(self) -> None:
        backend = create_backend(primary="ollama")
        assert isinstance(backend, OllamaBackend)

    def test_create_ollama_custom_host(self) -> None:
        backend = create_backend(
            primary="ollama",
            ollama_host="http://myhost:9999",
            ollama_model="llama3:8b",
        )
        assert isinstance(backend, OllamaBackend)
        assert backend._host == "http://myhost:9999"
        assert backend._model == "llama3:8b"

    def test_default_is_cody(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = create_backend()
        assert isinstance(backend, CodyBackend)


# --- OllamaBackend ---


class TestOllamaBackend:
    def test_init_defaults(self) -> None:
        backend = OllamaBackend()
        assert backend._host == "http://localhost:11434"
        assert backend._model == "qwen2.5:7b"
        assert backend._timeout == 120.0

    def test_init_custom(self) -> None:
        backend = OllamaBackend(
            host="http://myhost:9999/",
            model="llama3:8b",
            timeout=60.0,
        )
        assert backend._host == "http://myhost:9999"  # trailing slash stripped
        assert backend._model == "llama3:8b"
        assert backend._timeout == 60.0

    def test_build_messages_basic(self) -> None:
        backend = OllamaBackend()
        msgs = backend._build_messages(
            question="What is OAuth?",
            context="OAuth is an open standard.",
        )
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert "documentation assistant" in msgs[0]["content"].lower()
        assert msgs[1]["role"] == "user"
        assert "Context" in msgs[1]["content"]
        assert "OAuth is an open standard" in msgs[1]["content"]
        assert "What is OAuth?" in msgs[1]["content"]

    def test_build_messages_with_history(self) -> None:
        backend = OllamaBackend()
        history = [
            {"role": "user", "text": "What is auth?"},
            {"role": "assistant", "text": "Auth is authentication."},
        ]
        msgs = backend._build_messages(
            question="Tell me more",
            context="Some context",
            history=history,
        )
        assert len(msgs) == 4  # system + 2 history + current user
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "What is auth?"
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == "Auth is authentication."

    def test_build_messages_no_context(self) -> None:
        backend = OllamaBackend()
        msgs = backend._build_messages(question="Hello?", context="")
        user_msg = msgs[-1]["content"]
        assert "Context" not in user_msg
        assert "Hello?" in user_msg

    @pytest.mark.asyncio
    async def test_ask_success(self) -> None:
        backend = OllamaBackend()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "OAuth allows delegated access."},
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await backend.ask("What is OAuth?", "context here")

        assert isinstance(result, Answer)
        assert result.text == "OAuth allows delegated access."
        assert result.model == "qwen2.5:7b"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_ask_connection_error(self) -> None:
        import httpx

        backend = OllamaBackend()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await backend.ask("question", "context")

        assert result.text == ""
        assert "Cannot connect" in result.error

    @pytest.mark.asyncio
    async def test_ask_http_error(self) -> None:
        import httpx

        backend = OllamaBackend()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_request = httpx.Request("POST", "http://localhost:11434/api/chat")
            mock_response = httpx.Response(404, request=mock_request)
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Not Found", request=mock_request, response=mock_response
            )
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            result = await backend.ask("question", "context")

        assert result.text == ""
        assert "404" in result.error

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        backend = OllamaBackend()
        await backend.close()  # should not raise


# --- CodyBackend (unit tests — no real agent process) ---


class TestCodyBackend:
    def test_build_prompt_basic(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        prompt = backend._build_prompt("What is auth?", "Auth context here")
        assert "documentation assistant" in prompt.lower()
        assert "Auth context here" in prompt
        assert "What is auth?" in prompt

    def test_build_prompt_with_history(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        history = [
            {"role": "user", "text": "What is OAuth?"},
            {"role": "assistant", "text": "OAuth is..."},
        ]
        prompt = backend._build_prompt("Tell me more", "context", history)
        assert "Previous conversation" in prompt
        assert "What is OAuth?" in prompt
        assert "OAuth is..." in prompt

    def test_extract_response_with_assistant_message(self) -> None:
        response = {
            "messages": [
                {"speaker": "human", "text": "question"},
                {"speaker": "assistant", "text": "The answer is 42."},
            ]
        }
        assert CodyBackend._extract_response(response) == "The answer is 42."

    def test_extract_response_empty(self) -> None:
        assert CodyBackend._extract_response(None) == ""
        assert CodyBackend._extract_response({}) == ""
        assert CodyBackend._extract_response({"messages": []}) == ""

    def test_extract_response_no_assistant(self) -> None:
        response = {"messages": [{"speaker": "human", "text": "question"}]}
        assert CodyBackend._extract_response(response) == ""
