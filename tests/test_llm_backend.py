"""Tests for LLM backend implementations."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_qa.llm.backend import (
    Answer,
    CodyBackend,
    LLMBackend,
    OllamaBackend,
    _CONTEXT_TEMP_DIR,
    _JSONRPCHandler,
    _RPC_READ_TIMEOUT,
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

    def test_extract_response_error_type(self) -> None:
        response = {"type": "error", "messages": []}
        assert CodyBackend._extract_response(response) == ""

    def test_extract_response_unexpected_type_with_messages(self) -> None:
        response = {
            "type": "something_else",
            "messages": [{"speaker": "assistant", "text": "ok"}],
        }
        # Should still extract the text (with a warning logged)
        assert CodyBackend._extract_response(response) == "ok"


# --- CodyBackend: context file management ---


class TestWriteContextFiles:
    def test_empty_context(self) -> None:
        items, paths = CodyBackend._write_context_files("")
        assert items == []
        assert paths == []

    def test_single_chunk(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "doc_qa.llm.backend._CONTEXT_TEMP_DIR", str(tmp_path)
        )
        context = "[Source 1: api.md] (score: 0.95)\nAPI docs here."
        items, paths = CodyBackend._write_context_files(context)

        assert len(items) == 1
        assert len(paths) == 1
        assert os.path.isfile(paths[0])

        # Verify contextFile structure (matches CodyPy protocol)
        item = items[0]
        assert item["type"] == "file"
        assert item["uri"]["scheme"] == "file"
        assert item["uri"]["fsPath"] == paths[0]
        assert item["uri"]["path"] == paths[0]

        # Verify file content
        with open(paths[0]) as f:
            content = f.read()
        assert "API docs here" in content

    def test_multiple_chunks(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "doc_qa.llm.backend._CONTEXT_TEMP_DIR", str(tmp_path)
        )
        context = (
            "[Source 1: a.md] (score: 0.9)\nFirst chunk.\n\n"
            "[Source 2: b.md] (score: 0.8)\nSecond chunk."
        )
        items, paths = CodyBackend._write_context_files(context)

        assert len(items) == 2
        assert len(paths) == 2

        # Second chunk should have the [Source prefix restored
        with open(paths[1]) as f:
            content = f.read()
        assert content.startswith("[Source ")

    def test_cleanup_removes_files(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(
            "doc_qa.llm.backend._CONTEXT_TEMP_DIR", str(tmp_path)
        )
        context = "[Source 1: a.md] (score: 0.9)\nChunk."
        _, paths = CodyBackend._write_context_files(context)
        assert os.path.isfile(paths[0])

        CodyBackend._cleanup_context_files(paths)
        assert not os.path.isfile(paths[0])

    def test_cleanup_ignores_missing_files(self) -> None:
        # Should not raise
        CodyBackend._cleanup_context_files(["/nonexistent/file.txt"])


# --- CodyBackend: request building ---


class TestBuildRequestData:
    def test_structure(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        items = [{"type": "file", "uri": {"fsPath": "/tmp/c.txt", "path": "/tmp/c.txt"}}]
        data = backend._build_request_data("chat-123", "my prompt", items)

        assert data["id"] == "chat-123"
        msg = data["message"]
        assert msg["command"] == "submit"
        assert msg["text"] == "my prompt"
        assert msg["submitType"] == "user"
        assert msg["addEnhancedContext"] is False
        assert msg["contextFiles"] == items

    def test_empty_context_files(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        data = backend._build_request_data("chat-123", "prompt", [])
        assert data["message"]["contextFiles"] == []


# --- CodyBackend: prompt building with empty context ---


class TestBuildPromptContextHandling:
    def test_no_context_in_prompt_when_empty(self) -> None:
        """When context is empty (passed via contextFiles instead), prompt should not include Context section."""
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        prompt = backend._build_prompt("What is auth?", context="")
        assert "## Context" not in prompt
        assert "What is auth?" in prompt

    def test_context_included_when_provided(self) -> None:
        """Backward compat: context still works for inline use (e.g. Ollama path)."""
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        prompt = backend._build_prompt("What is auth?", context="Some docs here")
        assert "## Context" in prompt
        assert "Some docs here" in prompt


# --- CodyBackend: _new_chat ---


class TestNewChat:
    @pytest.mark.asyncio
    async def test_new_chat_creates_session(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        mock_rpc = AsyncMock()
        mock_rpc.request = AsyncMock(side_effect=["chat-abc", None])
        backend._rpc = mock_rpc

        chat_id = await backend._new_chat()

        assert chat_id == "chat-abc"
        assert mock_rpc.request.call_count == 2
        # First call: chat/new
        assert mock_rpc.request.call_args_list[0][0] == ("chat/new", None)
        # Second call: chat/setModel
        model_call = mock_rpc.request.call_args_list[1]
        assert model_call[0][0] == "chat/setModel"
        assert model_call[0][1]["id"] == "chat-abc"


# --- CodyBackend: _notify_context_files ---


class TestNotifyContextFiles:
    @pytest.mark.asyncio
    async def test_sends_didopen_for_each_file(self, tmp_path) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()

        # Create temp files
        f1 = tmp_path / "chunk_0.txt"
        f1.write_text("content of chunk 0")
        f2 = tmp_path / "chunk_1.txt"
        f2.write_text("content of chunk 1")

        items = [
            {"type": "file", "uri": {"fsPath": str(f1), "path": str(f1)}},
            {"type": "file", "uri": {"fsPath": str(f2), "path": str(f2)}},
        ]

        mock_rpc = AsyncMock()
        mock_rpc.notify = AsyncMock()
        backend._rpc = mock_rpc

        await backend._notify_context_files(items)

        assert mock_rpc.notify.call_count == 2
        # Verify textDocument/didOpen was called
        for call in mock_rpc.notify.call_args_list:
            assert call[0][0] == "textDocument/didOpen"
            params = call[0][1]
            assert "uri" in params
            assert "content" in params

    @pytest.mark.asyncio
    async def test_skips_missing_fspath(self) -> None:
        with patch("doc_qa.llm.backend._find_cody_binary", return_value="/usr/bin/cody"):
            backend = CodyBackend()
        mock_rpc = AsyncMock()
        mock_rpc.notify = AsyncMock()
        backend._rpc = mock_rpc

        items = [{"type": "file", "uri": {}}]
        await backend._notify_context_files(items)
        mock_rpc.notify.assert_not_called()


# --- _JSONRPCHandler: token usage in streaming ---


class TestStreamingTokenUsage:
    @pytest.mark.asyncio
    async def test_on_token_usage_called(self) -> None:
        """Verify on_token_usage callback is invoked with token usage data."""
        reader = AsyncMock(spec=["readuntil", "readexactly"])
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        handler = _JSONRPCHandler(reader, writer)

        # Build messages: streaming notification with tokenUsage, then final response
        streaming_msg = {
            "params": {
                "message": {
                    "isMessageInProgress": True,
                    "tokenUsage": {
                        "completionTokens": 50,
                        "promptTokens": 200,
                        "percentUsed": 0.25,
                    },
                    "messages": [
                        {"speaker": "assistant", "text": "partial answer"},
                    ],
                },
            },
        }
        final_msg = {
            "id": 1,
            "result": {"messages": [{"speaker": "assistant", "text": "done"}]},
        }

        messages = [streaming_msg, final_msg]
        call_idx = {"i": 0}

        async def mock_read_until(_sep):
            idx = call_idx["i"]
            body = json.dumps(messages[idx]).encode()
            return f"Content-Length: {len(body)}\r\n\r\n".encode()

        async def mock_read_exactly(n):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return json.dumps(messages[idx]).encode()

        reader.readuntil = mock_read_until
        reader.readexactly = mock_read_exactly

        usage_received = []
        tokens_received = []

        result = await handler.request_streaming(
            "chat/submitMessage",
            {"id": "test"},
            on_token=lambda t: tokens_received.append(t),
            on_token_usage=lambda u: usage_received.append(u),
        )

        assert len(usage_received) == 1
        assert usage_received[0]["completionTokens"] == 50
        assert usage_received[0]["promptTokens"] == 200
        assert len(tokens_received) == 1
        assert tokens_received[0] == "partial answer"

    @pytest.mark.asyncio
    async def test_on_token_usage_not_called_when_missing(self) -> None:
        """No callback invocation when tokenUsage is absent from notification."""
        reader = AsyncMock(spec=["readuntil", "readexactly"])
        writer = MagicMock()
        writer.write = MagicMock()
        writer.drain = AsyncMock()

        handler = _JSONRPCHandler(reader, writer)

        streaming_msg = {
            "params": {
                "message": {
                    "isMessageInProgress": True,
                    "messages": [
                        {"speaker": "assistant", "text": "partial"},
                    ],
                },
            },
        }
        final_msg = {"id": 1, "result": {"messages": []}}

        messages = [streaming_msg, final_msg]
        call_idx = {"i": 0}

        async def mock_read_until(_sep):
            idx = call_idx["i"]
            body = json.dumps(messages[idx]).encode()
            return f"Content-Length: {len(body)}\r\n\r\n".encode()

        async def mock_read_exactly(n):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return json.dumps(messages[idx]).encode()

        reader.readuntil = mock_read_until
        reader.readexactly = mock_read_exactly

        usage_received = []
        await handler.request_streaming(
            "chat/submitMessage",
            {"id": "test"},
            on_token_usage=lambda u: usage_received.append(u),
        )

        assert len(usage_received) == 0


# --- _JSONRPCHandler: read timeout ---


class TestRPCReadTimeout:
    def test_timeout_constant(self) -> None:
        assert _RPC_READ_TIMEOUT == 120
