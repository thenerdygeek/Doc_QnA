"""Tests for LLM test endpoints and model mapping."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from doc_qa.llm.models import format_cody_model, format_ollama_model


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def app_with_index(tmp_path: Path):
    """Create a FastAPI app backed by a populated test index."""
    from doc_qa.api.server import create_app
    from doc_qa.config import AppConfig
    from doc_qa.indexing.chunker import Chunk
    from doc_qa.indexing.indexer import DocIndex

    index_dir = tmp_path / "data" / "doc_qa_db"
    index = DocIndex(db_path=str(index_dir))

    # Add a single chunk so index is non-empty
    fp = str(tmp_path / "test.md")
    Path(fp).write_text("Test content.", encoding="utf-8")
    chunk = Chunk(
        chunk_id=f"{fp}#0",
        text="Test content.",
        file_path=fp,
        file_type="md",
        section_title="Test",
        section_level=1,
        chunk_index=0,
    )
    index.add_chunks([chunk], file_hash="hash")
    index.rebuild_fts_index()

    config = AppConfig()
    config.indexing.db_path = str(index_dir)
    app = create_app(repo_path=str(tmp_path), config=config)
    return app


@pytest.fixture
def empty_app(tmp_path: Path):
    """Create a FastAPI app with an empty index."""
    from doc_qa.api.server import create_app
    from doc_qa.config import AppConfig

    index_dir = tmp_path / "data" / "doc_qa_db"
    config = AppConfig()
    config.indexing.db_path = str(index_dir)
    app = create_app(repo_path=str(tmp_path), config=config)
    return app


# ── Enhanced Health Endpoint ─────────────────────────────────────


class TestEnhancedHealth:
    def test_health_returns_components(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        # Backward compat
        assert data["status"] == "ok"
        # New component info
        assert "components" in data
        assert data["components"]["index"]["ok"] is True
        assert data["components"]["index"]["chunks"] == 1

    def test_health_empty_index(self, empty_app) -> None:
        client = TestClient(empty_app)
        resp = client.get("/api/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert data["components"]["index"]["ok"] is True
        assert data["components"]["index"]["chunks"] == 0


# ── Cody Test Endpoint ───────────────────────────────────────────


class TestCodyTestEndpoint:
    def test_cody_test_missing_env_var(self, app_with_index) -> None:
        """When the env var doesn't exist, return error."""
        client = TestClient(app_with_index)
        # Use a definitely-unset env var name
        resp = client.post(
            "/api/llm/cody/test",
            json={"endpoint": "https://sourcegraph.com", "access_token_env": "NONEXISTENT_TOKEN_VAR_XYZ"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is False
        assert "NONEXISTENT_TOKEN_VAR_XYZ" in data["error"]

    @patch("doc_qa.llm.backend.CodyBackend.test_connection")
    def test_cody_test_success(self, mock_test, app_with_index, monkeypatch) -> None:
        """Successful Cody connection returns user + models."""
        monkeypatch.setenv("SRC_ACCESS_TOKEN", "sgp_test_token")
        mock_test.return_value = {
            "ok": True,
            "user": {"username": "alice", "email": "alice@example.com", "displayName": "Alice"},
            "models": [
                {"id": "anthropic::2025-01-01::claude-3.5-sonnet", "displayName": "Claude 3.5 Sonnet", "provider": "Anthropic", "thinking": False},
            ],
        }

        client = TestClient(app_with_index)
        resp = client.post(
            "/api/llm/cody/test",
            json={"endpoint": "https://sourcegraph.com", "access_token_env": "SRC_ACCESS_TOKEN"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["user"]["username"] == "alice"
        assert len(data["models"]) == 1

    @patch("doc_qa.llm.backend.CodyBackend.test_connection")
    def test_cody_test_auth_failure(self, mock_test, app_with_index, monkeypatch) -> None:
        """Auth failure returns error."""
        monkeypatch.setenv("SRC_ACCESS_TOKEN", "bad_token")
        mock_test.return_value = {
            "ok": False,
            "error": "Cody authentication failed: invalid token",
        }

        client = TestClient(app_with_index)
        resp = client.post(
            "/api/llm/cody/test",
            json={"endpoint": "https://sourcegraph.com", "access_token_env": "SRC_ACCESS_TOKEN"},
        )
        data = resp.json()
        assert data["ok"] is False
        assert "authentication failed" in data["error"]


# ── Ollama Test Endpoint ─────────────────────────────────────────


class TestOllamaTestEndpoint:
    @patch("httpx.AsyncClient")
    def test_ollama_test_success(self, MockClient, app_with_index) -> None:
        """Successful Ollama connection returns models."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {
                    "name": "qwen2.5:7b",
                    "details": {"family": "qwen2", "parameter_size": "7.6B"},
                },
                {
                    "name": "llama3.2:latest",
                    "details": {"family": "llama", "parameter_size": "3.2B"},
                },
            ]
        }

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        client = TestClient(app_with_index)
        resp = client.post(
            "/api/llm/ollama/test",
            json={"host": "http://localhost:11434"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert len(data["models"]) == 2
        assert data["models"][0]["id"] == "qwen2.5:7b"
        assert data["models"][0]["displayName"]  # non-empty

    @patch("httpx.AsyncClient")
    def test_ollama_test_connection_error(self, MockClient, app_with_index) -> None:
        """Connection error returns error message."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        client = TestClient(app_with_index)
        resp = client.post(
            "/api/llm/ollama/test",
            json={"host": "http://localhost:99999"},
        )
        data = resp.json()
        assert data["ok"] is False
        assert "Cannot connect" in data["error"]


# ── Model Mapping Unit Tests ─────────────────────────────────────


class TestFormatCodyModel:
    def test_standard_model_id(self) -> None:
        result = format_cody_model("anthropic::2025-01-01::claude-3.5-sonnet")
        assert result["id"] == "anthropic::2025-01-01::claude-3.5-sonnet"
        assert result["provider"] == "Anthropic"
        assert "Claude" in result["displayName"]
        assert "3.5" in result["displayName"]
        assert result["thinking"] is False

    def test_openai_model(self) -> None:
        result = format_cody_model("openai::2024-05-13::gpt-4o")
        assert result["provider"] == "OpenAI"
        assert "Gpt" in result["displayName"] or "GPT" in result["displayName"] or "4o" in result["displayName"]

    def test_thinking_model_by_name(self) -> None:
        result = format_cody_model("anthropic::2025-01-01::claude-3.5-sonnet-thinking")
        assert result["thinking"] is True

    def test_thinking_model_by_capabilities(self) -> None:
        result = format_cody_model(
            "anthropic::2025-01-01::claude-3.5-sonnet",
            capabilities={"thinking": True},
        )
        assert result["thinking"] is True

    def test_simple_model_id(self) -> None:
        """Model IDs without :: separators."""
        result = format_cody_model("claude-3-haiku")
        # When there's no :: separator, the whole string is both the slug and provider key
        assert result["displayName"]  # non-empty
        assert result["id"] == "claude-3-haiku"

    def test_google_provider(self) -> None:
        result = format_cody_model("google::2024-01-01::gemini-1.5-pro")
        assert result["provider"] == "Google"

    def test_fireworks_provider(self) -> None:
        result = format_cody_model("fireworks::v1::deepseek-v3")
        assert result["provider"] == "Fireworks"


class TestFormatOllamaModel:
    def test_model_with_tag(self) -> None:
        result = format_ollama_model({
            "name": "qwen2.5:7b",
            "details": {"family": "qwen2", "parameter_size": "7.6B"},
        })
        assert result["id"] == "qwen2.5:7b"
        assert "7b" in result["displayName"]
        assert result["family"] == "qwen2"
        assert result["size"] == "7.6B"

    def test_model_latest_tag(self) -> None:
        result = format_ollama_model({
            "name": "llama3.2:latest",
            "details": {"family": "llama", "parameter_size": "3.2B"},
        })
        assert result["id"] == "llama3.2:latest"
        # "latest" tag should not appear in display name
        assert "latest" not in result["displayName"]

    def test_model_no_details(self) -> None:
        result = format_ollama_model({"name": "mistral:7b"})
        assert result["id"] == "mistral:7b"
        assert result["family"] == ""
        assert result["size"] == ""

    def test_model_no_tag(self) -> None:
        result = format_ollama_model({"name": "phi3", "details": {}})
        assert result["id"] == "phi3"
        assert result["displayName"]  # non-empty
