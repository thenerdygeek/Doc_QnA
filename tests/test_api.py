"""Tests for the FastAPI server."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from doc_qa.indexing.chunker import Chunk
from doc_qa.indexing.indexer import DocIndex


@pytest.fixture
def app_with_index(tmp_path: Path):
    """Create a FastAPI app backed by a populated test index."""
    from doc_qa.api.server import create_app
    from doc_qa.config import AppConfig

    # Create test docs and index
    index_dir = tmp_path / "data" / "doc_qa_db"
    index = DocIndex(db_path=str(index_dir))

    docs = [
        ("auth.md", "Authentication", "OAuth 2.0 tokens authenticate users via identity provider."),
        ("deploy.md", "Deployment", "Docker containers deploy the app on Kubernetes with Helm charts."),
        ("api.md", "REST API", "REST API follows OpenAPI 3.0 spec with Bearer token authentication."),
    ]
    for fname, section, content in docs:
        fp = str(tmp_path / fname)
        Path(fp).write_text(content, encoding="utf-8")
        chunk = Chunk(
            chunk_id=f"{fp}#0",
            text=content,
            file_path=fp,
            file_type="md",
            section_title=section,
            section_level=1,
            chunk_index=0,
        )
        index.add_chunks([chunk], file_hash="test_hash")

    index.rebuild_fts_index()

    # Build config pointing to our test index
    config = AppConfig()
    config.indexing.db_path = str(index_dir)

    app = create_app(repo_path=str(tmp_path), config=config)
    return app


class TestHealthEndpoint:
    def test_health_returns_ok(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "components" in data
        assert data["components"]["index"]["ok"] is True
        assert data["components"]["index"]["chunks"] == 3


class TestStatsEndpoint:
    def test_stats_returns_counts(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 3
        assert data["total_files"] == 3
        assert "embedding_model" in data


class TestRetrieveEndpoint:
    def test_retrieve_returns_chunks(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.post("/api/retrieve", json={"question": "How does authentication work?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "chunks" in data
        assert len(data["chunks"]) > 0
        chunk = data["chunks"][0]
        assert "text" in chunk
        assert "score" in chunk
        assert "file_path" in chunk
        assert "section_title" in chunk

    def test_retrieve_respects_top_k(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.post("/api/retrieve", json={"question": "deployment", "top_k": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["chunks"]) <= 1

    def test_retrieve_returns_relevant_results(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.post("/api/retrieve", json={"question": "Docker Kubernetes deployment"})
        assert resp.status_code == 200
        data = resp.json()
        # Top result should be about deployment
        assert len(data["chunks"]) > 0
        top = data["chunks"][0]
        assert "deploy" in top["section_title"].lower() or "docker" in top["text"].lower()


class TestQueryEndpointValidation:
    def test_query_requires_question(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422  # validation error


# --- Session Store Tests ---


class TestSessionStore:
    def test_get_or_create_new_session(self) -> None:
        from doc_qa.api.server import _SessionStore

        store = _SessionStore()
        sid, session = store.get_or_create(None)
        assert isinstance(sid, str)
        assert len(sid) == 12
        assert session.history == []

    def test_get_or_create_returns_existing(self) -> None:
        from doc_qa.api.server import _SessionStore

        store = _SessionStore()
        sid1, session1 = store.get_or_create(None)
        session1.history.append({"role": "user", "text": "hello"})

        sid2, session2 = store.get_or_create(sid1)
        assert sid2 == sid1
        assert session2 is session1
        assert len(session2.history) == 1

    def test_unknown_session_creates_new(self) -> None:
        from doc_qa.api.server import _SessionStore

        store = _SessionStore()
        sid, session = store.get_or_create("nonexistent123")
        assert sid != "nonexistent123"
        assert session.history == []

    def test_expired_session_creates_new(self) -> None:
        from doc_qa.api.server import _Session, _SessionStore

        store = _SessionStore()
        sid1, session1 = store.get_or_create(None)
        # Manually expire by backdating
        session1.last_active = time.time() - 3600

        sid2, session2 = store.get_or_create(sid1)
        assert sid2 != sid1  # New session created
        assert session2 is not session1

    def test_cleanup_expired(self) -> None:
        from doc_qa.api.server import _SessionStore

        store = _SessionStore()
        sid1, s1 = store.get_or_create(None)
        sid2, s2 = store.get_or_create(None)

        # Expire first session
        s1.last_active = time.time() - 3600

        # Trigger cleanup via get_or_create
        store.get_or_create(None)
        assert sid1 not in store._sessions
        assert sid2 in store._sessions


class TestSession:
    def test_session_touch(self) -> None:
        from doc_qa.api.server import _Session

        session = _Session()
        old_time = session.last_active
        time.sleep(0.01)
        session.touch()
        assert session.last_active > old_time

    def test_session_not_expired(self) -> None:
        from doc_qa.api.server import _Session

        session = _Session()
        assert not session.is_expired(ttl=1800)

    def test_session_expired(self) -> None:
        from doc_qa.api.server import _Session

        session = _Session()
        session.last_active = time.time() - 3600
        assert session.is_expired(ttl=1800)
