"""Tests for the indexing API endpoints (/api/index/*)."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from doc_qa.indexing.job import SwapResult


# ── Fixtures ──────────────────────────────────────────────────────


def _make_docs(tmp_path: Path) -> Path:
    """Create a small docs directory for indexing tests."""
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "intro.md").write_text("# Intro\nWelcome to the docs.", encoding="utf-8")
    (docs / "guide.md").write_text("# Guide\nStep-by-step tutorial.", encoding="utf-8")
    return docs


@pytest.fixture
def app_with_index(tmp_path: Path):
    """Create a FastAPI app with mocked index (no ONNX model needed)."""
    from doc_qa.config import AppConfig

    mock_index = MagicMock()
    mock_index.count_rows.return_value = 1
    mock_index._table = MagicMock()

    mock_retriever = MagicMock()

    config = AppConfig()
    config.indexing.db_path = str(tmp_path / "data" / "doc_qa_db")

    with (
        patch("doc_qa.api.server.DocIndex", return_value=mock_index),
        patch("doc_qa.api.server.HybridRetriever", return_value=mock_retriever),
    ):
        from doc_qa.api.server import create_app

        app = create_app(repo_path=str(tmp_path), config=config)

    return app


# ── Cancel endpoint ──────────────────────────────────────────────


class TestCancelEndpoint:
    def test_cancel_when_idle_returns_409(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.post("/api/index/cancel")
        assert resp.status_code == 409
        assert "no indexing job" in resp.json()["detail"].lower()


# ── Status endpoint ──────────────────────────────────────────────


class TestStatusEndpoint:
    def test_status_when_idle(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/index/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "idle"

    def test_status_returns_state_field(self, app_with_index) -> None:
        """When idle (no job), only 'state' is returned."""
        client = TestClient(app_with_index)
        resp = client.get("/api/index/status")
        data = resp.json()
        assert "state" in data

    def test_status_has_full_fields_after_job(self, app_with_index, tmp_path: Path) -> None:
        """After a job runs, status includes progress fields."""
        docs_dir = _make_docs(tmp_path)
        client = TestClient(app_with_index)

        mock_swap = SwapResult(index=MagicMock(), retriever=MagicMock())
        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[]),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap),
            patch("doc_qa.indexing.job._cleanup_temp"),
            patch("doc_qa.api.server.save_config"),
        ):
            # Run a quick indexing job (0 files)
            client.get(f"/api/index/stream?action=start&repo_path={docs_dir}")

        resp = client.get("/api/index/status")
        data = resp.json()
        for key in ("state", "total_files", "processed_files", "total_chunks"):
            assert key in data


# ── Stream endpoint — validation ─────────────────────────────────


class TestStreamValidation:
    def test_start_without_repo_path_returns_400(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/index/stream?action=start")
        assert resp.status_code == 400
        assert "repo_path" in resp.json()["detail"].lower()

    def test_start_with_bad_path_returns_400(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/index/stream?action=start&repo_path=/nonexistent/path")
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_reconnect_when_idle_returns_204(self, app_with_index) -> None:
        client = TestClient(app_with_index)
        resp = client.get("/api/index/stream")
        assert resp.status_code == 204


# ── Stream endpoint — successful indexing ────────────────────────


class TestStreamIndexing:
    def test_start_returns_sse_events(self, app_with_index, tmp_path: Path) -> None:
        """Start indexing and verify SSE events are returned."""
        docs_dir = _make_docs(tmp_path)
        client = TestClient(app_with_index)

        mock_swap = SwapResult(index=MagicMock(), retriever=MagicMock())
        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[docs_dir / "intro.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={"chunks": [MagicMock(), MagicMock(), MagicMock()], "sections": 2, "skipped": False, "file_hash": "abc"}),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=3),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap),
            patch("doc_qa.indexing.job._cleanup_temp"),
            patch("doc_qa.api.server.save_config"),
        ):
            resp = client.get(
                f"/api/index/stream?action=start&repo_path={docs_dir}",
                headers={"Accept": "text/event-stream"},
            )
            assert resp.status_code == 200

            body = resp.text
            assert "event: status" in body
            assert "event: done" in body

    def test_start_while_running_returns_409(self, app_with_index, tmp_path: Path) -> None:
        """Starting a second job while one is running should return 409."""
        docs_dir = _make_docs(tmp_path)
        client = TestClient(app_with_index)

        block = threading.Event()
        entered = threading.Event()

        def _blocking_scan(*args, **kwargs):
            entered.set()
            block.wait(timeout=5)
            return []

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=_blocking_scan),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
            patch("doc_qa.api.server.save_config"),
        ):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    lambda: client.get(
                        f"/api/index/stream?action=start&repo_path={docs_dir}",
                    )
                )

                assert entered.wait(timeout=5)

                # Try to start second job — should get 409
                resp2 = client.get(
                    f"/api/index/stream?action=start&repo_path={docs_dir}",
                )
                assert resp2.status_code == 409

                block.set()
                future.result(timeout=10)


# ── Stream endpoint — cancel during indexing ─────────────────────


class TestStreamCancel:
    def test_cancel_running_job_returns_ok(self, app_with_index, tmp_path: Path) -> None:
        """Cancel a running indexing job via the cancel endpoint."""
        docs_dir = _make_docs(tmp_path)
        client = TestClient(app_with_index)

        block = threading.Event()
        entered = threading.Event()

        def _blocking_scan(*args, **kwargs):
            entered.set()
            block.wait(timeout=5)
            return []

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=_blocking_scan),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
            patch("doc_qa.api.server.save_config"),
        ):
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    lambda: client.get(
                        f"/api/index/stream?action=start&repo_path={docs_dir}",
                    )
                )

                assert entered.wait(timeout=5)

                resp = client.post("/api/index/cancel")
                assert resp.status_code == 200
                assert resp.json()["ok"] is True

                block.set()
                future.result(timeout=10)

                status_resp = client.get("/api/index/status")
                status = status_resp.json()
                assert status["state"] in ("cancelled", "idle")
