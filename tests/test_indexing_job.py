"""Tests for the IndexingJob + IndexingManager background indexing system."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_qa.config import AppConfig, DocRepoConfig, IndexingConfig
from doc_qa.indexing.job import (
    IndexingEvent,
    IndexingJob,
    IndexingState,
    SwapResult,
    _cleanup_temp,
    _parse_and_chunk,
    _process_file,
)
from doc_qa.indexing.manager import IndexingManager


# ── Helpers ──────────────────────────────────────────────────────


def _make_config(repo_path: str) -> AppConfig:
    """Create a minimal AppConfig for testing."""
    cfg = AppConfig()
    cfg.doc_repo = DocRepoConfig(path=repo_path)
    cfg.indexing = IndexingConfig(
        db_path="./data/test_db",
        chunk_size=256,
        chunk_overlap=25,
        min_chunk_size=50,
    )
    return cfg


def _make_test_repo(tmp_path: Path) -> Path:
    """Create a test docs repo with a few files."""
    repo = tmp_path / "docs"
    repo.mkdir()
    (repo / "intro.md").write_text("# Introduction\n\nWelcome to the docs.")
    (repo / "guide.md").write_text("# User Guide\n\nStep 1: install.\nStep 2: configure.")
    return repo


async def _noop_swap(result: SwapResult) -> None:
    """No-op swap callback for tests."""
    pass


# ── IndexingJob tests ────────────────────────────────────────────


class TestIndexingJob:
    """Unit tests for IndexingJob event-driven indexing pipeline."""

    def test_initial_state(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        assert job.state == IndexingState.idle
        assert not job.is_running
        assert not job.is_terminal
        assert job.total_files == 0
        assert job.processed_files == 0

    def test_subscribe_returns_empty_for_new_job(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        past, queue = job.subscribe()
        assert past == []
        assert isinstance(queue, asyncio.Queue)

    def test_unsubscribe_removes_queue(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        _, queue = job.subscribe()
        assert len(job._subscribers) == 1
        job.unsubscribe(queue)
        assert len(job._subscribers) == 0

    def test_unsubscribe_ignores_unknown_queue(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        unknown: asyncio.Queue = asyncio.Queue()
        job.unsubscribe(unknown)  # Should not raise

    def test_emit_pushes_to_subscribers(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        _, q1 = job.subscribe()
        _, q2 = job.subscribe()

        job._emit("test_event", {"key": "value"})

        assert not q1.empty()
        assert not q2.empty()
        evt = q1.get_nowait()
        assert evt.event == "test_event"
        assert evt.data == {"key": "value"}
        assert len(job._event_buffer) == 1

    def test_emit_sentinel_sends_none_to_all(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=str(tmp_path / "db"))

        _, q1 = job.subscribe()
        _, q2 = job.subscribe()

        job._emit_sentinel()

        assert q1.get_nowait() is None
        assert q2.get_nowait() is None

    @pytest.mark.asyncio
    async def test_full_successful_run(self, tmp_path: Path):
        """End-to-end run with mocked heavy I/O."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
        past, queue = job.subscribe()

        # Mock the heavy I/O operations
        mock_files = [repo / "intro.md", repo / "guide.md"]
        mock_swap_result = SwapResult(index=MagicMock(), retriever=MagicMock())
        on_swap = AsyncMock()

        # _parse_and_chunk returns chunks list (not count) — mock with 3 mock chunks per file
        mock_chunk = MagicMock()
        mock_chunk.file_path = str(mock_files[0])
        mock_parse_result = {
            "chunks": [mock_chunk, mock_chunk, mock_chunk],
            "sections": 2,
            "skipped": False,
            "file_hash": "abc123",
        }

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=mock_files),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value=mock_parse_result),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=3),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap_result),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()

            await job.run(on_swap)

        assert job.state == IndexingState.done
        assert job.is_terminal
        assert not job.is_running
        assert job.total_files == 2
        assert job.processed_files == 2
        assert job.total_chunks == 6  # 3 chunks * 2 files

        on_swap.assert_called_once_with(mock_swap_result)

        # Collect all events
        events: list[IndexingEvent] = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        event_types = [e.event for e in events]
        assert "status" in event_types
        assert "progress" in event_types
        assert "file_done" in event_types
        assert "done" in event_types

        # Check done event data
        done_events = [e for e in events if e.event == "done"]
        assert len(done_events) == 1
        assert done_events[0].data["total_files"] == 2
        assert done_events[0].data["total_chunks"] == 6

    @pytest.mark.asyncio
    async def test_cancel_during_indexing(self, tmp_path: Path):
        """Cancel after scanning but during file processing."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
        _, queue = job.subscribe()

        mock_files = [repo / "intro.md", repo / "guide.md"]

        call_count = 0
        mock_chunk = MagicMock()
        mock_chunk.file_path = str(mock_files[0])

        def _parse_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                # Cancel after first file
                job.cancel()
            return {"chunks": [mock_chunk, mock_chunk], "sections": 1, "skipped": False, "file_hash": "abc"}

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=mock_files),
            patch("doc_qa.indexing.job._parse_and_chunk", side_effect=_parse_side_effect),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=2),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._cleanup_temp") as mock_cleanup,
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()
            await job.run(AsyncMock())

        assert job.state == IndexingState.cancelled

        # Verify cleanup was called (once for pre-cleanup, once for cancel)
        assert mock_cleanup.call_count >= 1

        # Check cancelled event
        events = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        cancelled = [e for e in events if e.event == "cancelled"]
        assert len(cancelled) == 1

    @pytest.mark.asyncio
    async def test_error_during_scanning(self, tmp_path: Path):
        """Error in scan_files should emit error event."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
        _, queue = job.subscribe()

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=FileNotFoundError("No such dir")),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            await job.run(AsyncMock())

        assert job.state == IndexingState.error
        assert job.error_message == "No such dir"

        events = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        error_events = [e for e in events if e.event == "error"]
        assert len(error_events) == 1
        assert error_events[0].data["error"] == "No such dir"
        assert error_events[0].data["type"] == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_event_replay_for_late_subscriber(self, tmp_path: Path):
        """A subscriber joining mid-run should get replay of past events."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)

        # Emit some events manually
        job._emit("status", {"state": "scanning"})
        job._emit("progress", {"processed": 0, "total_files": 5})

        # Late subscriber should get replay
        past, queue = job.subscribe()
        assert len(past) == 2
        assert past[0].event == "status"
        assert past[1].event == "progress"

    @pytest.mark.asyncio
    async def test_empty_repo_completes_with_zero_files(self, tmp_path: Path):
        """An empty repo should still complete successfully."""
        repo = tmp_path / "empty_docs"
        repo.mkdir()
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
        _, queue = job.subscribe()

        mock_swap_result = SwapResult(index=MagicMock(), retriever=MagicMock())

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[]),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap_result),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()
            await job.run(AsyncMock())

        assert job.state == IndexingState.done
        assert job.total_files == 0
        assert job.total_chunks == 0

    @pytest.mark.asyncio
    async def test_state_transitions_in_order(self, tmp_path: Path):
        """State transitions follow: idle → scanning → indexing → rebuilding_fts → swapping → done."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
        _, queue = job.subscribe()

        mock_swap_result = SwapResult(index=MagicMock(), retriever=MagicMock())

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[repo / "intro.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={"chunks": [MagicMock()], "sections": 1, "skipped": False, "file_hash": "abc"}),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=1),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap_result),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()
            await job.run(AsyncMock())

        events = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        status_events = [e for e in events if e.event == "status"]
        states = [e.data["state"] for e in status_events]
        assert states == ["scanning", "indexing", "rebuilding_fts", "swapping"]


# ── _process_file tests ──────────────────────────────────────────


class TestProcessFile:
    """Unit tests for the _process_file helper."""

    def test_skips_unparseable_file(self, tmp_path: Path):
        """A file that yields no sections should be marked as skipped."""
        mock_index = MagicMock()
        with patch("doc_qa.indexing.job.parse_file", return_value=[]):
            result = _process_file(mock_index, str(tmp_path / "bad.xyz"), 256, 25, 50)

        assert result["skipped"] is True
        assert result["chunks"] == 0
        mock_index.upsert_file.assert_not_called()

    def test_handles_parse_exception(self, tmp_path: Path):
        """An exception during parsing should be caught and return skipped."""
        mock_index = MagicMock()
        with patch("doc_qa.indexing.job.parse_file", side_effect=Exception("Parse error")):
            result = _process_file(mock_index, str(tmp_path / "broken.md"), 256, 25, 50)

        assert result["skipped"] is True
        assert result["chunks"] == 0


# ── _cleanup_temp tests ──────────────────────────────────────────


class TestCleanupTemp:
    """Unit tests for temp directory cleanup."""

    def test_removes_existing_temp(self, tmp_path: Path):
        temp = tmp_path / "test_building"
        temp.mkdir()
        (temp / "data.bin").write_bytes(b"test")

        _cleanup_temp(str(temp))
        assert not temp.exists()

    def test_noop_if_not_exists(self, tmp_path: Path):
        _cleanup_temp(str(tmp_path / "nonexistent"))  # Should not raise


# ── IndexingManager tests ────────────────────────────────────────


class TestIndexingManager:
    """Unit tests for IndexingManager."""

    def test_initial_state(self):
        mgr = IndexingManager()
        assert mgr.current_job is None
        assert not mgr.is_running
        assert mgr.get_status()["state"] == "idle"

    @pytest.mark.asyncio
    async def test_start_creates_job(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        mgr = IndexingManager()

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[]),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=SwapResult(MagicMock(), MagicMock())),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()
            job = await mgr.start(
                repo_path=str(repo),
                config=cfg,
                db_path=str(tmp_path / "db"),
                on_swap=AsyncMock(),
            )

            assert mgr.current_job is job
            # Wait for background task to complete
            await asyncio.sleep(0.5)
            assert job.is_terminal

    @pytest.mark.asyncio
    async def test_reject_concurrent_start(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        mgr = IndexingManager()

        # Use a threading.Event to block the sync scan_files in executor
        block_event = threading.Event()

        def _blocking_scan(*args, **kwargs):
            block_event.wait(timeout=5)
            return []

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=_blocking_scan),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            await mgr.start(str(repo), cfg, str(tmp_path / "db"), AsyncMock())
            # Give the task a moment to enter scanning
            await asyncio.sleep(0.1)

            with pytest.raises(RuntimeError, match="already running"):
                await mgr.start(str(repo), cfg, str(tmp_path / "db2"), AsyncMock())

            # Unblock so cleanup happens
            block_event.set()
            await asyncio.sleep(0.5)

    @pytest.mark.asyncio
    async def test_cancel_running_job(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        mgr = IndexingManager()

        block_event = threading.Event()
        entered_scan = threading.Event()

        def _blocking_scan(*args, **kwargs):
            entered_scan.set()
            block_event.wait(timeout=5)
            return []

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=_blocking_scan),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            await mgr.start(str(repo), cfg, str(tmp_path / "db"), AsyncMock())
            # Wait until the scan is actually blocking
            await asyncio.get_running_loop().run_in_executor(None, entered_scan.wait, 5)

            mgr.cancel()
            block_event.set()
            await asyncio.sleep(0.5)

            assert mgr.current_job is not None
            assert mgr.current_job.state == IndexingState.cancelled

    def test_cancel_when_idle_raises(self):
        mgr = IndexingManager()
        with pytest.raises(RuntimeError, match="No indexing job"):
            mgr.cancel()

    @pytest.mark.asyncio
    async def test_get_status_while_running(self, tmp_path: Path):
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        mgr = IndexingManager()

        block_event = threading.Event()
        entered_scan = threading.Event()

        def _blocking_scan(*args, **kwargs):
            entered_scan.set()
            block_event.wait(timeout=5)
            return []

        with (
            patch("doc_qa.indexing.job.scan_files", side_effect=_blocking_scan),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            await mgr.start(str(repo), cfg, str(tmp_path / "db"), AsyncMock())
            # Wait until scan is actually running
            await asyncio.get_running_loop().run_in_executor(None, entered_scan.wait, 5)

            status = mgr.get_status()
            assert status["state"] == "scanning"
            assert status["repo_path"] == str(repo)

            block_event.set()
            await asyncio.sleep(0.5)
