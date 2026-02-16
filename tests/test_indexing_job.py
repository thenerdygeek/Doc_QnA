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
    _detect_changes_with_hashes,
    _normalize_path,
    _open_prod_index_readonly,
    _parse_and_chunk,
    _process_file,
    _reopen_from_prod,
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


# ── Incremental indexing tests ───────────────────────────────────


class TestIncrementalIndexing:
    """Tests for incremental indexing (skip unchanged files)."""

    def test_detect_changes_all_new(self, tmp_path: Path):
        """All files new when indexed_hashes is empty."""
        repo = _make_test_repo(tmp_path)
        files = [str(repo / "intro.md"), str(repo / "guide.md")]
        new, changed, deleted = _detect_changes_with_hashes(files, {})
        assert len(new) == 2
        assert changed == []
        assert deleted == []

    def test_detect_changes_unchanged(self, tmp_path: Path):
        """Files with matching hashes are not returned."""
        repo = _make_test_repo(tmp_path)
        from doc_qa.indexing.indexer import _compute_file_hash

        intro_path = str(repo / "intro.md")
        guide_path = str(repo / "guide.md")
        intro_hash = _compute_file_hash(intro_path)
        guide_hash = _compute_file_hash(guide_path)

        indexed_hashes = {
            _normalize_path(intro_path): intro_hash,
            _normalize_path(guide_path): guide_hash,
        }
        new, changed, deleted = _detect_changes_with_hashes(
            [intro_path, guide_path], indexed_hashes,
        )
        assert new == []
        assert changed == []
        assert deleted == []

    def test_detect_changes_modified(self, tmp_path: Path):
        """A file with a different hash should be detected as changed."""
        repo = _make_test_repo(tmp_path)
        intro_path = str(repo / "intro.md")
        guide_path = str(repo / "guide.md")

        from doc_qa.indexing.indexer import _compute_file_hash

        indexed_hashes = {
            _normalize_path(intro_path): "oldhash123",
            _normalize_path(guide_path): _compute_file_hash(guide_path),
        }
        new, changed, deleted = _detect_changes_with_hashes(
            [intro_path, guide_path], indexed_hashes,
        )
        assert new == []
        assert changed == [intro_path]
        assert deleted == []

    def test_detect_changes_deleted(self, tmp_path: Path):
        """A file in the index but not on disk should be deleted."""
        repo = _make_test_repo(tmp_path)
        intro_path = str(repo / "intro.md")

        indexed_hashes = {
            _normalize_path(intro_path): "somehash",
            "docs/removed.md": "oldhash",
        }
        new, changed, deleted = _detect_changes_with_hashes(
            [intro_path], indexed_hashes,
        )
        assert deleted == ["docs/removed.md"]

    def test_open_prod_index_returns_none_when_missing(self, tmp_path: Path):
        """Should return None if prod DB directory doesn't exist."""
        result = _open_prod_index_readonly(str(tmp_path / "nonexistent"), "model")
        assert result is None

    @pytest.mark.asyncio
    async def test_incremental_skips_unchanged_files(self, tmp_path: Path):
        """When force_reindex=False and all files are unchanged, job finishes immediately."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        from doc_qa.indexing.indexer import _compute_file_hash

        intro_path = str(repo / "intro.md")
        guide_path = str(repo / "guide.md")

        mock_prod_index = MagicMock()
        mock_prod_index.count_rows.return_value = 10
        mock_prod_index.get_indexed_file_hashes.return_value = {
            _normalize_path(intro_path): _compute_file_hash(intro_path),
            _normalize_path(guide_path): _compute_file_hash(guide_path),
        }

        job = IndexingJob(
            repo_path=str(repo), config=cfg, db_path=db_path,
            force_reindex=False,
        )
        _, queue = job.subscribe()

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[Path(intro_path), Path(guide_path)]),
            patch("doc_qa.indexing.job._open_prod_index_readonly", return_value=mock_prod_index),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            await job.run(AsyncMock())

        assert job.state == IndexingState.done
        # Should show all files as total but with skipped_unchanged
        events = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        done_events = [e for e in events if e.event == "done"]
        assert len(done_events) == 1
        assert done_events[0].data["skipped_unchanged"] == 2

    @pytest.mark.asyncio
    async def test_force_reindex_processes_all_files(self, tmp_path: Path):
        """When force_reindex=True, all files are processed even if unchanged."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        job = IndexingJob(
            repo_path=str(repo), config=cfg, db_path=db_path,
            force_reindex=True,
        )
        _, queue = job.subscribe()

        mock_chunk = MagicMock()
        mock_chunk.file_path = str(repo / "intro.md")
        mock_swap = SwapResult(index=MagicMock(), retriever=MagicMock())

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[repo / "intro.md", repo / "guide.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={"chunks": [mock_chunk], "sections": 1, "skipped": False, "file_hash": "abc"}),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=1),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_swap),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()
            await job.run(AsyncMock())

        assert job.state == IndexingState.done
        assert job.processed_files == 2  # All files processed

        events = []
        while not queue.empty():
            evt = queue.get_nowait()
            if evt is None:
                break
            events.append(evt)

        done_events = [e for e in events if e.event == "done"]
        assert done_events[0].data.get("skipped_unchanged", 0) == 0


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

    @pytest.mark.asyncio
    async def test_dead_task_detected_and_allows_restart(self, tmp_path: Path):
        """If the asyncio Task crashes but job state is still 'indexing',
        is_running should detect it, set error, and allow a new start."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        mgr = IndexingManager()

        # Create a job that crashes immediately
        async def _crashing_run(on_swap):
            raise RuntimeError("unexpected crash")

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[]),
            patch("doc_qa.indexing.job.DocIndex"),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            job = await mgr.start(str(repo), cfg, str(tmp_path / "db"), AsyncMock())
            # Manually replace job.run with one that crashes, and create a new task
            mgr._task.cancel()
            await asyncio.sleep(0.1)
            job._state = IndexingState.indexing  # simulate stuck state
            mgr._task = asyncio.create_task(_crashing_run(None))
            await asyncio.sleep(0.1)

            # Task is done (crashed) but job state is still 'indexing'
            assert mgr._task.done()
            assert job._state == IndexingState.indexing  # before is_running fixes it

            # is_running should detect the dead task and set error
            assert not mgr.is_running
            assert job.state == IndexingState.error

            # Now we should be able to start a new job
            with (
                patch("doc_qa.indexing.job.scan_files", return_value=[]),
                patch("doc_qa.indexing.job.DocIndex") as MockIdx,
                patch("doc_qa.indexing.job._atomic_swap", return_value=SwapResult(MagicMock(), MagicMock())),
                patch("doc_qa.indexing.job._cleanup_temp"),
            ):
                MockIdx.return_value.rebuild_fts_index = MagicMock()
                job2 = await mgr.start(str(repo), cfg, str(tmp_path / "db2"), AsyncMock())
                await asyncio.sleep(0.5)
                assert job2.is_terminal
                assert job2.state == IndexingState.done

    @pytest.mark.asyncio
    async def test_stale_job_replaced_on_start(self, tmp_path: Path):
        """A stale job (no events for > timeout) should be force-cleaned
        and replaced when start() is called again."""
        import time as _time

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
            await asyncio.get_running_loop().run_in_executor(None, entered_scan.wait, 5)

            assert mgr.is_running

            # Simulate stale: backdate all event timestamps and start_time
            for evt in mgr._job._event_buffer:
                evt.timestamp = _time.time() - 600  # 10 min ago
            mgr._job._start_time = _time.time() - 600

            assert mgr._is_stale()

            # Starting a new job should succeed (force-cleans the stale one)
            with (
                patch("doc_qa.indexing.job.scan_files", return_value=[]),
                patch("doc_qa.indexing.job.DocIndex") as MockIdx,
                patch("doc_qa.indexing.job._atomic_swap", return_value=SwapResult(MagicMock(), MagicMock())),
                patch("doc_qa.indexing.job._cleanup_temp"),
            ):
                MockIdx.return_value.rebuild_fts_index = MagicMock()
                job2 = await mgr.start(str(repo), cfg, str(tmp_path / "db2"), AsyncMock())
                block_event.set()  # unblock old scan
                await asyncio.sleep(0.5)
                assert job2.is_terminal
                assert job2.state == IndexingState.done

    @pytest.mark.asyncio
    async def test_stale_detected_when_no_events_emitted(self, tmp_path: Path):
        """Stale detection works even when the event buffer is empty
        (e.g., stuck in scanning before any events are emitted)."""
        import time as _time

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
            await asyncio.get_running_loop().run_in_executor(None, entered_scan.wait, 5)

            # Clear event buffer and backdate start_time
            mgr._job._event_buffer.clear()
            mgr._job._start_time = _time.time() - 600

            assert mgr._is_stale()
            block_event.set()
            await asyncio.sleep(0.5)


# ── Windows swap re-open tests ───────────────────────────────────


class TestWindowsSwapReopen:
    """Tests for the Windows post-finalization re-open fix."""

    @pytest.mark.asyncio
    async def test_windows_swap_reopens_from_prod(self, tmp_path: Path):
        """On Windows, on_swap should be called twice: once with temp path,
        once with reopened prod path after finalization."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        # Build a SwapResult that simulates Windows (temp_db_path is set)
        mock_windows_swap = SwapResult(
            index=MagicMock(),
            retriever=MagicMock(),
            temp_db_path=str(tmp_path / "temp_building"),
            prod_db_path=db_path,
        )
        mock_reopened_swap = SwapResult(
            index=MagicMock(),
            retriever=MagicMock(),
        )

        on_swap = AsyncMock()
        mock_chunk = MagicMock()
        mock_chunk.file_path = str(repo / "intro.md")

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[repo / "intro.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={
                "chunks": [mock_chunk], "sections": 1, "skipped": False, "file_hash": "abc",
            }),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=1),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_windows_swap),
            patch("doc_qa.indexing.job._finalize_swap_directories"),
            patch("doc_qa.indexing.job._reopen_from_prod", return_value=mock_reopened_swap),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()

            job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
            await job.run(on_swap)

        assert job.state == IndexingState.done

        # on_swap called twice: first with temp-path result, then with reopened prod result
        assert on_swap.call_count == 2
        on_swap.assert_any_call(mock_windows_swap)
        on_swap.assert_any_call(mock_reopened_swap)

    @pytest.mark.asyncio
    async def test_unix_swap_does_not_reopen(self, tmp_path: Path):
        """On Unix (temp_db_path=None), on_swap should only be called once."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        # Unix SwapResult: temp_db_path is None
        mock_unix_swap = SwapResult(
            index=MagicMock(),
            retriever=MagicMock(),
        )

        on_swap = AsyncMock()
        mock_chunk = MagicMock()
        mock_chunk.file_path = str(repo / "intro.md")

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[repo / "intro.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={
                "chunks": [mock_chunk], "sections": 1, "skipped": False, "file_hash": "abc",
            }),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=1),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_unix_swap),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()

            job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
            await job.run(on_swap)

        assert job.state == IndexingState.done
        # on_swap called only once on Unix — no re-open needed
        on_swap.assert_called_once_with(mock_unix_swap)

    @pytest.mark.asyncio
    async def test_windows_swap_handles_reopen_failure(self, tmp_path: Path):
        """If _reopen_from_prod returns None, on_swap should only be called once."""
        repo = _make_test_repo(tmp_path)
        cfg = _make_config(str(repo))
        db_path = str(tmp_path / "prod_db")

        mock_windows_swap = SwapResult(
            index=MagicMock(),
            retriever=MagicMock(),
            temp_db_path=str(tmp_path / "temp_building"),
            prod_db_path=db_path,
        )

        on_swap = AsyncMock()
        mock_chunk = MagicMock()
        mock_chunk.file_path = str(repo / "intro.md")

        with (
            patch("doc_qa.indexing.job.scan_files", return_value=[repo / "intro.md"]),
            patch("doc_qa.indexing.job._parse_and_chunk", return_value={
                "chunks": [mock_chunk], "sections": 1, "skipped": False, "file_hash": "abc",
            }),
            patch("doc_qa.indexing.job._bulk_add_chunks", return_value=1),
            patch("doc_qa.indexing.job.DocIndex") as MockIndex,
            patch("doc_qa.indexing.job._atomic_swap", return_value=mock_windows_swap),
            patch("doc_qa.indexing.job._finalize_swap_directories"),
            patch("doc_qa.indexing.job._reopen_from_prod", return_value=None),
            patch("doc_qa.indexing.job._cleanup_temp"),
        ):
            MockIndex.return_value.rebuild_fts_index = MagicMock()

            job = IndexingJob(repo_path=str(repo), config=cfg, db_path=db_path)
            await job.run(on_swap)

        assert job.state == IndexingState.done
        # on_swap called only once — reopen failed, so we don't call again
        on_swap.assert_called_once_with(mock_windows_swap)
