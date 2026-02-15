"""IndexingManager — coordinates background indexing jobs."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from doc_qa.config import AppConfig
from doc_qa.indexing.job import IndexingJob, IndexingState, OnSwapCallback

logger = logging.getLogger(__name__)

# If an indexing job hasn't emitted any event for this long (seconds),
# consider it stale.  Laptop sleep / network interruption can cause
# the asyncio task to survive but stop making real progress.
_STALE_JOB_TIMEOUT = 300  # 5 minutes


class IndexingManager:
    """Singleton-ish manager that enforces one indexing job at a time.

    The server creates a single instance and stores it on ``app.state``.
    """

    def __init__(self) -> None:
        self._job: IndexingJob | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    # ── Properties ────────────────────────────────────────────

    @property
    def current_job(self) -> IndexingJob | None:
        return self._job

    @property
    def is_running(self) -> bool:
        """True only if the job state is non-terminal AND the asyncio task is alive."""
        if self._job is None:
            return False
        if not self._job.is_running:
            return False
        # If the asyncio Task finished (crash, cancellation) but the job
        # state was never updated, treat it as not running.
        if self._task is not None and self._task.done():
            logger.warning("Indexing task finished but job state is '%s' — cleaning up", self._job.state.value)
            self._job._state = IndexingState.error
            self._job.error_message = "Indexing task stopped unexpectedly"
            return False
        return True

    def _is_stale(self) -> bool:
        """Check if the current job appears stale (no progress for a while)."""
        if self._job is None or not self._job.is_running:
            return False
        # Check the timestamp of the last emitted event
        if self._job._event_buffer:
            last_event_time = self._job._event_buffer[-1].timestamp
            if time.time() - last_event_time > _STALE_JOB_TIMEOUT:
                return True
        return False

    # ── Start / Cancel ────────────────────────────────────────

    async def start(
        self,
        repo_path: str,
        config: AppConfig,
        db_path: str,
        on_swap: OnSwapCallback,
    ) -> IndexingJob:
        """Launch a new indexing job. Raises if one is already running."""
        async with self._lock:
            if self.is_running:
                if self._is_stale():
                    logger.warning(
                        "Previous indexing job is stale (no progress for %ds) — cancelling it",
                        _STALE_JOB_TIMEOUT,
                    )
                    self._force_cleanup()
                else:
                    raise RuntimeError("An indexing job is already running")

            job = IndexingJob(repo_path=repo_path, config=config, db_path=db_path)
            self._job = job
            self._task = asyncio.create_task(job.run(on_swap))
            logger.info("Started indexing job for %s", repo_path)
            return job

    def cancel(self) -> None:
        """Cancel the current job. Raises if none is running."""
        if self._job is None or not self._job.is_running:
            raise RuntimeError("No indexing job is running")
        self._job.cancel()

    def _force_cleanup(self) -> None:
        """Force-cancel a stale or dead job so a new one can start."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        if self._job is not None:
            self._job._state = IndexingState.error
            self._job.error_message = "Replaced by new indexing request"
            self._job._emit_sentinel()
        self._job = None
        self._task = None

    # ── Status ────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Return a status snapshot suitable for a REST response."""
        if self._job is None:
            return {"state": IndexingState.idle.value}

        return {
            "state": self._job.state.value,
            "repo_path": self._job.repo_path,
            "total_files": self._job.total_files,
            "processed_files": self._job.processed_files,
            "total_chunks": self._job.total_chunks,
            "error": self._job.error_message,
        }
