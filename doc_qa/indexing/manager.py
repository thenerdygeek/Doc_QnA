"""IndexingManager — coordinates background indexing jobs."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from doc_qa.config import AppConfig
from doc_qa.indexing.job import IndexingJob, IndexingState, OnSwapCallback

logger = logging.getLogger(__name__)


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
        return self._job is not None and self._job.is_running

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
