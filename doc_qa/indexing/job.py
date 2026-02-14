"""Background indexing job with SSE event broadcasting."""

from __future__ import annotations

import asyncio
import enum
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from doc_qa.config import AppConfig
from doc_qa.indexing.chunker import Chunk, chunk_sections
from doc_qa.indexing.indexer import DocIndex, _compute_file_hash
from doc_qa.indexing.scanner import scan_files
from doc_qa.parsers.registry import parse_file
from doc_qa.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)


# ── State machine ────────────────────────────────────────────────


class IndexingState(str, enum.Enum):
    idle = "idle"
    scanning = "scanning"
    indexing = "indexing"
    rebuilding_fts = "rebuilding_fts"
    swapping = "swapping"
    done = "done"
    cancelled = "cancelled"
    error = "error"


_TERMINAL_STATES = frozenset({IndexingState.done, IndexingState.cancelled, IndexingState.error})


# ── Event dataclass ──────────────────────────────────────────────


@dataclass
class IndexingEvent:
    event: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


# ── Swap result ──────────────────────────────────────────────────


@dataclass
class SwapResult:
    index: DocIndex
    retriever: HybridRetriever


# ── Type alias for the on_swap callback ──────────────────────────

OnSwapCallback = Callable[[SwapResult], Coroutine[Any, Any, None]]


# ── IndexingJob ──────────────────────────────────────────────────


class IndexingJob:
    """A single background indexing run.

    Broadcasts events to SSE subscriber queues and supports event replay
    for reconnecting clients.
    """

    def __init__(
        self,
        repo_path: str,
        config: AppConfig,
        db_path: str,
    ) -> None:
        self.repo_path = repo_path
        self.config = config
        self.db_path = db_path

        self._state = IndexingState.idle
        self._cancel_event = asyncio.Event()
        self._subscribers: list[asyncio.Queue[IndexingEvent | None]] = []
        self._event_buffer: list[IndexingEvent] = []

        # Counters for status reporting
        self.total_files: int = 0
        self.processed_files: int = 0
        self.total_chunks: int = 0
        self.error_message: str | None = None
        self._start_time: float = 0.0

    # ── Properties ────────────────────────────────────────────

    @property
    def state(self) -> IndexingState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state not in _TERMINAL_STATES and self._state != IndexingState.idle

    @property
    def is_terminal(self) -> bool:
        return self._state in _TERMINAL_STATES

    # ── Subscriber management ─────────────────────────────────

    def subscribe(self) -> tuple[list[IndexingEvent], asyncio.Queue[IndexingEvent | None]]:
        """Subscribe to events. Returns (past_events, live_queue)."""
        queue: asyncio.Queue[IndexingEvent | None] = asyncio.Queue()
        self._subscribers.append(queue)
        return list(self._event_buffer), queue

    def unsubscribe(self, queue: asyncio.Queue[IndexingEvent | None]) -> None:
        """Remove a subscriber queue."""
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    # ── Event emission ────────────────────────────────────────

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        """Append to buffer and push to all subscriber queues."""
        evt = IndexingEvent(event=event, data=data)
        self._event_buffer.append(evt)
        for q in self._subscribers:
            q.put_nowait(evt)

    def _emit_sentinel(self) -> None:
        """Push None to all queues to signal end of stream."""
        for q in self._subscribers:
            q.put_nowait(None)

    def _set_state(self, new_state: IndexingState) -> None:
        self._state = new_state
        self._emit("status", {"state": new_state.value, "repo_path": self.repo_path})

    # ── Cancel ────────────────────────────────────────────────

    def cancel(self) -> None:
        """Request cancellation of the running job."""
        self._cancel_event.set()

    def _check_cancelled(self) -> None:
        """Raise asyncio.CancelledError if cancel was requested."""
        if self._cancel_event.is_set():
            raise asyncio.CancelledError("Indexing cancelled by user")

    # ── Main pipeline ─────────────────────────────────────────

    async def run(self, on_swap: OnSwapCallback) -> None:
        """Execute the full indexing pipeline.

        1. Scan files
        2. Parse + chunk + index each file into temp DB
        3. Rebuild FTS index
        4. Atomic swap (temp → prod)
        5. Done
        """
        loop = asyncio.get_running_loop()
        self._start_time = time.time()
        temp_db_path = f"{self.db_path}_building"

        try:
            # ── Phase 1: Scanning ────────────────────────────
            self._set_state(IndexingState.scanning)
            files = await loop.run_in_executor(None, scan_files, self.config.doc_repo)
            self.total_files = len(files)
            self._emit("progress", {
                "state": "scanning",
                "processed": 0,
                "total_files": self.total_files,
                "total_chunks": 0,
                "percent": 0,
                "message": f"Found {self.total_files} files",
            })
            self._check_cancelled()

            # ── Phase 2: Indexing into temp DB ───────────────
            self._set_state(IndexingState.indexing)

            # Clean up any leftover temp DB from a previous failed run
            await loop.run_in_executor(None, _cleanup_temp, temp_db_path)

            temp_index = DocIndex(
                db_path=temp_db_path,
                embedding_model=self.config.indexing.embedding_model,
            )

            for i, file_path in enumerate(files):
                self._check_cancelled()

                result = await loop.run_in_executor(
                    None,
                    _process_file,
                    temp_index,
                    str(file_path),
                    self.config.indexing.chunk_size,
                    self.config.indexing.chunk_overlap,
                    self.config.indexing.min_chunk_size,
                )

                self.processed_files = i + 1
                self.total_chunks += result["chunks"]

                self._emit("file_done", {
                    "file": str(file_path),
                    "file_index": i,
                    "total_files": self.total_files,
                    "chunks": result["chunks"],
                    "sections": result["sections"],
                    "skipped": result["skipped"],
                })

                percent = int((self.processed_files / max(self.total_files, 1)) * 100)
                self._emit("progress", {
                    "state": "indexing",
                    "processed": self.processed_files,
                    "total_files": self.total_files,
                    "total_chunks": self.total_chunks,
                    "percent": percent,
                })

            # ── Phase 3: Rebuild FTS ─────────────────────────
            self._check_cancelled()
            self._set_state(IndexingState.rebuilding_fts)
            await loop.run_in_executor(None, temp_index.rebuild_fts_index)

            # ── Phase 4: Atomic swap ─────────────────────────
            self._check_cancelled()
            self._set_state(IndexingState.swapping)
            swap_result = await loop.run_in_executor(
                None, _atomic_swap, temp_db_path, self.db_path, self.config.indexing.embedding_model,
            )

            # Notify server to hot-swap references
            await on_swap(swap_result)

            # ── Phase 5: Done ────────────────────────────────
            elapsed = round(time.time() - self._start_time, 1)
            self._state = IndexingState.done
            self._emit("done", {
                "total_files": self.total_files,
                "total_chunks": self.total_chunks,
                "elapsed": elapsed,
            })

        except asyncio.CancelledError:
            self._state = IndexingState.cancelled
            self._emit("cancelled", {"message": "Indexing was cancelled"})
            await loop.run_in_executor(None, _cleanup_temp, temp_db_path)
            logger.info("Indexing cancelled by user")

        except Exception as exc:
            self._state = IndexingState.error
            self.error_message = str(exc)
            self._emit("error", {"error": str(exc), "type": type(exc).__name__})
            await loop.run_in_executor(None, _cleanup_temp, temp_db_path)
            logger.exception("Indexing failed")

        finally:
            self._emit_sentinel()


# ── Helper functions (run in executor / sync) ────────────────────


def _process_file(
    temp_index: DocIndex,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> dict[str, Any]:
    """Parse, chunk, and upsert a single file into the temp index."""
    try:
        sections = parse_file(file_path)
        if not sections:
            return {"chunks": 0, "sections": 0, "skipped": True}

        chunks = chunk_sections(
            sections,
            file_path=file_path,
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            min_tokens=min_chunk_size,
        )
        if not chunks:
            return {"chunks": 0, "sections": len(sections), "skipped": True}

        added = temp_index.upsert_file(chunks, file_path)
        return {"chunks": added, "sections": len(sections), "skipped": False}

    except Exception as exc:
        logger.warning("Failed to process %s: %s", file_path, exc)
        return {"chunks": 0, "sections": 0, "skipped": True}


def _atomic_swap(
    temp_db_path: str,
    prod_db_path: str,
    embedding_model: str,
) -> SwapResult:
    """Atomically replace the production index with the temp one.

    Steps:
    1. Rename prod → backup
    2. Rename temp → prod
    3. Open new DocIndex + HybridRetriever
    4. Delete backup
    """
    prod = Path(prod_db_path)
    temp = Path(temp_db_path)
    backup = Path(f"{prod_db_path}_backup")

    # Remove stale backup if present
    if backup.exists():
        shutil.rmtree(backup)

    # Rename prod → backup (may not exist on first index)
    if prod.exists():
        prod.rename(backup)

    # Rename temp → prod
    temp.rename(prod)

    # Open fresh index + retriever
    new_index = DocIndex(db_path=prod_db_path, embedding_model=embedding_model)
    new_retriever = HybridRetriever(
        table=new_index._table,
        embedding_model=embedding_model,
    )

    # Clean up backup
    if backup.exists():
        shutil.rmtree(backup)

    logger.info("Atomic swap complete: %s → %s", temp_db_path, prod_db_path)
    return SwapResult(index=new_index, retriever=new_retriever)


def _cleanup_temp(temp_db_path: str) -> None:
    """Remove the temporary building directory if it exists."""
    p = Path(temp_db_path)
    if p.exists():
        shutil.rmtree(p)
        logger.info("Cleaned up temp index: %s", temp_db_path)
