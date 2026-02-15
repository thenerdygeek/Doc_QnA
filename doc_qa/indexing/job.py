"""Background indexing job with SSE event broadcasting."""

from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import gc
import logging
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

_IS_WINDOWS = sys.platform == "win32"

from doc_qa.config import AppConfig
from doc_qa.indexing.chunker import Chunk, chunk_sections
from doc_qa.indexing.indexer import DocIndex, _compute_file_hash
from doc_qa.indexing.scanner import scan_files
from doc_qa.parsers.registry import parse_file
from doc_qa.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ── Parallel processing settings ─────────────────────────────────
_MAX_WORKERS = 4          # parallel file parsing/chunking threads
_EMBED_BATCH_SIZE = 50    # files to accumulate before bulk embedding + insert
_MAX_EVENT_BUFFER = 200   # max events kept for SSE replay on reconnect


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
    temp_db_path: str | None = None  # set on Windows for deferred cleanup
    prod_db_path: str | None = None


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
        # Capped ring buffer — only the most recent events are kept for
        # SSE replay on reconnect.  For 2900+ file repos, keeping every
        # file_done event wastes memory and makes reconnect slow.
        self._event_buffer: deque[IndexingEvent] = deque(maxlen=_MAX_EVENT_BUFFER)

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
        """Subscribe to events. Returns (recent_events, live_queue).

        Only the most recent ``_MAX_EVENT_BUFFER`` events are replayed —
        reconnecting clients get current progress without replaying
        thousands of old file_done events.
        """
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

        Prevents the OS from sleeping while indexing is running.
        """
        from doc_qa.utils.keep_awake import keep_awake

        loop = asyncio.get_running_loop()
        self._start_time = time.time()
        temp_db_path = f"{self.db_path}_building"

        with keep_awake(f"Indexing {self.repo_path}"):
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

            chunk_size = self.config.indexing.chunk_size
            chunk_overlap = self.config.indexing.chunk_overlap
            min_chunk_size = self.config.indexing.min_chunk_size

            # Parse and chunk files in parallel, then batch-insert embeddings.
            # This gives ~3-4x speedup over sequential processing.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)
            pending_chunks: list[tuple[str, list[Chunk]]] = []  # (file_path, chunks)
            pending_hashes: list[str] = []

            try:
                # Submit parse jobs in batches
                batch_start = 0
                while batch_start < len(files):
                    self._check_cancelled()

                    # Submit a batch of files for parallel parsing
                    batch_end = min(batch_start + _EMBED_BATCH_SIZE, len(files))
                    batch_files = files[batch_start:batch_end]

                    futures = {
                        executor.submit(
                            _parse_and_chunk,
                            str(fp),
                            chunk_size,
                            chunk_overlap,
                            min_chunk_size,
                        ): str(fp)
                        for fp in batch_files
                    }

                    # Collect parse results
                    batch_chunks: list[Chunk] = []
                    batch_file_hashes: list[str] = []

                    for future in concurrent.futures.as_completed(futures):
                        self._check_cancelled()
                        file_path = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            logger.warning("Failed to process %s: %s", file_path, exc)
                            result = {"chunks": [], "sections": 0, "skipped": True, "file_hash": ""}

                        self.processed_files += 1
                        n_chunks = len(result["chunks"])
                        self.total_chunks += n_chunks

                        self._emit("file_done", {
                            "file": file_path,
                            "file_index": self.processed_files - 1,
                            "total_files": self.total_files,
                            "chunks": n_chunks,
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

                        if result["chunks"]:
                            batch_chunks.extend(result["chunks"])
                            batch_file_hashes.append(result["file_hash"])

                    # Bulk embed + insert the entire batch at once
                    if batch_chunks:
                        await loop.run_in_executor(
                            None,
                            _bulk_add_chunks,
                            temp_index,
                            batch_chunks,
                        )

                    batch_start = batch_end
            finally:
                executor.shutdown(wait=False)

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

            # Finalize directory renames (Windows deferred cleanup)
            if swap_result.temp_db_path is not None:
                await loop.run_in_executor(None, _finalize_swap_directories, swap_result)

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


def _parse_and_chunk(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> dict[str, Any]:
    """Parse and chunk a single file (CPU-only, no embedding or DB writes).

    This runs in a thread pool for parallelism.  Embedding and DB insertion
    happen in a separate bulk step so we can batch many files' chunks into
    one ``add_chunks`` call.
    """
    try:
        sections = parse_file(file_path)
        if not sections:
            return {"chunks": [], "sections": 0, "skipped": True, "file_hash": ""}

        chunks = chunk_sections(
            sections,
            file_path=file_path,
            max_tokens=chunk_size,
            overlap_tokens=chunk_overlap,
            min_tokens=min_chunk_size,
        )
        if not chunks:
            return {"chunks": [], "sections": len(sections), "skipped": True, "file_hash": ""}

        file_hash = _compute_file_hash(file_path)
        return {
            "chunks": chunks,
            "sections": len(sections),
            "skipped": False,
            "file_hash": file_hash,
        }

    except Exception as exc:
        logger.warning("Failed to process %s: %s", file_path, exc)
        return {"chunks": [], "sections": 0, "skipped": True, "file_hash": ""}


def _bulk_add_chunks(temp_index: DocIndex, chunks: list[Chunk]) -> int:
    """Embed and insert a batch of chunks from multiple files at once.

    Batching embeddings is much more efficient than per-file embedding
    because the ONNX model can process larger batches in one pass.
    """
    if not chunks:
        return 0

    # Group chunks by file for hash lookup
    file_hashes: dict[str, str] = {}
    for c in chunks:
        if c.file_path not in file_hashes:
            file_hashes[c.file_path] = _compute_file_hash(c.file_path)

    return temp_index.add_chunks(chunks, file_hashes)


def _process_file(
    temp_index: DocIndex,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
) -> dict[str, Any]:
    """Parse, chunk, and upsert a single file into the temp index.

    Kept for backward compatibility (used by incremental re-indexing).
    """
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
    """Replace the production index with the temp one.

    On Unix: atomic rename (prod → backup, temp → prod), then open.
    On Windows: open directly from temp path (avoids PermissionError from
    file locks), deferring directory renames to ``_finalize_swap_directories``
    after the old DocIndex references are released.
    """
    prod = Path(prod_db_path)
    temp = Path(temp_db_path)
    backup = Path(f"{prod_db_path}_backup")

    if _IS_WINDOWS:
        # Phase 1: open new index directly from temp — no renames yet.
        new_index = DocIndex(db_path=temp_db_path, embedding_model=embedding_model)
        new_retriever = HybridRetriever(
            table=new_index._table,
            embedding_model=embedding_model,
        )
        logger.info("Swap phase 1 (Windows): opened index from %s", temp_db_path)
        return SwapResult(
            index=new_index,
            retriever=new_retriever,
            temp_db_path=temp_db_path,
            prod_db_path=prod_db_path,
        )

    # Unix path: atomic renames are safe regardless of open handles.
    if backup.exists():
        shutil.rmtree(backup)
    if prod.exists():
        prod.rename(backup)
    temp.rename(prod)

    new_index = DocIndex(db_path=prod_db_path, embedding_model=embedding_model)
    new_retriever = HybridRetriever(
        table=new_index._table,
        embedding_model=embedding_model,
    )

    if backup.exists():
        shutil.rmtree(backup)

    logger.info("Atomic swap complete: %s → %s", temp_db_path, prod_db_path)
    return SwapResult(index=new_index, retriever=new_retriever)


def _finalize_swap_directories(swap_result: SwapResult) -> None:
    """Phase 2 (Windows only): rename directories after old references are released.

    Called in executor after ``on_swap`` has replaced all references to the old
    DocIndex/HybridRetriever, so file handles should now be closed.  Uses a
    retry loop with ``gc.collect()`` to handle delayed handle release.
    """
    if swap_result.temp_db_path is None:
        return  # Unix path — nothing to finalize.

    temp = Path(swap_result.temp_db_path)
    prod = Path(swap_result.prod_db_path)
    backup = Path(f"{swap_result.prod_db_path}_backup")

    # Force-close any lingering Python references.
    gc.collect()

    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            if backup.exists():
                shutil.rmtree(backup)
            if prod.exists():
                prod.rename(backup)
            if temp.exists():
                temp.rename(prod)
            # Clean up backup
            if backup.exists():
                shutil.rmtree(backup)
            logger.info("Swap phase 2 (Windows): directories finalized.")
            return
        except PermissionError:
            if attempt < max_attempts - 1:
                gc.collect()
                time.sleep(0.5 * (attempt + 1))
                logger.debug(
                    "Swap finalize retry %d/%d — waiting for file handles.",
                    attempt + 1,
                    max_attempts,
                )
            else:
                logger.warning(
                    "Could not finalize swap directories after %d attempts. "
                    "Index is serving from temp path: %s",
                    max_attempts,
                    swap_result.temp_db_path,
                )


def _cleanup_temp(temp_db_path: str) -> None:
    """Remove the temporary building directory if it exists."""
    p = Path(temp_db_path)
    if p.exists():
        shutil.rmtree(p)
        logger.info("Cleaned up temp index: %s", temp_db_path)
