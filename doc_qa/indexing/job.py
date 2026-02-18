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

import os

_IS_WINDOWS = sys.platform == "win32"

import pyarrow as pa

from doc_qa.config import AppConfig
from doc_qa.indexing.chunker import Chunk, chunk_sections
from doc_qa.indexing.indexer import DocIndex, _compute_file_hash
from doc_qa.indexing.scanner import scan_files
from doc_qa.parsers.date_extractor import extract_doc_date
from doc_qa.parsers.registry import parse_file
from doc_qa.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ── Parallel processing settings ─────────────────────────────────
_MAX_WORKERS = min(os.cpu_count() or 4, 16)  # scale with CPU, cap at 16
_EMBED_BATCH_SIZE = 75    # files to accumulate before bulk embedding + insert
_CHUNK_SUB_BATCH = 1000   # max chunks to embed+insert in one go (controls memory)
_MAX_EVENT_BUFFER = 500   # max events kept for SSE replay on reconnect


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
        force_reindex: bool = False,
    ) -> None:
        self.repo_path = repo_path
        self.config = config
        self.db_path = db_path
        self.force_reindex = force_reindex

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

        When ``force_reindex`` is False (default), performs incremental
        indexing: detects which files are new/changed/deleted compared to the
        production index and only processes those.  Unchanged file chunks are
        copied directly from the production DB (no re-embedding).

        When ``force_reindex`` is True, rebuilds the entire index from scratch.

        Pipeline:
        1. Scan files
        2. Detect changes (incremental) or process all (force)
        3. Parse + chunk + index into temp DB
        4. Rebuild FTS index
        5. Atomic swap (temp → prod)
        6. Done

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
            all_files = await loop.run_in_executor(None, scan_files, self.config.doc_repo)
            self._check_cancelled()

            # ── Incremental change detection ─────────────────
            files_to_process: list[Path] = all_files
            unchanged_files: set[str] = set()
            skipped_count = 0

            if not self.force_reindex:
                prod_index = _open_prod_index_readonly(
                    self.db_path, self.config.indexing.embedding_model,
                )
                if prod_index is not None:
                    indexed_hashes = await loop.run_in_executor(
                        None, prod_index.get_indexed_file_hashes,
                    )
                    if indexed_hashes:
                        new_files, changed_files, deleted_files = await loop.run_in_executor(
                            None,
                            _detect_changes_with_hashes,
                            [str(f) for f in all_files],
                            indexed_hashes,
                        )
                        files_to_process_set = set(new_files) | set(changed_files)
                        # Files in the prod index that are NOT changed/deleted
                        all_indexed = set(indexed_hashes.keys())
                        deleted_set = set(deleted_files)
                        unchanged_files = all_indexed - deleted_set - set(
                            _normalize_path(fp) for fp in changed_files
                        )

                        if not files_to_process_set and not deleted_files:
                            # Nothing changed — emit done immediately
                            self.total_files = len(all_files)
                            self.processed_files = len(all_files)
                            self.total_chunks = prod_index.count_rows()
                            elapsed = round(time.time() - self._start_time, 1)
                            self._state = IndexingState.done
                            self._emit("progress", {
                                "state": "done",
                                "processed": self.total_files,
                                "total_files": self.total_files,
                                "total_chunks": self.total_chunks,
                                "percent": 100,
                                "message": "All files are up to date",
                            })
                            self._emit("done", {
                                "total_files": self.total_files,
                                "total_chunks": self.total_chunks,
                                "elapsed": elapsed,
                                "skipped_unchanged": self.total_files,
                            })
                            return

                        files_to_process = [Path(f) for f in files_to_process_set]
                        skipped_count = len(unchanged_files)
                        logger.info(
                            "Incremental indexing: %d new/changed, %d unchanged (skipped), %d deleted",
                            len(files_to_process),
                            skipped_count,
                            len(deleted_files),
                        )

            self.total_files = len(files_to_process) + skipped_count
            self._emit("progress", {
                "state": "scanning",
                "processed": 0,
                "total_files": self.total_files,
                "total_chunks": 0,
                "percent": 0,
                "message": f"Found {len(all_files)} files"
                + (f" ({skipped_count} unchanged, skipped)" if skipped_count else ""),
            })

            # ── Phase 2: Indexing into temp DB ───────────────
            self._set_state(IndexingState.indexing)

            # Clean up any leftover temp DB from a previous failed run
            await loop.run_in_executor(None, _cleanup_temp, temp_db_path)

            temp_index = DocIndex(
                db_path=temp_db_path,
                embedding_model=self.config.indexing.embedding_model,
            )

            # Copy unchanged chunks from prod to temp (no re-embedding needed)
            if unchanged_files and not self.force_reindex:
                copied_chunks = await loop.run_in_executor(
                    None,
                    _copy_unchanged_chunks,
                    self.db_path,
                    temp_index,
                    unchanged_files,
                    self.config.indexing.embedding_model,
                )
                self.total_chunks += copied_chunks
                self.processed_files += skipped_count
                percent = int((self.processed_files / max(self.total_files, 1)) * 100)
                self._emit("progress", {
                    "state": "indexing",
                    "processed": self.processed_files,
                    "total_files": self.total_files,
                    "total_chunks": self.total_chunks,
                    "percent": percent,
                    "message": f"Copied {copied_chunks} chunks from {skipped_count} unchanged files",
                })

            chunk_size = self.config.indexing.chunk_size
            chunk_overlap = self.config.indexing.chunk_overlap
            min_chunk_size = self.config.indexing.min_chunk_size
            chunking_strategy = self.config.indexing.chunking_strategy
            embedding_model = self.config.indexing.embedding_model
            enable_parent_child = self.config.indexing.enable_parent_child
            parent_chunk_size = self.config.indexing.parent_chunk_size
            child_chunk_size = self.config.indexing.child_chunk_size

            # Parse and chunk files in parallel, then batch-insert embeddings.
            # This gives ~3-4x speedup over sequential processing.
            #
            # IMPORTANT: We use asyncio.wrap_future + asyncio.wait instead of
            # concurrent.futures.as_completed, because as_completed is a
            # BLOCKING iterator that would freeze the asyncio event loop and
            # prevent SSE events from being delivered to the frontend.
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS)

            try:
                # Submit parse jobs in batches
                batch_start = 0
                while batch_start < len(files_to_process):
                    self._check_cancelled()

                    # Submit a batch of files for parallel parsing
                    batch_end = min(batch_start + _EMBED_BATCH_SIZE, len(files_to_process))
                    batch_files = files_to_process[batch_start:batch_end]

                    # Submit to thread pool and wrap as asyncio futures
                    # so we can await them without blocking the event loop.
                    async_futures: dict[asyncio.Future, str] = {}
                    for fp in batch_files:
                        cf = executor.submit(
                            _parse_and_chunk,
                            str(fp),
                            chunk_size,
                            chunk_overlap,
                            min_chunk_size,
                            chunking_strategy,
                            embedding_model,
                            enable_parent_child,
                            parent_chunk_size,
                            child_chunk_size,
                        )
                        af = asyncio.wrap_future(cf, loop=loop)
                        async_futures[af] = str(fp)

                    # Collect parse results as they complete (non-blocking)
                    batch_chunks: list[Chunk] = []
                    batch_hashes: dict[str, str] = {}
                    batch_dates: dict[str, float] = {}
                    pending: set[asyncio.Future] = set(async_futures.keys())

                    while pending:
                        self._check_cancelled()
                        # Wait for at least one future to complete — yields
                        # control back to the event loop so SSE can flush.
                        # Timeout ensures cancel is checked every 2s even
                        # when futures are slow (e.g. HF Hub timeouts).
                        done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED,
                            timeout=2.0,
                        )
                        if not done:
                            continue  # Timeout — re-check cancel flag

                        for async_fut in done:
                            file_path = async_futures[async_fut]
                            try:
                                result = async_fut.result()
                            except Exception as exc:
                                logger.warning("Failed to process %s: %s", file_path, exc)
                                result = {"chunks": [], "sections": 0, "skipped": True, "file_hash": "", "doc_date": 0.0}

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
                            if result["file_hash"]:
                                batch_hashes[file_path] = result["file_hash"]
                            if result.get("doc_date", 0.0) > 0:
                                batch_dates[file_path] = result["doc_date"]

                    # Bulk embed + insert — sub-batched to control memory.
                    # A batch of 50 PDFs could produce 30K+ chunks; embedding
                    # all at once would use ~300+ MB for Python float lists.
                    if batch_chunks:
                        await loop.run_in_executor(
                            None,
                            _bulk_add_chunks,
                            temp_index,
                            batch_chunks,
                            batch_hashes,
                            batch_dates,
                        )

                    batch_start = batch_end
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

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

            # Windows deferred cleanup: rename temp → prod, then re-open
            # from prod path so the retriever isn't pointing at the
            # now-renamed temp directory.
            if swap_result.temp_db_path is not None:
                await loop.run_in_executor(None, _finalize_swap_directories, swap_result)
                # Re-open from production path so the retriever isn't pointing at
                # the now-renamed temp directory
                reopened = await loop.run_in_executor(
                    None, _reopen_from_prod, swap_result.prod_db_path, self.config.indexing.embedding_model,
                )
                if reopened is not None:
                    await on_swap(reopened)

            # ── Phase 5: Done ────────────────────────────────
            elapsed = round(time.time() - self._start_time, 1)
            self._state = IndexingState.done
            self._emit("done", {
                "total_files": self.total_files,
                "total_chunks": self.total_chunks,
                "elapsed": elapsed,
                "skipped_unchanged": skipped_count,
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
    chunking_strategy: str = "paragraph",
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    enable_parent_child: bool = False,
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
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

        if enable_parent_child:
            from doc_qa.indexing.chunker import chunk_sections_parent_child
            chunks = chunk_sections_parent_child(
                sections,
                file_path=file_path,
                parent_max_tokens=parent_chunk_size,
                child_max_tokens=child_chunk_size,
                overlap_tokens=chunk_overlap,
                min_tokens=min_chunk_size,
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
            )
        else:
            chunks = chunk_sections(
                sections,
                file_path=file_path,
                max_tokens=chunk_size,
                overlap_tokens=chunk_overlap,
                min_tokens=min_chunk_size,
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
            )
        if not chunks:
            return {"chunks": [], "sections": len(sections), "skipped": True, "file_hash": ""}

        file_hash = _compute_file_hash(file_path)
        doc_date = extract_doc_date(file_path)
        return {
            "chunks": chunks,
            "sections": len(sections),
            "skipped": False,
            "file_hash": file_hash,
            "doc_date": doc_date,
        }

    except Exception as exc:
        logger.warning("Failed to process %s: %s", file_path, exc)
        return {"chunks": [], "sections": 0, "skipped": True, "file_hash": "", "doc_date": 0.0}


def _bulk_add_chunks(
    temp_index: DocIndex,
    chunks: list[Chunk],
    file_hashes: dict[str, str] | None = None,
    doc_dates: dict[str, float] | None = None,
) -> int:
    """Embed and insert a batch of chunks from multiple files at once.

    Batching embeddings is much more efficient than per-file embedding
    because the ONNX model can process larger batches in one pass.

    When ``file_hashes`` is provided (from ``_parse_and_chunk`` results),
    avoids re-reading every file for SHA-256 — saves significant I/O
    for large repos.

    For very large batches (e.g. 50 PDFs × 600 chunks = 30K chunks),
    processes in sub-batches of ``_CHUNK_SUB_BATCH`` to control peak
    memory usage during embedding.
    """
    if not chunks:
        return 0

    # Build hash lookup — prefer pre-computed hashes, fall back to on-demand
    if file_hashes is None:
        file_hashes = {}
    hashes: dict[str, str] = dict(file_hashes)
    for c in chunks:
        if c.file_path not in hashes:
            hashes[c.file_path] = _compute_file_hash(c.file_path)

    total_added = 0

    # Sub-batch to control memory: embedding 30K chunks at once would
    # allocate ~300+ MB for Python float lists.
    for i in range(0, len(chunks), _CHUNK_SUB_BATCH):
        sub = chunks[i : i + _CHUNK_SUB_BATCH]
        total_added += temp_index.add_chunks(sub, hashes, doc_dates=doc_dates)

    return total_added


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


def _reopen_from_prod(prod_db_path: str, embedding_model: str) -> SwapResult | None:
    """Re-open the index from the production path after Windows finalization.

    After ``_finalize_swap_directories`` renames temp → prod, the retriever
    still holds file handles to the old temp path.  This function opens a
    fresh DocIndex + HybridRetriever from the (now-populated) production
    directory so subsequent queries hit valid Lance manifests.
    """
    prod = Path(prod_db_path)
    if not prod.exists():
        logger.warning("Production path %s does not exist after finalization", prod_db_path)
        return None
    try:
        new_index = DocIndex(db_path=prod_db_path, embedding_model=embedding_model)
        new_retriever = HybridRetriever(
            table=new_index._table,
            embedding_model=embedding_model,
        )
        logger.info("Re-opened index from production path: %s", prod_db_path)
        return SwapResult(index=new_index, retriever=new_retriever)
    except Exception as exc:
        logger.warning("Failed to re-open index from production path %s: %s", prod_db_path, exc)
        return None


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
                    "Index is serving from temp path: %s. "
                    "Queries may fail with corrupt Lance manifest errors. "
                    "Try closing other programs that access the index and restart.",
                    max_attempts,
                    swap_result.temp_db_path,
                )


def _cleanup_temp(temp_db_path: str) -> None:
    """Remove the temporary building directory if it exists."""
    p = Path(temp_db_path)
    if p.exists():
        shutil.rmtree(p)
        logger.info("Cleaned up temp index: %s", temp_db_path)


def _normalize_path(file_path: str) -> str:
    """Normalize file path to forward slashes for cross-platform consistency."""
    return file_path.replace("\\", "/")


def _open_prod_index_readonly(
    db_path: str, embedding_model: str,
) -> DocIndex | None:
    """Open the production index for read-only change detection.

    Returns None if the production DB doesn't exist or is empty.
    """
    prod = Path(db_path)
    if not prod.exists():
        return None
    try:
        idx = DocIndex(db_path=db_path, embedding_model=embedding_model)
        if idx.count_rows() == 0:
            return None
        return idx
    except Exception as exc:
        logger.warning("Could not open production index for change detection: %s", exc)
        return None


def _detect_changes_with_hashes(
    file_paths: list[str],
    indexed_hashes: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Compare current files against indexed hashes.

    Like ``DocIndex.detect_changes`` but takes pre-loaded hashes
    (avoids re-reading the prod table inside the temp index context).

    Returns:
        (new_files, changed_files, deleted_files)
    """
    from doc_qa.indexing.indexer import _compute_file_hash

    norm_to_orig = {_normalize_path(fp): fp for fp in file_paths}
    current_set = set(norm_to_orig.keys())
    indexed_set = set(indexed_hashes.keys())

    new_files: list[str] = []
    changed_files: list[str] = []

    for norm_fp, orig_fp in norm_to_orig.items():
        if norm_fp not in indexed_set:
            new_files.append(orig_fp)
        else:
            current_hash = _compute_file_hash(orig_fp)
            if current_hash != indexed_hashes[norm_fp]:
                changed_files.append(orig_fp)

    deleted_files = list(indexed_set - current_set)

    logger.info(
        "Change detection: %d new, %d changed, %d deleted",
        len(new_files), len(changed_files), len(deleted_files),
    )
    return new_files, changed_files, deleted_files


def _copy_unchanged_chunks(
    prod_db_path: str,
    temp_index: DocIndex,
    unchanged_files: set[str],
    embedding_model: str,
) -> int:
    """Copy chunk rows from production index to temp for unchanged files.

    This avoids re-embedding unchanged files — the existing vectors are
    copied directly, which is orders of magnitude faster.

    Uses filtered queries instead of full-table materialisation so that
    only matching rows are loaded into memory.  This is critical for
    large repos (2900+ files / 100K+ chunks) where the full Arrow table
    would exceed available RAM.

    Returns the number of chunks copied.
    """
    if not unchanged_files:
        return 0

    # Number of file paths per WHERE clause.  200 OR conditions is well
    # within LanceDB's query parser limits and keeps each result set small.
    _QUERY_BATCH = 200

    try:
        prod_index = DocIndex(db_path=prod_db_path, embedding_model=embedding_model)

        unchanged_list = list(unchanged_files)
        total_copied = 0

        for batch_start in range(0, len(unchanged_list), _QUERY_BATCH):
            batch_files = unchanged_list[batch_start : batch_start + _QUERY_BATCH]

            # Build an OR filter — only matching rows are loaded from disk.
            conditions = []
            for fp in batch_files:
                safe = fp.replace("\\", "\\\\").replace('"', '\\"')
                conditions.append(f'file_path = "{safe}"')
            where_clause = " OR ".join(conditions)

            # Filtered query: returns only rows for these files.
            # .search() without a vector query acts as a full-scan with
            # server-side WHERE, avoiding full-table Arrow materialisation.
            filtered = (
                prod_index._table.search()
                .where(where_clause, prefilter=True)
                .limit(500_000)
                .to_arrow()
            )

            if filtered.num_rows == 0:
                continue

            # Drop internal columns (_distance, _score, _rowid) that
            # LanceDB search may add — they're not part of our schema.
            drop_cols = [c for c in filtered.column_names if c.startswith("_")]
            if drop_cols:
                filtered = filtered.drop(drop_cols)

            temp_index._table.add(filtered)
            total_copied += filtered.num_rows
            del filtered

        if total_copied > 0:
            logger.info("Copied %d unchanged chunks to temp index.", total_copied)
        return total_copied

    except Exception as exc:
        logger.warning("Failed to copy unchanged chunks: %s — will re-index all.", exc)
        return 0
