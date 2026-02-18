"""LanceDB indexer — stores and retrieves document chunks with embeddings."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from doc_qa.indexing.chunker import Chunk
from doc_qa.indexing.embedder import embed_texts, get_embedding_dimension

logger = logging.getLogger(__name__)


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for change detection."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _normalize_path(file_path: str) -> str:
    """Normalize file path to forward slashes for cross-platform consistency."""
    return file_path.replace("\\", "/")


class DocIndex:
    """Manages the LanceDB vector store for document chunks.

    Handles:
    - Table creation with schema (text, vector, metadata)
    - BM25 full-text index for hybrid search
    - Upsert via merge_insert for incremental re-indexing
    - File hash tracking for change detection
    """

    TABLE_NAME = "doc_chunks"

    def __init__(
        self,
        db_path: str = "./data/doc_qa_db",
        embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    ) -> None:
        import lancedb

        self._db_path = db_path
        self._embedding_model = embedding_model
        self._dim = get_embedding_dimension(embedding_model)

        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(db_path)
        self._table = self._get_or_create_table()

    def _get_or_create_table(self) -> Any:
        """Get existing table or create a new one with the correct schema."""
        if self.TABLE_NAME in self._db.list_tables().tables:
            table = self._db.open_table(self.TABLE_NAME)
            logger.info(
                "Opened existing table '%s' with %d rows.",
                self.TABLE_NAME,
                table.count_rows(),
            )
            # Migrate: add doc_date column if missing (pre-v2 indexes)
            if "doc_date" not in table.schema.names:
                self._migrate_add_doc_date(table)
                # Migration may have replaced the table reference
                if hasattr(self, "_table") and self._table is not None:
                    table = self._table
            # Migrate: add content_type column if missing (pre-v3 indexes)
            if "content_type" not in table.schema.names:
                self._migrate_add_content_type(table)
                # Migration may have replaced the table reference
                if hasattr(self, "_table") and self._table is not None:
                    table = self._table
            # Migrate: add parent_chunk_id + parent_text if missing (pre-v4 indexes)
            if "parent_chunk_id" not in table.schema.names:
                self._migrate_add_parent_columns(table)
                if hasattr(self, "_table") and self._table is not None:
                    table = self._table
            return table

        # Create with explicit schema
        schema = pa.schema([
            pa.field("chunk_id", pa.utf8()),
            pa.field("text", pa.utf8()),
            pa.field("vector", pa.list_(pa.float32(), self._dim)),
            pa.field("file_path", pa.utf8()),
            pa.field("file_type", pa.utf8()),
            pa.field("section_title", pa.utf8()),
            pa.field("section_level", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("file_hash", pa.utf8()),
            pa.field("doc_date", pa.float64()),
            pa.field("content_type", pa.utf8()),
            pa.field("parent_chunk_id", pa.utf8()),
            pa.field("parent_text", pa.utf8()),
        ])

        table = self._db.create_table(self.TABLE_NAME, schema=schema)
        logger.info("Created new table '%s'.", self.TABLE_NAME)
        return table

    def _migrate_add_doc_date(self, table: Any) -> None:
        """Add doc_date column to an existing table (migration for pre-v2 indexes).

        Sets all existing rows to 0.0 (unknown date = treated as oldest).
        Next incremental or full re-index will backfill actual dates.
        """
        try:
            # Read all data, add column, rewrite
            arrow_table = table.to_arrow()
            n = arrow_table.num_rows
            doc_date_col = pa.array([0.0] * n, type=pa.float64())
            new_table = arrow_table.append_column("doc_date", doc_date_col)
            # Drop and recreate with new schema
            self._db.drop_table(self.TABLE_NAME)
            new_lance_table = self._db.create_table(self.TABLE_NAME, new_table)
            logger.info("Migrated table '%s': added doc_date column (%d rows).", self.TABLE_NAME, n)
            # Update internal reference
            self._table = new_lance_table
        except Exception:
            logger.warning(
                "Failed to migrate doc_date column — dates will be unavailable until re-index.",
                exc_info=True,
            )

    def _migrate_add_content_type(self, table: Any) -> None:
        """Add content_type column to an existing table (migration for pre-v3 indexes).

        Sets all existing rows to ``"prose"`` (default).
        Next incremental or full re-index will backfill actual content types.
        """
        try:
            arrow_table = table.to_arrow()
            n = arrow_table.num_rows
            content_type_col = pa.array(["prose"] * n, type=pa.utf8())
            new_table = arrow_table.append_column("content_type", content_type_col)
            self._db.drop_table(self.TABLE_NAME)
            new_lance_table = self._db.create_table(self.TABLE_NAME, new_table)
            logger.info(
                "Migrated table '%s': added content_type column (%d rows).",
                self.TABLE_NAME, n,
            )
            self._table = new_lance_table
        except Exception:
            logger.warning(
                "Failed to migrate content_type column — types will be unavailable until re-index.",
                exc_info=True,
            )

    def _migrate_add_parent_columns(self, table: Any) -> None:
        """Add parent_chunk_id and parent_text columns (migration for pre-v4 indexes).

        Existing chunks become self-parenting: parent_chunk_id="" and
        parent_text=chunk.text (backward compatible — retrievers can check
        for non-empty parent_text to use parent context).
        """
        try:
            arrow_table = table.to_arrow()
            n = arrow_table.num_rows
            # Self-parenting: empty parent_chunk_id, parent_text = own text
            parent_id_col = pa.array([""] * n, type=pa.utf8())
            parent_text_col = arrow_table.column("text")
            new_table = arrow_table.append_column("parent_chunk_id", parent_id_col)
            new_table = new_table.append_column("parent_text", parent_text_col)
            self._db.drop_table(self.TABLE_NAME)
            new_lance_table = self._db.create_table(self.TABLE_NAME, new_table)
            logger.info(
                "Migrated table '%s': added parent columns (%d rows).",
                self.TABLE_NAME, n,
            )
            self._table = new_lance_table
        except Exception:
            logger.warning(
                "Failed to migrate parent columns — parent-child retrieval unavailable until re-index.",
                exc_info=True,
            )

    def _create_fts_index(self) -> None:
        """Create or rebuild the BM25 full-text search index on the text column."""
        try:
            self._table.create_fts_index("text", replace=True)
            logger.info("BM25 full-text index created on 'text' column.")
        except Exception:
            logger.warning("Failed to create FTS index — hybrid search may be unavailable.", exc_info=True)

    def get_indexed_file_hashes(self) -> dict[str, str]:
        """Return {file_path: file_hash} for all currently indexed files.

        Only loads the two metadata columns — avoids materializing the
        full table (text + vectors) which can be 100+ MB at scale.
        """
        if self._table.count_rows() == 0:
            return {}

        # Select only the columns we need BEFORE converting to Arrow.
        # LanceDB supports column projection via to_arrow(columns=[...]).
        try:
            arrow_table = self._table.to_arrow(columns=["file_path", "file_hash"])
        except TypeError:
            # Older LanceDB versions may not support columns kwarg
            arrow_table = self._table.to_arrow().select(["file_path", "file_hash"])

        result: dict[str, str] = {}
        for fp, fh in zip(
            arrow_table.column("file_path").to_pylist(),
            arrow_table.column("file_hash").to_pylist(),
        ):
            if fp not in result:
                result[fp] = fh
        return result

    def detect_changes(self, file_paths: list[str]) -> tuple[list[str], list[str], list[str]]:
        """Compare current files against index to find changes.

        Args:
            file_paths: List of file paths currently in the doc repo.

        Returns:
            Tuple of (new_files, changed_files, deleted_files).
        """
        indexed = self.get_indexed_file_hashes()
        # Build lookup with normalized paths for cross-platform consistency.
        # Indexed paths are already normalized (stored with forward slashes).
        norm_to_orig = {_normalize_path(fp): fp for fp in file_paths}
        current_set = set(norm_to_orig.keys())
        indexed_set = set(indexed.keys())

        new_files: list[str] = []
        changed_files: list[str] = []

        for norm_fp, orig_fp in norm_to_orig.items():
            if norm_fp not in indexed_set:
                new_files.append(orig_fp)
            else:
                current_hash = _compute_file_hash(orig_fp)
                if current_hash != indexed[norm_fp]:
                    changed_files.append(orig_fp)

        deleted_files = list(indexed_set - current_set)

        logger.info(
            "Change detection: %d new, %d changed, %d deleted.",
            len(new_files),
            len(changed_files),
            len(deleted_files),
        )
        return new_files, changed_files, deleted_files

    def delete_file_chunks(self, file_path: str) -> int:
        """Delete all chunks for a specific file. Returns count deleted."""
        before = self._table.count_rows()
        safe_path = _normalize_path(file_path).replace("\\", "\\\\").replace('"', '\\"')
        self._table.delete(f'file_path = "{safe_path}"')
        after = self._table.count_rows()
        deleted = before - after
        if deleted > 0:
            logger.info("Deleted %d chunks for %s.", deleted, file_path)
        return deleted

    def add_chunks(
        self,
        chunks: list[Chunk],
        file_hash: str | dict[str, str],
        doc_dates: dict[str, float] | None = None,
    ) -> int:
        """Embed and add chunks to the index.

        Args:
            chunks: List of Chunk objects to add.
            file_hash: Either a single SHA-256 hash (all chunks from one file)
                or a dict of ``{file_path: hash}`` for multi-file batches.
            doc_dates: Optional dict of ``{file_path: unix_timestamp}`` for
                version-aware retrieval. Defaults to 0.0 (unknown).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Embed all chunk texts in one batch (much faster than per-file)
        texts = [c.text for c in chunks]
        import time as _time
        _t0 = _time.time()
        vectors = embed_texts(texts, model_name=self._embedding_model)
        logger.info("[PERF] Embedding %d chunks: %.2fs (%.0f chunks/sec)",
                    len(texts), _time.time() - _t0,
                    len(texts) / max(_time.time() - _t0, 0.001))

        # Resolve hash per chunk
        if isinstance(file_hash, str):
            hash_lookup: dict[str, str] = {}  # single hash for all
        else:
            hash_lookup = file_hash

        date_lookup = doc_dates or {}

        # Build records
        records = []
        for chunk, vector in zip(chunks, vectors):
            norm_path = _normalize_path(chunk.file_path)
            h = hash_lookup.get(chunk.file_path, file_hash if isinstance(file_hash, str) else "")
            records.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "vector": vector.tolist(),
                "file_path": norm_path,
                "file_type": chunk.file_type,
                "section_title": chunk.section_title,
                "section_level": chunk.section_level,
                "chunk_index": chunk.chunk_index,
                "file_hash": h,
                "doc_date": date_lookup.get(chunk.file_path, 0.0),
                "content_type": chunk.metadata.get("content_type", "prose"),
                "parent_chunk_id": chunk.parent_chunk_id,
                "parent_text": chunk.parent_text,
            })

        _t1 = _time.time()
        self._table.add(records)
        logger.info("[PERF] DB insert %d records: %.2fs", len(records), _time.time() - _t1)
        n_files = len(set(_normalize_path(c.file_path) for c in chunks))
        logger.info("Added %d chunks from %d file(s).", len(records), n_files)
        return len(records)

    def upsert_file(self, chunks: list[Chunk], file_path: str) -> int:
        """Delete old chunks for a file and insert new ones (atomic upsert).

        Old chunks are snapshotted before deletion. If ``add_chunks`` fails
        (e.g. embedding error), the snapshot is restored so no data is lost.

        Args:
            chunks: New chunks for this file.
            file_path: Path to the source file.

        Returns:
            Number of chunks added.
        """
        file_hash = _compute_file_hash(file_path)
        norm_path = _normalize_path(file_path)

        # Snapshot existing rows for this file so we can restore on failure.
        # Use a filtered query instead of loading the entire table.
        backup_rows: list[dict] | None = None
        if self._table.count_rows() > 0:
            try:
                safe_path = norm_path.replace("\\", "\\\\").replace('"', '\\"')
                filtered = self._table.search().where(
                    f'file_path = "{safe_path}"', prefilter=True
                ).limit(10000).to_arrow()
                if filtered.num_rows > 0:
                    backup_rows = filtered.to_pylist()
            except Exception:
                backup_rows = None

        self.delete_file_chunks(norm_path)
        try:
            return self.add_chunks(chunks, file_hash)
        except Exception:
            # Restore old data so it isn't permanently lost.
            if backup_rows:
                try:
                    self._table.add(backup_rows)
                    logger.warning(
                        "add_chunks failed for %s — restored %d old chunks.",
                        file_path,
                        len(backup_rows),
                    )
                except Exception:
                    logger.error(
                        "add_chunks failed for %s AND restore failed — data lost.",
                        file_path,
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "add_chunks failed for %s — no backup to restore.",
                    file_path,
                )
            raise

    def rebuild_fts_index(self) -> None:
        """Rebuild the full-text search index after bulk changes."""
        if self._table.count_rows() > 0:
            self._create_fts_index()

    def count_rows(self) -> int:
        """Return total number of chunks in the index."""
        return self._table.count_rows()

    def count_files(self) -> int:
        """Return number of unique files in the index."""
        if self._table.count_rows() == 0:
            return 0
        try:
            col = self._table.to_arrow(columns=["file_path"]).column("file_path")
        except TypeError:
            col = self._table.to_arrow().select(["file_path"]).column("file_path")
        return len(set(col.to_pylist()))

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        return {
            "total_chunks": self.count_rows(),
            "total_files": self.count_files(),
            "db_path": self._db_path,
            "embedding_model": self._embedding_model,
            "embedding_dim": self._dim,
        }
