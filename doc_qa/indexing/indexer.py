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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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
        ])

        table = self._db.create_table(self.TABLE_NAME, schema=schema)
        logger.info("Created new table '%s'.", self.TABLE_NAME)
        return table

    def _create_fts_index(self) -> None:
        """Create or rebuild the BM25 full-text search index on the text column."""
        try:
            self._table.create_fts_index("text", replace=True)
            logger.info("BM25 full-text index created on 'text' column.")
        except Exception:
            logger.warning("Failed to create FTS index — hybrid search may be unavailable.", exc_info=True)

    def get_indexed_file_hashes(self) -> dict[str, str]:
        """Return {file_path: file_hash} for all currently indexed files."""
        if self._table.count_rows() == 0:
            return {}

        # Use PyArrow directly — no pandas dependency
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
        current_set = set(file_paths)
        indexed_set = set(indexed.keys())

        new_files: list[str] = []
        changed_files: list[str] = []

        for fp in file_paths:
            if fp not in indexed_set:
                new_files.append(fp)
            else:
                current_hash = _compute_file_hash(fp)
                if current_hash != indexed[fp]:
                    changed_files.append(fp)

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
        safe_path = file_path.replace('"', '\\"')
        self._table.delete(f'file_path = "{safe_path}"')
        after = self._table.count_rows()
        deleted = before - after
        if deleted > 0:
            logger.info("Deleted %d chunks for %s.", deleted, file_path)
        return deleted

    def add_chunks(self, chunks: list[Chunk], file_hash: str) -> int:
        """Embed and add chunks to the index.

        Args:
            chunks: List of Chunk objects to add.
            file_hash: SHA-256 hash of the source file.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Embed all chunk texts
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts, model_name=self._embedding_model)

        # Build records
        records = []
        for chunk, vector in zip(chunks, vectors):
            records.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "vector": vector.tolist(),
                "file_path": chunk.file_path,
                "file_type": chunk.file_type,
                "section_title": chunk.section_title,
                "section_level": chunk.section_level,
                "chunk_index": chunk.chunk_index,
                "file_hash": file_hash,
            })

        self._table.add(records)
        logger.info("Added %d chunks for %s.", len(records), chunks[0].file_path)
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

        # Snapshot existing rows for this file so we can restore on failure.
        backup_rows: list[dict] | None = None
        if self._table.count_rows() > 0:
            try:
                at = self._table.to_arrow()
                mask = pc.equal(at.column("file_path"), file_path)
                filtered = at.filter(mask)
                if filtered.num_rows > 0:
                    backup_rows = filtered.to_pylist()
            except Exception:
                backup_rows = None

        self.delete_file_chunks(file_path)
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
