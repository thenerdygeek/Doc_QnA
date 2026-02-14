"""Tests for the LanceDB indexer."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from doc_qa.indexing.chunker import Chunk
from doc_qa.indexing.indexer import DocIndex


@pytest.fixture
def tmp_index(tmp_path: Path) -> DocIndex:
    """Create a temporary DocIndex."""
    return DocIndex(db_path=str(tmp_path / "test_db"))


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    """Create a sample markdown file."""
    p = tmp_path / "sample.md"
    p.write_text("# Sample\n\nThis is a test document.", encoding="utf-8")
    return p


def _make_chunks(file_path: str, n: int = 3) -> list[Chunk]:
    """Create n test chunks for a given file."""
    return [
        Chunk(
            chunk_id=f"{file_path}#{i}",
            text=f"Chunk {i} content for testing. " * 10,
            file_path=file_path,
            file_type="md",
            section_title=f"Section {i}",
            section_level=1,
            chunk_index=i,
        )
        for i in range(n)
    ]


class TestDocIndex:
    def test_create_empty_index(self, tmp_index: DocIndex) -> None:
        """New index should have zero rows."""
        assert tmp_index.count_rows() == 0
        assert tmp_index.count_files() == 0

    def test_add_chunks(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """Adding chunks should increase row count."""
        chunks = _make_chunks(str(sample_doc))
        n = tmp_index.add_chunks(chunks, file_hash="abc123")
        assert n == 3
        assert tmp_index.count_rows() == 3

    def test_upsert_file(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """Upserting should replace old chunks for the same file."""
        fp = str(sample_doc)

        # Insert initial chunks
        chunks1 = _make_chunks(fp, n=3)
        tmp_index.add_chunks(chunks1, file_hash="hash1")
        assert tmp_index.count_rows() == 3

        # Upsert with different chunks — should delete old and add new
        chunks2 = _make_chunks(fp, n=5)
        tmp_index.upsert_file(chunks2, fp)
        assert tmp_index.count_rows() == 5

    def test_delete_file_chunks(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """Deleting chunks for a file should remove only those chunks."""
        fp1 = str(sample_doc)
        fp2 = fp1 + ".other"

        # Create a second file
        Path(fp2).write_text("other content", encoding="utf-8")

        tmp_index.add_chunks(_make_chunks(fp1, 3), "h1")
        tmp_index.add_chunks(_make_chunks(fp2, 2), "h2")
        assert tmp_index.count_rows() == 5

        deleted = tmp_index.delete_file_chunks(fp1)
        assert deleted == 3
        assert tmp_index.count_rows() == 2

    def test_detect_changes_new_file(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """A file not in the index should be detected as new."""
        new, changed, deleted = tmp_index.detect_changes([str(sample_doc)])
        assert str(sample_doc) in new
        assert changed == []
        assert deleted == []

    def test_detect_changes_unchanged(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """A file with matching hash should not appear in changes."""
        fp = str(sample_doc)
        chunks = _make_chunks(fp)
        tmp_index.upsert_file(chunks, fp)

        new, changed, deleted = tmp_index.detect_changes([fp])
        assert new == []
        assert changed == []
        assert deleted == []

    def test_detect_changes_modified(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """A file with different hash should be detected as changed."""
        fp = str(sample_doc)
        chunks = _make_chunks(fp)
        tmp_index.add_chunks(chunks, file_hash="old_hash_that_wont_match")

        new, changed, deleted = tmp_index.detect_changes([fp])
        assert new == []
        assert fp in changed
        assert deleted == []

    def test_detect_changes_deleted(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """A file in index but not on disk should be detected as deleted."""
        fp = str(sample_doc)
        chunks = _make_chunks(fp)
        tmp_index.add_chunks(chunks, file_hash="hash1")

        # Pass empty file list — the indexed file should show as deleted
        new, changed, deleted = tmp_index.detect_changes([])
        assert new == []
        assert changed == []
        assert fp in deleted

    def test_stats(self, tmp_index: DocIndex, sample_doc: Path) -> None:
        """Stats should reflect current index state."""
        fp = str(sample_doc)
        tmp_index.add_chunks(_make_chunks(fp, 5), "h1")

        stats = tmp_index.stats()
        assert stats["total_chunks"] == 5
        assert stats["total_files"] == 1
        assert stats["embedding_dim"] == 384

    def test_reopen_persists(self, tmp_path: Path, sample_doc: Path) -> None:
        """Data should persist across DocIndex instances."""
        db_path = str(tmp_path / "persist_db")
        fp = str(sample_doc)

        # Create and populate
        idx1 = DocIndex(db_path=db_path)
        idx1.add_chunks(_make_chunks(fp, 4), "h1")
        assert idx1.count_rows() == 4

        # Reopen
        idx2 = DocIndex(db_path=db_path)
        assert idx2.count_rows() == 4
