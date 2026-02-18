"""Tests for parent-child two-tier chunking."""
from __future__ import annotations

import pytest

from doc_qa.indexing.chunker import Chunk, chunk_sections_parent_child
from doc_qa.parsers.registry import ParsedSection


def _make_sections(texts: list[str], title: str = "Test") -> list[ParsedSection]:
    """Helper to build ParsedSection objects from plain text strings."""
    return [
        ParsedSection(title=title, level=1, content=t)
        for t in texts
    ]


class TestChunkSectionsParentChild:
    """Tests for chunk_sections_parent_child()."""

    def test_empty_sections(self) -> None:
        """Empty input produces no chunks."""
        assert chunk_sections_parent_child([], file_path="test.md") == []

    def test_child_count_gte_parent_count(self) -> None:
        """Child count should be >= parent count (each parent produces ≥1 child)."""
        # Long section that will produce multiple parents, each splitting into children
        long_text = ("This is a paragraph of test content. " * 100 + "\n\n") * 5
        sections = _make_sections([long_text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/big.md",
            parent_max_tokens=200,
            child_max_tokens=50,
        )
        assert len(children) > 0
        # Count unique parents
        parent_ids = {c.parent_chunk_id for c in children}
        assert len(children) >= len(parent_ids)

    def test_every_child_has_parent_id_and_text(self) -> None:
        """Every child must have valid parent_chunk_id and parent_text."""
        text = ("Sentence about testing. " * 60 + "\n\n") * 3
        sections = _make_sections([text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/test.md",
            parent_max_tokens=200,
            child_max_tokens=50,
        )
        for child in children:
            assert child.parent_chunk_id, f"Chunk {child.chunk_id} has no parent_chunk_id"
            assert child.parent_text, f"Chunk {child.chunk_id} has no parent_text"

    def test_parent_text_contains_child_text(self) -> None:
        """Each child's text should be a substring of its parent_text."""
        text = ("Word " * 80 + "\n\n") * 4
        sections = _make_sections([text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/test.md",
            parent_max_tokens=200,
            child_max_tokens=50,
        )
        for child in children:
            # child.text should appear in parent_text (allowing for whitespace)
            assert child.text.strip()[:50] in child.parent_text, (
                f"Child text not found in parent text for {child.chunk_id}"
            )

    def test_sequential_unique_chunk_ids(self) -> None:
        """All child chunk_ids must be unique and sequential."""
        text = ("Content here. " * 50 + "\n\n") * 3
        sections = _make_sections([text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/test.md",
            parent_max_tokens=200,
            child_max_tokens=50,
        )
        ids = [c.chunk_id for c in children]
        assert len(ids) == len(set(ids)), "chunk_ids are not unique"
        # Check sequential numbering
        for i, child in enumerate(children):
            assert child.chunk_index == i

    def test_self_parenting_for_small_sections(self) -> None:
        """A parent small enough to fit in child_max_tokens becomes self-parenting."""
        # Very short section that fits within child_max_tokens
        short_text = "A short section."
        sections = _make_sections([short_text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/small.md",
            parent_max_tokens=1024,
            child_max_tokens=256,
        )
        assert len(children) >= 1
        for child in children:
            # Self-parenting: text == parent_text
            assert child.text == child.parent_text

    def test_file_path_preserved(self) -> None:
        """File path is propagated to all children."""
        text = "Some content. " * 30
        sections = _make_sections([text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/readme.md",
        )
        for child in children:
            assert child.file_path == "docs/readme.md"

    def test_section_metadata_preserved(self) -> None:
        """Section title and level are propagated from parents to children."""
        sections = [
            ParsedSection(title="Getting Started", level=2, content="Some content. " * 50),
        ]
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/guide.md",
            parent_max_tokens=200,
            child_max_tokens=50,
        )
        for child in children:
            assert child.section_title == "Getting Started"
            assert child.section_level == 2

    def test_parent_id_format(self) -> None:
        """Parent chunk IDs should follow the expected format."""
        text = "Test content. " * 30
        sections = _make_sections([text])
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/test.md",
        )
        for child in children:
            assert child.parent_chunk_id.startswith("docs/test.md#parent_")

    def test_multiple_sections_produce_multiple_parents(self) -> None:
        """Multiple sections should produce children from different parents."""
        sections = [
            ParsedSection(title="Section A", level=1, content="Alpha content. " * 40),
            ParsedSection(title="Section B", level=1, content="Beta content. " * 40),
        ]
        children = chunk_sections_parent_child(
            sections,
            file_path="docs/multi.md",
            parent_max_tokens=100,
            child_max_tokens=50,
        )
        parent_ids = {c.parent_chunk_id for c in children}
        # With two sections and small max_tokens, we should get multiple parent groups
        assert len(parent_ids) >= 2
