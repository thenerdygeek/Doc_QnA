"""Tests for the section chunker."""

from __future__ import annotations

import pytest

from doc_qa.indexing.chunker import Chunk, chunk_sections
from doc_qa.parsers.base import ParsedSection


def _make_section(
    title: str = "Test",
    content: str = "Content",
    level: int = 1,
) -> ParsedSection:
    return ParsedSection(
        title=title,
        content=content,
        level=level,
        file_path="test.md",
        file_type="md",
    )


class TestChunker:
    def test_single_small_section(self) -> None:
        """A small section should produce exactly one chunk."""
        sections = [_make_section("Intro", "Short text.")]
        chunks = chunk_sections(sections, "test.md")

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "test.md#0"
        assert chunks[0].section_title == "Intro"
        assert "Intro" in chunks[0].text
        assert "Short text" in chunks[0].text

    def test_chunk_ids_are_sequential(self) -> None:
        """Chunk IDs should be file_path#0, file_path#1, etc."""
        # Each section must be >= min_tokens (100) to avoid merging
        sections = [
            _make_section("A", "word " * 120),
            _make_section("B", "word " * 120),
            _make_section("C", "word " * 120),
        ]
        chunks = chunk_sections(sections, "doc.md")

        assert [c.chunk_id for c in chunks] == ["doc.md#0", "doc.md#1", "doc.md#2"]
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_small_sections_merged(self) -> None:
        """Sections below min_tokens should be merged with the next."""
        sections = [
            _make_section("A", "x" * 20),  # ~5 tokens — below min
            _make_section("B", "y" * 20),  # ~5 tokens — below min
        ]
        chunks = chunk_sections(sections, "test.md", min_tokens=50)

        # Both should be merged into one chunk
        assert len(chunks) == 1
        assert "A" in chunks[0].text
        assert "x" * 20 in chunks[0].text
        assert "y" * 20 in chunks[0].text

    def test_large_section_split(self) -> None:
        """A section exceeding max_tokens should be split."""
        # 600 tokens worth of text (~2400 chars)
        long_content = "\n\n".join(f"Paragraph {i}. " + "word " * 50 for i in range(10))
        sections = [_make_section("Big Section", long_content)]
        chunks = chunk_sections(sections, "test.md", max_tokens=200)

        assert len(chunks) > 1
        # All chunks should have the same section title
        assert all(c.section_title == "Big Section" for c in chunks)
        # All chunks should have sequential IDs
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_code_block_kept_intact(self) -> None:
        """Code blocks should not be split even if they exceed max_tokens."""
        code = "```java\n" + "int x = 1;\n" * 200 + "```"
        sections = [_make_section("Code", code)]
        chunks = chunk_sections(sections, "test.md", max_tokens=50)

        # Should be one chunk despite being large (code block preserved)
        assert len(chunks) == 1
        assert "```java" in chunks[0].text

    def test_empty_sections_produce_no_chunks(self) -> None:
        """Empty section list should produce no chunks."""
        chunks = chunk_sections([], "test.md")
        assert chunks == []

    def test_file_type_preserved(self) -> None:
        """Chunk file_type should match the source section."""
        sections = [_make_section("A", "content")]
        chunks = chunk_sections(sections, "test.md")
        assert chunks[0].file_type == "md"

    def test_estimate_tokens(self) -> None:
        """Chunk token estimation should work."""
        sections = [_make_section("Title", "a" * 400)]
        chunks = chunk_sections(sections, "test.md")
        assert chunks[0].estimate_tokens() > 0

    def test_overlap_present_when_splitting(self) -> None:
        """When a section is split, consecutive chunks should have overlapping content."""
        # Create distinct paragraphs
        paragraphs = [f"Paragraph {i}: " + "word " * 40 for i in range(8)]
        content = "\n\n".join(paragraphs)
        sections = [_make_section("Overlap Test", content)]
        chunks = chunk_sections(sections, "test.md", max_tokens=150, overlap_tokens=30)

        assert len(chunks) >= 2
        # Check that some text from end of chunk N appears in chunk N+1
        for i in range(len(chunks) - 1):
            # Last paragraph of chunk i should appear in chunk i+1
            # (due to overlap)
            chunk_i_lines = chunks[i].text.strip().split("\n")
            chunk_next_text = chunks[i + 1].text
            # At least some content should overlap
            last_line = chunk_i_lines[-1].strip()
            if last_line:
                assert (
                    last_line in chunk_next_text
                    or any(word in chunk_next_text for word in last_line.split()[:3])
                ), f"No overlap found between chunk {i} and {i + 1}"


class TestChunkDataclass:
    def test_chunk_fields(self) -> None:
        chunk = Chunk(
            chunk_id="test.md#0",
            text="Hello world",
            file_path="test.md",
            file_type="md",
            section_title="Greeting",
            section_level=1,
            chunk_index=0,
        )
        assert chunk.chunk_id == "test.md#0"
        assert chunk.file_path == "test.md"
        assert chunk.section_title == "Greeting"
        assert chunk.estimate_tokens() == len("Hello world") // 4
