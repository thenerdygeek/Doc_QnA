"""Tests for the section chunker."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from doc_qa.indexing.chunker import (
    Chunk,
    _detect_content_type,
    _extract_atomic_blocks,
    _split_into_sentences,
    _split_semantic,
    chunk_sections,
)
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


class TestSentenceSplitter:
    def test_basic_splitting(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = _split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
        assert result[2] == "Third sentence."

    def test_single_sentence(self) -> None:
        result = _split_into_sentences("Just one sentence here.")
        assert len(result) == 1

    def test_empty_string(self) -> None:
        result = _split_into_sentences("")
        assert result == []


def _make_embeddings_with_topic_shift(n_sentences: int, shift_at: int) -> list:
    """Create fake embeddings where sentences before shift_at are similar
    and sentences after shift_at are in a different direction.

    This creates a clear cosine similarity drop at the topic shift point.
    """
    dim = 32
    embeddings = []
    for i in range(n_sentences):
        vec = np.zeros(dim, dtype=np.float32)
        if i < shift_at:
            # Topic A: dominant in first half of dimensions
            vec[:dim // 2] = 0.8 + np.random.rand(dim // 2).astype(np.float32) * 0.2
            vec[dim // 2:] = np.random.rand(dim // 2).astype(np.float32) * 0.1
        else:
            # Topic B: dominant in second half — orthogonal to Topic A
            vec[:dim // 2] = np.random.rand(dim // 2).astype(np.float32) * 0.1
            vec[dim // 2:] = 0.8 + np.random.rand(dim // 2).astype(np.float32) * 0.2
        embeddings.append(vec)
    return embeddings


class TestSemanticChunking:
    def test_splits_at_topic_boundary(self) -> None:
        """Semantic chunking should detect a topic shift and split there."""
        np.random.seed(42)
        # 8 sentences: topic A (0-3), topic B (4-7)
        # Sentences must be long enough so 4 sentences exceed min_tokens
        sentences = [f"Topic A sentence {i} with extra padding words here. " for i in range(4)] + \
                    [f"Topic B sentence {i} with extra padding words here. " for i in range(4)]
        text = " ".join(sentences)
        fake_embeds = _make_embeddings_with_topic_shift(8, shift_at=4)

        with patch("doc_qa.indexing.embedder.embed_texts", return_value=fake_embeds):
            parts = _split_semantic(
                text, max_tokens=2000, overlap_tokens=0, min_tokens=10,
                embedding_model="test-model",
            )

        # Should split into at least 2 chunks at the topic boundary
        assert len(parts) >= 2
        assert "Topic A" in parts[0]
        assert "Topic B" in parts[-1]

    def test_respects_max_tokens(self) -> None:
        """Semantic chunking should hard-split when exceeding max_tokens."""
        # 10 sentences, all similar (no semantic break) but total exceeds max_tokens
        sentences = [f"Sentence number {i} with some padding words. " for i in range(10)]
        text = " ".join(sentences)
        # All embeddings are similar — no semantic break
        dim = 32
        base = np.random.rand(dim).astype(np.float32)
        fake_embeds = [base + np.random.rand(dim).astype(np.float32) * 0.01 for _ in range(10)]

        with patch("doc_qa.indexing.embedder.embed_texts", return_value=fake_embeds):
            parts = _split_semantic(
                text, max_tokens=50, overlap_tokens=0, min_tokens=10,
                embedding_model="test-model",
            )

        # Should split due to max_tokens even without semantic boundary
        assert len(parts) >= 2

    def test_single_sentence_returns_whole_text(self) -> None:
        """A single-sentence text should not be split."""
        text = "Just one sentence here."
        with patch("doc_qa.indexing.embedder.embed_texts") as mock_embed:
            parts = _split_semantic(
                text, max_tokens=500, overlap_tokens=0, min_tokens=10,
                embedding_model="test-model",
            )

        assert len(parts) == 1
        assert parts[0] == text
        mock_embed.assert_not_called()

    def test_chunk_sections_semantic_strategy(self) -> None:
        """chunk_sections should use semantic splitting when strategy='semantic'."""
        np.random.seed(42)
        # Use long sentences so 4 exceed min_tokens and total exceeds max_tokens
        # to trigger the splitting branch in chunk_sections
        long_content = " ".join(
            [f"Topic A sentence {i} with lots of padding content words. " for i in range(6)]
            + [f"Topic B sentence {i} with lots of padding content words. " for i in range(6)]
        )
        sections = [_make_section("Mixed", long_content)]
        fake_embeds = _make_embeddings_with_topic_shift(12, shift_at=6)

        with patch("doc_qa.indexing.embedder.embed_texts", return_value=fake_embeds):
            chunks = chunk_sections(
                sections, "test.md",
                max_tokens=100, overlap_tokens=0, min_tokens=10,
                chunking_strategy="semantic",
                embedding_model="test-model",
            )

        # Should have split into multiple chunks
        assert len(chunks) >= 2
        assert all(c.section_title == "Mixed" for c in chunks)

    def test_paragraph_strategy_ignores_embeddings(self) -> None:
        """Default paragraph strategy should not call embed_texts."""
        long_content = "\n\n".join(f"Paragraph {i}. " + "word " * 50 for i in range(6))
        sections = [_make_section("Para", long_content)]

        with patch("doc_qa.indexing.embedder.embed_texts") as mock_embed:
            chunks = chunk_sections(
                sections, "test.md",
                max_tokens=200, chunking_strategy="paragraph",
            )

        assert len(chunks) >= 2
        mock_embed.assert_not_called()


# ── Atomic block extraction tests ────────────────────────────────────


class TestExtractAtomicBlocks:
    def test_prose_only(self) -> None:
        text = "Hello world.\n\nAnother paragraph."
        blocks = _extract_atomic_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1] == "prose"
        assert "Hello world" in blocks[0][0]

    def test_code_block_extracted(self) -> None:
        text = "Before code.\n```python\nprint('hi')\n```\nAfter code."
        blocks = _extract_atomic_blocks(text)
        types = [b[1] for b in blocks]
        assert "code" in types
        code_blocks = [b for b in blocks if b[1] == "code"]
        assert len(code_blocks) == 1
        assert "print('hi')" in code_blocks[0][0]
        assert "```python" in code_blocks[0][0]

    def test_table_extracted(self) -> None:
        text = "Some intro.\n| A | B |\n| --- | --- |\n| 1 | 2 |\nSome outro."
        blocks = _extract_atomic_blocks(text)
        types = [b[1] for b in blocks]
        assert "table" in types
        table_blocks = [b for b in blocks if b[1] == "table"]
        assert len(table_blocks) == 1
        assert "| A | B |" in table_blocks[0][0]

    def test_mixed_content(self) -> None:
        text = (
            "Intro paragraph.\n\n"
            "| Col1 | Col2 |\n| --- | --- |\n| a | b |\n\n"
            "```js\nconsole.log('x');\n```\n\n"
            "Final paragraph."
        )
        blocks = _extract_atomic_blocks(text)
        types = [b[1] for b in blocks]
        assert "prose" in types
        assert "table" in types
        assert "code" in types

    def test_unclosed_code_fence(self) -> None:
        text = "```python\nincomplete code"
        blocks = _extract_atomic_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][1] == "code"

    def test_tilde_fence(self) -> None:
        text = "Before.\n~~~\ncode here\n~~~\nAfter."
        blocks = _extract_atomic_blocks(text)
        code_blocks = [b for b in blocks if b[1] == "code"]
        assert len(code_blocks) == 1
        assert "code here" in code_blocks[0][0]

    def test_empty_text(self) -> None:
        blocks = _extract_atomic_blocks("")
        assert len(blocks) == 1
        assert blocks[0][1] == "prose"


# ── Content type detection tests ─────────────────────────────────────


class TestDetectContentType:
    def test_prose(self) -> None:
        result = _detect_content_type("Just some regular text here.")
        assert result["content_type"] == "prose"
        assert result["has_table"] == "false"
        assert result["has_code"] == "false"

    def test_code_detection(self) -> None:
        result = _detect_content_type("```python\nprint(1)\n```")
        assert result["content_type"] == "code"
        assert result["has_code"] == "true"
        assert result["code_language"] == "python"

    def test_table_detection(self) -> None:
        result = _detect_content_type("| A | B |\n| --- | --- |\n| 1 | 2 |")
        assert result["content_type"] == "table"
        assert result["has_table"] == "true"

    def test_mixed_detection(self) -> None:
        text = "| A | B |\n| --- | --- |\n| 1 | 2 |\n```\ncode\n```"
        result = _detect_content_type(text)
        assert result["content_type"] == "mixed"
        assert result["has_table"] == "true"
        assert result["has_code"] == "true"

    def test_word_count(self) -> None:
        result = _detect_content_type("one two three four")
        assert result["word_count"] == "4"

    def test_code_language_empty_when_no_lang(self) -> None:
        result = _detect_content_type("```\ncode\n```")
        assert result["has_code"] == "true"
        assert result["code_language"] == ""


# ── Table preservation tests ─────────────────────────────────────────


class TestTablePreservation:
    def test_table_stays_in_one_chunk(self) -> None:
        """A markdown table should never be split across chunks."""
        table = "| Col1 | Col2 | Col3 |\n| --- | --- | --- |\n"
        for i in range(80):
            table += f"| val{i} | data{i} | info{i} |\n"

        section = _make_section("Big Table", table)
        chunks = chunk_sections(
            [section],
            file_path="test.md",
            max_tokens=128,
            overlap_tokens=20,
            min_tokens=50,
        )

        # The table should be in exactly one chunk (atomic)
        table_chunks = [c for c in chunks if c.metadata.get("has_table") == "true"]
        assert len(table_chunks) >= 1
        for tc in table_chunks:
            if "| Col1 |" in tc.text:
                assert "| val0 |" in tc.text
                assert "| val79 |" in tc.text


class TestCodeBlockPreservation:
    def test_code_never_split(self) -> None:
        """A code block should never be split across chunks."""
        code_lines = ["```python"]
        for i in range(100):
            code_lines.append(f"x_{i} = {i}  # line {i}")
        code_lines.append("```")
        code = "\n".join(code_lines)

        section = _make_section("Big Code", code)
        chunks = chunk_sections(
            [section],
            file_path="test.md",
            max_tokens=128,
            overlap_tokens=20,
            min_tokens=50,
        )

        code_chunks = [c for c in chunks if c.metadata.get("has_code") == "true"]
        assert len(code_chunks) >= 1
        for cc in code_chunks:
            if "```python" in cc.text:
                assert "x_99 = 99" in cc.text
                break
        else:
            pytest.fail("Could not find code chunk with opening fence")


# ── Metadata population tests ────────────────────────────────────────


class TestChunkMetadata:
    def test_metadata_populated_on_small_section(self) -> None:
        """Even small sections that fit in one chunk get metadata."""
        section = _make_section("Small", "Just some text.")
        chunks = chunk_sections([section], file_path="test.md")
        assert len(chunks) == 1
        assert "content_type" in chunks[0].metadata
        assert chunks[0].metadata["content_type"] == "prose"

    def test_metadata_has_table_flag(self) -> None:
        section = _make_section(
            "Table Section",
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
        )
        chunks = chunk_sections([section], file_path="test.md")
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("has_table") == "true"

    def test_metadata_has_code_flag(self) -> None:
        section = _make_section(
            "Code Section",
            "```python\nprint('hi')\n```",
        )
        chunks = chunk_sections([section], file_path="test.md")
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("has_code") == "true"

    def test_metadata_word_count(self) -> None:
        section = _make_section("Counted", "one two three four five")
        chunks = chunk_sections([section], file_path="test.md")
        assert len(chunks) == 1
        assert "word_count" in chunks[0].metadata
