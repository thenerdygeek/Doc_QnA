"""Section-based chunking with overflow splitting and merging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from doc_qa.parsers.base import ParsedSection

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""

    chunk_id: str  # "file_path#chunk_index"
    text: str
    file_path: str
    file_type: str
    section_title: str
    section_level: int
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)

    def estimate_tokens(self) -> int:
        """Approximate token count (~4 chars per token)."""
        return len(self.text) // 4


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token for English)."""
    return len(text) // 4


def _split_at_paragraphs(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text at paragraph boundaries with overlap.

    Tries to split at double-newlines (paragraph boundaries).
    Falls back to single-newline splits if paragraphs are too large.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _estimate_tokens(para)

        # If a single paragraph exceeds max, split it further
        if para_tokens > max_tokens:
            # Flush current
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            # Split long paragraph at sentence boundaries or hard limit
            lines = para.split("\n")
            sub_current: list[str] = []
            sub_tokens = 0
            for line in lines:
                line_tokens = _estimate_tokens(line)
                if sub_tokens + line_tokens > max_tokens and sub_current:
                    chunks.append("\n".join(sub_current))
                    # Overlap: keep last portion
                    overlap_chars = overlap_tokens * 4
                    last_text = "\n".join(sub_current)
                    if len(last_text) > overlap_chars:
                        overlap_text = last_text[-overlap_chars:]
                        # Trim to word boundary — find first space
                        space_idx = overlap_text.find(" ")
                        if space_idx != -1 and space_idx < len(overlap_text) - 1:
                            overlap_text = overlap_text[space_idx + 1:]
                        sub_current = [overlap_text]
                        sub_tokens = _estimate_tokens(overlap_text)
                    else:
                        sub_current = []
                        sub_tokens = 0
                sub_current.append(line)
                sub_tokens += line_tokens
            if sub_current:
                chunks.append("\n".join(sub_current))
            continue

        # Check if adding this paragraph exceeds limit
        if current_tokens + para_tokens > max_tokens and current:
            chunks.append("\n\n".join(current))

            # Overlap: keep the last paragraph(s) that fit within overlap budget
            overlap_paras: list[str] = []
            overlap_count = 0
            for prev_para in reversed(current):
                prev_tokens = _estimate_tokens(prev_para)
                if overlap_count + prev_tokens > overlap_tokens:
                    break
                overlap_paras.insert(0, prev_para)
                overlap_count += prev_tokens

            current = overlap_paras
            current_tokens = overlap_count

        current.append(para)
        current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _is_code_block(text: str) -> bool:
    """Check if the text is primarily a code block."""
    stripped = text.strip()
    return stripped.startswith("```") and stripped.endswith("```")


def chunk_sections(
    sections: list[ParsedSection],
    file_path: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    min_tokens: int = 100,
) -> list[Chunk]:
    """Convert parsed sections into embeddable chunks.

    Strategy:
    1. Each section becomes a candidate chunk
    2. Sections > max_tokens: split at paragraph boundaries with overlap
    3. Sections < min_tokens: merge with the next section
    4. Code blocks: keep intact (never split mid-code)

    Args:
        sections: Parsed sections from a document parser.
        file_path: Source file path for chunk_id generation.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.
        min_tokens: Minimum tokens — merge smaller sections.

    Returns:
        List of Chunk objects ready for embedding.
    """
    if not sections:
        return []

    # Determine file_type from first section
    file_type = sections[0].file_type if sections else ""

    # Phase 1: Merge small sections
    merged: list[ParsedSection] = []
    pending: ParsedSection | None = None

    for section in sections:
        if pending is not None:
            combined_tokens = _estimate_tokens(pending.full_text) + _estimate_tokens(
                section.full_text
            )
            if _estimate_tokens(pending.full_text) < min_tokens and combined_tokens <= max_tokens:
                # Merge into pending
                pending = ParsedSection(
                    title=pending.title,
                    content=pending.content + "\n\n" + section.full_text,
                    level=pending.level,
                    file_path=pending.file_path,
                    file_type=pending.file_type,
                    metadata=pending.metadata,
                )
                continue
            else:
                merged.append(pending)
                pending = section
                continue
        pending = section

    if pending is not None:
        merged.append(pending)

    # Phase 2: Split oversized sections, preserve code blocks
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in merged:
        full_text = section.full_text
        tokens = _estimate_tokens(full_text)

        if tokens <= max_tokens or _is_code_block(section.content):
            # Fits in one chunk or is a code block — keep intact
            chunks.append(
                Chunk(
                    chunk_id=f"{file_path}#{chunk_index}",
                    text=full_text,
                    file_path=file_path,
                    file_type=file_type,
                    section_title=section.title,
                    section_level=section.level,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
        else:
            # Split at paragraph boundaries
            parts = _split_at_paragraphs(full_text, max_tokens, overlap_tokens)
            for part in parts:
                chunks.append(
                    Chunk(
                        chunk_id=f"{file_path}#{chunk_index}",
                        text=part,
                        file_path=file_path,
                        file_type=file_type,
                        section_title=section.title,
                        section_level=section.level,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

    logger.info(
        "Chunked %s: %d sections → %d chunks",
        file_path,
        len(sections),
        len(chunks),
    )
    return chunks
