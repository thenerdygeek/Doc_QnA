"""Section-based chunking with overflow splitting and merging.

Supports two splitting strategies for oversized sections:
- ``"paragraph"`` (default): splits at ``\\n\\n`` boundaries.
- ``"semantic"``: splits at topic boundaries detected by cosine
  similarity drops between consecutive sentence embeddings.

Tables and code blocks are treated as atomic units: they are never
split across chunk boundaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from doc_qa.parsers.base import ParsedSection

logger = logging.getLogger(__name__)

# Regex for detecting table lines (pipe-delimited)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|")

# Regex for detecting code fence markers
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


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
    parent_chunk_id: str = ""
    parent_text: str = ""

    def estimate_tokens(self) -> int:
        """Approximate token count (~4 chars per token)."""
        return len(self.text) // 4


def _estimate_tokens(text: str) -> int:
    """Approximate token count (~4 chars per token for English)."""
    return len(text) // 4


def _extract_atomic_blocks(text: str) -> list[tuple[str, str]]:
    """Extract atomic blocks from text, preserving code blocks and tables intact.

    Uses a state machine to track fence open/close (code blocks between
    ``` markers) and contiguous pipe-line sequences (tables).

    Returns:
        List of ``(block_text, block_type)`` tuples where *block_type* is
        one of ``"prose"``, ``"code"``, or ``"table"``.
    """
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []

    prose_lines: list[str] = []
    code_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False
    fence_marker = ""  # the actual fence string (e.g. "```" or "~~~~")

    def _flush_prose() -> None:
        if prose_lines:
            blocks.append(("\n".join(prose_lines), "prose"))
            prose_lines.clear()

    def _flush_table() -> None:
        if table_lines:
            blocks.append(("\n".join(table_lines), "table"))
            table_lines.clear()

    for line in lines:
        if in_code:
            code_lines.append(line)
            stripped = line.strip()
            # Close fence: same or more backticks/tildes of the same char
            if stripped.startswith(fence_marker[0]) and len(stripped) >= len(fence_marker):
                # Ensure it is purely the fence char (no trailing content except whitespace)
                fence_chars = stripped.rstrip()
                if all(c == fence_marker[0] for c in fence_chars):
                    blocks.append(("\n".join(code_lines), "code"))
                    code_lines.clear()
                    in_code = False
            continue

        # Check for fence opening
        m = _FENCE_RE.match(line.strip())
        if m:
            _flush_table()
            _flush_prose()
            fence_marker = m.group(1)
            in_code = True
            code_lines.append(line)
            continue

        # Check for table line
        if _TABLE_LINE_RE.match(line):
            _flush_prose()
            table_lines.append(line)
            continue

        # Regular prose line
        if table_lines:
            _flush_table()
        prose_lines.append(line)

    # Flush remaining
    if in_code and code_lines:
        # Unclosed code fence — treat as code block anyway
        blocks.append(("\n".join(code_lines), "code"))
    elif code_lines:
        blocks.append(("\n".join(code_lines), "code"))
    _flush_table()
    _flush_prose()

    return blocks


def _apply_table_prefix(text: str, meta: dict[str, str]) -> str:
    """Prepend a table-header prefix for better embedding quality.

    When the chunk is detected as table content and has ``table_headers``
    metadata (set by the PDF parser), prepends a short description so the
    embedding model captures column semantics.
    """
    ct = meta.get("content_type", "")
    headers = meta.get("table_headers", "")
    if ct in ("table", "mixed") and headers:
        return f"Table with columns: {headers}.\n\n{text}"
    return text


def _detect_content_type(text: str) -> dict[str, str]:
    """Detect the content type of a text block.

    Returns:
        Dict with keys:
        - ``content_type``: ``"prose"``, ``"code"``, ``"table"``, or ``"mixed"``
        - ``has_table``: ``"true"`` or ``"false"``
        - ``has_code``: ``"true"`` or ``"false"``
        - ``code_language``: detected language from first fence or ``""``
        - ``word_count``: word count as string
    """
    has_table = False
    has_code = False
    code_language = ""

    for line in text.split("\n"):
        stripped = line.strip()
        if _TABLE_LINE_RE.match(stripped):
            has_table = True
        m = _FENCE_RE.match(stripped)
        if m:
            has_code = True
            if not code_language:
                # Language is any text after the fence marker on the same line
                after_fence = stripped[len(m.group(1)):].strip()
                if after_fence:
                    code_language = after_fence.split()[0]

    if has_table and has_code:
        content_type = "mixed"
    elif has_table:
        content_type = "table"
    elif has_code:
        content_type = "code"
    else:
        content_type = "prose"

    return {
        "content_type": content_type,
        "has_table": str(has_table).lower(),
        "has_code": str(has_code).lower(),
        "code_language": code_language,
        "word_count": str(len(text.split())),
    }


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


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using regex heuristic."""
    raw = _SENTENCE_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def _split_semantic(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    min_tokens: int,
    embedding_model: str,
) -> list[str]:
    """Split text at semantic boundaries detected by embedding similarity drops.

    1. Split text into sentences.
    2. Embed each sentence.
    3. Compute cosine similarity between consecutive sentence embeddings.
    4. Identify breakpoints where similarity drops below (mean - 1 std).
    5. Build chunks respecting *max_tokens*, splitting at semantic boundaries
       when the current chunk is already above *min_tokens*.
    """
    from doc_qa.indexing.embedder import embed_texts

    sentences = _split_into_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # Embed all sentences in one batch
    embeddings = embed_texts(sentences, model_name=embedding_model)
    if len(embeddings) < 2:
        return [text]

    # Cosine similarity between consecutive sentence embeddings
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
        similarities.append(float(np.dot(a, b) / denom))

    # Breakpoint threshold: 1 std below mean = likely topic shift
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    threshold = mean_sim - std_sim

    # Build chunks, splitting at semantic boundaries
    chunks: list[str] = []
    current_sentences: list[str] = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        stok = _estimate_tokens(sentence)

        # Hard split: would exceed max_tokens
        if current_tokens + stok > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Overlap: keep last sentences that fit in overlap budget
            overlap_sents: list[str] = []
            overlap_tok = 0
            for prev in reversed(current_sentences):
                pt = _estimate_tokens(prev)
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap_sents.insert(0, prev)
                overlap_tok += pt
            current_sentences = overlap_sents
            current_tokens = overlap_tok

        # Semantic split: similarity dip AND we already have enough content
        elif (
            i > 0
            and similarities[i - 1] < threshold
            and current_tokens >= min_tokens
            and current_sentences
        ):
            chunks.append(" ".join(current_sentences))
            overlap_sents = []
            overlap_tok = 0
            for prev in reversed(current_sentences):
                pt = _estimate_tokens(prev)
                if overlap_tok + pt > overlap_tokens:
                    break
                overlap_sents.insert(0, prev)
                overlap_tok += pt
            current_sentences = overlap_sents
            current_tokens = overlap_tok

        current_sentences.append(sentence)
        current_tokens += stok

    if current_sentences:
        chunks.append(" ".join(current_sentences))

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
    chunking_strategy: str = "paragraph",
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
) -> list[Chunk]:
    """Convert parsed sections into embeddable chunks.

    Strategy:
    1. Each section becomes a candidate chunk
    2. Sections > max_tokens: split using the chosen strategy with overlap
    3. Sections < min_tokens: merge with the next section
    4. Code blocks: keep intact (never split mid-code)

    Args:
        sections: Parsed sections from a document parser.
        file_path: Source file path for chunk_id generation.
        max_tokens: Maximum tokens per chunk.
        overlap_tokens: Token overlap between consecutive chunks.
        min_tokens: Minimum tokens — merge smaller sections.
        chunking_strategy: ``"paragraph"`` (fast, default) or
            ``"semantic"`` (embedding-based topic detection).
        embedding_model: Model name for semantic chunking embeddings.

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

    # Phase 2: Split oversized sections, preserve code blocks and tables
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in merged:
        full_text = section.full_text
        tokens = _estimate_tokens(full_text)

        if tokens <= max_tokens or _is_code_block(section.content):
            # Fits in one chunk or is a code block — keep intact
            meta = _detect_content_type(full_text)
            meta.update(section.metadata)
            chunks.append(
                Chunk(
                    chunk_id=f"{file_path}#{chunk_index}",
                    text=_apply_table_prefix(full_text, meta),
                    file_path=file_path,
                    file_type=file_type,
                    section_title=section.title,
                    section_level=section.level,
                    chunk_index=chunk_index,
                    metadata=meta,
                )
            )
            chunk_index += 1
        else:
            # Extract atomic blocks (code/table kept intact, prose split)
            atomic = _extract_atomic_blocks(full_text)
            for block_text, block_type in atomic:
                block_text_stripped = block_text.strip()
                if not block_text_stripped:
                    continue

                if block_type in ("code", "table"):
                    # Atomic: never split
                    meta = _detect_content_type(block_text_stripped)
                    meta.update(section.metadata)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{file_path}#{chunk_index}",
                            text=_apply_table_prefix(block_text_stripped, meta),
                            file_path=file_path,
                            file_type=file_type,
                            section_title=section.title,
                            section_level=section.level,
                            chunk_index=chunk_index,
                            metadata=meta,
                        )
                    )
                    chunk_index += 1
                else:
                    # Prose: split using the chosen strategy
                    if _estimate_tokens(block_text_stripped) <= max_tokens:
                        parts = [block_text_stripped]
                    elif chunking_strategy == "semantic":
                        parts = _split_semantic(
                            block_text_stripped, max_tokens, overlap_tokens,
                            min_tokens, embedding_model,
                        )
                    else:
                        parts = _split_at_paragraphs(
                            block_text_stripped, max_tokens, overlap_tokens,
                        )
                    for part in parts:
                        part_stripped = part.strip()
                        if not part_stripped:
                            continue
                        meta = _detect_content_type(part_stripped)
                        meta.update(section.metadata)
                        chunks.append(
                            Chunk(
                                chunk_id=f"{file_path}#{chunk_index}",
                                text=_apply_table_prefix(part_stripped, meta),
                                file_path=file_path,
                                file_type=file_type,
                                section_title=section.title,
                                section_level=section.level,
                                chunk_index=chunk_index,
                                metadata=meta,
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


def chunk_sections_parent_child(
    sections: list[ParsedSection],
    file_path: str,
    parent_max_tokens: int = 1024,
    child_max_tokens: int = 256,
    overlap_tokens: int = 50,
    min_tokens: int = 100,
    chunking_strategy: str = "paragraph",
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
) -> list[Chunk]:
    """Two-tier parent-child chunking for precision retrieval + full-context LLM.

    Phase 1: Create parent-sized chunks using standard ``chunk_sections``.
    Phase 2: Split each parent into child-sized pieces.
    Each child stores its parent's ID and full text for context expansion.

    If a parent is already small enough (≤ child_max_tokens), it becomes
    its own child (self-parenting).

    Args:
        sections: Parsed sections from a document parser.
        file_path: Source file path.
        parent_max_tokens: Maximum tokens for parent chunks.
        child_max_tokens: Maximum tokens for child chunks.
        overlap_tokens: Token overlap for splitting.
        min_tokens: Minimum tokens before merging.
        chunking_strategy: Splitting strategy for prose blocks.
        embedding_model: Model name for semantic chunking.

    Returns:
        List of child Chunks, each with parent_chunk_id and parent_text set.
    """
    if not sections:
        return []

    # Phase 1: Create parent-sized chunks
    parents = chunk_sections(
        sections,
        file_path=file_path,
        max_tokens=parent_max_tokens,
        overlap_tokens=overlap_tokens,
        min_tokens=min_tokens,
        chunking_strategy=chunking_strategy,
        embedding_model=embedding_model,
    )

    if not parents:
        return []

    # Phase 2: Split each parent into child-sized pieces
    children: list[Chunk] = []
    child_index = 0

    for parent in parents:
        parent_id = f"{file_path}#parent_{parent.chunk_index}"
        parent_text = parent.text
        parent_tokens = _estimate_tokens(parent_text)

        if parent_tokens <= child_max_tokens:
            # Self-parenting: parent is small enough to be its own child
            children.append(
                Chunk(
                    chunk_id=f"{file_path}#{child_index}",
                    text=parent_text,
                    file_path=parent.file_path,
                    file_type=parent.file_type,
                    section_title=parent.section_title,
                    section_level=parent.section_level,
                    chunk_index=child_index,
                    metadata=parent.metadata.copy(),
                    parent_chunk_id=parent_id,
                    parent_text=parent_text,
                )
            )
            child_index += 1
        else:
            # Split parent into child-sized pieces
            child_texts = _split_at_paragraphs(
                parent_text, child_max_tokens, overlap_tokens,
            )
            for child_text in child_texts:
                child_text_stripped = child_text.strip()
                if not child_text_stripped:
                    continue
                meta = _detect_content_type(child_text_stripped)
                meta.update(parent.metadata)
                children.append(
                    Chunk(
                        chunk_id=f"{file_path}#{child_index}",
                        text=child_text_stripped,
                        file_path=parent.file_path,
                        file_type=parent.file_type,
                        section_title=parent.section_title,
                        section_level=parent.section_level,
                        chunk_index=child_index,
                        metadata=meta,
                        parent_chunk_id=parent_id,
                        parent_text=parent_text,
                    )
                )
                child_index += 1

    logger.info(
        "Parent-child chunked %s: %d parents → %d children",
        file_path,
        len(parents),
        len(children),
    )
    return children
