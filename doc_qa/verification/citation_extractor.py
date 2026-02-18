"""Extract numbered citation references from LLM-generated answers.

Maps ``[1]``, ``[2]``, etc. in the answer text to the corresponding
source chunks from retrieval, producing a ``CitationMapping`` list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class CitationMapping:
    """A single citation reference linking answer text to a source chunk."""

    number: int
    chunk_id: str
    file_path: str
    section_title: str
    score: float


def extract_citations(
    answer_text: str,
    sources: list,
) -> list[CitationMapping]:
    """Extract numbered citations from ``answer_text`` and map to sources.

    Args:
        answer_text: The LLM-generated answer containing ``[1]``, ``[2]``, etc.
        sources: Ordered list of source chunks (must have ``chunk_id``,
            ``file_path``, ``section_title``, ``score`` attributes).

    Returns:
        De-duplicated list of ``CitationMapping`` for valid in-range references,
        ordered by citation number.
    """
    if not answer_text or not sources:
        return []

    matches = _CITATION_RE.findall(answer_text)
    if not matches:
        return []

    seen: set[int] = set()
    citations: list[CitationMapping] = []

    for raw_num in matches:
        num = int(raw_num)
        if num < 1 or num > len(sources):
            continue  # out of range
        if num in seen:
            continue  # already mapped
        seen.add(num)

        src = sources[num - 1]  # 1-indexed → 0-indexed
        citations.append(CitationMapping(
            number=num,
            chunk_id=getattr(src, "chunk_id", ""),
            file_path=getattr(src, "file_path", ""),
            section_title=getattr(src, "section_title", ""),
            score=round(getattr(src, "score", 0.0), 4),
        ))

    citations.sort(key=lambda c: c.number)
    return citations
