"""Document relevance grader for CRAG (Corrective Retrieval-Augmented Generation).

Grades retrieved chunks as RELEVANT, PARTIAL, or IRRELEVANT using a single
batched LLM call, then returns structured GradedChunk results.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from doc_qa.llm.prompt_templates import DOCUMENT_GRADING
from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Maximum characters per chunk included in the grading prompt.
_CHUNK_TRUNCATION = 500

# Pattern to parse a grading response line.
# Matches: "Chunk 1: RELEVANT — some reason" or "Chunk 2: PARTIAL - reason"
_GRADE_LINE_RE = re.compile(
    r"Chunk\s+(\d+)\s*:\s*(RELEVANT|PARTIAL|IRRELEVANT)\s*[—\-]\s*(.*)",
    re.IGNORECASE,
)

_VALID_GRADES = frozenset({"relevant", "partial", "irrelevant"})


@dataclass
class GradedChunk:
    """A retrieved chunk annotated with a relevance grade."""

    chunk: RetrievedChunk
    grade: str  # "relevant" | "partial" | "irrelevant"
    reasoning: str


def _format_chunks_for_grading(chunks: list[RetrievedChunk]) -> str:
    """Format chunks into a numbered list for the grading prompt.

    Each chunk is truncated to ``_CHUNK_TRUNCATION`` characters to keep the
    prompt compact while retaining enough content for the LLM to judge
    relevance.
    """
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = chunk.text[:_CHUNK_TRUNCATION]
        if len(chunk.text) > _CHUNK_TRUNCATION:
            text += "..."
        parts.append(f"[Chunk {idx}]: {text}")
    return "\n\n".join(parts)


def _parse_grading_response(
    response: str,
    chunks: list[RetrievedChunk],
) -> list[GradedChunk]:
    """Parse the LLM grading response into :class:`GradedChunk` objects.

    If parsing fails for any chunk (missing line, bad format), that chunk is
    conservatively assigned ``"relevant"`` so it is not discarded.

    Returns:
        A list of ``GradedChunk`` with the same length and order as *chunks*.
    """
    # Build a map: 1-based index -> (grade, reasoning)
    parsed: dict[int, tuple[str, str]] = {}
    for line in response.splitlines():
        m = _GRADE_LINE_RE.search(line)
        if m:
            idx = int(m.group(1))
            grade = m.group(2).lower()
            reasoning = m.group(3).strip()
            if grade in _VALID_GRADES:
                parsed[idx] = (grade, reasoning)

    graded: list[GradedChunk] = []
    for idx, chunk in enumerate(chunks, start=1):
        if idx in parsed:
            grade, reasoning = parsed[idx]
        else:
            grade = "relevant"
            reasoning = "Grade not parsed — kept as relevant (conservative fallback)."
            logger.debug(
                "No grade parsed for chunk %d (%s); defaulting to relevant.",
                idx,
                chunk.chunk_id,
            )
        graded.append(GradedChunk(chunk=chunk, grade=grade, reasoning=reasoning))

    return graded


async def grade_documents(
    query: str,
    chunks: list[RetrievedChunk],
    llm_backend,
) -> list[GradedChunk]:
    """Grade a list of retrieved chunks for relevance to *query*.

    All chunks are sent in a single batched LLM call for efficiency.  If the
    LLM response is completely unparseable, every chunk is conservatively kept
    as ``"relevant"``.

    Args:
        query: The user's original question.
        chunks: Retrieved document chunks to grade.
        llm_backend: An LLM backend instance exposing an ``ask`` method.

    Returns:
        A list of :class:`GradedChunk` in the same order as *chunks*.
    """
    if not chunks:
        return []

    formatted_chunks = _format_chunks_for_grading(chunks)
    prompt = DOCUMENT_GRADING.format(query=query, chunks=formatted_chunks)

    logger.info("Grading %d chunks for query: %.80s...", len(chunks), query)

    try:
        answer = await llm_backend.ask(
            question=prompt,
            context="",
        )
    except Exception:
        logger.warning(
            "LLM grading call failed — keeping all chunks as relevant.",
            exc_info=True,
        )
        return [
            GradedChunk(
                chunk=c,
                grade="relevant",
                reasoning="LLM grading failed — kept as relevant (fallback).",
            )
            for c in chunks
        ]

    if answer.error:
        logger.warning(
            "LLM returned error during grading (%s) — keeping all chunks.",
            answer.error,
        )
        return [
            GradedChunk(
                chunk=c,
                grade="relevant",
                reasoning=f"LLM error during grading: {answer.error}",
            )
            for c in chunks
        ]

    graded = _parse_grading_response(answer.text, chunks)

    relevant_count = sum(1 for g in graded if g.grade == "relevant")
    partial_count = sum(1 for g in graded if g.grade == "partial")
    irrelevant_count = sum(1 for g in graded if g.grade == "irrelevant")
    logger.info(
        "Grading complete: %d relevant, %d partial, %d irrelevant.",
        relevant_count,
        partial_count,
        irrelevant_count,
    )

    return graded
