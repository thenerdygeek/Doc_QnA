"""Active learning: boost/penalize chunk scores based on user feedback."""

from __future__ import annotations

import logging

from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def apply_chunk_feedback_boost(
    chunks: list[RetrievedChunk],
    feedback_scores: dict[str, float],
    max_boost: float = 0.10,
) -> list[RetrievedChunk]:
    """Adjust chunk scores based on accumulated user feedback.

    Each chunk's score is shifted by ``feedback_score * max_boost``
    where ``feedback_score`` is in [-1.0, 1.0] (from
    ``get_chunk_feedback_scores``).

    Scores are clamped to [0, 1] and the list is re-sorted descending.

    Args:
        chunks: Retrieved chunks with scores to adjust (modified in-place).
        feedback_scores: Mapping of chunk_id → score in [-1, 1].
        max_boost: Maximum absolute score shift.  Defaults to 0.10 to
            prevent feedback from dominating retrieval quality.

    Returns:
        The same list, re-sorted by adjusted score descending.
    """
    if not chunks or not feedback_scores:
        return chunks

    adjusted = 0
    for chunk in chunks:
        fb = feedback_scores.get(chunk.chunk_id)
        if fb is not None:
            chunk.score += fb * max_boost
            chunk.score = max(0.0, min(1.0, chunk.score))
            adjusted += 1

    if adjusted:
        chunks.sort(key=lambda c: c.score, reverse=True)
        logger.info("Feedback boost: adjusted %d/%d chunk scores.", adjusted, len(chunks))

    return chunks
