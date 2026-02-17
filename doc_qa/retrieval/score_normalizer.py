"""Score normalization for heterogeneous retrieval pipelines.

Different retrieval/reranking stages produce scores on incompatible scales:
  - RRF (Reciprocal Rank Fusion): rank-based, typically 0.01–0.03
  - Cosine similarity: [-1, 1], often [0, 1] for normalized embeddings
  - Cross-encoder logits: unbounded, typically -3 to +10

This module normalizes all scores to a common [0, 1] range so that
downstream components (confidence scoring, CRAG grading, min_score
filtering) work correctly.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc_qa.retrieval.retriever import RetrievedChunk


def normalize_min_max(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Normalize chunk scores to [0, 1] via min-max scaling.

    Best for bounded, rank-based scores (RRF, native hybrid, cosine).
    With a single chunk the score is set to 1.0.  With identical scores
    all chunks receive 1.0.

    Args:
        chunks: Chunks with raw scores (modified in-place).

    Returns:
        The same list with ``score`` fields updated.
    """
    if not chunks:
        return chunks

    scores = [c.score for c in chunks]
    lo = min(scores)
    hi = max(scores)

    if hi == lo:
        for c in chunks:
            c.score = 1.0
    else:
        span = hi - lo
        for c in chunks:
            c.score = (c.score - lo) / span

    return chunks


def normalize_sigmoid(
    chunks: list[RetrievedChunk],
    shift: float = 0.0,
    scale: float = 1.0,
) -> list[RetrievedChunk]:
    """Normalize chunk scores to (0, 1) via sigmoid transformation.

    Best for unbounded scores (cross-encoder logits).

    The transformation is::

        normalized = 1 / (1 + exp(-(score - shift) * scale))

    *shift* centers the sigmoid (scores below shift map to <0.5),
    *scale* controls steepness (higher = sharper transition).

    Args:
        chunks: Chunks with raw logit scores (modified in-place).
        shift: Centering point for the sigmoid.
        scale: Steepness multiplier.

    Returns:
        The same list with ``score`` fields updated.
    """
    for c in chunks:
        c.score = 1.0 / (1.0 + math.exp(-(c.score - shift) * scale))
    return chunks


def filter_by_score(
    chunks: list[RetrievedChunk],
    min_score: float,
) -> list[RetrievedChunk]:
    """Keep only chunks whose normalized score meets *min_score*.

    Should be called **after** normalization so the threshold operates
    on a [0, 1] scale.

    Args:
        chunks: Chunks with normalized scores.
        min_score: Minimum score threshold (0–1).

    Returns:
        Filtered list preserving original order.
    """
    return [c for c in chunks if c.score >= min_score]
