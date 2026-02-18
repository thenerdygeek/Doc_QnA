"""Adaptive score filtering and dynamic top-k for retrieval.

After reranking normalizes all scores to [0, 1], these functions
analyse the score distribution to make smarter filtering decisions
than fixed thresholds.

Adaptive min_score
    Instead of a fixed threshold that either over- or under-filters
    depending on the query, compute a threshold from the actual score
    distribution (mean - std * factor, floored at a configurable minimum).

Dynamic top_k
    Instead of always returning a fixed number of chunks, detect natural
    "gaps" in the sorted scores — a large drop between consecutive scores
    signals the boundary between relevant and irrelevant results.
"""

from __future__ import annotations

import logging
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
_DEFAULT_STD_FACTOR = 1.0     # mean - (factor * std)
_DEFAULT_FLOOR = 0.15         # never drop below this
_DEFAULT_GAP_THRESHOLD = 0.10 # score drop that signals a boundary
_MIN_RESULTS = 2              # always return at least this many


def compute_adaptive_min_score(
    chunks: list[RetrievedChunk],
    std_factor: float = _DEFAULT_STD_FACTOR,
    floor: float = _DEFAULT_FLOOR,
) -> float:
    """Compute a dynamic min-score threshold from the score distribution.

    Algorithm: ``threshold = mean(scores) - std_factor * stdev(scores)``,
    clamped to ``[floor, 1.0]``.  This adapts to query difficulty:

    - **Easy queries** (high, tight scores): threshold is high, weak
      results filtered aggressively.
    - **Hard queries** (low, spread scores): threshold drops, keeping
      more candidates for the LLM to reason over.

    Args:
        chunks: Reranked chunks with normalized scores in [0, 1].
        std_factor: How many standard deviations below the mean.
        floor: Absolute minimum threshold.

    Returns:
        Adaptive min-score in [floor, 1.0].
    """
    if len(chunks) < 3:
        return floor

    scores = [c.score for c in chunks]
    mean = statistics.mean(scores)
    std = statistics.stdev(scores)

    threshold = mean - std_factor * std
    threshold = max(threshold, floor)
    threshold = min(threshold, 1.0)

    logger.debug(
        "Adaptive min_score: mean=%.3f std=%.3f → threshold=%.3f",
        mean, std, threshold,
    )
    return threshold


def compute_dynamic_top_k(
    chunks: list[RetrievedChunk],
    base_k: int = 5,
    max_k: int = 10,
    gap_threshold: float = _DEFAULT_GAP_THRESHOLD,
) -> int:
    """Determine how many results to include based on score gaps.

    Scans sorted scores top-down for a "natural break" — a gap larger
    than *gap_threshold* between consecutive scores.  If found, cuts at
    that boundary.  Otherwise returns *base_k*.

    The result is clamped to ``[min_results, max_k]``.

    Args:
        chunks: Reranked chunks sorted by score descending.
        base_k: Default number of results when no clear gap exists.
        max_k: Hard upper limit.
        gap_threshold: Minimum score drop to trigger a cut.

    Returns:
        Number of chunks to keep.
    """
    n = len(chunks)
    if n <= _MIN_RESULTS:
        return n

    scores = [c.score for c in chunks]

    # Scan for the first large gap (top-down, starting after min_results)
    for i in range(_MIN_RESULTS, min(n, max_k)):
        gap = scores[i - 1] - scores[i]
        if gap >= gap_threshold:
            logger.debug(
                "Dynamic top_k: gap=%.3f at position %d (scores %.3f→%.3f).",
                gap, i, scores[i - 1], scores[i],
            )
            return i

    # No clear gap — use base_k, bounded by available chunks
    k = min(base_k, n, max_k)
    logger.debug("Dynamic top_k: no gap found, using base_k=%d.", k)
    return k
