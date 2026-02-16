"""Near-duplicate deduplication for version-aware retrieval.

When multiple versions of the same document exist in the index,
retrieved chunks may contain near-identical content from different
files. This module detects such near-duplicates and keeps only the
chunk from the most recently dated file.

This runs AFTER retrieval and BEFORE reranking, on the candidate
pool (typically 20 chunks) — so the pairwise comparison is cheap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# Default cosine similarity threshold for near-duplicate detection.
# 0.95 is high enough to catch true duplicates (same paragraph with
# minor formatting differences) without false positives on topically
# similar but distinct content.
DEDUP_THRESHOLD = 0.95


def deduplicate_near_duplicates(
    chunks: list[RetrievedChunk],
    threshold: float = DEDUP_THRESHOLD,
) -> list[RetrievedChunk]:
    """Remove near-duplicate chunks, keeping the one from the newest file.

    Two chunks are considered near-duplicates when:
    1. They come from DIFFERENT files (same-file duplicates are already
       handled by the file diversity cap)
    2. Their embedding cosine similarity exceeds ``threshold``

    When a near-duplicate pair is found, the chunk with the lower
    ``doc_date`` (older file) is removed. If both have ``doc_date == 0.0``
    (unknown), the one with the lower retrieval score is removed.

    Args:
        chunks: Candidate chunks from the retriever (with vectors loaded).
        threshold: Cosine similarity threshold for near-duplicate detection.

    Returns:
        Filtered list preserving original order, with duplicates removed.
    """
    if len(chunks) <= 1:
        return chunks

    # Build vector matrix from loaded vectors
    has_vectors = all(c.vector is not None for c in chunks)
    if not has_vectors:
        # No vectors available — skip dedup (shouldn't happen in normal flow)
        logger.debug("Skipping dedup: vectors not loaded on chunks.")
        return chunks

    n = len(chunks)
    vecs = np.array([c.vector for c in chunks], dtype=np.float32)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    vecs_normed = vecs / norms

    # Compute pairwise cosine similarity matrix (N×N)
    sim_matrix = vecs_normed @ vecs_normed.T

    # Find near-duplicate pairs from different files
    remove_indices: set[int] = set()

    for i in range(n):
        if i in remove_indices:
            continue
        for j in range(i + 1, n):
            if j in remove_indices:
                continue

            # Only consider pairs from different files
            if chunks[i].file_path == chunks[j].file_path:
                continue

            if sim_matrix[i, j] >= threshold:
                # Near-duplicate found — remove the older one
                loser = _pick_loser(chunks[i], chunks[j], i, j)
                remove_indices.add(loser)
                winner = i if loser == j else j
                logger.debug(
                    "Dedup: %.3f similarity — keeping %s (date=%.0f), "
                    "removing %s (date=%.0f)",
                    sim_matrix[i, j],
                    chunks[winner].file_path,
                    chunks[winner].doc_date,
                    chunks[loser].file_path,
                    chunks[loser].doc_date,
                )

    if remove_indices:
        logger.info(
            "Near-duplicate dedup: removed %d/%d chunks (threshold=%.2f).",
            len(remove_indices), n, threshold,
        )

    return [c for idx, c in enumerate(chunks) if idx not in remove_indices]


def _pick_loser(
    a: RetrievedChunk,
    b: RetrievedChunk,
    idx_a: int,
    idx_b: int,
) -> int:
    """Decide which chunk to remove from a near-duplicate pair.

    Priority:
    1. Remove the one with the older doc_date (lower timestamp)
    2. If dates are equal (both 0.0 = unknown), remove the one with lower score
    """
    if a.doc_date != b.doc_date:
        # Remove the older document's chunk
        return idx_a if a.doc_date < b.doc_date else idx_b

    # Dates equal (likely both 0.0) — keep the higher-scored chunk
    return idx_a if a.score < b.score else idx_b
