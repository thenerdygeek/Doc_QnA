"""Reranker — re-scores chunks using embedding similarity.

Uses bi-encoder cosine similarity between query and chunk embeddings
for a second-pass scoring refinement. This is lighter than a true
cross-encoder but still improves result quality by re-scoring all
candidates with a unified metric.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from doc_qa.indexing.embedder import embed_query, embed_texts

if TYPE_CHECKING:
    from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks by computing fresh cosine similarity scores.

    The retriever's initial scores may mix distance metrics (BM25 vs cosine).
    This reranker normalizes by computing cosine similarity for all chunks
    against the query embedding in a single batch.

    Args:
        query: User query string.
        chunks: Candidate chunks from the retriever.
        model_name: Embedding model name.
        top_k: Max results to return (None = return all, reranked).

    Returns:
        Chunks reranked by cosine similarity, highest first.
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    # Use cached vectors if available (avoid re-embedding)
    if all(c.vector is not None for c in chunks):
        query_vec = embed_query(query, model_name=model_name)
        chunk_vecs = [np.array(c.vector, dtype=np.float32) for c in chunks]
    else:
        query_vec = embed_query(query, model_name=model_name)
        chunk_vecs = embed_texts([c.text for c in chunks], model_name=model_name)

    # Compute cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    scores: list[float] = []
    for vec in chunk_vecs:
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
        scores.append(float(np.dot(query_norm, vec_norm)))

    # Pair scores with chunks and sort
    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    result: list[RetrievedChunk] = []
    for score, chunk in scored:
        chunk.score = score
        result.append(chunk)

    if top_k is not None:
        result = result[:top_k]

    logger.info(
        "Reranked %d chunks → top score=%.3f, bottom=%.3f",
        len(result),
        result[0].score if result else 0.0,
        result[-1].score if result else 0.0,
    )
    return result
