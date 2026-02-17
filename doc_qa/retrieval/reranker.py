"""Cross-encoder reranker for second-pass scoring."""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce_model = None
_ce_model_name = ""
_ce_lock = threading.Lock()


def _get_cross_encoder(model_name: str):
    """Lazy-load cross-encoder model (singleton, thread-safe)."""
    global _ce_model, _ce_model_name
    if _ce_model is not None and _ce_model_name == model_name:
        return _ce_model
    with _ce_lock:
        if _ce_model is not None and _ce_model_name == model_name:
            return _ce_model
        from sentence_transformers import CrossEncoder

        cache_dir = os.environ.get(
            "FASTEMBED_CACHE_PATH",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "models"),
        )
        logger.info("Loading cross-encoder: %s", model_name)
        _ce_model = CrossEncoder(model_name, cache_folder=cache_dir)
        _ce_model_name = model_name
        return _ce_model


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    model_name: str = _DEFAULT_MODEL,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Rerank chunks using a cross-encoder model.

    Cross-encoders jointly attend to query+document pairs, providing
    much more accurate relevance scores than bi-encoder cosine similarity
    (typically 20-35% improvement in ranking quality).

    Args:
        query: User query string.
        chunks: Candidate chunks from the retriever.
        model_name: Cross-encoder model name.
        top_k: Max results to return (None = return all, reranked).

    Returns:
        Chunks reranked by cross-encoder score, highest first.
    """
    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    ce = _get_cross_encoder(model_name)
    pairs = [[query, c.text] for c in chunks]
    scores = ce.predict(pairs).tolist()

    scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    result: list[RetrievedChunk] = []
    for score, chunk in scored:
        chunk.score = score
        result.append(chunk)

    if top_k is not None:
        result = result[:top_k]

    # Normalize cross-encoder logits to (0, 1) via sigmoid.
    # Raw scores are unbounded (typically -3 to +10); downstream components
    # (confidence scoring, CRAG grading) expect [0, 1].
    from doc_qa.retrieval.score_normalizer import normalize_sigmoid
    normalize_sigmoid(result)

    logger.info(
        "Reranked %d chunks → top score=%.3f, bottom=%.3f",
        len(result),
        result[0].score if result else 0.0,
        result[-1].score if result else 0.0,
    )
    return result
