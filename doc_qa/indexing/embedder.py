"""FastEmbed wrapper for generating text embeddings."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Lazy singleton — model is loaded on first use
_model: object | None = None
_model_name: str = ""
_model_lock = threading.Lock()


def _get_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> object:
    """Get or create the FastEmbed TextEmbedding model (lazy singleton)."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None and _model_name == model_name:
            return _model
        from fastembed import TextEmbedding

        logger.info("Loading embedding model: %s", model_name)
        _model = TextEmbedding(model_name)
        _model_name = model_name
        logger.info("Embedding model loaded.")
        return _model


def get_embedding_dimension(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> int:
    """Return the embedding dimension for the given model.

    Known dimensions to avoid loading the model just for this:
    - all-MiniLM-L6-v2: 384
    - all-mpnet-base-v2: 768
    - nomic-embed-text-v1.5: 768
    """
    known = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "nomic-ai/nomic-embed-text-v1.5": 768,
    }
    if model_name in known:
        return known[model_name]
    # Fallback: embed a dummy string to get the dimension
    vecs = embed_texts(["test"], model_name=model_name)
    return len(vecs[0])


def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> list[NDArray[np.float32]]:
    """Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed.
        model_name: FastEmbed model identifier.
        batch_size: Batch size for embedding (larger = faster but more memory).

    Returns:
        List of numpy arrays, each of shape (dimension,).
    """
    if not texts:
        return []

    model = _get_model(model_name)
    # FastEmbed's embed() returns a generator of numpy arrays
    embeddings = list(model.embed(texts, batch_size=batch_size))

    logger.debug("Embedded %d texts (dim=%d).", len(embeddings), len(embeddings[0]))
    return embeddings


def embed_query(
    query: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> NDArray[np.float32]:
    """Embed a single query string. Uses query_embed for asymmetric models."""
    model = _get_model(model_name)

    # Some models have a separate query embedding method
    if hasattr(model, "query_embed"):
        result = list(model.query_embed(query))
        return result[0]

    # Fallback to regular embed
    result = list(model.embed([query]))
    return result[0]
