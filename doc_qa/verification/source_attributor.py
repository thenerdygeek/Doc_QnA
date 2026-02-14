"""Sentence-level source attribution using embedding similarity.

Splits an answer into sentences, embeds them alongside the retrieved chunks,
and assigns each sentence to the most similar source chunk.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from doc_qa.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton for the embedding model
# ---------------------------------------------------------------------------

_embedding_model = None
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Minimum cosine similarity to consider a match.
_ATTRIBUTION_THRESHOLD = 0.3


def _get_embedding_model():
    """Return a lazily-initialised fastembed TextEmbedding model."""
    global _embedding_model  # noqa: PLW0603
    if _embedding_model is None:
        from fastembed import TextEmbedding

        _embedding_model = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME)
        logger.info("Initialized fastembed model: %s", _EMBEDDING_MODEL_NAME)
    return _embedding_model


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Attribution:
    """Maps a single sentence to its most similar source chunk."""

    sentence: str
    source_index: int
    similarity: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Split on sentence-ending punctuation followed by whitespace or end-of-string.
_SENTENCE_RE = re.compile(r"(?<=[.!?\n])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using regex on sentence-ending punctuation.

    Keeps the punctuation attached to the preceding sentence.  Empty
    fragments are discarded.
    """
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (numpy-free)."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        norm_a += ai * ai
        norm_b += bi * bi

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the fastembed model."""
    model = _get_embedding_model()
    # fastembed returns a generator of numpy arrays; convert to plain lists.
    return [vec.tolist() for vec in model.embed(texts)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attribute_sources(
    answer_text: str,
    chunks: list[RetrievedChunk],
) -> list[Attribution]:
    """Attribute each sentence of *answer_text* to the most similar chunk.

    Args:
        answer_text: The generated answer to attribute.
        chunks: The retrieved source chunks to match against.

    Returns:
        A list of :class:`Attribution` objects, one per sentence.  Sentences
        that do not meet the similarity threshold are assigned
        ``source_index = -1``.
    """
    if not answer_text.strip() or not chunks:
        return []

    sentences = _split_sentences(answer_text)
    if not sentences:
        return []

    chunk_texts = [c.text for c in chunks]

    # Embed everything in one batch for efficiency.
    all_texts = sentences + chunk_texts
    all_vectors = _embed_texts(all_texts)

    sentence_vectors = all_vectors[: len(sentences)]
    chunk_vectors = all_vectors[len(sentences) :]

    attributions: list[Attribution] = []
    for s_idx, s_vec in enumerate(sentence_vectors):
        best_idx = -1
        best_sim = -1.0
        for c_idx, c_vec in enumerate(chunk_vectors):
            sim = _cosine_similarity(s_vec, c_vec)
            if sim > best_sim:
                best_sim = sim
                best_idx = c_idx

        # Apply threshold
        if best_sim < _ATTRIBUTION_THRESHOLD:
            best_idx = -1

        attributions.append(
            Attribution(
                sentence=sentences[s_idx],
                source_index=best_idx,
                similarity=max(best_sim, 0.0),
            )
        )

    logger.info(
        "Attributed %d sentence(s) to %d chunk(s); %d below threshold.",
        len(sentences),
        len(chunks),
        sum(1 for a in attributions if a.source_index == -1),
    )

    return attributions
