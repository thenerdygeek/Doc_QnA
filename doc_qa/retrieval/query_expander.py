"""HyDE (Hypothetical Document Embeddings) and multi-query expansion.

Generates a hypothetical answer to the query using the LLM, then embeds
that answer instead of the raw question.  The hypothesis is typically
closer in embedding space to real relevant documents than the short
original query, improving recall by 10-15% on vague/short queries.

Also provides LLM-based query expansion: generate alternative phrasings
of a question so that retrieval can find documents that use different
terminology, then merge results with RRF.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np

from doc_qa.indexing.embedder import embed_query as _embed_query
from doc_qa.llm.prompt_templates import HYDE_GENERATION, QUERY_EXPANSION

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


async def generate_hypothetical_document(
    question: str,
    llm_backend,
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
) -> NDArray[np.float32]:
    """Generate a hypothetical answer and return its embedding.

    The LLM produces a plausible (but possibly inaccurate) answer.
    We don't care about factual accuracy — only that the generated
    text uses vocabulary and phrasing similar to actual documents,
    making the resulting embedding a better search probe.

    Args:
        question: The user's original query.
        llm_backend: Any LLM backend with an ``ask()`` method.
        embedding_model: Model to use for embedding the hypothesis.

    Returns:
        Embedding vector of the hypothetical document.
    """
    prompt = HYDE_GENERATION.format(question=question)
    response = await llm_backend.ask(question=prompt, context="")

    if response.error:
        raise RuntimeError(f"HyDE LLM error: {response.error}")
    if not response.text or not response.text.strip():
        raise RuntimeError("HyDE LLM returned empty response")

    hypothetical_doc = response.text

    logger.info(
        "HyDE generated %d-char hypothesis for query: %.60s...",
        len(hypothetical_doc),
        question,
    )

    return _embed_query(hypothetical_doc, model_name=embedding_model)


async def generate_combined_embedding(
    question: str,
    llm_backend,
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5",
    hyde_weight: float = 0.7,
) -> NDArray[np.float32]:
    """Generate a weighted combination of HyDE and original query embeddings.

    Combines the hypothetical document embedding with the original query
    embedding using a weighted average, then L2-normalizes the result.
    This preserves the query's intent signal while gaining HyDE's
    vocabulary-matching advantage.

    Falls back to the original query embedding on any LLM failure.

    Args:
        question: The user's original query.
        llm_backend: Any LLM backend with an ``ask()`` method.
        embedding_model: Model to use for embedding.
        hyde_weight: Weight for the HyDE embedding (0-1).  The original
            query gets weight ``1 - hyde_weight``.

    Returns:
        L2-normalized combined embedding vector.
    """
    original_vec = _embed_query(question, model_name=embedding_model)

    try:
        hyde_vec = await generate_hypothetical_document(
            question, llm_backend, embedding_model=embedding_model,
        )
    except Exception as exc:
        logger.warning("HyDE generation failed, using original embedding: %s", exc)
        return original_vec

    combined = hyde_weight * hyde_vec + (1.0 - hyde_weight) * original_vec
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm

    logger.info("HyDE combined embedding (weight=%.2f).", hyde_weight)
    return combined


async def expand_query(
    question: str,
    llm_backend,
    n_variants: int = 3,
) -> list[str]:
    """Generate alternative phrasings of the question via LLM.

    Returns ``[original, variant1, variant2, ...]``.  Falls back to
    ``[original]`` on any failure so the pipeline always has at least
    the user's original query to work with.

    Args:
        question: The user's original query.
        llm_backend: Any LLM backend with an ``ask()`` method.
        n_variants: Maximum number of alternative phrasings to generate
            (in addition to the original).

    Returns:
        A list starting with the original question followed by up to
        *n_variants* alternative phrasings.
    """
    try:
        prompt = QUERY_EXPANSION.format(n_variants=n_variants, question=question)
        response = await llm_backend.ask(question=prompt, context="")
        raw_text = response.text.strip()

        variants: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading numbered list markers like "1. ", "2) ", etc.
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and cleaned != question:
                variants.append(cleaned)

        # Cap at n_variants alternatives
        variants = variants[:n_variants]

        if variants:
            logger.info(
                "Query expansion produced %d variant(s) for: %.60s...",
                len(variants),
                question,
            )

        return [question] + variants
    except Exception as exc:
        logger.warning("Query expansion failed, returning original only: %s", exc)
        return [question]
