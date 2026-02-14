"""CRAG query rewriting and corrective re-retrieval logic.

Implements the Corrective Retrieval-Augmented Generation flow:
grade initial results, rewrite the query when too many chunks are irrelevant,
re-retrieve, and merge the candidate pools.
"""

from __future__ import annotations

import logging

from doc_qa.llm.prompt_templates import QUERY_REWRITE
from doc_qa.retrieval.retriever import HybridRetriever, RetrievedChunk
from doc_qa.verification.grader import GradedChunk, grade_documents

logger = logging.getLogger(__name__)


def should_rewrite(
    graded: list[GradedChunk],
    threshold: float = 0.5,
) -> bool:
    """Decide whether the query should be rewritten.

    Returns ``True`` if more than *threshold* fraction of graded chunks are
    ``"irrelevant"``.  An empty list returns ``False`` (nothing to rewrite).
    """
    if not graded:
        return False

    irrelevant_count = sum(1 for g in graded if g.grade == "irrelevant")
    fraction = irrelevant_count / len(graded)
    logger.debug(
        "Irrelevant fraction: %.2f (%d/%d), threshold: %.2f",
        fraction,
        irrelevant_count,
        len(graded),
        threshold,
    )
    return fraction > threshold


async def rewrite_query(
    original_query: str,
    graded_chunks: list[GradedChunk],
    llm_backend,
) -> str:
    """Use the LLM to rewrite a query for better retrieval.

    Partial-match chunks are included as context so the LLM can pick up on
    relevant terminology that may improve the rewritten query.

    Args:
        original_query: The user's original question.
        graded_chunks: Previously graded chunks (partial matches used as hints).
        llm_backend: An LLM backend instance exposing an ``ask`` method.

    Returns:
        The rewritten query string.  Falls back to the original query on error.
    """
    # Collect partial-match context for the rewrite prompt.
    partial_texts: list[str] = []
    for gc in graded_chunks:
        if gc.grade == "partial":
            partial_texts.append(gc.chunk.text[:300])

    partial_context = (
        "\n---\n".join(partial_texts) if partial_texts else "(no partial matches)"
    )

    prompt = QUERY_REWRITE.format(
        original_query=original_query,
        partial_context=partial_context,
    )

    logger.info("Rewriting query: %.80s...", original_query)

    try:
        answer = await llm_backend.ask(question=prompt, context="")
    except Exception:
        logger.warning(
            "LLM query-rewrite failed — using original query.",
            exc_info=True,
        )
        return original_query

    if answer.error or not answer.text.strip():
        logger.warning(
            "Query rewrite returned error or empty — using original query."
        )
        return original_query

    rewritten = answer.text.strip()
    logger.info("Rewritten query: %.120s", rewritten)
    return rewritten


def merge_candidates(
    graded: list[GradedChunk],
    new_chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge original RELEVANT chunks with newly retrieved chunks.

    Original chunks graded as ``"relevant"`` are kept and placed first.
    New chunks are appended, skipping duplicates (by ``chunk_id``).

    Args:
        graded: Graded original chunks.
        new_chunks: Freshly retrieved chunks from the rewritten query.

    Returns:
        Deduplicated list of :class:`RetrievedChunk`.
    """
    seen_ids: set[str] = set()
    merged: list[RetrievedChunk] = []

    # Keep original relevant chunks first (they are known-good).
    for gc in graded:
        if gc.grade == "relevant" and gc.chunk.chunk_id not in seen_ids:
            seen_ids.add(gc.chunk.chunk_id)
            merged.append(gc.chunk)

    # Append new results, deduplicating.
    for chunk in new_chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            merged.append(chunk)

    logger.debug(
        "Merged candidates: %d kept-relevant + %d new = %d total (deduped).",
        sum(1 for g in graded if g.grade == "relevant"),
        len(new_chunks),
        len(merged),
    )
    return merged


async def corrective_retrieve(
    query: str,
    initial_chunks: list[RetrievedChunk],
    llm_backend,
    retriever: HybridRetriever,
    max_rewrites: int = 2,
    candidate_pool: int = 20,
    min_score: float = 0.3,
) -> tuple[list[RetrievedChunk], bool]:
    """Full CRAG flow: grade, optionally rewrite, re-retrieve, and merge.

    1. Grade the initial chunks for relevance.
    2. If too many are irrelevant, rewrite the query and re-retrieve.
    3. Merge the original relevant chunks with the new results.
    4. Repeat up to *max_rewrites* times if still insufficient.

    Args:
        query: The user's original question.
        initial_chunks: Chunks from the initial retrieval.
        llm_backend: An LLM backend instance exposing an ``ask`` method.
        retriever: A :class:`HybridRetriever` for re-retrieval.
        max_rewrites: Maximum number of rewrite-and-retrieve cycles.
        candidate_pool: Number of candidates to fetch per retrieval.
        min_score: Minimum score threshold for retrieval.

    Returns:
        A tuple of ``(final_chunks, was_rewritten)`` where *was_rewritten*
        indicates whether at least one query rewrite occurred.
    """
    if not initial_chunks:
        return [], False

    current_query = query
    current_chunks = initial_chunks
    was_rewritten = False

    for iteration in range(max_rewrites):
        # Step 1: Grade current chunks.
        graded = await grade_documents(current_query, current_chunks, llm_backend)

        # Step 2: Decide whether to rewrite.
        if not should_rewrite(graded):
            # Good enough — return only relevant + partial chunks, drop irrelevant.
            final = [
                gc.chunk for gc in graded if gc.grade in ("relevant", "partial")
            ]
            logger.info(
                "CRAG iteration %d: no rewrite needed, returning %d chunks.",
                iteration + 1,
                len(final),
            )
            return final if final else current_chunks, was_rewritten

        # Step 3: Rewrite and re-retrieve.
        logger.info(
            "CRAG iteration %d: rewriting query (too many irrelevant chunks).",
            iteration + 1,
        )
        rewritten = await rewrite_query(current_query, graded, llm_backend)
        was_rewritten = True

        new_chunks = retriever.search(
            query=rewritten,
            top_k=candidate_pool,
            min_score=min_score,
        )

        # Step 4: Merge relevant originals with new results.
        current_chunks = merge_candidates(graded, new_chunks)
        current_query = rewritten

    # Exhausted all rewrites — do one final grading pass to filter.
    graded = await grade_documents(current_query, current_chunks, llm_backend)
    final = [gc.chunk for gc in graded if gc.grade in ("relevant", "partial")]
    logger.info(
        "CRAG exhausted %d rewrites, returning %d chunks.",
        max_rewrites,
        len(final),
    )
    return final if final else current_chunks, was_rewritten
