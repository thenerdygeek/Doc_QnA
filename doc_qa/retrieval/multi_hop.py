"""Multi-hop reasoning for complex cross-document questions.

After an initial answer, asks the LLM to detect knowledge gaps, generates
follow-up sub-queries, retrieves additional context, and merges results.
Max hops is configurable (default 2) to bound latency.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_QUERY_LINE_RE = re.compile(r"^\d+\.\s*(.+)", re.MULTILINE)


async def detect_knowledge_gaps(
    question: str,
    initial_answer: str,
    context_preview: str,
    llm_backend,
) -> list[str]:
    """Ask the LLM to identify unanswered aspects and return follow-up sub-queries.

    Returns:
        List of follow-up query strings (max 3), or empty if answer is complete.
    """
    from doc_qa.llm.prompt_templates import MULTI_HOP_GAP_DETECTION

    prompt = MULTI_HOP_GAP_DETECTION.format(
        question=question,
        answer=initial_answer[:1000],
        context_preview=context_preview[:1500],
    )

    try:
        result = await llm_backend.ask(
            question=prompt,
            context="",
            history=None,
        )
        text = result.text.strip()

        if text.upper().startswith("NONE"):
            return []

        sub_queries = _QUERY_LINE_RE.findall(text)
        return [q.strip() for q in sub_queries[:3] if q.strip()]
    except Exception as exc:
        logger.warning("Gap detection failed: %s", exc)
        return []


async def multi_hop_retrieve(
    question: str,
    initial_chunks: list,
    initial_answer: str,
    context_preview: str,
    retriever,
    llm_backend,
    max_hops: int = 2,
    candidate_pool: int = 20,
    min_score: float = 0.3,
) -> tuple[list, bool]:
    """Iterative multi-hop retrieval: detect gaps -> retrieve -> merge.

    Args:
        question: Original user question.
        initial_chunks: Chunks from the first retrieval pass.
        initial_answer: Answer generated from initial context.
        context_preview: Text preview of initial context.
        retriever: ``HybridRetriever`` instance.
        llm_backend: LLM backend with ``ask()`` method.
        max_hops: Maximum number of additional retrieval rounds.
        candidate_pool: Number of candidates per sub-query retrieval.
        min_score: Minimum relevance score for retrieved chunks.

    Returns:
        Tuple of (merged_chunks, had_new_context) where ``had_new_context``
        indicates whether any new chunks were added.
    """
    seen_ids = {getattr(c, "chunk_id", id(c)) for c in initial_chunks}
    all_chunks = list(initial_chunks)
    had_new = False

    current_answer = initial_answer
    current_context = context_preview

    for hop in range(max_hops):
        sub_queries = await detect_knowledge_gaps(
            question, current_answer, current_context, llm_backend,
        )
        if not sub_queries:
            logger.debug("Multi-hop: no gaps detected at hop %d", hop + 1)
            break

        logger.info("Multi-hop hop %d: %d sub-queries", hop + 1, len(sub_queries))

        # Retrieve for each sub-query and merge (dedupe by chunk_id)
        new_in_hop = []
        for sq in sub_queries:
            try:
                candidates = retriever.search(
                    query=sq,
                    top_k=candidate_pool,
                    min_score=min_score,
                )
                for c in candidates:
                    cid = getattr(c, "chunk_id", id(c))
                    if cid not in seen_ids:
                        seen_ids.add(cid)
                        new_in_hop.append(c)
            except Exception as exc:
                logger.warning("Multi-hop sub-query retrieval failed: %s", exc)

        if not new_in_hop:
            logger.debug("Multi-hop: no new chunks at hop %d", hop + 1)
            break

        all_chunks.extend(new_in_hop)
        had_new = True

        # Update context preview for next gap detection round
        current_context = "\n\n".join(
            getattr(c, "text", "")[:300] for c in all_chunks[-5:]
        )

    return all_chunks, had_new
