"""Multi-intent detection and query decomposition."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from doc_qa.intelligence.intent_classifier import IntentMatch, classify_intent
from doc_qa.llm.prompt_templates import QUERY_DECOMPOSITION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SubQuery:
    """A single sub-query extracted from a multi-intent question."""

    query_text: str
    intent: IntentMatch


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""

    original: str
    sub_queries: list[SubQuery]
    is_multi_intent: bool


# ---------------------------------------------------------------------------
# Multi-intent detection (heuristic)
# ---------------------------------------------------------------------------

_COORDINATION_MARKERS = re.compile(
    r"\b(?:and\s+also|plus\s+show|as\s+well\s+as|additionally|along\s+with)\b",
    re.IGNORECASE,
)

_OUTPUT_VERBS = re.compile(
    r"\b(?:show|draw|diagram|list|compare|explain|generate|create|give|write|illustrate|visualize|describe)\b",
    re.IGNORECASE,
)


def detect_multi_intent(query: str) -> bool:
    """Heuristic check for whether a query contains multiple output intents.

    Requires both a coordination marker ("and also", "plus show", etc.) and
    at least two distinct output verbs.
    """
    if not _COORDINATION_MARKERS.search(query):
        return False

    verb_matches = _OUTPUT_VERBS.findall(query)
    # Deduplicate (case-insensitive) to count distinct verbs
    unique_verbs = {v.lower() for v in verb_matches}
    return len(unique_verbs) >= 2


# ---------------------------------------------------------------------------
# LLM decomposition response parsing
# ---------------------------------------------------------------------------

_SUB_QUERY_LINE = re.compile(r"^SUB-QUERY\s+\d+:\s*(.+)", re.MULTILINE)
_NUMBERED_ITEM = re.compile(r"^\d+[.)]\s*(.+)", re.MULTILINE)


def _parse_sub_queries(response: str) -> list[str]:
    """Extract sub-queries from the LLM decomposition response.

    Tries ``SUB-QUERY N:`` format first, falls back to numbered list items.
    """
    matches = _SUB_QUERY_LINE.findall(response)
    if matches:
        return [m.strip() for m in matches if m.strip()]

    numbered = _NUMBERED_ITEM.findall(response)
    if numbered:
        return [n.strip() for n in numbered if n.strip()]

    # Last resort: return the whole response as a single sub-query
    stripped = response.strip()
    return [stripped] if stripped else []


# ---------------------------------------------------------------------------
# Query decomposition
# ---------------------------------------------------------------------------


async def decompose_query(
    query: str,
    llm_backend,
    max_sub_queries: int = 3,
) -> DecomposedQuery:
    """Decompose a query into sub-queries when multiple intents are detected.

    Only invokes the LLM when :func:`detect_multi_intent` returns ``True``.
    For single-intent queries the original query is returned as-is with its
    classified intent.

    Args:
        query: The user's documentation question.
        llm_backend: An ``LLMBackend`` instance.
        max_sub_queries: Maximum number of sub-queries to produce.

    Returns:
        A ``DecomposedQuery`` with classified sub-queries.
    """
    if not detect_multi_intent(query):
        intent = await classify_intent(query, llm_backend)
        return DecomposedQuery(
            original=query,
            sub_queries=[SubQuery(query_text=query, intent=intent)],
            is_multi_intent=False,
        )

    # Multi-intent detected — ask LLM to decompose
    try:
        prompt = QUERY_DECOMPOSITION.format(
            query=query,
            max_sub_queries=max_sub_queries,
        )
        answer = await llm_backend.ask(question=prompt, context="")

        if answer.error:
            logger.warning("LLM decomposition failed: %s; treating as single query.", answer.error)
            intent = await classify_intent(query, llm_backend)
            return DecomposedQuery(
                original=query,
                sub_queries=[SubQuery(query_text=query, intent=intent)],
                is_multi_intent=False,
            )

        raw_parts = _parse_sub_queries(answer.text)

        # Cap at max_sub_queries
        parts = raw_parts[:max_sub_queries]

        if len(parts) <= 1:
            # LLM decided it was really a single query after all
            intent = await classify_intent(query, llm_backend)
            return DecomposedQuery(
                original=query,
                sub_queries=[SubQuery(query_text=query, intent=intent)],
                is_multi_intent=False,
            )

        # Classify each sub-query
        sub_queries: list[SubQuery] = []
        for part in parts:
            intent = await classify_intent(part, llm_backend)
            sub_queries.append(SubQuery(query_text=part, intent=intent))

        logger.info(
            "Decomposed query into %d sub-queries: %s",
            len(sub_queries),
            [sq.query_text[:60] for sq in sub_queries],
        )

        return DecomposedQuery(
            original=query,
            sub_queries=sub_queries,
            is_multi_intent=True,
        )

    except Exception:
        logger.exception("Unexpected error during query decomposition.")
        intent = await classify_intent(query, llm_backend)
        return DecomposedQuery(
            original=query,
            sub_queries=[SubQuery(query_text=query, intent=intent)],
            is_multi_intent=False,
        )


# ---------------------------------------------------------------------------
# Complexity assessment
# ---------------------------------------------------------------------------

_ENTITY_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
)

_CAUSAL_CHAIN = re.compile(
    r"\b(?:because|therefore|as\s+a\s+result|consequently|if\s+.+\s+then|causes?|leads?\s+to|affects?)\b",
    re.IGNORECASE,
)

_CROSS_DOC = re.compile(
    r"\b(?:according\s+to|in\s+the\s+.+\s+(?:docs?|documentation|section|page|guide)|"
    r"as\s+(?:described|mentioned|documented)\s+in|reference|see\s+also)\b",
    re.IGNORECASE,
)


def assess_complexity(query: str) -> str:
    """Assess the complexity of a query.

    Returns:
        ``"simple"`` for straightforward factual queries, or
        ``"multi_hop"`` for queries that likely require reasoning across
        multiple document sections or entities.
    """
    signals = 0

    # Count distinct named entities (capitalised multi-word sequences)
    entities = set(_ENTITY_PATTERN.findall(query))
    # Filter out common English words that happen to start sentences
    stop = {"The", "This", "That", "What", "How", "When", "Where", "Why", "Which", "Who"}
    entities -= stop
    if len(entities) >= 3:
        signals += 1

    # Causal chains
    if _CAUSAL_CHAIN.search(query):
        signals += 1

    # Cross-doc references
    if _CROSS_DOC.search(query):
        signals += 1

    return "multi_hop" if signals >= 2 else "simple"
