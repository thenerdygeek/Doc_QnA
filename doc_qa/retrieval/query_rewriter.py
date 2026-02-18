"""Multi-turn conversational query rewriter.

Before retrieval, rewrites a follow-up question into a standalone query
using conversation history.  The original question is preserved for
display and LLM generation — only retrieval uses the rewritten form.

Short-circuits via a cheap regex heuristic (``needs_rewrite``) so that
standalone questions skip the LLM call entirely.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate the query depends on prior conversation context
_NEEDS_REWRITE_RE = re.compile(
    r"""(?xi)          # case-insensitive, verbose
    \b(
        it | its | that | those | this | these | them
        | the\s+same | the\s+above | the\s+previous
        | tell\s+me\s+more | what\s+about | how\s+about
        | can\s+you\s+explain | can\s+you\s+elaborate
        | continue | go\s+on | expand\s+on
        | more\s+details? | more\s+info
        | also | additionally | furthermore
        | the\s+one | which\s+one
    )\b
    """
)

# Queries longer than this with no pronoun/reference markers are likely standalone
_STANDALONE_MIN_WORDS = 12


def needs_rewrite(question: str, history: list[dict]) -> bool:
    """Cheap heuristic: decide whether the query needs LLM rewriting.

    Returns False (skip rewrite) when:
    - No history at all
    - The query is long enough and contains no context-dependent references
    """
    if not history:
        return False

    words = question.split()
    has_reference = bool(_NEEDS_REWRITE_RE.search(question))

    # Short queries with any reference pattern → rewrite
    if has_reference:
        return True

    # Long, self-contained queries → skip rewrite
    if len(words) >= _STANDALONE_MIN_WORDS:
        return False

    # Short queries without clear references — err on the side of rewriting
    # if there's meaningful history (at least one full exchange)
    return len(history) >= 2


async def rewrite_query(
    question: str,
    history: list[dict],
    llm_backend,
    max_history_turns: int = 4,
) -> str:
    """Rewrite a follow-up question into a standalone query using history.

    Args:
        question: The user's raw follow-up question.
        history: Conversation history (list of ``{role, text}`` dicts).
        llm_backend: LLM backend with an ``ask()`` method.
        max_history_turns: Maximum number of recent history entries to include.

    Returns:
        The rewritten standalone question, or the original on failure.
    """
    from doc_qa.llm.prompt_templates import CONVERSATIONAL_REWRITE

    # Truncate history to recent turns
    recent = history[-max_history_turns:] if len(history) > max_history_turns else history

    history_text = "\n".join(
        f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['text'][:300]}"
        for h in recent
    )

    prompt = CONVERSATIONAL_REWRITE.format(
        history=history_text,
        question=question,
    )

    try:
        result = await llm_backend.ask(
            question=prompt,
            context="",
            history=None,
        )
        rewritten = result.text.strip()
        # Basic sanity: not empty, not identical to the original, not too long
        if rewritten and rewritten != question and len(rewritten) < 500:
            logger.info("Query rewritten: '%s' → '%s'", question, rewritten)
            return rewritten
        logger.debug("Rewrite returned unchanged or invalid result")
    except Exception as exc:
        logger.warning("Query rewrite failed, using original: %s", exc)

    return question
