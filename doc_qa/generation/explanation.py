"""Default generator — passes through to LLM with standard prompt."""

from __future__ import annotations

import logging

from doc_qa.generation.base import GenerationResult, OutputGenerator

logger = logging.getLogger(__name__)


class ExplanationGenerator(OutputGenerator):
    """Pass-through generator that delegates directly to the LLM backend.

    This is the default/fallback behaviour: no special format instructions
    are injected.  The question and context are forwarded as-is and the
    answer is wrapped in a ``GenerationResult``.
    """

    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        answer = await llm_backend.ask(question, context, history)

        if answer.error:
            logger.warning("LLM returned an error: %s", answer.error)

        return GenerationResult(
            text=answer.text,
            model=answer.model,
        )
