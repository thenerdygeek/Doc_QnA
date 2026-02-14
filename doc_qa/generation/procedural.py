"""Step-by-step instruction generator."""

from __future__ import annotations

import logging
import re

from doc_qa.generation.base import GenerationResult, OutputGenerator
from doc_qa.llm.prompt_templates import PROCEDURAL_GENERATION

logger = logging.getLogger(__name__)

# Matches at least one numbered list item (e.g. "1. ", "2. ").
_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s", re.MULTILINE)


class ProceduralGenerator(OutputGenerator):
    """Inject a procedural format instruction and validate numbered steps.

    Post-processing ensures the response contains at least one numbered
    list item.  If not, a warning is logged but the text is returned
    as-is (the LLM may have chosen a different valid structure).
    """

    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        format_instruction = PROCEDURAL_GENERATION
        augmented_question = f"{question}\n\n{format_instruction}"

        answer = await llm_backend.ask(augmented_question, context, history)

        if answer.error:
            logger.warning(
                "LLM error during procedural generation: %s", answer.error
            )
            return GenerationResult(
                text=answer.text,
                format_instruction=format_instruction,
                model=answer.model,
            )

        if not _NUMBERED_LIST_RE.search(answer.text):
            logger.warning(
                "Procedural response does not contain a numbered list."
            )

        return GenerationResult(
            text=answer.text,
            format_instruction=format_instruction,
            model=answer.model,
        )
