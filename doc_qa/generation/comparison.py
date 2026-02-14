"""Comparison/table generator."""

from __future__ import annotations

import logging
import re

from doc_qa.generation.base import GenerationResult, OutputGenerator
from doc_qa.llm.prompt_templates import COMPARISON_TABLE_GENERATION

logger = logging.getLogger(__name__)

# Matches a markdown table header row followed by a separator row.
# e.g.:  | Feature | A | B |
#        |---------|---|---|
_TABLE_RE = re.compile(
    r"^\|.+\|[ \t]*\n\|[-| :]+\|",
    re.MULTILINE,
)


class ComparisonGenerator(OutputGenerator):
    """Inject a comparison-table format instruction and validate the output.

    Post-processing performs a basic check that the response contains at
    least one markdown table (header row + separator row).  If not, a
    warning is logged but the text is returned as-is.
    """

    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        format_instruction = COMPARISON_TABLE_GENERATION
        augmented_question = f"{question}\n\n{format_instruction}"

        answer = await llm_backend.ask(augmented_question, context, history)

        if answer.error:
            logger.warning(
                "LLM error during comparison generation: %s", answer.error
            )
            return GenerationResult(
                text=answer.text,
                format_instruction=format_instruction,
                model=answer.model,
            )

        if not _TABLE_RE.search(answer.text):
            logger.warning(
                "Comparison response does not contain a markdown table."
            )

        return GenerationResult(
            text=answer.text,
            format_instruction=format_instruction,
            model=answer.model,
        )
