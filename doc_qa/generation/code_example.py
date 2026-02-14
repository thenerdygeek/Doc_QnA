"""Code example generator."""

from __future__ import annotations

import logging
import re

from doc_qa.generation.base import GenerationResult, OutputGenerator
from doc_qa.llm.prompt_templates import CODE_EXAMPLE_GENERATION

logger = logging.getLogger(__name__)

# Matches a bare code fence (``` with no language tag) at the start of a line.
_BARE_FENCE_RE = re.compile(r"^```\s*$", re.MULTILINE)


class CodeExampleGenerator(OutputGenerator):
    """Inject a code-example format instruction and post-process the output.

    The ``sub_type`` from the intent match (e.g. ``curl``, ``json``, ``yaml``,
    ``graphql``) is forwarded as the requested code format so that the LLM
    produces the right kind of example.

    Post-processing ensures that every code fence has an explicit language tag.
    """

    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        code_format = getattr(intent_match, "sub_type", None) or "code"

        format_instruction = CODE_EXAMPLE_GENERATION.format(code_format=code_format)
        augmented_question = f"{question}\n\n{format_instruction}"

        answer = await llm_backend.ask(augmented_question, context, history)

        if answer.error:
            logger.warning("LLM error during code example generation: %s", answer.error)
            return GenerationResult(
                text=answer.text,
                format_instruction=format_instruction,
                model=answer.model,
            )

        text = _ensure_language_tags(answer.text, code_format)

        return GenerationResult(
            text=text,
            format_instruction=format_instruction,
            model=answer.model,
        )


def _ensure_language_tags(text: str, default_lang: str) -> str:
    """Replace bare ```` ``` ```` fences with ```` ```<lang> ````.

    If a fenced code block has no language tag, insert *default_lang* so
    that renderers can apply syntax highlighting.
    """
    return _BARE_FENCE_RE.sub(f"```{default_lang}", text)
