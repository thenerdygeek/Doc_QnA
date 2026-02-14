"""Route intent to the correct generator and execute."""

from __future__ import annotations

import logging
import re

from doc_qa.config import GenerationConfig, IntelligenceConfig
from doc_qa.generation.base import GenerationResult, OutputGenerator
from doc_qa.generation.code_example import CodeExampleGenerator
from doc_qa.generation.comparison import ComparisonGenerator
from doc_qa.generation.diagram import DiagramGenerator
from doc_qa.generation.explanation import ExplanationGenerator
from doc_qa.generation.procedural import ProceduralGenerator
from doc_qa.intelligence.intent_classifier import IntentMatch, OutputIntent
from doc_qa.llm.prompt_templates import DIAGRAM_SUGGEST

logger = logging.getLogger(__name__)

# Maps each OutputIntent value to its specialized generator class.
_GENERATOR_MAP: dict[OutputIntent, type[OutputGenerator]] = {
    OutputIntent.DIAGRAM: DiagramGenerator,
    OutputIntent.CODE_EXAMPLE: CodeExampleGenerator,
    OutputIntent.COMPARISON_TABLE: ComparisonGenerator,
    OutputIntent.PROCEDURAL: ProceduralGenerator,
    OutputIntent.EXPLANATION: ExplanationGenerator,
}

# ── Diagram suggestion parsing ───────────────────────────────────────

_SUGGEST_RE = re.compile(r"^SUGGEST_DIAGRAM:\s*(YES|NO)", re.MULTILINE | re.IGNORECASE)
_SUGGEST_TYPE_RE = re.compile(r"^DIAGRAM_TYPE:\s*(\S+)", re.MULTILINE | re.IGNORECASE)


def _parse_diagram_suggestion(text: str) -> tuple[bool, str]:
    """Parse the DIAGRAM_SUGGEST LLM response.

    Returns (should_suggest, diagram_type).
    """
    suggest_m = _SUGGEST_RE.search(text)
    if not suggest_m or suggest_m.group(1).upper() != "YES":
        return False, "none"

    type_m = _SUGGEST_TYPE_RE.search(text)
    diagram_type = type_m.group(1).strip().lower() if type_m else "flowchart"
    if diagram_type == "none":
        return False, "none"

    return True, diagram_type


# ── Strategy resolution ──────────────────────────────────────────────


def resolve_generation_strategy(
    match: IntentMatch,
    config: GenerationConfig,
    *,
    intent_confidence_high: float = 0.85,
    intent_confidence_medium: float = 0.65,
) -> dict:
    """Decide which generator(s) to use based on intent confidence.

    Returns a dict with:
      - ``"generator"``: the intent name to use for the primary generator.
      - ``"include_explanation"``: whether to also include an explanation pass.

    Thresholds:
      - confidence >= *intent_confidence_high*  -> specialized only
      - confidence >= *intent_confidence_medium* -> specialized + explanation
      - below medium                            -> explanation only
    """
    confidence = match.confidence

    if confidence >= intent_confidence_high:
        return {
            "generator": match.intent.value,
            "include_explanation": False,
        }

    if confidence >= intent_confidence_medium:
        return {
            "generator": match.intent.value,
            "include_explanation": True,
        }

    return {
        "generator": OutputIntent.EXPLANATION.value,
        "include_explanation": False,
    }


async def _maybe_suggest_diagram(
    question: str,
    context: str,
    history: list[dict] | None,
    llm_backend,
    explanation_text: str,
    gen_config: GenerationConfig,
) -> GenerationResult | None:
    """Check if an explanation answer would benefit from a diagram.

    Returns a DiagramGenerator result if yes, None otherwise.
    """
    # Lightweight LLM check
    answer_preview = explanation_text[:500]
    suggest_prompt = DIAGRAM_SUGGEST.format(
        question=question,
        answer_preview=answer_preview,
    )
    answer = await llm_backend.ask(suggest_prompt, context="")
    if answer.error:
        logger.warning("Diagram suggestion check failed: %s", answer.error)
        return None

    should_suggest, diagram_type = _parse_diagram_suggestion(answer.text)
    if not should_suggest:
        logger.info("Diagram suggestion: NO — skipping proactive diagram.")
        return None

    logger.info("Diagram suggestion: YES (type=%s) — generating proactive diagram.", diagram_type)

    # Create a synthetic IntentMatch for the diagram generator
    synthetic_match = IntentMatch(
        intent=OutputIntent.DIAGRAM,
        confidence=0.90,
        matched_pattern="proactive_suggestion",
        sub_type=diagram_type,
    )

    try:
        generator = DiagramGenerator(gen_config=gen_config)
        return await generator.generate(
            question, context, history, llm_backend, synthetic_match
        )
    except Exception:
        logger.exception("Proactive diagram generation failed.")
        return None


async def route_and_generate(
    question: str,
    context: str,
    history: list[dict] | None,
    llm_backend,
    intent_match: IntentMatch,
    gen_config: GenerationConfig | None = None,
    intel_config: IntelligenceConfig | None = None,
) -> GenerationResult:
    """Map intent to generator, run it, and fall back on error.

    The routing logic:
      1. Resolve the generation strategy (specialized vs. explanation).
      2. Instantiate the matching ``OutputGenerator``.
      3. Call ``generate()``.
      4. On any exception, fall back to ``ExplanationGenerator``.
      5. If the result is an explanation and ``suggest_diagrams`` is enabled,
         check if a proactive diagram would improve the answer.
    """
    # Use defaults when configs are not provided
    effective_gen = gen_config or GenerationConfig()
    effective_intel = intel_config or IntelligenceConfig()

    strategy = resolve_generation_strategy(
        intent_match,
        effective_gen,
        intent_confidence_high=effective_intel.intent_confidence_high,
        intent_confidence_medium=effective_intel.intent_confidence_medium,
    )

    generator_name = strategy["generator"]
    include_explanation = strategy["include_explanation"]

    logger.info(
        "Generation strategy: generator=%s, include_explanation=%s (confidence=%.2f)",
        generator_name,
        include_explanation,
        intent_match.confidence,
    )

    # Look up the generator class.
    generator_cls = _GENERATOR_MAP.get(intent_match.intent)
    if generator_cls is None or generator_name == OutputIntent.EXPLANATION.value:
        generator_cls = ExplanationGenerator

    try:
        # DiagramGenerator accepts gen_config for validate-repair settings;
        # other generators use a no-arg constructor.
        if generator_cls is DiagramGenerator:
            generator = generator_cls(gen_config=effective_gen)
        else:
            generator = generator_cls()
        result = await generator.generate(
            question, context, history, llm_backend, intent_match
        )
    except Exception:
        logger.exception(
            "Specialized generator %s failed — falling back to ExplanationGenerator.",
            generator_cls.__name__,
        )
        fallback = ExplanationGenerator()
        result = await fallback.generate(
            question, context, history, llm_backend, intent_match
        )

    # If strategy says to include an explanation alongside the specialized
    # output (medium-confidence band), append an explanation pass.
    if include_explanation and generator_cls is not ExplanationGenerator:
        try:
            explanation = ExplanationGenerator()
            expl_result = await explanation.generate(
                question, context, history, llm_backend, intent_match
            )
            result = GenerationResult(
                text=f"{result.text}\n\n---\n\n{expl_result.text}",
                diagrams=result.diagrams,
                format_instruction=result.format_instruction,
                model=result.model or expl_result.model,
            )
        except Exception:
            logger.exception(
                "Supplementary explanation generation failed — returning "
                "specialized result only."
            )

    # ── Phase 2: Proactive diagram suggestion ────────────────────────
    # If the final output is an explanation (no diagrams), and the config
    # enables proactive suggestions, check if a diagram would help.
    if (
        effective_gen.suggest_diagrams
        and not result.diagrams
        and generator_cls is ExplanationGenerator
    ):
        try:
            diagram_result = await _maybe_suggest_diagram(
                question=question,
                context=context,
                history=history,
                llm_backend=llm_backend,
                explanation_text=result.text,
                gen_config=effective_gen,
            )
            if diagram_result and diagram_result.diagrams:
                # Append diagram section after the explanation
                result = GenerationResult(
                    text=f"{result.text}\n\n---\n\n{diagram_result.text}",
                    diagrams=diagram_result.diagrams,
                    format_instruction=result.format_instruction,
                    model=result.model,
                )
                logger.info("Proactive diagram appended to explanation.")
        except Exception:
            logger.exception("Proactive diagram suggestion failed — continuing without.")

    return result
