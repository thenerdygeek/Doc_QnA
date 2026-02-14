"""Mermaid diagram generator with plan-then-generate pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from doc_qa.config import GenerationConfig
from doc_qa.generation.base import GenerationResult, OutputGenerator
from doc_qa.llm.prompt_templates import (
    DIAGRAM_GENERATION,
    DIAGRAM_GENERATION_PLANNED,
    DIAGRAM_PLAN,
    DIAGRAM_REPAIR,
)

logger = logging.getLogger(__name__)

# Graceful import of the Mermaid validator — it may not exist yet.
try:
    from doc_qa.verification.mermaid_validator import MermaidValidator  # type: ignore[import-untyped]

    _HAS_VALIDATOR = True
except ImportError:
    _HAS_VALIDATOR = False
    logger.warning(
        "MermaidValidator not available — diagram validation will be skipped."
    )

_MERMAID_FENCE_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)

_DEFAULT_MAX_REPAIR_ATTEMPTS = 3

# ── Plan parsing ─────────────────────────────────────────────────────

_PLAN_TYPE_RE = re.compile(r"^DIAGRAM_TYPE:\s*(.+)", re.MULTILINE)
_PLAN_TITLE_RE = re.compile(r"^TITLE:\s*(.+)", re.MULTILINE)
_PLAN_ENTITY_RE = re.compile(r"^-\s+([^:]+):\s*(.+)", re.MULTILINE)
_PLAN_REL_RE = re.compile(r"^-\s+(.+?)\s*->\s*(.+?):\s*(.+)", re.MULTILINE)


@dataclass
class DiagramPlan:
    """Structured output from the diagram planning step."""

    diagram_type: str = "flowchart"
    title: str = ""
    entities: list[tuple[str, str]] = field(default_factory=list)  # (name, role)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)  # (from, to, label)


def _parse_diagram_plan(text: str) -> DiagramPlan | None:
    """Parse the LLM's structured plan response.

    Returns None if the response cannot be parsed meaningfully.
    """
    type_m = _PLAN_TYPE_RE.search(text)
    title_m = _PLAN_TITLE_RE.search(text)

    if not type_m:
        return None

    diagram_type = type_m.group(1).strip().lower()
    # Normalise common variations
    type_map = {
        "flowchart": "flowchart",
        "flow chart": "flowchart",
        "graph": "flowchart",
        "sequence": "sequence",
        "sequencediagram": "sequence",
        "sequence diagram": "sequence",
        "classdiagram": "classDiagram",
        "class diagram": "classDiagram",
        "class": "classDiagram",
        "erdiagram": "erDiagram",
        "er diagram": "erDiagram",
        "er": "erDiagram",
        "statediagram": "stateDiagram",
        "state diagram": "stateDiagram",
        "state": "stateDiagram",
    }
    diagram_type = type_map.get(diagram_type, diagram_type)

    # Split at ENTITIES: and RELATIONSHIPS: headers to parse sections
    entities_section = ""
    rel_section = ""
    upper = text.upper()
    ent_idx = upper.find("ENTITIES:")
    rel_idx = upper.find("RELATIONSHIPS:")

    if ent_idx != -1 and rel_idx != -1:
        entities_section = text[ent_idx:rel_idx]
        rel_section = text[rel_idx:]
    elif ent_idx != -1:
        entities_section = text[ent_idx:]
    elif rel_idx != -1:
        rel_section = text[rel_idx:]

    entities = [
        (m.group(1).strip(), m.group(2).strip())
        for m in _PLAN_ENTITY_RE.finditer(entities_section)
    ]
    relationships = [
        (m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
        for m in _PLAN_REL_RE.finditer(rel_section)
    ]

    # Need at least 2 entities or 1 relationship to be useful
    if len(entities) < 2 and len(relationships) < 1:
        return None

    return DiagramPlan(
        diagram_type=diagram_type,
        title=title_m.group(1).strip() if title_m else "",
        entities=entities,
        relationships=relationships,
    )


def _extract_mermaid_blocks(text: str) -> list[str]:
    """Extract all mermaid fenced code blocks from *text*."""
    return _MERMAID_FENCE_RE.findall(text)


class DiagramGenerator(OutputGenerator):
    """Generate Mermaid diagrams with a plan-then-generate pipeline.

    Pipeline:
      1. **Plan** -- ask the LLM to identify entities, relationships,
         and the optimal diagram type (lightweight structured output).
      2. **Generate** -- ask the LLM with a plan-enriched prompt that
         constrains the Mermaid generation to the planned structure.
      3. **Validate** -- extract ````` ```mermaid ````` blocks and run them
         through ``MermaidValidator`` (if available).
      4. **Repair** -- on validation failure, send the error and the
         broken diagram back to the LLM.  Repeat up to
         ``max_repair_attempts`` times.

    Falls back to the unplanned prompt if the planning step fails.
    """

    def __init__(self, gen_config: GenerationConfig | None = None) -> None:
        self._gen_config = gen_config or GenerationConfig()
        self._max_repairs = self._gen_config.max_diagram_retries

    async def _plan_diagram(
        self,
        question: str,
        context: str,
        llm_backend,
    ) -> DiagramPlan | None:
        """Ask the LLM to plan the diagram structure before generating Mermaid."""
        plan_prompt = DIAGRAM_PLAN.format(question=question)
        # Pass context as doc context (not as part of the question)
        answer = await llm_backend.ask(plan_prompt, context)

        if answer.error:
            logger.warning("Diagram planning LLM error: %s", answer.error)
            return None

        plan = _parse_diagram_plan(answer.text)
        if plan:
            logger.info(
                "Diagram plan: type=%s, entities=%d, relationships=%d",
                plan.diagram_type,
                len(plan.entities),
                len(plan.relationships),
            )
        else:
            logger.info("Diagram planning produced unparseable output — skipping plan.")
        return plan

    def _build_planned_prompt(self, plan: DiagramPlan) -> str:
        """Convert a DiagramPlan into an enriched generation prompt."""
        entities_text = "\n".join(
            f"- {name}: {role}" for name, role in plan.entities
        )
        rel_text = "\n".join(
            f"- {src} -> {dst}: {label}"
            for src, dst, label in plan.relationships
        )

        # Map plan type to Mermaid declaration
        type_decl_map = {
            "flowchart": "graph TD",
            "sequence": "sequenceDiagram",
            "classDiagram": "classDiagram",
            "erDiagram": "erDiagram",
            "stateDiagram": "stateDiagram-v2",
        }
        diagram_type = type_decl_map.get(plan.diagram_type, "graph TD")

        return DIAGRAM_GENERATION_PLANNED.format(
            diagram_type=diagram_type,
            title=plan.title or "Untitled",
            entities=entities_text or "- (use entities from context)",
            relationships=rel_text or "- (determine from context)",
        )

    async def generate(
        self,
        question: str,
        context: str,
        history: list[dict] | None,
        llm_backend,
        intent_match,
    ) -> GenerationResult:
        # Step 1: Plan the diagram (lightweight LLM call)
        plan = await self._plan_diagram(question, context, llm_backend)

        # Step 2: Generate Mermaid — use planned prompt if plan succeeded
        if plan:
            diagram_type = plan.diagram_type
            format_instruction = self._build_planned_prompt(plan)
        else:
            # Fallback to original unplanned flow
            diagram_type = getattr(intent_match, "sub_type", None) or "flowchart"
            format_instruction = DIAGRAM_GENERATION.format(diagram_type=diagram_type)

        augmented_question = f"{question}\n\n{format_instruction}"
        answer = await llm_backend.ask(augmented_question, context, history)

        if answer.error:
            logger.warning("LLM error during diagram generation: %s", answer.error)
            return GenerationResult(
                text=answer.text,
                format_instruction=format_instruction,
                model=answer.model,
            )

        text = answer.text
        diagrams = _extract_mermaid_blocks(text)

        # Step 3: Validate and repair
        if diagrams and _HAS_VALIDATOR:
            validator = MermaidValidator(
                node_script_path=self._gen_config.node_script_path,
                mode=self._gen_config.mermaid_validation,
            )
            for attempt in range(1, self._max_repairs + 1):
                all_valid = True
                for diagram in diagrams:
                    result = validator.validate(diagram)
                    if not result["valid"]:
                        all_valid = False
                        val_error = result.get("error", "Unknown validation error")
                        logger.info(
                            "Diagram validation failed (attempt %d/%d): %s",
                            attempt,
                            self._max_repairs,
                            val_error,
                        )
                        repair_prompt = DIAGRAM_REPAIR.format(
                            error=val_error,
                            diagram=diagram,
                        )
                        repair_prompt = (
                            f"Original question: {question}\n\n{repair_prompt}"
                        )
                        repair_answer = await llm_backend.ask(
                            repair_prompt, context, history
                        )
                        if repair_answer.error:
                            logger.warning(
                                "LLM error during diagram repair (attempt %d/%d): %s",
                                attempt,
                                self._max_repairs,
                                repair_answer.error,
                            )
                            continue

                        text = repair_answer.text
                        diagrams = _extract_mermaid_blocks(text)
                        break  # re-validate from scratch with the new text

                if all_valid:
                    logger.info("All diagrams validated successfully.")
                    break
            else:
                logger.warning(
                    "Diagram validation exhausted %d repair attempts — returning as-is.",
                    self._max_repairs,
                )

        return GenerationResult(
            text=text,
            diagrams=diagrams or None,
            format_instruction=format_instruction,
            model=answer.model,
        )
