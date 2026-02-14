"""Two-tier intent classification: fast heuristic regex patterns first, LLM fallback second."""

from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass

from doc_qa.llm.prompt_templates import INTENT_CLASSIFICATION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class OutputIntent(enum.Enum):
    """The output format intent detected from a user query."""

    DIAGRAM = "DIAGRAM"
    CODE_EXAMPLE = "CODE_EXAMPLE"
    COMPARISON_TABLE = "COMPARISON_TABLE"
    PROCEDURAL = "PROCEDURAL"
    EXPLANATION = "EXPLANATION"


@dataclass
class IntentMatch:
    """Result of intent classification."""

    intent: OutputIntent
    confidence: float  # 0-1
    matched_pattern: str
    sub_type: str


# ---------------------------------------------------------------------------
# Compiled regex patterns — require BOTH topic noun AND request verb
# ---------------------------------------------------------------------------

# ── Diagram patterns ──

_DIAGRAM_TOPIC = re.compile(
    r"\b(?:flow|architecture|sequence|relationship|how\s+\w+\s+interact)\b",
    re.IGNORECASE,
)
_DIAGRAM_VERB = re.compile(
    r"\b(?:diagram|draw|visualize|illustrate|show\s+\w+\s+flow)\b",
    re.IGNORECASE,
)
_DIAGRAM_EXPLICIT = re.compile(r"\bmermaid\b", re.IGNORECASE)

# ── Code patterns ──

_CODE_FORMAT = re.compile(
    r"\b(?:curl|API|endpoint|code\s+(?:example|sample)|(?:json|yaml)\s+example)\b",
    re.IGNORECASE,
)
_CODE_VERB = re.compile(
    r"\b(?:show|give|generate|write|create)\b",
    re.IGNORECASE,
)

# ── Comparison patterns ──

_COMPARISON_TOPIC = re.compile(
    r"\b(?:difference|compare|vs\.?|versus|contrast)\b",
    re.IGNORECASE,
)
_COMPARISON_VERB = re.compile(
    r"\b(?:between|table|compare)\b",
    re.IGNORECASE,
)
_COMPARISON_PHRASE = re.compile(
    r"\b(?:pros\s+and\s+cons|advantages\s+(?:and\s+)?disadvantages|tradeoffs?|trade[\s-]offs?)\b",
    re.IGNORECASE,
)

# ── Procedural patterns ──

_PROCEDURAL_TOPIC = re.compile(
    r"\b(?:how\s+do\s+I|how\s+to|how\s+can\s+I|steps\s+to|guide|setup|configure|install)\b",
    re.IGNORECASE,
)
_PROCEDURAL_VERB = re.compile(
    r"\b(?:set\s+up|configure|install|deploy|migrate|upgrade|implement)\b",
    re.IGNORECASE,
)

# ── Diagram sub-type detection ──

_SUBTYPE_SEQUENCE = re.compile(
    r"\b(?:sequence|message\s+(?:flow|passing)|between\s+\w+\s+and|interaction)\b",
    re.IGNORECASE,
)
_SUBTYPE_ER = re.compile(
    r"\b(?:entity|relationship|ER\s+diagram|data\s+model|schema)\b",
    re.IGNORECASE,
)
_SUBTYPE_CLASS = re.compile(
    r"\b(?:class\s+diagram|inheritance|interface|UML)\b",
    re.IGNORECASE,
)
_SUBTYPE_STATE = re.compile(
    r"\b(?:state\s+(?:machine|diagram)|lifecycle|transition)\b",
    re.IGNORECASE,
)

# ── Code sub-type detection ──

_SUBTYPE_CURL = re.compile(r"\b(?:curl|REST|HTTP\s+request)\b", re.IGNORECASE)
_SUBTYPE_GRAPHQL = re.compile(r"\b(?:graphql|mutation|subscription)\b", re.IGNORECASE)
_SUBTYPE_GRPC = re.compile(r"\b(?:grpc|protobuf|proto)\b", re.IGNORECASE)
_SUBTYPE_YAML = re.compile(r"\b(?:yaml|yml|kubernetes|k8s|helm|docker[\s-]?compose)\b", re.IGNORECASE)
_SUBTYPE_JSON = re.compile(r"\b(?:json|payload|request\s+body)\b", re.IGNORECASE)
_SUBTYPE_HTTP = re.compile(r"\b(?:HTTP|endpoint|header)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Heuristic classification
# ---------------------------------------------------------------------------


def _detect_diagram_subtype(query: str) -> str:
    """Detect the specific diagram sub-type from the query text."""
    if _SUBTYPE_SEQUENCE.search(query):
        return "sequence"
    if _SUBTYPE_ER.search(query):
        return "erDiagram"
    if _SUBTYPE_CLASS.search(query):
        return "classDiagram"
    if _SUBTYPE_STATE.search(query):
        return "stateDiagram"
    return "flowchart"


def _detect_code_subtype(query: str) -> str:
    """Detect the specific code format sub-type from the query text."""
    if _SUBTYPE_CURL.search(query):
        return "curl"
    if _SUBTYPE_GRAPHQL.search(query):
        return "graphql"
    if _SUBTYPE_GRPC.search(query):
        return "grpc"
    if _SUBTYPE_YAML.search(query):
        return "yaml"
    if _SUBTYPE_JSON.search(query):
        return "json"
    if _SUBTYPE_HTTP.search(query):
        return "http"
    return "none"


def classify_by_heuristic(query: str) -> IntentMatch | None:
    """Fast heuristic intent classification using regex patterns.

    Returns None if no high-confidence match is found, signalling
    that the LLM fallback should be used.
    """
    # Diagram: explicit mermaid keyword is always high-confidence
    if _DIAGRAM_EXPLICIT.search(query):
        return IntentMatch(
            intent=OutputIntent.DIAGRAM,
            confidence=0.95,
            matched_pattern="explicit_mermaid",
            sub_type=_detect_diagram_subtype(query),
        )

    # Diagram: topic + verb
    if _DIAGRAM_TOPIC.search(query) and _DIAGRAM_VERB.search(query):
        return IntentMatch(
            intent=OutputIntent.DIAGRAM,
            confidence=0.92,
            matched_pattern="diagram_topic_verb",
            sub_type=_detect_diagram_subtype(query),
        )

    # Code example: format keyword + verb
    if _CODE_FORMAT.search(query) and _CODE_VERB.search(query):
        return IntentMatch(
            intent=OutputIntent.CODE_EXAMPLE,
            confidence=0.90,
            matched_pattern="code_format_verb",
            sub_type=_detect_code_subtype(query),
        )

    # Comparison: standalone phrase (e.g. "pros and cons")
    if _COMPARISON_PHRASE.search(query):
        return IntentMatch(
            intent=OutputIntent.COMPARISON_TABLE,
            confidence=0.93,
            matched_pattern="comparison_phrase",
            sub_type="none",
        )

    # Comparison: topic + verb
    if _COMPARISON_TOPIC.search(query) and _COMPARISON_VERB.search(query):
        return IntentMatch(
            intent=OutputIntent.COMPARISON_TABLE,
            confidence=0.90,
            matched_pattern="comparison_topic_verb",
            sub_type="none",
        )

    # Procedural: topic + verb
    if _PROCEDURAL_TOPIC.search(query) and _PROCEDURAL_VERB.search(query):
        return IntentMatch(
            intent=OutputIntent.PROCEDURAL,
            confidence=0.88,
            matched_pattern="procedural_topic_verb",
            sub_type="none",
        )

    # No high-confidence heuristic match
    return None


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_INTENT_NAMES = {member.value: member for member in OutputIntent}

_LLM_INTENT_RE = re.compile(r"^Intent:\s*(\S+)", re.MULTILINE)
_LLM_SUBTYPE_RE = re.compile(r"^Sub-type:\s*(.+)", re.MULTILINE)
_LLM_REASONING_RE = re.compile(r"^Reasoning:", re.MULTILINE)


def _parse_llm_intent(response_text: str) -> IntentMatch:
    """Parse an LLM classification response with Reasoning/Intent/Sub-type lines.

    Falls back to fuzzy matching when the expected format is not found.
    """
    intent_match = _LLM_INTENT_RE.search(response_text)
    subtype_match = _LLM_SUBTYPE_RE.search(response_text)
    has_reasoning = _LLM_REASONING_RE.search(response_text)

    sub_type = subtype_match.group(1).strip() if subtype_match else "none"

    # Exact match on the Intent: line
    if intent_match:
        raw = intent_match.group(1).strip().upper()
        if raw in _INTENT_NAMES:
            confidence = 0.85 if has_reasoning else 0.70
            return IntentMatch(
                intent=_INTENT_NAMES[raw],
                confidence=confidence,
                matched_pattern="llm_classification",
                sub_type=sub_type,
            )

    # Fuzzy fallback: scan the whole response for any known intent name
    upper = response_text.upper()
    for name, member in _INTENT_NAMES.items():
        if name in upper:
            logger.debug("Fuzzy-matched intent %s from LLM response.", name)
            return IntentMatch(
                intent=member,
                confidence=0.60,
                matched_pattern="llm_fuzzy",
                sub_type=sub_type,
            )

    # Total failure — default to EXPLANATION
    logger.warning("Could not parse intent from LLM response, defaulting to EXPLANATION.")
    return IntentMatch(
        intent=OutputIntent.EXPLANATION,
        confidence=0.50,
        matched_pattern="llm_parse_failure",
        sub_type="none",
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def classify_intent(query: str, llm_backend) -> IntentMatch:
    """Classify the output intent of a query.

    Uses fast heuristic patterns first, falls back to LLM classification
    when heuristics are not confident enough.

    Args:
        query: The user's documentation question.
        llm_backend: An ``LLMBackend`` instance (used only when heuristics
            return ``None``).

    Returns:
        An ``IntentMatch`` with the detected intent, confidence, and sub-type.
    """
    # Tier 1: fast heuristic
    heuristic_result = classify_by_heuristic(query)
    if heuristic_result is not None:
        logger.info(
            "Heuristic intent: %s (confidence=%.2f, pattern=%s)",
            heuristic_result.intent.value,
            heuristic_result.confidence,
            heuristic_result.matched_pattern,
        )
        return heuristic_result

    # Tier 2: LLM fallback
    try:
        prompt = INTENT_CLASSIFICATION.format(query=query)
        answer = await llm_backend.ask(question=prompt, context="")
        if answer.error:
            logger.warning("LLM intent classification failed: %s", answer.error)
            return IntentMatch(
                intent=OutputIntent.EXPLANATION,
                confidence=0.50,
                matched_pattern="llm_error_fallback",
                sub_type="none",
            )

        result = _parse_llm_intent(answer.text)
        logger.info(
            "LLM intent: %s (confidence=%.2f, pattern=%s)",
            result.intent.value,
            result.confidence,
            result.matched_pattern,
        )
        return result

    except Exception:
        logger.exception("Unexpected error during LLM intent classification.")
        return IntentMatch(
            intent=OutputIntent.EXPLANATION,
            confidence=0.50,
            matched_pattern="llm_exception_fallback",
            sub_type="none",
        )
