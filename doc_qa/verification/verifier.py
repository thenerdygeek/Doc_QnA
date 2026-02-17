"""Generate-then-verify pipeline: second LLM call to check answer accuracy.

Sends the original question, the generated answer, and the source texts to
the LLM and asks it to verify factual accuracy.  The response is parsed into
a structured ``VerificationResult``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from doc_qa.llm.prompt_templates import VERIFICATION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    """Outcome of a verification check on a generated answer."""

    passed: bool
    confidence: float
    issues: list[str] = field(default_factory=list)
    suggested_fix: str | None = None


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_VERDICT_RE = re.compile(
    r"Verdict\s*[:—\-]\s*(PASS(?:ED)?|FAIL(?:ED)?|YES|NO|CORRECT|INCORRECT)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(r"Confidence\s*[:—\-]\s*([\d.]+(?:\s*/\s*[\d.]+)?)", re.IGNORECASE)
_ISSUES_RE = re.compile(r"Issues\s*[:—\-]\s*(.+)", re.IGNORECASE)
_FIX_RE = re.compile(r"Suggested\s+fix\s*[:—\-]\s*(.+)", re.IGNORECASE)

_PASS_SYNONYMS = frozenset({"pass", "passed", "yes", "correct"})
_FAIL_SYNONYMS = frozenset({"fail", "failed", "no", "incorrect"})


def _try_parse_json(response: str) -> VerificationResult | None:
    """Try to parse the response as JSON (some LLMs return structured output)."""
    import json

    # Try to find a JSON block in the response
    for start_marker in ("{", "```json\n", "```\n"):
        idx = response.find(start_marker)
        if idx == -1:
            continue
        candidate = response[idx:]
        if candidate.startswith("```"):
            # Strip code fence
            candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
            end = candidate.find("```")
            if end != -1:
                candidate = candidate[:end]
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                continue

            # Extract verdict
            verdict_raw = str(data.get("verdict", data.get("passed", ""))).lower()
            if verdict_raw in _PASS_SYNONYMS or verdict_raw == "true":
                passed = True
            elif verdict_raw in _FAIL_SYNONYMS or verdict_raw == "false":
                passed = False
            else:
                continue  # Unrecognisable verdict, skip JSON parse

            # Extract confidence
            conf_raw = data.get("confidence", 0.5)
            confidence = float(conf_raw)
            confidence = max(0.0, min(1.0, confidence))

            # Extract issues
            issues_raw = data.get("issues", [])
            if isinstance(issues_raw, list):
                issues = [str(i) for i in issues_raw if str(i).lower() != "none"]
            elif isinstance(issues_raw, str):
                issues = [] if issues_raw.lower() == "none" else [issues_raw]
            else:
                issues = []

            # Extract suggested fix
            fix_raw = data.get("suggested_fix", data.get("suggestedFix", None))
            suggested_fix = None
            if fix_raw and str(fix_raw).lower() != "none":
                suggested_fix = str(fix_raw)

            return VerificationResult(
                passed=passed,
                confidence=confidence,
                issues=issues,
                suggested_fix=suggested_fix,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return None


def _parse_confidence(raw: str) -> float:
    """Parse confidence from various formats: '0.85', '8/10', '85%'."""
    raw = raw.strip().rstrip("%")
    if "/" in raw:
        parts = raw.split("/")
        try:
            return max(0.0, min(1.0, float(parts[0].strip()) / float(parts[1].strip())))
        except (ValueError, ZeroDivisionError):
            return 0.5
    try:
        val = float(raw)
        if val > 10:
            # Percentage: 85 → 0.85
            val = val / 100.0
        elif val > 1.0 and val == int(val):
            # Integer scale: 8 → 0.8 (but NOT 1.5 which is just out-of-range)
            val = val / 10.0
        return max(0.0, min(1.0, val))
    except ValueError:
        return 0.5


def _parse_verification_response(response: str) -> VerificationResult:
    """Extract verdict, confidence, issues, and fix from the LLM response.

    Tries JSON parsing first, then falls back to regex extraction.
    Falls back to safe defaults when any field cannot be parsed.
    """
    # Try JSON first (some models return structured output)
    json_result = _try_parse_json(response)
    if json_result is not None:
        return json_result

    # -- Verdict (regex) --
    m_verdict = _VERDICT_RE.search(response)
    if m_verdict:
        verdict_word = m_verdict.group(1).lower()
        passed = verdict_word in _PASS_SYNONYMS
    else:
        passed = True  # conservative fallback

    # -- Confidence --
    m_confidence = _CONFIDENCE_RE.search(response)
    if m_confidence:
        confidence = _parse_confidence(m_confidence.group(1))
    else:
        confidence = 0.5

    # -- Issues --
    m_issues = _ISSUES_RE.search(response)
    if m_issues:
        raw = m_issues.group(1).strip()
        if raw.lower() == "none":
            issues: list[str] = []
        else:
            issues = [s.strip() for s in raw.split(",") if s.strip()]
    else:
        issues = ["verification parse error"]

    # -- Suggested fix --
    m_fix = _FIX_RE.search(response)
    if m_fix:
        raw_fix = m_fix.group(1).strip()
        suggested_fix: str | None = None if raw_fix.lower() == "none" else raw_fix
    else:
        suggested_fix = None

    # If we could not parse the verdict at all, flag that in issues
    if m_verdict is None and not issues:
        issues = ["verification parse error"]

    return VerificationResult(
        passed=passed,
        confidence=confidence,
        issues=issues,
        suggested_fix=suggested_fix,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def verify_answer(
    question: str,
    answer: str,
    source_texts: list[str],
    llm_backend,
) -> VerificationResult:
    """Run a verification LLM call to check *answer* against *source_texts*.

    Args:
        question: The user's original question.
        answer: The generated answer text to verify.
        source_texts: The raw source document texts that the answer should
            be grounded in.
        llm_backend: An LLM backend exposing an ``ask`` method
            (see :class:`doc_qa.llm.backend.LLMBackend`).

    Returns:
        A :class:`VerificationResult` indicating whether the answer passes
        verification.
    """
    if not answer.strip():
        return VerificationResult(
            passed=True,
            confidence=1.0,
            issues=[],
            suggested_fix=None,
        )

    sources_block = "\n\n---\n\n".join(source_texts) if source_texts else "(no sources provided)"

    prompt = VERIFICATION.format(
        sources=sources_block,
        answer=answer,
        question=question,
    )

    logger.info(
        "Verifying answer (%.60s...) against %d source(s).",
        answer.replace("\n", " "),
        len(source_texts),
    )

    try:
        llm_answer = await llm_backend.ask(question=prompt, context="")
    except Exception:
        logger.warning("Verification LLM call failed — assuming pass.", exc_info=True)
        return VerificationResult(
            passed=True,
            confidence=0.5,
            issues=["verification call failed"],
            suggested_fix=None,
        )

    if llm_answer.error:
        logger.warning("Verification LLM returned error: %s", llm_answer.error)
        return VerificationResult(
            passed=True,
            confidence=0.5,
            issues=[f"verification error: {llm_answer.error}"],
            suggested_fix=None,
        )

    result = _parse_verification_response(llm_answer.text)

    logger.info(
        "Verification result: passed=%s, confidence=%.2f, issues=%s",
        result.passed,
        result.confidence,
        result.issues or "none",
    )

    return result
