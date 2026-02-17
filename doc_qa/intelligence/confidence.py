"""Confidence scoring combining retrieval and verification signals.

Produces a composite score that can drive an abstention decision when the
system is not confident enough to provide a reliable answer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from doc_qa.config import VerificationConfig
from doc_qa.verification.verifier import VerificationResult

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceAssessment:
    """Combined confidence assessment for a generated answer."""

    score: float
    retrieval_signal: float
    verification_signal: float
    should_abstain: bool
    abstain_reason: str | None = None
    caveat_added: bool = False


def compute_confidence(
    retrieval_scores: list[float],
    verification: VerificationResult | None,
    config: VerificationConfig,
) -> ConfidenceAssessment:
    """Compute a confidence score from retrieval and verification signals.

    The final score is a weighted combination:
        ``0.4 * retrieval_signal + 0.6 * verification_signal``

    Retrieval signal penalties:
        - All scores below 0.3: heavy penalty (signal halved).
        - Single-source reliance (top-1 >> top-2): 0.15 penalty.

    Verification signal:
        - If *verification* is ``None``, uses 0.7 as a neutral default.
        - If present, uses ``verification.confidence`` with a 0.2 penalty
          if the verification did not pass.

    Args:
        retrieval_scores: Similarity scores from retrieval (assumed 0-1),
            in descending order.
        verification: Optional verification result from the verify step.
        config: Verification configuration (thresholds, abstain flag).

    Returns:
        A :class:`ConfidenceAssessment` with the composite score and
        abstention decision.
    """
    retrieval_signal = _compute_retrieval_signal(retrieval_scores)
    verification_signal = _compute_verification_signal(verification)

    combined = 0.4 * retrieval_signal + 0.6 * verification_signal
    combined = max(0.0, min(1.0, combined))

    should_abstain = False
    caveat_added = False
    abstain_reason: str | None = None

    caveat_threshold = getattr(config, "caveat_threshold", 0.4)

    if combined >= config.confidence_threshold:
        # High confidence — normal answer
        pass
    elif combined >= caveat_threshold:
        # Moderate confidence — answer with caveat
        caveat_added = True
    elif config.abstain_on_low_confidence:
        # Low confidence — abstain
        should_abstain = True
        abstain_reason = (
            f"Confidence {combined:.2f} is below threshold "
            f"{caveat_threshold:.2f}"
        )

    assessment = ConfidenceAssessment(
        score=combined,
        retrieval_signal=retrieval_signal,
        verification_signal=verification_signal,
        should_abstain=should_abstain,
        abstain_reason=abstain_reason,
        caveat_added=caveat_added,
    )

    logger.info(
        "Confidence: score=%.2f (retrieval=%.2f, verification=%.2f), "
        "abstain=%s, caveat=%s",
        combined,
        retrieval_signal,
        verification_signal,
        should_abstain,
        caveat_added,
    )

    return assessment


def _compute_retrieval_signal(scores: list[float]) -> float:
    """Derive the retrieval confidence signal from similarity scores.

    Args:
        scores: Retrieval scores in descending order (0-1 range).

    Returns:
        A float in [0, 1].
    """
    if not scores:
        return 0.0

    # Average of all scores as baseline.
    avg = sum(scores) / len(scores)

    signal = avg

    # Penalty: all scores below 0.3 => halve the signal.
    if all(s < 0.3 for s in scores):
        signal *= 0.5

    # Penalty: single-source reliance.
    # If top-1 is much larger than top-2, the answer relies on a single source
    # which is risky.
    if len(scores) >= 2:
        gap = scores[0] - scores[1]
        if gap > 0.3:
            signal = max(0.0, signal - 0.15)

    return max(0.0, min(1.0, signal))


def _compute_verification_signal(verification: VerificationResult | None) -> float:
    """Derive the verification confidence signal.

    Args:
        verification: The verification result, or ``None`` if verification
            was not performed.

    Returns:
        A float in [0, 1].
    """
    if verification is None:
        return 0.7  # neutral default

    signal = verification.confidence

    if not verification.passed:
        signal = max(0.0, signal - 0.2)

    return max(0.0, min(1.0, signal))
