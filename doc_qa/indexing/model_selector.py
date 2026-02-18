"""Embedding model auto-selection based on system resources.

Resolves the ``"auto"`` sentinel to a concrete model name by checking
available RAM.  Non-``"auto"`` values pass through unchanged.
"""

from __future__ import annotations

import logging

from doc_qa.utils.system_info import get_total_ram_gb

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────

EMBEDDING_MODELS: dict[str, str] = {
    "auto": "Auto-detect (recommended)",
    "nomic-ai/nomic-embed-text-v1.5": "Nomic Embed v1.5 (768d, 8192 tokens)",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM L6 v2 (384d, 256 tokens)",
}

_MODEL_NOMIC = "nomic-ai/nomic-embed-text-v1.5"
_MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"

# Machines with >= 12 GB total RAM get the Nomic model.
# 12 GB (not 16 GB) so that a 16 GB machine comfortably qualifies
# after accounting for OS and other processes.
RAM_THRESHOLD_GB: float = 12.0


def resolve_model_name(configured: str) -> str:
    """Resolve ``"auto"`` to a concrete model name based on system RAM.

    Non-``"auto"`` values pass through unchanged.

    Returns:
        A concrete HuggingFace model identifier.
    """
    if configured != "auto":
        return configured

    ram_gb = get_total_ram_gb()
    if ram_gb >= RAM_THRESHOLD_GB:
        chosen = _MODEL_NOMIC
        logger.info(
            "Auto-select: %.1f GB RAM (>= %.0f GB threshold) → %s",
            ram_gb, RAM_THRESHOLD_GB, chosen,
        )
    else:
        chosen = _MODEL_MINILM
        logger.info(
            "Auto-select: %.1f GB RAM (< %.0f GB threshold) → %s",
            ram_gb, RAM_THRESHOLD_GB, chosen,
        )
    return chosen


def get_model_info() -> dict:
    """Return model selection metadata for the API/UI.

    Returns:
        Dict with keys: ``configured``, ``resolved``, ``ram_gb``,
        ``ram_sufficient``.
    """
    # Import here to read the *current* config (may be hot-reloaded)
    from doc_qa.config import load_config

    cfg = load_config()
    configured = cfg.indexing.embedding_model
    ram_gb = get_total_ram_gb()
    resolved = resolve_model_name(configured)

    return {
        "configured": configured,
        "resolved": resolved,
        "ram_gb": round(ram_gb, 1),
        "ram_sufficient": ram_gb >= RAM_THRESHOLD_GB,
    }
