"""Model name mapping for Cody and Ollama backends."""

from __future__ import annotations

import re

# ── Provider prefix mapping ──────────────────────────────────────

_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "fireworks": "Fireworks",
    "mistral": "Mistral",
    "meta": "Meta",
    "amazon": "Amazon",
    "cohere": "Cohere",
}

# Keywords that indicate a thinking/reasoning model
_THINKING_KEYWORDS = {"thinking", "think", "reasoning", "o1", "o3", "deepseek-r1"}


def format_cody_model(raw_id: str, capabilities: dict | None = None) -> dict:
    """Parse a Cody model ID into a display-friendly dict.

    Args:
        raw_id: Raw model ID like ``"anthropic::2025-01-01::claude-3.5-sonnet"``.
        capabilities: Optional capabilities dict from the ``chat/models`` RPC.

    Returns:
        ``{"id": str, "displayName": str, "provider": str, "thinking": bool}``
    """
    parts = raw_id.split("::")
    model_slug = parts[-1] if parts else raw_id

    # Provider from first segment
    provider_key = parts[0].lower() if parts else ""
    provider = _PROVIDER_MAP.get(provider_key, provider_key.title())

    # Build display name: "claude-3.5-sonnet" → "Claude 3.5 Sonnet"
    display_name = _humanize_model_name(model_slug)

    # Detect thinking capability
    thinking = False
    if capabilities and capabilities.get("thinking"):
        thinking = True
    elif any(kw in raw_id.lower() for kw in _THINKING_KEYWORDS):
        thinking = True

    return {
        "id": raw_id,
        "displayName": display_name,
        "provider": provider,
        "thinking": thinking,
    }


def _humanize_model_name(slug: str) -> str:
    """Convert a model slug like ``claude-3.5-sonnet`` to ``Claude 3.5 Sonnet``."""
    # Replace hyphens with spaces, but keep version numbers together (e.g., "3.5")
    # Strategy: split on hyphens, title-case words, rejoin
    tokens = slug.split("-")
    result: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # If token is a number and next token starts with a digit, join as version
        if re.match(r"^\d+$", token) and i + 1 < len(tokens) and re.match(r"^\d", tokens[i + 1]):
            result.append(f"{token}.{tokens[i + 1]}")
            i += 2
        else:
            result.append(token.capitalize() if not re.match(r"^\d", token) else token)
            i += 1
    return " ".join(result)


def format_ollama_model(model_data: dict) -> dict:
    """Parse an Ollama ``/api/tags`` model entry into a display-friendly dict.

    Args:
        model_data: Single model entry from Ollama's ``/api/tags`` response.

    Returns:
        ``{"id": str, "displayName": str, "family": str, "size": str}``
    """
    name: str = model_data.get("name", "")

    # Split "qwen2.5:7b" → base="qwen2.5", tag="7b"
    if ":" in name:
        base, tag = name.rsplit(":", 1)
    else:
        base, tag = name, ""

    display_name = _humanize_model_name(base)
    if tag and tag != "latest":
        display_name += f" ({tag})"

    details = model_data.get("details", {})
    family = details.get("family", "")
    size = details.get("parameter_size", "")

    return {
        "id": name,
        "displayName": display_name,
        "family": family,
        "size": size,
    }
