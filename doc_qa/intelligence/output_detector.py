"""Post-generation format detection via regex.

Scans LLM-generated response text for structured content blocks
(Mermaid diagrams, code blocks, Markdown tables, numbered lists).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DetectedFormats:
    """Formats detected in an LLM response."""

    has_mermaid: bool = False
    has_code_blocks: bool = False
    has_table: bool = False
    has_numbered_list: bool = False
    mermaid_blocks: list[str] = field(default_factory=list)
    code_blocks: list[tuple[str, str]] = field(default_factory=list)
    code_languages: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_MERMAID_RE = re.compile(
    r"```mermaid\s*\n(.*?)```",
    re.DOTALL,
)

_CODE_BLOCK_RE = re.compile(
    r"```(\w*)\s*\n(.*?)```",
    re.DOTALL,
)

# Markdown table: header row, separator row (---|---), and data rows
# The separator must contain at least one column with dashes.
_TABLE_SEPARATOR_RE = re.compile(
    r"^\|.*\|\s*\n\|[\s\-:|]+\|\s*\n(?:\|.*\|\s*\n?)+",
    re.MULTILINE,
)

# Numbered list: at least 3 consecutive numbered items (1. ... 2. ... 3. ...)
_NUMBERED_LIST_RE = re.compile(
    r"(?:^|\n)\s*1[.)]\s+.+\n\s*2[.)]\s+.+\n\s*3[.)]\s+.+",
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def detect_response_formats(text: str) -> DetectedFormats:
    """Detect structured content formats in LLM-generated response text.

    Args:
        text: The full response text from the LLM.

    Returns:
        A ``DetectedFormats`` instance describing what was found.
    """
    result = DetectedFormats()

    # Mermaid diagrams
    mermaid_matches = _MERMAID_RE.findall(text)
    if mermaid_matches:
        result.has_mermaid = True
        result.mermaid_blocks = [m.strip() for m in mermaid_matches]

    # Code blocks (excluding mermaid)
    for lang, content in _CODE_BLOCK_RE.findall(text):
        lang_lower = lang.lower()
        if lang_lower == "mermaid":
            continue
        result.code_blocks.append((lang, content.strip()))
        if lang:
            result.code_languages.add(lang_lower)

    if result.code_blocks:
        result.has_code_blocks = True

    # Markdown tables
    if _TABLE_SEPARATOR_RE.search(text):
        result.has_table = True

    # Numbered lists (>=3 consecutive)
    if _NUMBERED_LIST_RE.search(text):
        result.has_numbered_list = True

    return result
