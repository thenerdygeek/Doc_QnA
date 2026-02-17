"""Markdown parser using mistletoe AST."""

from __future__ import annotations

import logging
from pathlib import Path

from doc_qa.parsers.base import ParsedSection, Parser

logger = logging.getLogger(__name__)


def _render_inline(token: object) -> str:
    """Recursively render inline tokens to plain text."""
    # Import here to keep module-level import light
    from mistletoe.span_token import RawText

    if isinstance(token, RawText):
        return token.content

    # For tokens with children, recurse
    children = getattr(token, "children", None)
    if children:
        return "".join(_render_inline(child) for child in children)

    # Fallback
    content = getattr(token, "content", None)
    if content:
        return str(content)
    return ""


def _render_block(token: object) -> str:
    """Render a block-level token to text."""
    from mistletoe.block_token import (
        BlockCode,
        CodeFence,
        List,
        ListItem,
        Paragraph,
        Quote,
        Table,
        ThematicBreak,
    )

    if isinstance(token, Paragraph):
        return "".join(_render_inline(child) for child in token.children)

    if isinstance(token, CodeFence):
        lang = getattr(token, "language", "") or ""
        # CodeFence children are RawText with the code content
        code = "".join(_render_inline(child) for child in token.children).strip()
        return f"```{lang}\n{code}\n```"

    if isinstance(token, BlockCode):
        code = "".join(_render_inline(child) for child in token.children).strip()
        return f"```\n{code}\n```"

    if isinstance(token, Quote):
        parts = []
        for child in token.children:
            text = _render_block(child)
            for line in text.split("\n"):
                parts.append(f"> {line}")
        return "\n".join(parts)

    if isinstance(token, List):
        items: list[str] = []
        for i, child in enumerate(token.children, 1):
            if isinstance(child, ListItem):
                text = " ".join(
                    _render_block(sub).strip() for sub in child.children
                ).strip()
                # Check if ordered via the start attribute on the List
                prefix = f"{i}." if getattr(token, "start", None) is not None else "-"
                items.append(f"{prefix} {text}")
        return "\n".join(items)

    if isinstance(token, Table):
        rows: list[str] = []
        header = getattr(token, "header", None)
        if header:
            cells = [_render_inline(cell) for cell in header.children]
            rows.append(" | ".join(cells))
            rows.append(" | ".join("---" for _ in cells))
        for row in token.children:
            cells = [_render_inline(cell) for cell in row.children]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    if isinstance(token, ThematicBreak):
        return "---"

    # Fallback — try to render children
    children = getattr(token, "children", None)
    if children:
        return "\n\n".join(_render_block(child) for child in children)

    return ""


class MarkdownParser(Parser):
    """Parse Markdown files using mistletoe AST."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown"]

    def parse(self, file_path: Path) -> list[ParsedSection]:
        try:
            return self._parse_ast(file_path)
        except ImportError:
            logger.error(
                "mistletoe not installed — cannot parse %s. "
                "Install with: pip install mistletoe",
                file_path.name,
            )
            return []
        except Exception:
            logger.exception("Failed to parse %s", file_path)
            return []

    def _parse_ast(self, file_path: Path) -> list[ParsedSection]:
        """Parse markdown into heading-based sections via mistletoe AST."""
        from mistletoe import Document
        from mistletoe.block_token import Heading

        with open(file_path, encoding="utf-8") as f:
            doc = Document(f)

        sections: list[ParsedSection] = []
        current_title = ""
        current_level = 1
        current_blocks: list[str] = []

        def _flush() -> None:
            content = "\n\n".join(b for b in current_blocks if b.strip())
            if current_title or content:
                # Detect tables (lines with |) and code (``` blocks) for metadata
                has_table = any(
                    line.strip().startswith("|") and line.strip().endswith("|")
                    for line in content.split("\n")
                    if line.strip()
                )
                has_code = "```" in content
                if has_table and has_code:
                    ct = "mixed"
                elif has_table:
                    ct = "table"
                elif has_code:
                    ct = "code"
                else:
                    ct = "prose"
                meta = {
                    "content_type": ct,
                    "has_table": str(has_table).lower(),
                    "has_code": str(has_code).lower(),
                }
                sections.append(
                    ParsedSection(
                        title=current_title,
                        content=content,
                        level=current_level,
                        file_path=str(file_path),
                        file_type="md",
                        metadata=meta,
                    )
                )

        for child in doc.children:
            if isinstance(child, Heading):
                # Flush previous section
                _flush()
                current_title = _render_inline(child).strip()
                current_level = child.level
                current_blocks = []
            else:
                rendered = _render_block(child)
                if rendered.strip():
                    current_blocks.append(rendered)

        # Flush final section
        _flush()

        # If no headings found, treat entire file as one section
        if not sections or (len(sections) == 1 and not sections[0].title):
            with open(file_path, encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                sections = [
                    ParsedSection(
                        title=file_path.stem,
                        content=content,
                        level=1,
                        file_path=str(file_path),
                        file_type="md",
                    )
                ]

        logger.info("Parsed %s: %d sections", file_path.name, len(sections))
        return sections
