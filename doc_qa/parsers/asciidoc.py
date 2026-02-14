"""AsciiDoc parser using Asciidoctor → DocBook XML → xml.etree."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from doc_qa.parsers.base import ParsedSection, Parser

logger = logging.getLogger(__name__)

_DOCBOOK_NS = "http://docbook.org/ns/docbook"
_NS = {"db": _DOCBOOK_NS}


def _has_asciidoctor() -> bool:
    """Check if asciidoctor is available on PATH."""
    return shutil.which("asciidoctor") is not None


def _get_text_content(element: ET.Element) -> str:
    """Recursively extract all text content from an XML element.

    Handles mixed content like <para>text <emphasis>bold</emphasis> more</para>.
    """
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        parts.append(_get_text_content(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _extract_section_content(section: ET.Element) -> str:
    """Extract text content from a DocBook section, excluding nested sections."""
    parts: list[str] = []

    for child in section:
        tag = child.tag.replace(f"{{{_DOCBOOK_NS}}}", "")

        # Skip nested sections — they'll be processed separately
        if tag == "section":
            continue
        # Skip the title — handled separately
        if tag == "title":
            continue

        if tag == "programlisting":
            # Code block — preserve as-is
            lang = child.get("language", "")
            code = _get_text_content(child).strip()
            if lang:
                parts.append(f"```{lang}\n{code}\n```")
            else:
                parts.append(f"```\n{code}\n```")
        elif tag in ("note", "tip", "warning", "important", "caution"):
            # Admonition — prefix with label
            label = tag.upper()
            text = _get_text_content(child).strip()
            parts.append(f"{label}: {text}")
        elif tag == "table":
            # Table — extract cell text row by row
            rows: list[str] = []
            for row in child.findall(".//db:row", _NS):
                cells = [_get_text_content(cell).strip() for cell in row]
                rows.append(" | ".join(cells))
            if rows:
                parts.append("\n".join(rows))
        elif tag == "itemizedlist" or tag == "orderedlist":
            # List items
            for i, item in enumerate(child.findall("db:listitem", _NS), 1):
                text = _get_text_content(item).strip()
                prefix = f"{i}." if tag == "orderedlist" else "-"
                parts.append(f"{prefix} {text}")
        else:
            # Generic element — extract text
            text = _get_text_content(child).strip()
            if text:
                parts.append(text)

    return "\n\n".join(parts)


def _walk_sections(
    element: ET.Element,
    file_path: str,
    level: int = 1,
) -> list[ParsedSection]:
    """Recursively walk DocBook sections and extract ParsedSection objects."""
    sections: list[ParsedSection] = []

    for section in element.findall("db:section", _NS):
        title_el = section.find("db:title", _NS)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        content = _extract_section_content(section)

        if title or content:
            sections.append(
                ParsedSection(
                    title=title,
                    content=content,
                    level=level,
                    file_path=file_path,
                    file_type="adoc",
                )
            )

        # Recurse into nested sections
        sections.extend(_walk_sections(section, file_path, level + 1))

    return sections


class AsciiDocParser(Parser):
    """Parse AsciiDoc files via Asciidoctor → DocBook XML → xml.etree."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".adoc", ".asciidoc", ".asc"]

    def parse(self, file_path: Path) -> list[ParsedSection]:
        if not _has_asciidoctor():
            logger.warning(
                "Asciidoctor not installed — skipping %s. "
                "Install with: gem install asciidoctor",
                file_path.name,
            )
            return []

        try:
            return self._parse_via_docbook(file_path)
        except Exception:
            logger.exception("Failed to parse %s", file_path)
            return []

    def _parse_via_docbook(self, file_path: Path) -> list[ParsedSection]:
        """Convert .adoc → DocBook XML via asciidoctor, then parse XML."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / (file_path.stem + ".docbook.xml")

            result = subprocess.run(
                [
                    "asciidoctor",
                    "-b", "docbook",
                    "-o", str(tmp_path),
                    "--base-dir", str(file_path.parent),
                    "--safe-mode", "unsafe",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(
                    "Asciidoctor failed for %s: %s",
                    file_path.name,
                    result.stderr.strip(),
                )
                return []

            tree = ET.parse(tmp_path)
            root = tree.getroot()

            # DocBook 5 root can be <article> or <book>
            sections = _walk_sections(root, str(file_path))

            # If no sections found, extract top-level content as a single section
            if not sections:
                title_el = root.find("db:title", _NS) or root.find("db:info/db:title", _NS)
                title = (
                    title_el.text.strip() if title_el is not None and title_el.text else file_path.stem
                )
                content = _extract_section_content(root)
                if content:
                    sections.append(
                        ParsedSection(
                            title=title,
                            content=content,
                            level=1,
                            file_path=str(file_path),
                            file_type="adoc",
                        )
                    )

            logger.info("Parsed %s: %d sections", file_path.name, len(sections))
            return sections
