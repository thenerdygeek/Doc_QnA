"""PDF parser using pdfplumber for text extraction."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from doc_qa.parsers.base import ParsedSection, Parser

logger = logging.getLogger(__name__)

# Heuristic for detecting headings: short lines in ALL CAPS or title case
# that appear after whitespace and before body text.
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:\d+\.[\d.]*\s+.+)"  # numbered headings: "1.2 Foo Bar"
    r"|(?:[A-Z][A-Z\s]{2,50})"  # ALL CAPS headings
    r")$"
)


class PDFParser(Parser):
    """Parses PDF documents into sections using pdfplumber.

    Strategy:
        - Extract text page by page.
        - Attempt to detect section headings via heuristics.
        - If no headings found, treat each page as a section.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: Path) -> list[ParsedSection]:
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed — skipping %s", file_path.name)
            return []

        try:
            return self._extract_sections(file_path)
        except Exception:
            logger.warning("Failed to parse PDF %s", file_path.name, exc_info=True)
            return []

    @staticmethod
    def _extract_tables_from_page(page: object) -> list[str]:
        """Extract tables from a pdfplumber page and format as markdown.

        Uses ``page.extract_tables()`` (pdfplumber built-in) and converts
        each table to a markdown pipe table.

        Args:
            page: A pdfplumber page object.

        Returns:
            List of markdown-formatted table strings.
        """
        tables: list[str] = []
        try:
            raw_tables = page.extract_tables()  # type: ignore[attr-defined]
        except Exception:
            return []

        if not raw_tables:
            return []

        for raw_table in raw_tables:
            if not raw_table or not raw_table[0]:
                continue
            md_rows: list[str] = []
            for row_idx, row in enumerate(raw_table):
                # Replace None cells with empty string
                cells = [(cell or "").replace("\n", " ").strip() for cell in row]
                md_rows.append("| " + " | ".join(cells) + " |")
                # Add header separator after first row
                if row_idx == 0:
                    md_rows.append("| " + " | ".join("---" for _ in cells) + " |")
            tables.append("\n".join(md_rows))

        return tables

    def _extract_sections(self, file_path: Path) -> list[ParsedSection]:
        import pdfplumber

        fp = str(file_path)
        sections: list[ParsedSection] = []

        with pdfplumber.open(file_path) as pdf:
            if not pdf.pages:
                return []

            # First pass: extract all text and try heading detection
            page_texts: list[str] = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                # Extract tables from the page and append as markdown
                md_tables = self._extract_tables_from_page(page)
                page_text = text.strip()
                if md_tables:
                    page_text = page_text + "\n\n" + "\n\n".join(md_tables)
                page_texts.append(page_text)

            full_text = "\n\n".join(page_texts)

            # Try font-size-based heading detection first
            font_sections = self._split_by_font_size(pdf, fp)
            if font_sections:
                return font_sections

            # Fall back to regex-based heading detection
            heading_sections = self._split_by_headings(full_text, fp)
            if heading_sections:
                return heading_sections

            # Fallback: one section per page
            for i, text in enumerate(page_texts):
                if not text.strip():
                    continue
                sections.append(
                    ParsedSection(
                        title=f"Page {i + 1}",
                        content=text,
                        level=1,
                        file_path=fp,
                        file_type="pdf",
                        metadata=self._make_metadata(text),
                    )
                )

        return sections

    def _split_by_font_size(self, pdf, file_path: str) -> list[ParsedSection]:
        """Detect headings by font size — larger than body text = heading."""
        from collections import Counter

        # Collect font sizes across all pages
        all_sizes: list[float] = []
        page_chars: list[list[dict]] = []
        for page in pdf.pages:
            chars = page.chars or []
            page_chars.append(chars)
            all_sizes.extend(float(c.get("size", 0)) for c in chars if c.get("text", "").strip())

        if not all_sizes:
            return []

        # Most common size = body text
        size_counts = Counter(round(s, 1) for s in all_sizes)
        body_size = size_counts.most_common(1)[0][0]
        heading_threshold = body_size * 1.2

        sections: list[ParsedSection] = []
        current_title = ""
        current_lines: list[str] = []

        for chars in page_chars:
            if not chars:
                continue
            # Group chars into lines by y-coordinate (top)
            lines_by_y: dict[float, list[dict]] = {}
            for c in chars:
                y = round(float(c.get("top", 0)), 0)
                lines_by_y.setdefault(y, []).append(c)

            for y in sorted(lines_by_y):
                line_chars = lines_by_y[y]
                text = "".join(c.get("text", "") for c in line_chars).strip()
                if not text:
                    continue
                # Check if this line's average font size indicates a heading
                avg_size = sum(float(c.get("size", 0)) for c in line_chars) / len(line_chars)
                if round(avg_size, 1) >= heading_threshold and len(text) < 100:
                    # Save previous section
                    if current_lines:
                        content = "\n".join(current_lines).strip()
                        if content:
                            sections.append(ParsedSection(
                                title=current_title or file_path,
                                content=content,
                                level=1,
                                file_path=file_path,
                                file_type="pdf",
                                metadata=self._make_metadata(content),
                            ))
                    current_title = text
                    current_lines = []
                else:
                    current_lines.append(text)

        # Final section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(ParsedSection(
                    title=current_title or file_path,
                    content=content,
                    level=1,
                    file_path=file_path,
                    file_type="pdf",
                    metadata=self._make_metadata(content),
                ))

        if len(sections) >= 2:
            return sections
        return []

    def _split_by_headings(
        self, full_text: str, file_path: str
    ) -> list[ParsedSection]:
        """Try to detect headings and split text into sections."""
        lines = full_text.split("\n")
        sections: list[ParsedSection] = []
        current_title = ""
        current_lines: list[str] = []

        for line in lines:
            stripped = line.strip()
            if self._is_heading(stripped):
                # Save previous section
                if current_lines:
                    content = "\n".join(current_lines).strip()
                    if content:
                        sections.append(
                            ParsedSection(
                                title=current_title or file_path,
                                content=content,
                                level=1,
                                file_path=file_path,
                                file_type="pdf",
                                metadata=self._make_metadata(content),
                            )
                        )
                current_title = stripped
                current_lines = []
            else:
                current_lines.append(line)

        # Final section
        if current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    ParsedSection(
                        title=current_title or file_path,
                        content=content,
                        level=1,
                        file_path=file_path,
                        file_type="pdf",
                        metadata=self._make_metadata(content),
                    )
                )

        # Only use heading-based splitting if we found at least 2 headings
        if len(sections) >= 2:
            return sections
        return []

    @staticmethod
    def _make_metadata(content: str) -> dict[str, str]:
        """Build content-type metadata for a PDF section."""
        has_table = any(
            line.strip().startswith("|") and line.strip().endswith("|")
            for line in content.split("\n")
            if line.strip()
        )
        ct = "table" if has_table else "prose"
        return {
            "content_type": ct,
            "has_table": str(has_table).lower(),
            "has_code": "false",
        }

    @staticmethod
    def _is_heading(line: str) -> bool:
        """Heuristic: is this line likely a section heading?"""
        if not line or len(line) > 80:
            return False
        if len(line) < 3:
            return False
        return bool(_HEADING_RE.match(line))
