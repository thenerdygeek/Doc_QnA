"""Parser registry — dispatches files to the correct parser by extension."""

from __future__ import annotations

import logging
from pathlib import Path

from doc_qa.parsers.asciidoc import AsciiDocParser
from doc_qa.parsers.base import ParsedSection, Parser
from doc_qa.parsers.markdown import MarkdownParser
from doc_qa.parsers.pdf import PDFParser
from doc_qa.parsers.plantuml import PlantUMLParser

logger = logging.getLogger(__name__)

_PARSERS: list[Parser] = [
    AsciiDocParser(),
    MarkdownParser(),
    PlantUMLParser(),
    PDFParser(),
]


def get_parser(file_path: Path) -> Parser | None:
    """Find a parser that can handle the given file."""
    for parser in _PARSERS:
        if parser.can_parse(file_path):
            return parser
    return None


def parse_file(file_path: Path) -> list[ParsedSection]:
    """Parse a file using the appropriate parser.

    Returns empty list if no parser is available or parsing fails.
    """
    path = Path(file_path)
    parser = get_parser(path)
    if parser is None:
        logger.debug("No parser for %s (extension: %s)", path.name, path.suffix)
        return []
    return parser.parse(path)
