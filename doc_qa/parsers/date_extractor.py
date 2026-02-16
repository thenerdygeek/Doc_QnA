"""Extract document dates from file metadata and content.

Provides a best-effort document date for version-aware retrieval.
The date is used to prefer newer documents when near-duplicate
chunks are found across multiple files.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# PDF date format: D:YYYYMMDDHHmmSS+HH'mm' (timezone part is optional)
_PDF_DATE_RE = re.compile(
    r"D:(\d{4})(\d{2})(\d{2})"
    r"(?:(\d{2})(\d{2})(\d{2}))?"
)

# AsciiDoc date attributes (first 50 lines)
_ADOC_DATE_RE = re.compile(
    r"^:(?:revdate|date):\s*(.+)$", re.MULTILINE
)

# Common date formats to try when parsing date strings
_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%B %d, %Y",       # January 15, 2025
    "%b %d, %Y",       # Jan 15, 2025
    "%d %B %Y",         # 15 January 2025
    "%d %b %Y",         # 15 Jan 2025
    "%m/%d/%Y",
]


def extract_doc_date(file_path: str) -> float:
    """Extract the best-effort document date as a Unix timestamp.

    Priority chain:
    1. Embedded metadata (PDF creation/mod date, MD frontmatter, AsciiDoc attrs)
    2. Filesystem mtime (always available as fallback)

    Returns:
        Unix timestamp (seconds since epoch). Always > 0 due to mtime fallback.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    # Try format-specific extraction first
    embedded_date = 0.0
    try:
        if ext == ".pdf":
            embedded_date = _extract_pdf_date(file_path)
        elif ext in (".md", ".markdown"):
            embedded_date = _extract_markdown_date(file_path)
        elif ext in (".adoc", ".asciidoc"):
            embedded_date = _extract_asciidoc_date(file_path)
    except Exception as exc:
        logger.debug("Date extraction failed for %s: %s", path.name, exc)

    if embedded_date > 0:
        return embedded_date

    # Fallback: filesystem modification time
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0


def _extract_pdf_date(file_path: str) -> float:
    """Extract date from PDF metadata (/ModDate or /CreationDate)."""
    try:
        import pdfplumber
    except ImportError:
        return 0.0

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata or {}

            # Prefer ModDate (last modified) over CreationDate
            for key in ("ModDate", "modDate", "mod_date",
                        "CreationDate", "creationDate", "creation_date"):
                raw = metadata.get(key, "")
                if raw:
                    ts = _parse_pdf_date(raw)
                    if ts > 0:
                        return ts
    except Exception:
        pass

    return 0.0


def _parse_pdf_date(raw: str) -> float:
    """Parse PDF date format D:YYYYMMDDHHmmSS into Unix timestamp."""
    match = _PDF_DATE_RE.search(str(raw))
    if not match:
        # Try as a plain date string
        return _parse_date_string(str(raw))

    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)

    try:
        dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, OverflowError):
        return 0.0


def _extract_markdown_date(file_path: str) -> float:
    """Extract date from markdown YAML frontmatter."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(4096)  # Read enough for frontmatter
    except OSError:
        return 0.0

    # Check for YAML frontmatter (--- delimited)
    if not content.startswith("---"):
        return 0.0

    end = content.find("\n---", 3)
    if end == -1:
        return 0.0

    frontmatter = content[3:end]

    try:
        import yaml
        data = yaml.safe_load(frontmatter)
        if not isinstance(data, dict):
            return 0.0
    except Exception:
        return 0.0

    # Look for date fields
    for key in ("date", "updated", "last_modified", "modified", "created"):
        val = data.get(key)
        if val is None:
            continue
        if isinstance(val, datetime):
            return val.timestamp()
        if isinstance(val, date):
            # yaml.safe_load parses "date: 2025-03-15" as datetime.date
            dt = datetime(val.year, val.month, val.day, tzinfo=timezone.utc)
            return dt.timestamp()
        if isinstance(val, str):
            ts = _parse_date_string(val)
            if ts > 0:
                return ts

    return 0.0


def _extract_asciidoc_date(file_path: str) -> float:
    """Extract date from AsciiDoc :revdate: or :date: attribute."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            # Read only the header area (first 50 lines)
            header_lines = []
            for i, line in enumerate(f):
                if i >= 50:
                    break
                header_lines.append(line)
    except OSError:
        return 0.0

    header = "".join(header_lines)
    match = _ADOC_DATE_RE.search(header)
    if match:
        return _parse_date_string(match.group(1).strip())

    return 0.0


def _parse_date_string(raw: str) -> float:
    """Try multiple date formats to parse a date string into a timestamp."""
    raw = raw.strip()
    if not raw:
        return 0.0

    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            # If no timezone info, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue

    return 0.0
