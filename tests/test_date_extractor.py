"""Tests for document date extraction."""

import os
import tempfile
import time

import pytest

from doc_qa.parsers.date_extractor import (
    _extract_asciidoc_date,
    _extract_markdown_date,
    _parse_date_string,
    _parse_pdf_date,
    extract_doc_date,
)


class TestParseDateString:
    def test_iso_date(self):
        ts = _parse_date_string("2025-03-15")
        assert ts > 0
        # March 15 2025 — just verify it's in the right ballpark (2025)
        assert 1735689600 < ts < 1767225600  # between Jan 1 2025 and Jan 1 2026

    def test_iso_datetime(self):
        ts = _parse_date_string("2025-03-15T10:30:00")
        assert ts > 0

    def test_us_date(self):
        ts = _parse_date_string("03/15/2025")
        assert ts > 0

    def test_verbose_date(self):
        ts = _parse_date_string("January 15, 2025")
        assert ts > 0

    def test_empty_string(self):
        assert _parse_date_string("") == 0.0

    def test_garbage(self):
        assert _parse_date_string("not a date") == 0.0

    def test_iso_with_timezone(self):
        ts = _parse_date_string("2025-03-15T10:30:00Z")
        assert ts > 0


class TestParsePdfDate:
    def test_standard_format(self):
        ts = _parse_pdf_date("D:20250315103000")
        assert ts > 0

    def test_date_only(self):
        ts = _parse_pdf_date("D:20250315")
        assert ts > 0

    def test_with_timezone(self):
        ts = _parse_pdf_date("D:20250315103000+05'30'")
        assert ts > 0

    def test_empty(self):
        assert _parse_pdf_date("") == 0.0

    def test_fallback_to_string_parse(self):
        ts = _parse_pdf_date("2025-03-15")
        assert ts > 0


class TestExtractMarkdownDate:
    def test_with_frontmatter(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\ntitle: Test\ndate: 2025-03-15\n---\n\n# Hello\n")
        ts = _extract_markdown_date(str(md))
        assert ts > 0

    def test_updated_field(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\nupdated: 2025-06-01\n---\n\nContent\n")
        ts = _extract_markdown_date(str(md))
        assert ts > 0

    def test_no_frontmatter(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Hello\n\nNo frontmatter here.\n")
        assert _extract_markdown_date(str(md)) == 0.0

    def test_empty_frontmatter(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\ntitle: Test\n---\n\nNo date.\n")
        assert _extract_markdown_date(str(md)) == 0.0


class TestExtractAsciidocDate:
    def test_revdate(self, tmp_path):
        adoc = tmp_path / "doc.adoc"
        adoc.write_text("= Title\n:revdate: 2025-03-15\n\nContent\n")
        ts = _extract_asciidoc_date(str(adoc))
        assert ts > 0

    def test_date_attribute(self, tmp_path):
        adoc = tmp_path / "doc.adoc"
        adoc.write_text("= Title\n:date: 2025-03-15\n\nContent\n")
        ts = _extract_asciidoc_date(str(adoc))
        assert ts > 0

    def test_no_date(self, tmp_path):
        adoc = tmp_path / "doc.adoc"
        adoc.write_text("= Title\n\nNo date attribute.\n")
        assert _extract_asciidoc_date(str(adoc)) == 0.0


class TestExtractDocDate:
    def test_markdown_with_date(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("---\ndate: 2025-03-15\n---\n\n# Test\n")
        ts = extract_doc_date(str(md))
        assert ts > 0

    def test_falls_back_to_mtime(self, tmp_path):
        txt = tmp_path / "plain.txt"
        txt.write_text("No embedded date possible.")
        ts = extract_doc_date(str(txt))
        # Should be close to current time (file just created)
        assert abs(ts - time.time()) < 5

    def test_asciidoc_with_date(self, tmp_path):
        adoc = tmp_path / "doc.adoc"
        adoc.write_text("= Title\n:revdate: 2025-01-01\n\nContent\n")
        ts = extract_doc_date(str(adoc))
        assert ts > 0

    def test_nonexistent_file(self):
        ts = extract_doc_date("/nonexistent/file.pdf")
        assert ts == 0.0
