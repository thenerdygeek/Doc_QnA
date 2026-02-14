"""Tests for document parsers."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from doc_qa.parsers.base import ParsedSection
from doc_qa.parsers.markdown import MarkdownParser
from doc_qa.parsers.plantuml import PlantUMLParser
from doc_qa.parsers.registry import get_parser, parse_file


@pytest.fixture
def tmp_md(tmp_path: Path) -> Path:
    """Create a sample Markdown file."""
    content = dedent("""\
        # Authentication

        This section describes the auth flow.

        ## JWT Tokens

        JWT tokens are used for stateless auth.

        ```java
        String token = Jwts.builder()
            .setSubject(user.getId())
            .signWith(key)
            .compact();
        ```

        ## OAuth2

        OAuth2 is used for third-party integrations.

        - Authorization Code flow
        - Client Credentials flow
    """)
    p = tmp_path / "auth.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def tmp_puml(tmp_path: Path) -> Path:
    """Create a sample PlantUML sequence diagram."""
    content = dedent("""\
        @startuml
        title Authentication Flow

        participant "User" as user
        participant "API Gateway" as gw
        participant "Auth Service" as auth
        database "User DB" as db

        user -> gw : POST /login (credentials)
        gw -> auth : validate(credentials)
        auth -> db : SELECT user WHERE email=?
        db --> auth : user record
        auth --> gw : JWT token
        gw --> user : 200 OK (token)

        note right of auth : Token expires in 24h

        @enduml
    """)
    p = tmp_path / "auth-flow.puml"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def tmp_md_no_headings(tmp_path: Path) -> Path:
    """Create a Markdown file with no headings."""
    content = "Just some plain text without any headings.\n\nAnother paragraph."
    p = tmp_path / "plain.md"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def tmp_puml_component(tmp_path: Path) -> Path:
    """Create a PlantUML component diagram."""
    content = dedent("""\
        @startuml
        title System Architecture

        component "Web App" as web
        component "API Server" as api
        database "PostgreSQL" as db

        web --> api : REST
        api --> db : JDBC

        @enduml
    """)
    p = tmp_path / "system.puml"
    p.write_text(content, encoding="utf-8")
    return p


# --- Markdown Parser Tests ---


class TestMarkdownParser:
    def test_parse_sections(self, tmp_md: Path) -> None:
        parser = MarkdownParser()
        sections = parser.parse(tmp_md)

        assert len(sections) == 3
        assert sections[0].title == "Authentication"
        assert sections[0].level == 1
        assert "auth flow" in sections[0].content

        assert sections[1].title == "JWT Tokens"
        assert sections[1].level == 2
        assert "stateless" in sections[1].content

        assert sections[2].title == "OAuth2"
        assert sections[2].level == 2
        assert "Client Credentials" in sections[2].content

    def test_code_block_preserved(self, tmp_md: Path) -> None:
        parser = MarkdownParser()
        sections = parser.parse(tmp_md)

        jwt_section = sections[1]
        assert "```java" in jwt_section.content
        assert "Jwts.builder()" in jwt_section.content

    def test_no_headings_fallback(self, tmp_md_no_headings: Path) -> None:
        parser = MarkdownParser()
        sections = parser.parse(tmp_md_no_headings)

        assert len(sections) == 1
        assert sections[0].title == "plain"  # stem of filename
        assert "plain text" in sections[0].content

    def test_file_type_is_md(self, tmp_md: Path) -> None:
        parser = MarkdownParser()
        sections = parser.parse(tmp_md)
        assert all(s.file_type == "md" for s in sections)

    def test_supported_extensions(self) -> None:
        parser = MarkdownParser()
        assert ".md" in parser.supported_extensions
        assert ".markdown" in parser.supported_extensions

    def test_can_parse(self, tmp_md: Path) -> None:
        parser = MarkdownParser()
        assert parser.can_parse(tmp_md)
        assert not parser.can_parse(Path("test.pdf"))

    def test_estimate_tokens(self, tmp_md: Path) -> None:
        parser = MarkdownParser()
        sections = parser.parse(tmp_md)
        for section in sections:
            tokens = section.estimate_tokens()
            assert tokens > 0
            assert tokens == len(section.full_text) // 4


# --- PlantUML Parser Tests ---


class TestPlantUMLParser:
    def test_parse_sequence_diagram(self, tmp_puml: Path) -> None:
        parser = PlantUMLParser()
        sections = parser.parse(tmp_puml)

        assert len(sections) == 1
        section = sections[0]
        assert section.title == "Authentication Flow"
        assert section.file_type == "puml"
        assert "Sequence diagram" in section.content

    def test_participants_extracted(self, tmp_puml: Path) -> None:
        parser = PlantUMLParser()
        sections = parser.parse(tmp_puml)
        content = sections[0].content

        assert "User" in content
        assert "API Gateway" in content
        assert "Auth Service" in content
        assert "User DB" in content

    def test_messages_extracted(self, tmp_puml: Path) -> None:
        parser = PlantUMLParser()
        sections = parser.parse(tmp_puml)
        content = sections[0].content

        assert "POST /login" in content
        assert "validate(credentials)" in content
        assert "JWT token" in content

    def test_notes_extracted(self, tmp_puml: Path) -> None:
        parser = PlantUMLParser()
        sections = parser.parse(tmp_puml)
        content = sections[0].content
        assert "Token expires in 24h" in content

    def test_component_diagram(self, tmp_puml_component: Path) -> None:
        parser = PlantUMLParser()
        sections = parser.parse(tmp_puml_component)

        assert len(sections) == 1
        content = sections[0].content
        assert "System Architecture" in sections[0].title
        assert "Web App" in content
        assert "API Server" in content
        assert "PostgreSQL" in content

    def test_supported_extensions(self) -> None:
        parser = PlantUMLParser()
        assert ".puml" in parser.supported_extensions
        assert ".plantuml" in parser.supported_extensions


# --- Registry Tests ---


class TestRegistry:
    def test_get_parser_md(self) -> None:
        parser = get_parser(Path("test.md"))
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_puml(self) -> None:
        parser = get_parser(Path("test.puml"))
        assert isinstance(parser, PlantUMLParser)

    def test_get_parser_unknown(self) -> None:
        parser = get_parser(Path("test.xyz"))
        assert parser is None

    def test_parse_file_md(self, tmp_md: Path) -> None:
        sections = parse_file(tmp_md)
        assert len(sections) == 3
        assert sections[0].title == "Authentication"

    def test_parse_file_unknown(self, tmp_path: Path) -> None:
        p = tmp_path / "test.xyz"
        p.write_text("unknown format", encoding="utf-8")
        sections = parse_file(p)
        assert sections == []

    def test_get_parser_pdf(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = get_parser(Path("report.pdf"))
        assert isinstance(parser, PDFParser)

    def test_parse_file_pdf(self, tmp_path: Path) -> None:
        """Registry routes .pdf files to PDFParser (returns [] if pdfplumber missing)."""
        p = tmp_path / "test.pdf"
        p.write_bytes(b"%PDF-1.4 fake")
        sections = parse_file(p)
        # Either returns parsed sections or [] (if pdfplumber not installed / bad PDF)
        assert isinstance(sections, list)


# --- PDF Parser Tests ---


class TestPDFParser:
    def test_supported_extensions(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        assert ".pdf" in parser.supported_extensions

    def test_can_parse_pdf(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        assert parser.can_parse(Path("report.pdf"))
        assert not parser.can_parse(Path("readme.md"))

    def test_parse_invalid_pdf_returns_empty(self, tmp_path: Path) -> None:
        from doc_qa.parsers.pdf import PDFParser

        p = tmp_path / "bad.pdf"
        p.write_bytes(b"this is not a PDF")
        parser = PDFParser()
        sections = parser.parse(p)
        assert sections == []

    def test_is_heading_heuristics(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        # Numbered heading
        assert parser._is_heading("1.2 Architecture Overview")
        # ALL CAPS heading
        assert parser._is_heading("SYSTEM REQUIREMENTS")
        # Too long — not a heading
        assert not parser._is_heading("A" * 81)
        # Too short
        assert not parser._is_heading("Hi")
        # Normal text
        assert not parser._is_heading("This is just a normal paragraph of text.")
        # Empty
        assert not parser._is_heading("")

    def test_split_by_headings_returns_sections(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        text = (
            "1.1 Introduction\n"
            "This is the introduction.\n"
            "More intro text.\n"
            "1.2 Architecture\n"
            "The system uses microservices.\n"
            "1.3 Deployment\n"
            "We deploy on Kubernetes."
        )
        sections = parser._split_by_headings(text, "test.pdf")
        assert len(sections) == 3
        assert sections[0].title == "1.1 Introduction"
        assert "introduction" in sections[0].content.lower()
        assert sections[1].title == "1.2 Architecture"
        assert sections[2].title == "1.3 Deployment"
        for s in sections:
            assert s.file_type == "pdf"
            assert s.file_path == "test.pdf"

    def test_split_by_headings_insufficient_returns_empty(self) -> None:
        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        # Only one heading — should return [] (needs ≥2)
        text = "1.1 Introduction\nSome text here."
        sections = parser._split_by_headings(text, "test.pdf")
        assert sections == []

    def test_parse_real_pdf(self, tmp_path: Path) -> None:
        """Integration test with a real (minimal) PDF created via pdfplumber-compatible format."""
        try:
            import pdfplumber
        except ImportError:
            pytest.skip("pdfplumber not installed")

        # Create a minimal real PDF using reportlab if available, otherwise skip
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
        except ImportError:
            pytest.skip("reportlab not installed (needed to create test PDFs)")

        pdf_path = tmp_path / "test_doc.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(72, 700, "1.1 Introduction")
        c.drawString(72, 680, "This document describes the system.")
        c.drawString(72, 660, "1.2 Architecture")
        c.drawString(72, 640, "The system uses a layered architecture.")
        c.save()

        from doc_qa.parsers.pdf import PDFParser

        parser = PDFParser()
        sections = parser.parse(pdf_path)
        # Should get at least page-based sections
        assert len(sections) >= 1
        assert all(s.file_type == "pdf" for s in sections)
