"""Tests for file scanner and deduplication."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_qa.config import DocRepoConfig
from doc_qa.indexing.scanner import scan_files


@pytest.fixture
def doc_repo(tmp_path: Path) -> Path:
    """Create a mock doc repo with various file types."""
    docs = tmp_path / "docs"
    docs.mkdir()

    # Normal files
    (docs / "getting-started.md").write_text("# Getting Started\nIntro text.")
    (docs / "architecture.adoc").write_text("= Architecture\nOverview.")
    (docs / "auth-flow.puml").write_text("@startuml\ntitle Auth\n@enduml")

    # Duplicate set: source + rendered
    (docs / "deploy-flow.puml").write_text("@startuml\ntitle Deploy\n@enduml")
    (docs / "deploy-flow.png").write_bytes(b"fake png")
    (docs / "deploy-flow.svg").write_text("<svg>fake</svg>")

    # Another duplicate: adoc + pdf
    (docs / "api-guide.adoc").write_text("= API Guide\nDetails.")
    (docs / "api-guide.pdf").write_bytes(b"fake pdf")

    # Build directory (should be excluded)
    build = docs / "build"
    build.mkdir()
    (build / "output.md").write_text("# Build output")

    return docs


class TestScanner:
    def test_finds_supported_files(self, doc_repo: Path) -> None:
        config = DocRepoConfig(
            path=str(doc_repo),
            patterns=["**/*.md", "**/*.adoc", "**/*.puml", "**/*.pdf"],
            exclude=["**/build/**"],
        )
        files = scan_files(config)
        names = {f.name for f in files}

        assert "getting-started.md" in names
        assert "architecture.adoc" in names
        assert "auth-flow.puml" in names

    def test_excludes_build_dir(self, doc_repo: Path) -> None:
        config = DocRepoConfig(
            path=str(doc_repo),
            patterns=["**/*.md"],
            exclude=["**/build/**"],
        )
        files = scan_files(config)
        names = {f.name for f in files}

        assert "output.md" not in names

    def test_dedup_puml_over_png_svg(self, doc_repo: Path) -> None:
        """PlantUML source should be preferred over rendered PNG/SVG."""
        config = DocRepoConfig(
            path=str(doc_repo),
            patterns=["**/*.puml", "**/*.png", "**/*.svg"],
            exclude=[],
        )
        files = scan_files(config)
        names = {f.name for f in files}

        assert "deploy-flow.puml" in names
        assert "deploy-flow.png" not in names
        assert "deploy-flow.svg" not in names

    def test_dedup_adoc_over_pdf(self, doc_repo: Path) -> None:
        """AsciiDoc source should be preferred over compiled PDF."""
        config = DocRepoConfig(
            path=str(doc_repo),
            patterns=["**/*.adoc", "**/*.pdf"],
            exclude=[],
        )
        files = scan_files(config)
        names = {f.name for f in files}

        assert "api-guide.adoc" in names
        assert "api-guide.pdf" not in names

    def test_invalid_path_raises(self) -> None:
        config = DocRepoConfig(path="/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            scan_files(config)

    def test_results_sorted(self, doc_repo: Path) -> None:
        config = DocRepoConfig(
            path=str(doc_repo),
            patterns=["**/*.md", "**/*.adoc", "**/*.puml"],
            exclude=["**/build/**"],
        )
        files = scan_files(config)
        assert files == sorted(files, key=lambda p: str(p))

    def test_empty_repo(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        config = DocRepoConfig(path=str(empty), patterns=["**/*.md"])
        files = scan_files(config)
        assert files == []
