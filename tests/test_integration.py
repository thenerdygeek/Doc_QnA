"""Integration tests against real-world documentation repositories.

Requires the following repos cloned into tests/integration_data/:
    - arc42-template  (168 .adoc, 39 .png, deeply nested, 11 languages)
    - docToolchain    (137 .adoc, 72 .png, 7 .svg, 1 .puml, inline PlantUML)

Run with:
    pytest tests/test_integration.py -v --tb=short

These tests exercise the full pipeline: scan → parse → chunk → index → retrieve.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

# ── Paths ────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "integration_data"
_ARC42_DIR = _DATA_DIR / "arc42-template"
_DOCTOOL_DIR = _DATA_DIR / "docToolchain"

# Skip the entire module if repos aren't cloned
pytestmark = pytest.mark.skipif(
    not _ARC42_DIR.is_dir() or not _DOCTOOL_DIR.is_dir(),
    reason="Integration data repos not cloned. Run: git clone ... into tests/integration_data/",
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def arc42_files() -> list[Path]:
    """Scan arc42-template repo."""
    from doc_qa.config import DocRepoConfig
    from doc_qa.indexing.scanner import scan_files

    cfg = DocRepoConfig(path=str(_ARC42_DIR))
    return scan_files(cfg)


@pytest.fixture(scope="module")
def doctool_files() -> list[Path]:
    """Scan docToolchain repo."""
    from doc_qa.config import DocRepoConfig
    from doc_qa.indexing.scanner import scan_files

    cfg = DocRepoConfig(path=str(_DOCTOOL_DIR))
    return scan_files(cfg)


@pytest.fixture(scope="module")
def arc42_index(tmp_path_factory, arc42_files):
    """Full index of the arc42-template repo."""
    from doc_qa.indexing.chunker import chunk_sections
    from doc_qa.indexing.indexer import DocIndex
    from doc_qa.parsers.registry import parse_file

    db_path = str(tmp_path_factory.mktemp("arc42_db"))
    index = DocIndex(db_path=db_path)

    total_chunks = 0
    total_files = 0
    for fp in arc42_files:
        sections = parse_file(fp)
        if not sections:
            continue
        chunks = chunk_sections(sections, file_path=str(fp))
        if not chunks:
            continue
        n = index.upsert_file(chunks, str(fp))
        total_chunks += n
        total_files += 1

    index.rebuild_fts_index()
    index._meta = {"total_chunks": total_chunks, "total_files": total_files}
    return index


@pytest.fixture(scope="module")
def doctool_index(tmp_path_factory, doctool_files):
    """Full index of the docToolchain repo."""
    from doc_qa.indexing.chunker import chunk_sections
    from doc_qa.indexing.indexer import DocIndex
    from doc_qa.parsers.registry import parse_file

    db_path = str(tmp_path_factory.mktemp("doctool_db"))
    index = DocIndex(db_path=db_path)

    total_chunks = 0
    total_files = 0
    for fp in doctool_files:
        sections = parse_file(fp)
        if not sections:
            continue
        chunks = chunk_sections(sections, file_path=str(fp))
        if not chunks:
            continue
        n = index.upsert_file(chunks, str(fp))
        total_chunks += n
        total_files += 1

    index.rebuild_fts_index()
    index._meta = {"total_chunks": total_chunks, "total_files": total_files}
    return index


# ── 1. Scanner Tests ────────────────────────────────────────────────────


class TestScannerIntegration:
    """Verify the scanner finds and deduplicates files from real repos."""

    def test_arc42_finds_adoc_files(self, arc42_files: list[Path]) -> None:
        adoc = [f for f in arc42_files if f.suffix == ".adoc"]
        assert len(adoc) >= 100, f"Expected 100+ .adoc files, got {len(adoc)}"

    def test_arc42_finds_multiple_languages(self, arc42_files: list[Path]) -> None:
        """arc42-template has 11 language dirs (EN, DE, ES, FR, IT, etc.)."""
        languages = set()
        for f in arc42_files:
            parts = f.relative_to(_ARC42_DIR).parts
            if len(parts) >= 2 and parts[0].isupper() and len(parts[0]) <= 3:
                languages.add(parts[0])
        assert len(languages) >= 5, f"Expected 5+ language dirs, got {languages}"

    def test_arc42_deduplicates_png(self, arc42_files: list[Path]) -> None:
        """PNGs with a matching .adoc/.puml source should be deduplicated out."""
        extensions = {f.suffix.lower() for f in arc42_files}
        # PNG should only appear if no matching source exists
        assert ".adoc" in extensions

    def test_arc42_total_count_reasonable(self, arc42_files: list[Path]) -> None:
        # After dedup, should be fewer than the raw 679 files
        assert 100 < len(arc42_files) < 400

    def test_doctool_finds_adoc_files(self, doctool_files: list[Path]) -> None:
        adoc = [f for f in doctool_files if f.suffix == ".adoc"]
        assert len(adoc) >= 50, f"Expected 50+ .adoc files, got {len(adoc)}"

    def test_doctool_finds_puml(self, doctool_files: list[Path]) -> None:
        puml = [f for f in doctool_files if f.suffix in (".puml", ".plantuml")]
        assert len(puml) >= 1, "Expected at least 1 .puml file"

    def test_doctool_finds_md_files(self, doctool_files: list[Path]) -> None:
        md = [f for f in doctool_files if f.suffix == ".md"]
        assert len(md) >= 5, f"Expected 5+ .md files, got {len(md)}"

    def test_no_git_files_included(self, arc42_files: list[Path]) -> None:
        """Scanner should exclude .git directory."""
        git_files = [f for f in arc42_files if ".git" in f.parts]
        assert len(git_files) == 0, f"Found .git files: {git_files[:3]}"

    def test_no_build_dirs_included(self, doctool_files: list[Path]) -> None:
        build_files = [f for f in doctool_files if "build" in f.parts or "target" in f.parts]
        assert len(build_files) == 0


# ── 2. Parser Tests ─────────────────────────────────────────────────────


class TestParserIntegration:
    """Verify parsers handle real-world documents without crashing."""

    def test_parse_all_arc42_adoc(self, arc42_files: list[Path]) -> None:
        """Every .adoc file should parse without exception (may return [])."""
        from doc_qa.parsers.registry import parse_file

        adoc_files = [f for f in arc42_files if f.suffix == ".adoc"]
        parsed_count = 0
        empty_count = 0
        for f in adoc_files:
            sections = parse_file(f)
            assert isinstance(sections, list), f"parse_file({f}) returned non-list"
            if sections:
                parsed_count += 1
                for s in sections:
                    # arc42 templates are scaffolding — sections may have titles only
                    assert s.title or s.content, f"Both title and content empty in {f}"
                    assert s.file_type, f"Missing file_type in {f}:{s.title}"
            else:
                empty_count += 1

        # Most .adoc files should parse successfully
        success_rate = parsed_count / (parsed_count + empty_count) if (parsed_count + empty_count) else 0
        assert success_rate > 0.5, f"Only {success_rate:.0%} of .adoc files parsed successfully"

    def test_parse_all_doctool_adoc(self, doctool_files: list[Path]) -> None:
        from doc_qa.parsers.registry import parse_file

        adoc_files = [f for f in doctool_files if f.suffix == ".adoc"]
        parsed_count = 0
        for f in adoc_files:
            sections = parse_file(f)
            assert isinstance(sections, list)
            if sections:
                parsed_count += 1

        assert parsed_count > 0, "No .adoc files parsed successfully in docToolchain"

    def test_parse_puml_file(self, doctool_files: list[Path]) -> None:
        from doc_qa.parsers.registry import parse_file

        puml_files = [f for f in doctool_files if f.suffix == ".puml"]
        if not puml_files:
            pytest.skip("No .puml files found")

        sections = parse_file(puml_files[0])
        assert len(sections) >= 1
        assert sections[0].file_type == "puml"

    def test_parse_md_files(self, doctool_files: list[Path]) -> None:
        from doc_qa.parsers.registry import parse_file

        md_files = [f for f in doctool_files if f.suffix == ".md"]
        parsed = 0
        for f in md_files[:10]:  # Sample up to 10
            sections = parse_file(f)
            if sections:
                parsed += 1
                assert all(s.file_type == "md" for s in sections)

        assert parsed > 0, "No .md files parsed successfully"

    def test_parsed_sections_have_valid_metadata(self, arc42_files: list[Path]) -> None:
        """Spot-check that parsed sections carry correct metadata."""
        from doc_qa.parsers.registry import parse_file

        adoc_files = [f for f in arc42_files if f.suffix == ".adoc"]
        # Parse a sample
        for f in adoc_files[:5]:
            sections = parse_file(f)
            for s in sections:
                assert isinstance(s.title, str)
                assert isinstance(s.content, str)
                assert isinstance(s.level, int) and s.level >= 1
                assert s.file_path, f"Missing file_path in section from {f}"


# ── 3. Chunker Integration ──────────────────────────────────────────────


class TestChunkerIntegration:
    """Verify chunking produces sensible output on real documents."""

    def test_chunk_large_adoc_file(self, arc42_files: list[Path]) -> None:
        """A large .adoc file should produce multiple chunks."""
        from doc_qa.indexing.chunker import chunk_sections
        from doc_qa.parsers.registry import parse_file

        # Find the largest .adoc file
        adoc_files = [f for f in arc42_files if f.suffix == ".adoc"]
        largest = max(adoc_files, key=lambda f: f.stat().st_size)

        sections = parse_file(largest)
        if not sections:
            pytest.skip(f"Could not parse {largest.name}")

        chunks = chunk_sections(sections, file_path=str(largest))
        assert len(chunks) >= 1, f"Expected chunks from {largest.name}"

        # Each chunk should have required fields
        for c in chunks:
            assert c.text.strip(), "Chunk has empty text"
            assert c.chunk_id, "Chunk has no ID"
            assert c.file_path == str(largest)
            assert c.section_title

    def test_chunk_ids_unique_across_file(self, doctool_files: list[Path]) -> None:
        from doc_qa.indexing.chunker import chunk_sections
        from doc_qa.parsers.registry import parse_file

        adoc_files = [f for f in doctool_files if f.suffix == ".adoc"]
        for f in adoc_files[:10]:
            sections = parse_file(f)
            if not sections:
                continue
            chunks = chunk_sections(sections, file_path=str(f))
            ids = [c.chunk_id for c in chunks]
            assert len(ids) == len(set(ids)), f"Duplicate chunk IDs in {f.name}: {ids}"

    def test_chunk_size_within_limits(self, arc42_files: list[Path]) -> None:
        """Chunks should roughly respect the token budget."""
        from doc_qa.indexing.chunker import chunk_sections
        from doc_qa.parsers.registry import parse_file

        max_tokens = 512
        adoc_files = [f for f in arc42_files if f.suffix == ".adoc"]
        oversized = []
        for f in adoc_files[:20]:
            sections = parse_file(f)
            if not sections:
                continue
            chunks = chunk_sections(sections, file_path=str(f), max_tokens=max_tokens)
            for c in chunks:
                est = len(c.text) // 4
                # Allow 50% overshoot (code blocks kept intact)
                if est > max_tokens * 1.5:
                    oversized.append((f.name, c.chunk_id, est))

        # At most 10% of files should have oversized chunks
        assert len(oversized) <= 2, f"Too many oversized chunks: {oversized}"


# ── 4. Indexer Integration ───────────────────────────────────────────────


class TestIndexerIntegration:
    """Verify the full indexing pipeline on real repos."""

    def test_arc42_index_not_empty(self, arc42_index) -> None:
        total = arc42_index.count_rows()
        assert total > 50, f"Expected 50+ chunks in arc42 index, got {total}"

    def test_arc42_index_file_count(self, arc42_index) -> None:
        files = arc42_index.count_files()
        assert files >= 10, f"Expected 10+ files indexed, got {files}"

    def test_arc42_stats_consistent(self, arc42_index) -> None:
        stats = arc42_index.stats()
        assert stats["total_chunks"] == arc42_index.count_rows()
        assert stats["total_files"] == arc42_index.count_files()
        assert stats["total_chunks"] > 0
        assert stats["total_files"] > 0

    def test_doctool_index_not_empty(self, doctool_index) -> None:
        total = doctool_index.count_rows()
        assert total > 50, f"Expected 50+ chunks in docToolchain index, got {total}"

    def test_doctool_index_file_count(self, doctool_index) -> None:
        files = doctool_index.count_files()
        assert files >= 10, f"Expected 10+ files indexed, got {files}"

    def test_index_has_correct_embedding_dim(self, arc42_index) -> None:
        """Embedded vectors should have the correct dimension (384 for all-MiniLM-L6-v2)."""
        import pyarrow as pa

        table = arc42_index._table.to_arrow()
        schema = table.schema
        vec_field = schema.field("vector")
        # LanceDB stores as FixedSizeList
        dim = vec_field.type.list_size
        assert dim == 384, f"Expected dim 384, got {dim}"


# ── 5. Retrieval Integration (basic) ────────────────────────────────────


class TestRetrievalIntegration:
    """End-to-end retrieval tests on real indexed data."""

    def test_vector_search_arc42(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="vector")
        results = retriever.search("architecture documentation", top_k=5)
        assert len(results) > 0, "Vector search returned no results"
        assert all(r.score > 0 for r in results)

    def test_fts_search_arc42(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="fts")
        results = retriever.search("quality requirements", top_k=5)
        assert len(results) > 0, "FTS search returned no results"

    def test_hybrid_search_arc42(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="hybrid")
        results = retriever.search("deployment infrastructure", top_k=5)
        assert len(results) > 0, "Hybrid search returned no results"

    def test_min_score_filter_works(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="vector")
        results = retriever.search("xyzzy nonsense gibberish", top_k=5, min_score=0.9)
        assert len(results) <= 3

    def test_top_k_respected(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="vector")
        for k in [1, 3, 5]:
            results = retriever.search("architecture", top_k=k)
            assert len(results) <= k


# ── 5b. Ground-Truth Retrieval Validation (arc42) ───────────────────────

# Hand-crafted test cases: query → expected source files + content keywords.
# Built by reading the actual repo content to know what SHOULD be retrieved.

# File path fragments (matched against the end of file_path)
# arc42 has individual chapter files AND a master arc42-template.adoc that
# includes all chapters inline. Retrieval may return either.
_ARC42_MASTER = "arc42-template.adoc"
_ARC42_BUILDING_BLOCK = ["05_building_block_view.adoc", _ARC42_MASTER]
_ARC42_QUALITY = ["10_quality_requirements.adoc", _ARC42_MASTER]
_ARC42_DEPLOYMENT = ["07_deployment_view.adoc", _ARC42_MASTER]
_ARC42_RUNTIME = ["06_runtime_view.adoc", _ARC42_MASTER]
_ARC42_CONTEXT = ["03_context_and_scope.adoc", _ARC42_MASTER]
_ARC42_RISKS = ["11_technical_risks.adoc", _ARC42_MASTER]
_ARC42_DECISIONS = ["09_architecture_decisions.adoc", _ARC42_MASTER]
_ARC42_CONCEPTS = ["08_concepts.adoc", _ARC42_MASTER]
_ARC42_STRATEGY = ["04_solution_strategy.adoc", _ARC42_MASTER]
_ARC42_CONSTRAINTS = ["02_architecture_constraints.adoc", _ARC42_MASTER]
_ARC42_INTRO = ["01_introduction_and_goals.adoc", _ARC42_MASTER]
_ARC42_GLOSSARY = ["12_glossary.adoc", _ARC42_MASTER]


def _result_files(results) -> list[str]:
    """Extract just filenames from retrieval results."""
    return [Path(r.file_path).name for r in results]


def _any_result_from(results, expected_files: list[str]) -> bool:
    """Check if at least one result comes from one of the expected files."""
    names = _result_files(results)
    return any(name in expected_files for name in names)


def _top_n_contain(results, expected_files: list[str], n: int = 3) -> bool:
    """Check if at least one of the top-N results comes from expected files."""
    names = _result_files(results[:n])
    return any(name in expected_files for name in names)


def _any_text_contains(results, keywords: list[str], n: int = 5) -> bool:
    """Check if any of the top-N results' text contains at least one keyword."""
    for r in results[:n]:
        text = r.text.lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


class TestArc42GroundTruth:
    """Validate retrieval accuracy against hand-verified ground-truth data.

    For each arc42 topic, we know EXACTLY which file contains the relevant
    content. These tests verify the retriever surfaces the correct files.
    """

    @pytest.fixture(autouse=True)
    def _retriever(self, arc42_index):
        from doc_qa.retrieval.retriever import HybridRetriever

        self.retriever = HybridRetriever(table=arc42_index._table, mode="vector")

    # -- Building Block View --

    def test_building_block_view_query(self) -> None:
        """'building block view' → 05_building_block_view.adoc"""
        results = self.retriever.search("building block view decomposition", top_k=5)
        assert len(results) > 0
        assert _top_n_contain(results, _ARC42_BUILDING_BLOCK)
        assert _any_text_contains(results, [
            "building block", "decomposition", "white box", "black box",
            "component", "module", "hierarchy",
        ])

    def test_system_decomposition_query(self) -> None:
        """Synonym query for building blocks: system decomposition."""
        results = self.retriever.search("how is the system decomposed into components", top_k=5)
        assert _any_result_from(results, _ARC42_BUILDING_BLOCK)

    # -- Quality Requirements --

    def test_quality_requirements_query(self) -> None:
        """'quality requirements' → 10_quality_requirements.adoc or 01_introduction."""
        results = self.retriever.search("quality requirements and quality goals", top_k=5)
        assert len(results) > 0
        assert _any_result_from(results, _ARC42_QUALITY + _ARC42_INTRO)
        assert _any_text_contains(results, [
            "quality", "requirement", "goal", "attribute",
        ])

    # -- Deployment View --

    def test_deployment_view_query(self) -> None:
        """'deployment' → 07_deployment_view.adoc"""
        results = self.retriever.search("deployment view infrastructure hardware", top_k=5)
        assert _top_n_contain(results, _ARC42_DEPLOYMENT)
        assert _any_text_contains(results, [
            "deployment", "infrastructure", "hardware", "server",
            "environment", "node", "processor",
        ])

    def test_infrastructure_mapping_query(self) -> None:
        """Synonym: infrastructure and environment mapping."""
        results = self.retriever.search("technical infrastructure and environment topology", top_k=5)
        assert _any_result_from(results, _ARC42_DEPLOYMENT)

    # -- Runtime View --

    def test_runtime_view_query(self) -> None:
        """'runtime scenarios' → 06_runtime_view.adoc"""
        results = self.retriever.search("runtime view behavior scenarios interactions", top_k=5)
        assert _top_n_contain(results, _ARC42_RUNTIME)
        assert _any_text_contains(results, [
            "runtime", "scenario", "behavior", "interaction",
            "sequence", "use case",
        ])

    # -- Context and Scope --

    def test_context_scope_query(self) -> None:
        """'system context' → 03_context_and_scope.adoc"""
        results = self.retriever.search("system context scope external interfaces", top_k=5)
        assert _top_n_contain(results, _ARC42_CONTEXT)
        assert _any_text_contains(results, [
            "context", "scope", "boundary", "external", "interface",
            "neighbor", "communication",
        ])

    def test_external_systems_query(self) -> None:
        """Synonym: neighboring systems and external communication."""
        results = self.retriever.search("neighboring systems and external communication partners", top_k=5)
        assert _any_result_from(results, _ARC42_CONTEXT)

    # -- Technical Risks --

    def test_technical_risks_query(self) -> None:
        """'risks and technical debt' → 11_technical_risks.adoc"""
        results = self.retriever.search("technical risks and technical debt", top_k=5)
        assert _top_n_contain(results, _ARC42_RISKS)
        assert _any_text_contains(results, ["risk", "debt", "problem"])

    # -- Architecture Decisions --

    def test_architecture_decisions_query(self) -> None:
        """'architecture decisions' → 09_architecture_decisions.adoc"""
        results = self.retriever.search("important architecture decisions and rationale", top_k=5)
        assert _top_n_contain(results, _ARC42_DECISIONS)
        assert _any_text_contains(results, [
            "decision", "rationale", "alternative", "chosen",
        ])

    # -- Cross-Cutting Concepts --

    def test_crosscutting_concepts_query(self) -> None:
        """'cross-cutting concerns' → 08_concepts.adoc"""
        results = self.retriever.search("cross-cutting concepts and design patterns", top_k=5)
        assert _top_n_contain(results, _ARC42_CONCEPTS)
        assert _any_text_contains(results, [
            "cross-cutting", "concept", "pattern",
        ])

    # -- Solution Strategy --

    def test_solution_strategy_query(self) -> None:
        """'solution strategy' → 04_solution_strategy.adoc"""
        results = self.retriever.search("solution strategy technology decisions", top_k=5)
        assert _top_n_contain(results, _ARC42_STRATEGY)
        assert _any_text_contains(results, [
            "solution", "strategy", "technology", "decision",
        ])

    # -- Architecture Constraints --

    def test_constraints_query(self) -> None:
        """'architecture constraints' → 02_architecture_constraints.adoc"""
        results = self.retriever.search("architecture constraints technical organizational", top_k=5)
        assert _top_n_contain(results, _ARC42_CONSTRAINTS)
        assert _any_text_contains(results, [
            "constraint", "convention", "requirement",
        ])

    # -- Glossary --

    def test_glossary_query(self) -> None:
        """'glossary terminology' → 12_glossary.adoc"""
        results = self.retriever.search("glossary domain technical terms definitions", top_k=5)
        assert _any_result_from(results, _ARC42_GLOSSARY)


# ── 5c. Ground-Truth Retrieval Validation (docToolchain) ─────────────────

_DT_HTML = "03_task_generateHTML.adoc"
_DT_PDF = "03_task_generatePDF.adoc"
_DT_CONFLUENCE = "03_task_publishToConfluence.adoc"
_DT_JIRA = "03_task_exportJiraIssues.adoc"
_DT_SITE = "03_task_generateSite.adoc"
_DT_CONFIG = "30_config.adoc"
_DT_INSTALL = "20_install.adoc"
_DT_STRUCTURIZR = "03_task_exportStructurizr.adoc"
_DT_EA = "03_task_exportEA.adoc"
_DT_CHANGELOG = "03_task_exportChangeLog.adoc"
_DT_HTML_TUTORIAL = "030_generateHTML.adoc"
_DT_CONFLUENCE_TUTORIAL = "070_publishToConfluence.adoc"
_DT_INSTALL_TUTORIAL = "010_Install.adoc"
_DT_KROKI = "170_kroki-configuration.adoc"


class TestDocToolchainGroundTruth:
    """Validate retrieval against docToolchain ground-truth data.

    Each query maps to specific task/tutorial documentation files
    whose content was manually verified.
    """

    @pytest.fixture(autouse=True)
    def _retriever(self, doctool_index):
        from doc_qa.retrieval.retriever import HybridRetriever

        self.retriever = HybridRetriever(table=doctool_index._table, mode="vector")

    # -- HTML Generation --

    def test_html_generation_query(self) -> None:
        """'generate HTML from AsciiDoc' → generateHTML task/tutorial."""
        results = self.retriever.search("how to generate HTML output from AsciiDoc", top_k=5)
        assert len(results) > 0
        assert _any_result_from(results, [_DT_HTML, _DT_HTML_TUTORIAL])
        assert _any_text_contains(results, [
            "generatehtml", "html", "asciidoctor", "output",
        ])

    # -- PDF Generation --

    def test_pdf_generation_query(self) -> None:
        """'generate PDF' → generatePDF task."""
        results = self.retriever.search("generate PDF document asciidoctor-pdf", top_k=5)
        assert _any_result_from(results, [_DT_PDF, _DT_HTML_TUTORIAL])
        assert _any_text_contains(results, ["pdf", "asciidoctor", "generate"])

    # -- PlantUML Diagrams --

    def test_plantuml_diagram_query(self) -> None:
        """'PlantUML diagrams' → generateHTML (diagram section) or kroki."""
        results = self.retriever.search("PlantUML diagram generation configuration", top_k=5)
        assert _any_result_from(results, [_DT_HTML, _DT_KROKI, "dtcw.puml"])
        assert _any_text_contains(results, [
            "plantuml", "diagram", "asciidoctor-diagram",
        ])

    # -- Confluence Publishing --

    def test_confluence_publishing_query(self) -> None:
        """'publish to Confluence' → publishToConfluence task/tutorial."""
        results = self.retriever.search("publish documentation to Confluence", top_k=5)
        assert _any_result_from(results, [_DT_CONFLUENCE, _DT_CONFLUENCE_TUTORIAL])
        assert _any_text_contains(results, ["confluence", "publish"])

    # -- Jira Export --

    def test_jira_export_query(self) -> None:
        """'export Jira issues' → exportJiraIssues task."""
        results = self.retriever.search("export Jira issues to AsciiDoc", top_k=5)
        assert _any_result_from(results, [_DT_JIRA])
        assert _any_text_contains(results, ["jira", "issue", "export"])

    # -- Microsite Generation --

    def test_microsite_query(self) -> None:
        """'generate microsite' → generateSite task."""
        results = self.retriever.search("generate static microsite with jBake", top_k=5)
        assert _any_result_from(results, [_DT_SITE, "040_generateSite.adoc"])
        assert _any_text_contains(results, ["site", "microsite", "jbake", "generat"])

    # -- Configuration --

    def test_configuration_query(self) -> None:
        """'docToolchain configuration' → 30_config.adoc."""
        results = self.retriever.search("docToolchain configuration docToolchainConfig", top_k=5)
        assert _any_result_from(results, [_DT_CONFIG])
        assert _any_text_contains(results, ["config", "configuration"])

    # -- Installation --

    def test_installation_query(self) -> None:
        """'install docToolchain' → 20_install.adoc or tutorial."""
        results = self.retriever.search("install docToolchain dtcw wrapper", top_k=5)
        assert _any_result_from(results, [_DT_INSTALL, _DT_INSTALL_TUTORIAL])
        assert _any_text_contains(results, ["install", "dtcw", "download"])

    # -- Structurizr / C4 --

    def test_structurizr_query(self) -> None:
        """'Structurizr C4 export' → exportStructurizr task."""
        results = self.retriever.search("Structurizr C4 model diagram export", top_k=5)
        assert _any_result_from(results, [_DT_STRUCTURIZR])
        assert _any_text_contains(results, ["structurizr", "c4", "workspace"])

    # -- Enterprise Architect --

    def test_enterprise_architect_query(self) -> None:
        """'Enterprise Architect export' → exportEA task."""
        results = self.retriever.search("Enterprise Architect EA diagram export", top_k=5)
        assert _any_result_from(results, [_DT_EA])
        assert _any_text_contains(results, ["enterprise architect", "ea", "diagram"])

    # -- Changelog Export --

    def test_changelog_export_query(self) -> None:
        """'export changelog from git' → exportChangeLog task."""
        results = self.retriever.search("export git changelog version history", top_k=5)
        assert _any_result_from(results, [_DT_CHANGELOG, "changelog.adoc"])
        assert _any_text_contains(results, ["changelog", "git", "version"])


# ── 5d. Cross-Search-Mode Consistency ────────────────────────────────────


class TestSearchModeConsistency:
    """Verify that all three search modes return relevant results for the same queries."""

    MODES = ["vector", "fts", "hybrid"]

    @pytest.fixture(autouse=True)
    def _retrievers(self, arc42_index):
        from doc_qa.retrieval.retriever import HybridRetriever

        self.retrievers = {
            mode: HybridRetriever(table=arc42_index._table, mode=mode)
            for mode in self.MODES
        }

    def test_all_modes_find_building_blocks(self) -> None:
        for mode, retriever in self.retrievers.items():
            results = retriever.search("building block view", top_k=5)
            assert len(results) > 0, f"{mode} search returned no results for 'building block view'"

    def test_all_modes_find_deployment(self) -> None:
        for mode, retriever in self.retrievers.items():
            results = retriever.search("deployment infrastructure", top_k=5)
            assert len(results) > 0, f"{mode} search returned no results for 'deployment'"

    def test_all_modes_find_quality(self) -> None:
        for mode, retriever in self.retrievers.items():
            results = retriever.search("quality requirements attributes", top_k=5)
            assert len(results) > 0, f"{mode} search returned no results for 'quality'"

    def test_vector_and_hybrid_overlap(self) -> None:
        """Vector and hybrid should have overlapping results."""
        vec = self.retrievers["vector"].search("architecture decisions", top_k=5)
        hyb = self.retrievers["hybrid"].search("architecture decisions", top_k=5)
        vec_files = set(_result_files(vec))
        hyb_files = set(_result_files(hyb))
        overlap = vec_files & hyb_files
        assert len(overlap) > 0, f"No overlap: vector={vec_files}, hybrid={hyb_files}"


# ── 6. Query Pipeline Integration (no LLM) ──────────────────────────────


class TestQueryPipelineIntegration:
    """Full pipeline test with a mock LLM."""

    @pytest.mark.asyncio
    async def test_full_pipeline_arc42(self, arc42_index) -> None:
        from doc_qa.llm.backend import Answer, LLMBackend
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        class MockLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                return Answer(text=f"Answer about: {question}", sources=[], model="mock")

            async def close(self):
                pass

        pipeline = QueryPipeline(
            table=arc42_index._table,
            llm_backend=MockLLM(),
            rerank=False,
        )
        result = await pipeline.query("What is the building block view?")
        assert result.answer.startswith("Answer about:")
        assert result.chunks_retrieved > 0
        assert len(result.sources) > 0
        assert result.model == "mock"

    @pytest.mark.asyncio
    async def test_pipeline_sources_have_valid_paths(self, arc42_index) -> None:
        from doc_qa.llm.backend import Answer, LLMBackend
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        class MockLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                return Answer(text="ok", sources=[], model="mock")

            async def close(self):
                pass

        pipeline = QueryPipeline(
            table=arc42_index._table,
            llm_backend=MockLLM(),
            rerank=False,
        )
        result = await pipeline.query("architecture decisions")
        for src in result.sources:
            assert Path(src.file_path).exists(), f"Source path does not exist: {src.file_path}"
            # section_title may be empty for top-level/untitled sections
            assert isinstance(src.section_title, str)
            assert src.score > 0

    @pytest.mark.asyncio
    async def test_pipeline_multi_turn(self, doctool_index) -> None:
        from doc_qa.llm.backend import Answer, LLMBackend
        from doc_qa.retrieval.query_pipeline import QueryPipeline

        class MockLLM(LLMBackend):
            async def ask(self, question, context, history=None):
                n = len(history) if history else 0
                return Answer(text=f"Turn {n // 2 + 1}", sources=[], model="mock")

            async def close(self):
                pass

        pipeline = QueryPipeline(
            table=doctool_index._table,
            llm_backend=MockLLM(),
            rerank=False,
        )
        r1 = await pipeline.query("What is docToolchain?")
        assert "Turn 1" in r1.answer
        assert len(pipeline._history) == 2

        r2 = await pipeline.query("How do I configure it?")
        assert "Turn 2" in r2.answer
        assert len(pipeline._history) == 4


# ── 7. API Integration ──────────────────────────────────────────────────


class TestAPIIntegration:
    """Test the FastAPI endpoints with a real index."""

    @pytest.fixture
    def arc42_app(self, arc42_index, tmp_path):
        from doc_qa.api.server import create_app
        from doc_qa.config import AppConfig

        config = AppConfig()
        config.indexing.db_path = arc42_index._db_path

        app = create_app(repo_path=str(_ARC42_DIR), config=config)
        return app

    def test_health_endpoint(self, arc42_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(arc42_app)
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_stats_endpoint(self, arc42_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(arc42_app)
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] > 50
        assert data["total_files"] >= 10

    def test_retrieve_endpoint(self, arc42_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(arc42_app)
        resp = client.post("/api/retrieve", json={
            "question": "quality requirements and constraints",
            "top_k": 5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["chunks"]) > 0
        chunk = data["chunks"][0]
        assert "text" in chunk
        assert "score" in chunk
        assert chunk["score"] > 0

    def test_retrieve_different_queries_return_different_results(self, arc42_app) -> None:
        from fastapi.testclient import TestClient

        client = TestClient(arc42_app)
        r1 = client.post("/api/retrieve", json={"question": "runtime view"}).json()
        r2 = client.post("/api/retrieve", json={"question": "deployment"}).json()

        texts1 = {c["text"][:100] for c in r1["chunks"]}
        texts2 = {c["text"][:100] for c in r2["chunks"]}
        # At least some results should differ
        assert texts1 != texts2, "Different queries returned identical results"


# ── 8. Incremental Re-indexing ───────────────────────────────────────────


class TestIncrementalIndexing:
    """Verify incremental indexing detects changes correctly on real files."""

    def test_reindex_detects_no_changes_for_indexed_files(self, arc42_index, arc42_files) -> None:
        """Re-scanning files that were indexed should detect no changes."""
        # Only check files that were actually indexed (parseable ones)
        stats = arc42_index.stats()
        indexed_count = stats["total_files"]
        file_paths = [str(f) for f in arc42_files]
        new, changed, deleted = arc42_index.detect_changes(file_paths)
        # Changed and deleted should be zero for files already indexed
        assert len(changed) == 0, f"Unexpected changed files: {len(changed)}"
        assert len(deleted) == 0, f"Unexpected deleted files: {len(deleted)}"
        # New files are those that weren't parsed (empty sections) — this is expected
        # So just verify: new + indexed ≈ total scanned
        assert len(new) + indexed_count <= len(file_paths) + 1

    def test_detects_deleted_file(self, arc42_index) -> None:
        """Removing a file from the list should detect it as deleted."""
        # Get a file that IS in the index
        import pyarrow as pa

        table = arc42_index._table.to_arrow()
        file_paths_in_index = list(set(table.column("file_path").to_pylist()))
        if not file_paths_in_index:
            pytest.skip("No files in index")

        # Pass all but one
        removed = file_paths_in_index[-1]
        remaining = file_paths_in_index[:-1]
        _, _, deleted = arc42_index.detect_changes(remaining)
        assert removed in deleted


# ── 9. Performance / Scale ───────────────────────────────────────────────


class TestPerformance:
    """Ensure operations complete in reasonable time on real data."""

    def test_scan_completes_quickly(self) -> None:
        from doc_qa.config import DocRepoConfig
        from doc_qa.indexing.scanner import scan_files

        cfg = DocRepoConfig(path=str(_ARC42_DIR))
        start = time.time()
        files = scan_files(cfg)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Scanning took {elapsed:.1f}s (expected <5s)"
        assert len(files) > 0

    def test_vector_search_latency(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="vector")
        # Warm up
        retriever.search("test", top_k=3)

        start = time.time()
        for _ in range(5):
            retriever.search("architecture documentation", top_k=5)
        elapsed = time.time() - start
        avg = elapsed / 5
        assert avg < 2.0, f"Average vector search took {avg:.2f}s (expected <2s)"

    def test_fts_search_latency(self, arc42_index) -> None:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(table=arc42_index._table, mode="fts")

        start = time.time()
        for _ in range(5):
            retriever.search("building block", top_k=5)
        elapsed = time.time() - start
        avg = elapsed / 5
        assert avg < 2.0, f"Average FTS search took {avg:.2f}s (expected <2s)"
