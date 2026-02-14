"""Configuration loading from YAML."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DocRepoConfig:
    path: str = ""
    patterns: list[str] = field(
        default_factory=lambda: ["**/*.adoc", "**/*.md", "**/*.puml", "**/*.pdf"]
    )
    exclude: list[str] = field(
        default_factory=lambda: ["**/build/**", "**/target/**", "**/node_modules/**", "**/.git/**"]
    )


@dataclass
class IndexingConfig:
    db_path: str = "./data/doc_qa_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class RetrievalConfig:
    search_mode: str = "hybrid"
    top_k: int = 5
    candidate_pool: int = 20
    min_score: float = 0.3
    max_chunks_per_file: int = 2
    rerank: bool = True
    max_query_length: int = 2000
    max_history_turns: int = 10


@dataclass
class CodyConfig:
    access_token_env: str = "SRC_ACCESS_TOKEN"
    endpoint: str = "https://sourcegraph.com"
    agent_binary: str | None = None
    model: str = "anthropic::2025-01-01::claude-3.5-sonnet"


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"


@dataclass
class LLMConfig:
    primary: str = "cody"
    fallback: str = "ollama"


@dataclass
class APIConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    session_ttl: int = 1800


@dataclass
class IntelligenceConfig:
    enable_intent_classification: bool = True
    intent_confidence_high: float = 0.85
    intent_confidence_medium: float = 0.65
    enable_multi_intent: bool = True
    max_sub_queries: int = 3


@dataclass
class GenerationConfig:
    enable_diagrams: bool = True
    mermaid_validation: str = "auto"  # "node" | "regex" | "auto" | "none"
    node_script_path: str = "scripts/validate_mermaid.mjs"
    max_diagram_retries: int = 3
    suggest_diagrams: bool = True  # proactive diagram suggestion for explanations


@dataclass
class VerificationConfig:
    enable_verification: bool = True
    enable_crag: bool = True
    confidence_threshold: float = 0.4
    max_crag_rewrites: int = 2
    abstain_on_low_confidence: bool = True


@dataclass
class StreamingConfig:
    enable_sse: bool = True
    sse_ping_interval: int = 15
    max_concurrent_streams: int = 20


@dataclass
class DatabaseConfig:
    url: str | None = None  # postgresql+asyncpg://user:pass@host:port/dbname


@dataclass
class AppConfig:
    doc_repo: DocRepoConfig = field(default_factory=DocRepoConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cody: CodyConfig = field(default_factory=CodyConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    api: APIConfig = field(default_factory=APIConfig)
    intelligence: IntelligenceConfig | None = None
    generation: GenerationConfig | None = None
    verification: VerificationConfig | None = None
    streaming: StreamingConfig | None = None
    database: DatabaseConfig | None = None


def _apply_dict(target: Any, data: dict[str, Any]) -> None:
    """Apply dictionary values onto a dataclass instance."""
    for key, value in data.items():
        if hasattr(target, key):
            current = getattr(target, key)
            if current is not None and value is not None:
                expected_type = type(current)
                actual_type = type(value)
                # Allow int → float coercion
                if expected_type is float and actual_type is int:
                    value = float(value)
                # Guard against bool being subclass of int
                elif expected_type is int and actual_type is bool:
                    logger.warning(
                        "Config type mismatch for '%s': expected %s, got %s — skipping.",
                        key, expected_type.__name__, actual_type.__name__,
                    )
                    continue
                elif not isinstance(value, expected_type):
                    logger.warning(
                        "Config type mismatch for '%s': expected %s, got %s — skipping.",
                        key, expected_type.__name__, actual_type.__name__,
                    )
                    continue
            setattr(target, key, value)


def resolve_db_path(config: AppConfig, repo_path: str) -> str:
    """Resolve db_path relative to repo_path if not absolute."""
    db_path = config.indexing.db_path
    if not Path(db_path).is_absolute():
        db_path = str(Path(repo_path) / db_path)
    return db_path


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration from YAML file, falling back to defaults."""
    cfg = AppConfig()

    if config_path is None:
        config_path = Path("config.yaml")

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        if "doc_repo" in raw:
            _apply_dict(cfg.doc_repo, raw["doc_repo"])
        if "indexing" in raw:
            _apply_dict(cfg.indexing, raw["indexing"])
        if "retrieval" in raw:
            _apply_dict(cfg.retrieval, raw["retrieval"])
        if "llm" in raw:
            _apply_dict(cfg.llm, raw["llm"])
        if "cody" in raw:
            _apply_dict(cfg.cody, raw["cody"])
        if "ollama" in raw:
            _apply_dict(cfg.ollama, raw["ollama"])
        if "api" in raw:
            _apply_dict(cfg.api, raw["api"])
        if "intelligence" in raw:
            cfg.intelligence = IntelligenceConfig()
            _apply_dict(cfg.intelligence, raw["intelligence"])
        if "generation" in raw:
            cfg.generation = GenerationConfig()
            _apply_dict(cfg.generation, raw["generation"])
        if "verification" in raw:
            cfg.verification = VerificationConfig()
            _apply_dict(cfg.verification, raw["verification"])
        if "streaming" in raw:
            cfg.streaming = StreamingConfig()
            _apply_dict(cfg.streaming, raw["streaming"])
        if "database" in raw:
            cfg.database = DatabaseConfig()
            _apply_dict(cfg.database, raw["database"])

    return cfg


# Sections that require a server restart to take effect
UNSAFE_SECTIONS = frozenset({"indexing"})

# Sections that are optional (may be None on AppConfig)
_OPTIONAL_SECTIONS = frozenset(
    {"intelligence", "generation", "verification", "streaming", "database"}
)

# Factory map for optional sections (name → default constructor)
_OPTIONAL_FACTORIES: dict[str, type] = {
    "intelligence": IntelligenceConfig,
    "generation": GenerationConfig,
    "verification": VerificationConfig,
    "streaming": StreamingConfig,
    "database": DatabaseConfig,
}


def config_to_dict(cfg: AppConfig) -> dict[str, Any]:
    """Convert an AppConfig to a plain dict, redacting secrets.

    - ``None`` optional sections are omitted.
    - ``cody.access_token_env`` value is replaced with ``"***"``
      so credentials never leak to the browser.
    """
    result: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        if value is None:
            continue
        result[f.name] = dataclasses.asdict(value)

    # Redact the Cody access-token env-var name
    if "cody" in result:
        result["cody"]["access_token_env"] = "***"

    return result


def save_config(cfg: AppConfig, config_path: Path | None = None) -> None:
    """Write the current config to YAML on disk.

    Preserves only the known config sections.  The file is
    overwritten atomically (write-to-temp then rename).
    """
    if config_path is None:
        config_path = Path("config.yaml")

    data = config_to_dict(cfg)
    # Do NOT persist the redacted token — restore from live config
    if "cody" in data:
        data["cody"]["access_token_env"] = cfg.cody.access_token_env

    tmp = config_path.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    tmp.replace(config_path)
