"""File discovery and deduplication for doc repos."""

from __future__ import annotations

import logging
from pathlib import Path

from doc_qa.config import DocRepoConfig

logger = logging.getLogger(__name__)

# Format priority — higher number = preferred when duplicates exist.
# Source formats are preferred over rendered/compiled formats.
_FORMAT_PRIORITY: dict[str, int] = {
    ".puml": 10,
    ".adoc": 10,
    ".md": 10,
    ".svg": 5,
    ".html": 5,
    ".png": 1,
    ".pdf": 1,
}

# Groups of extensions that represent the same content in different forms.
# Within each group, only the highest-priority format is kept.
_DEDUP_GROUPS: list[set[str]] = [
    {".puml", ".png", ".svg"},  # diagram source + rendered
    {".adoc", ".pdf", ".html"},  # doc source + compiled
    {".md", ".pdf", ".html"},  # doc source + compiled
]


def _get_dedup_group(ext: str) -> set[str] | None:
    """Find which dedup group an extension belongs to."""
    for group in _DEDUP_GROUPS:
        if ext in group:
            return group
    return None


def _should_exclude(file_path: Path, repo_root: Path, exclude_patterns: list[str]) -> bool:
    """Check if a file matches any exclusion pattern."""
    rel = file_path.relative_to(repo_root)
    rel_parts = rel.parts  # e.g., ("docs", "build", "output.md")
    for pattern in exclude_patterns:
        # Extract directory component from glob pattern
        clean = pattern.replace("**/", "").replace("/**", "").strip("/")
        # Match against exact path components (not substring)
        if clean in rel_parts:
            return True
    return False


def scan_files(config: DocRepoConfig) -> list[Path]:
    """Scan a documentation repo and return deduplicated file list.

    1. Glob for all supported patterns
    2. Exclude build/target/node_modules directories
    3. Deduplicate by filename stem (prefer source formats)

    Args:
        config: DocRepoConfig with path, patterns, and exclude settings.

    Returns:
        Sorted list of Path objects to process.
    """
    repo_root = Path(config.path)
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Doc repo path not found: {repo_root}")

    # 1. Collect all matching files
    all_files: dict[Path, str] = {}  # path → extension
    for pattern in config.patterns:
        for match in repo_root.glob(pattern):
            if match.is_file() and not _should_exclude(match, repo_root, config.exclude):
                all_files[match] = match.suffix.lower()

    logger.info("Found %d files before deduplication.", len(all_files))

    # 2. Group by stem for deduplication
    # Key: (parent_dir, stem) → list of (path, ext, priority)
    stem_groups: dict[tuple[Path, str], list[tuple[Path, str, int]]] = {}
    for path, ext in all_files.items():
        key = (path.parent, path.stem)
        priority = _FORMAT_PRIORITY.get(ext, 0)
        stem_groups.setdefault(key, []).append((path, ext, priority))

    # 3. Deduplicate — keep highest priority per stem group
    result: list[Path] = []
    for (parent, stem), entries in stem_groups.items():
        if len(entries) == 1:
            result.append(entries[0][0])
            continue

        # Check if any pair belongs to the same dedup group
        kept: set[Path] = set()
        processed_groups: set[frozenset[str]] = set()

        for path, ext, priority in entries:
            group = _get_dedup_group(ext)
            if group is None:
                # Not in any dedup group — always keep
                kept.add(path)
                continue

            frozen_group = frozenset(group)
            if frozen_group in processed_groups:
                continue

            # Find all entries in this dedup group and keep the highest priority
            in_group = [(p, e, pr) for p, e, pr in entries if e in group]
            if in_group:
                best = max(in_group, key=lambda x: x[2])
                kept.add(best[0])
                if len(in_group) > 1:
                    dupes = [p.name for p, _, _ in in_group if p != best[0]]
                    logger.debug(
                        "Dedup: keeping %s, skipping %s",
                        best[0].name,
                        ", ".join(dupes),
                    )
            processed_groups.add(frozen_group)

        # Add entries not in any dedup group
        for path, ext, _ in entries:
            if _get_dedup_group(ext) is None and path not in kept:
                kept.add(path)

        result.extend(kept)

    result.sort(key=lambda p: str(p))
    logger.info("After deduplication: %d files.", len(result))
    return result
