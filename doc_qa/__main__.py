"""CLI entry point for doc-qa-tool."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from doc_qa.config import load_config, resolve_db_path


def cmd_index(args: argparse.Namespace) -> None:
    """Index documentation from the given repo path."""
    import time

    from doc_qa.indexing.chunker import chunk_sections
    from doc_qa.indexing.indexer import DocIndex
    from doc_qa.indexing.scanner import scan_files
    from doc_qa.parsers.registry import parse_file

    config = load_config(Path(args.config) if args.config else None)

    repo_path = Path(args.repo)
    if not repo_path.is_dir():
        print(f"Error: '{repo_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    config.doc_repo.path = str(repo_path)

    # Resolve db_path relative to the repo if it's relative
    db_path = resolve_db_path(config, str(repo_path))

    print(f"Scanning '{repo_path}' for documents...")
    files = scan_files(config.doc_repo)
    print(f"Found {len(files)} files.")

    if not files:
        print("No supported files found. Nothing to index.")
        return

    # Initialize index
    index = DocIndex(db_path=db_path, embedding_model=config.indexing.embedding_model)

    # Detect changes for incremental indexing
    file_paths = [str(f) for f in files]
    new_files, changed_files, deleted_files = index.detect_changes(file_paths)

    # Delete removed files
    for fp in deleted_files:
        index.delete_file_chunks(fp)
        print(f"  Removed: {Path(fp).name}")

    files_to_process = new_files + changed_files
    if not files_to_process and not deleted_files:
        stats = index.stats()
        print(f"\nIndex is up-to-date. {stats['total_chunks']} chunks from {stats['total_files']} files.")
        return

    if not files_to_process:
        index.rebuild_fts_index()
        stats = index.stats()
        print(f"\nDone. {stats['total_chunks']} chunks from {stats['total_files']} files.")
        return

    skipped = len(files) - len(files_to_process)
    if skipped > 0:
        print(f"Skipping {skipped} unchanged files.")
    print(f"Processing {len(files_to_process)} files...\n")

    start = time.time()
    total_chunks = 0
    total_files = 0

    for fp in files_to_process:
        file_path = Path(fp)
        sections = parse_file(file_path)
        if not sections:
            print(f"  {file_path.name}: skipped (no parseable content)")
            continue

        chunks = chunk_sections(
            sections,
            file_path=fp,
            max_tokens=config.indexing.chunk_size,
            overlap_tokens=config.indexing.chunk_overlap,
            min_tokens=config.indexing.min_chunk_size,
        )

        if not chunks:
            continue

        n = index.upsert_file(chunks, fp)
        total_chunks += n
        total_files += 1
        label = "new" if fp in new_files else "updated"
        print(f"  {file_path.name}: {len(sections)} sections -> {n} chunks [{label}]")

    # Rebuild FTS index after all changes
    index.rebuild_fts_index()

    elapsed = time.time() - start
    stats = index.stats()
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Index: {stats['total_chunks']} chunks from {stats['total_files']} files.")
    print(f"DB: {db_path}")


def cmd_query(args: argparse.Namespace) -> None:
    """Query indexed documentation."""
    import asyncio

    from doc_qa.indexing.indexer import DocIndex
    from doc_qa.llm.backend import create_backend
    from doc_qa.retrieval.query_pipeline import QueryPipeline

    config = load_config(Path(args.config) if args.config else None)

    repo_path = Path(args.repo)
    if not repo_path.is_dir():
        print(f"Error: '{repo_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Resolve db_path
    db_path = resolve_db_path(config, str(repo_path))

    # Open existing index
    index = DocIndex(db_path=db_path, embedding_model=config.indexing.embedding_model)
    if index.count_rows() == 0:
        print("Error: Index is empty. Run 'doc-qa index' first.", file=sys.stderr)
        sys.exit(1)

    print(f"Index: {index.count_rows()} chunks from {index.count_files()} files.\n")

    # Retrieval-only mode — no LLM needed
    if args.retrieval_only:
        from doc_qa.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(
            table=index._table,
            embedding_model=config.indexing.embedding_model,
            mode=config.retrieval.search_mode,
        )
        chunks = retriever.search(
            query=args.question,
            top_k=config.retrieval.top_k,
            min_score=config.retrieval.min_score,
        )
        if not chunks:
            print("No relevant chunks found.")
            return

        for i, c in enumerate(chunks, 1):
            name = Path(c.file_path).name
            print(f"[{i}] {name} > {c.section_title} (score: {c.score:.3f})")
            print(f"    {c.text[:200]}...")
            print()
        return

    # Set up LLM backend
    backend = create_backend(
        primary=config.llm.primary,
        fallback=config.llm.fallback,
        cody_endpoint=config.cody.endpoint,
        cody_model=config.cody.model,
        cody_binary=config.cody.agent_binary,
        workspace_root=str(repo_path),
        ollama_host=config.ollama.host,
        ollama_model=config.ollama.model,
    )

    # Set up query pipeline
    pipeline = QueryPipeline(
        table=index._table,
        llm_backend=backend,
        embedding_model=config.indexing.embedding_model,
        search_mode=config.retrieval.search_mode,
        candidate_pool=config.retrieval.candidate_pool,
        top_k=config.retrieval.top_k,
        min_score=config.retrieval.min_score,
        max_chunks_per_file=config.retrieval.max_chunks_per_file,
        rerank=config.retrieval.rerank,
        max_history_turns=config.retrieval.max_history_turns,
    )

    async def _run_query() -> None:
        try:
            result = await pipeline.query(args.question)

            if result.error:
                print(f"Error: {result.error}", file=sys.stderr)
                return

            print(result.answer)

            if result.sources:
                print("\n--- Sources ---")
                for src in result.sources:
                    name = Path(src.file_path).name
                    print(f"  [{name}] {src.section_title} (score: {src.score:.3f})")
        finally:
            await backend.close()

    asyncio.run(_run_query())


def cmd_eval(args: argparse.Namespace) -> None:
    """Run retrieval evaluation."""
    from doc_qa.eval.evaluator import evaluate, format_report, load_test_cases
    from doc_qa.indexing.indexer import DocIndex
    from doc_qa.retrieval.retriever import HybridRetriever

    config = load_config(Path(args.config) if args.config else None)

    repo_path = Path(args.repo)
    if not repo_path.is_dir():
        print(f"Error: '{repo_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Resolve db_path
    db_path = resolve_db_path(config, str(repo_path))

    # Load test cases
    cases = load_test_cases(args.test_cases)
    if not cases:
        print("Error: No test cases found.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(cases)} test cases.\n")

    # Open index
    index = DocIndex(db_path=db_path, embedding_model=config.indexing.embedding_model)
    if index.count_rows() == 0:
        print("Error: Index is empty. Run 'doc-qa index' first.", file=sys.stderr)
        sys.exit(1)

    # Set up retriever
    retriever = HybridRetriever(
        table=index._table,
        embedding_model=config.indexing.embedding_model,
        mode=config.retrieval.search_mode,
    )

    # Run evaluation
    k = config.retrieval.top_k
    summary = evaluate(cases, retriever, k=k)

    # Print report
    print(format_report(summary, k=k))

    # Exit code based on pass/fail
    if not summary.passed():
        sys.exit(1)


def cmd_db(args: argparse.Namespace) -> None:
    """Run database migrations."""
    import os

    from doc_qa.db.engine import get_database_url

    config = load_config(Path(args.config) if args.config else None)
    db_url = get_database_url(config.database.url if config.database else None)
    if not db_url:
        print(
            "Error: No database URL configured.\n"
            "Set DOC_QA_DATABASE_URL env var or add database.url to config.yaml.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Alembic needs the sync URL form for its config, but our env.py resolves it
    os.environ["DOC_QA_DATABASE_URL"] = db_url

    from alembic import command as alembic_cmd
    from alembic.config import Config as AlembicConfig

    alembic_ini = Path(__file__).resolve().parent.parent.parent / "alembic.ini"
    alembic_cfg = AlembicConfig(str(alembic_ini))

    action = args.action
    if action == "upgrade":
        revision = args.revision or "head"
        print(f"Upgrading database to {revision}...")
        alembic_cmd.upgrade(alembic_cfg, revision)
        print("Done.")
    elif action == "downgrade":
        revision = args.revision or "-1"
        print(f"Downgrading database by {revision}...")
        alembic_cmd.downgrade(alembic_cfg, revision)
        print("Done.")


def _ensure_embedding_model(model_name: str) -> None:
    """Pre-download the embedding model, then enable offline mode.

    After this call, fastembed / huggingface_hub will NOT make any
    network requests — all model data is served from the local cache.

    Download strategy (automatic fallback):
      1. Try fastembed's built-in loader (uses HuggingFace Hub).
      2. If that fails, download from Google Cloud Storage and retry.
      3. If both fail, print instructions and exit.
    """
    import os

    from doc_qa.indexing.embedder import (
        _get_model,
        download_model_from_gcs,
        get_cache_dir,
    )

    cache_dir = get_cache_dir()
    print(f"Loading embedding model: {model_name}")
    print(f"  Cache: {cache_dir}")

    # ── Attempt 1: fastembed's built-in loader ──
    try:
        _get_model(model_name)
        print("Embedding model ready.\n")
    except Exception as first_err:
        # ── Attempt 2: download from GCS and retry ──
        print(f"\n  Standard download failed: {first_err}")
        print("  Trying alternative download from Google Cloud Storage...")
        try:
            download_model_from_gcs(model_name, cache_dir)
            _get_model(model_name)
            print("Embedding model ready (via GCS).\n")
        except Exception as gcs_err:
            print(f"\nError: All automatic download methods failed.", file=sys.stderr)
            print(f"  Standard: {first_err}", file=sys.stderr)
            print(f"  GCS:      {gcs_err}", file=sys.stderr)
            print(
                "\nThe embedding model (~90 MB) must be downloaded once.\n"
                "\n"
                "Option 1 — Download in your browser, then install:\n"
                "  1. Open this URL in your browser:\n"
                "     https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz\n"
                "  2. Run:\n"
                "     doc-qa install-model /path/to/sentence-transformers-all-MiniLM-L6-v2.tar.gz\n"
                "\n"
                "Option 2 — Copy from another machine:\n"
                "  Copy the data/models/ folder from a machine that already has the model.\n"
                f"\nCache directory: {cache_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Lock down: no more network calls from fastembed / huggingface_hub
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def cmd_download_model(args: argparse.Namespace) -> None:
    """Download the embedding model for offline use.

    Uses Google Cloud Storage directly — more reliable on corporate
    networks where HuggingFace may be blocked or throttled.
    """
    config = load_config(Path(args.config) if args.config else None)
    model_name = config.indexing.embedding_model

    from doc_qa.indexing.embedder import (
        _get_model,
        download_model_from_gcs,
        embed_texts,
        get_cache_dir,
    )

    cache_dir = get_cache_dir()
    print(f"Downloading embedding model: {model_name}")
    print(f"  Cache directory: {cache_dir}\n")

    try:
        # Download from GCS (bypasses HuggingFace entirely)
        download_model_from_gcs(model_name, cache_dir)

        # Load into fastembed to verify it works
        _get_model(model_name)

        # Quick sanity check — embed a test string
        vecs = embed_texts(["test"])
        print(f"\nModel downloaded and verified (dim={len(vecs[0])}).")
        print(f"Cache: {cache_dir}")
        print("\nYou can now run 'doc-qa serve' offline.")
        print("To use on another machine, copy the data/models/ folder.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        print(
            "\nAutomatic download failed. Manual steps:\n"
            "\n"
            "Option 1 — Download in your browser, then install:\n"
            "  1. Open this URL in your browser:\n"
            "     https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz\n"
            "  2. Run:\n"
            "     doc-qa install-model /path/to/sentence-transformers-all-MiniLM-L6-v2.tar.gz\n"
            "\n"
            "Option 2 — Behind a corporate proxy:\n"
            "  set HTTPS_PROXY=http://your-proxy:8080\n"
            "  doc-qa download-model\n"
            "\n"
            "Option 3 — Copy from another machine:\n"
            "  Copy the data/models/ folder from a machine that already has the model.\n"
            f"\nCache directory: {cache_dir}",
            file=sys.stderr,
        )
        sys.exit(1)


def cmd_install_model(args: argparse.Namespace) -> None:
    """Install the embedding model from a locally downloaded tar.gz file.

    For users who downloaded the model via browser because automatic
    download failed (e.g., corporate proxy blocking Python requests).
    """
    import os
    import tarfile

    from doc_qa.indexing.embedder import (
        _cleanup_corrupt_hf_cache,
        _get_model,
        embed_texts,
        get_cache_dir,
    )

    tar_path = Path(args.file)
    if not tar_path.is_file():
        print(f"Error: '{tar_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if not tar_path.name.endswith((".tar.gz", ".tgz")):
        print(f"Error: Expected a .tar.gz file, got '{tar_path.name}'.", file=sys.stderr)
        sys.exit(1)

    config = load_config(Path(args.config) if args.config else None)
    model_name = config.indexing.embedding_model
    cache_dir = get_cache_dir()

    print(f"Installing embedding model from: {tar_path}")
    print(f"  Cache directory: {cache_dir}\n")

    # 1. Create cache dir
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Clean up any corrupt HF cache from previous failed downloads
    print("  Cleaning up any corrupt cache...")
    _cleanup_corrupt_hf_cache(cache_dir, model_name)

    # 3. Extract tar.gz
    print(f"  Extracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=cache_dir)
    except Exception as e:
        print(f"\nError extracting: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Verify model.onnx exists
    model_dir = Path(cache_dir) / "fast-all-MiniLM-L6-v2"
    onnx_file = model_dir / "model.onnx"
    if not onnx_file.is_file():
        print(f"\nError: Expected file not found: {onnx_file}", file=sys.stderr)
        print("The tar.gz may have a different structure.", file=sys.stderr)
        # List what was extracted
        if model_dir.exists():
            print(f"\nContents of {model_dir}:", file=sys.stderr)
            for f in sorted(model_dir.iterdir()):
                print(f"  {f.name}", file=sys.stderr)
        sys.exit(1)

    # 5. Load into fastembed and verify
    print("  Loading model into fastembed...")
    try:
        _get_model(model_name)
        vecs = embed_texts(["test"])
        print(f"\nModel installed and verified (dim={len(vecs[0])}).")
        print(f"Cache: {cache_dir}")
        print("\nYou can now run 'doc-qa serve' — no internet required.")
    except Exception as e:
        print(f"\nError: Model extracted but failed to load: {e}", file=sys.stderr)
        print("Try deleting the cache and re-installing:", file=sys.stderr)
        print(f"  rm -rf {cache_dir}", file=sys.stderr)
        print(f"  doc-qa install-model {tar_path}", file=sys.stderr)
        sys.exit(1)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from doc_qa.api.server import create_app

    config = load_config(Path(args.config) if args.config else None)

    # ── Pre-download embedding model (internet required on first run only) ──
    _ensure_embedding_model(config.indexing.embedding_model)

    # Resolve repo path: CLI flag → config.yaml → None (configure from UI)
    repo_str: str | None = args.repo
    if not repo_str and config.doc_repo.path:
        repo_str = config.doc_repo.path

    if repo_str:
        repo_path = Path(repo_str)
        if not repo_path.is_dir():
            if args.repo:
                # Explicit --repo flag: hard error (user typo)
                print(f"Error: '{repo_path}' is not a valid directory.", file=sys.stderr)
                sys.exit(1)
            else:
                # Stale path from config.yaml: warn and continue without repo
                print(f"Warning: saved repo path '{repo_path}' not found on this machine.")
                print("Configure the correct path in Settings > Indexing.\n")
                repo_str = None
        else:
            repo_str = str(repo_path)
    else:
        print("No --repo specified and no doc_repo.path in config.yaml.")
        print("Start the UI and configure the repository path in Settings > Indexing.\n")

    host = args.host or config.api.host
    port = args.port or config.api.port

    app = create_app(repo_path=repo_str or "", config=config)
    print(f"Starting server at http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    print("Server is fully offline (except Cody/Ollama LLM backends).\n")
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="doc-qa",
        description="Documentation Q&A tool powered by RAG + Cody AI",
    )
    parser.add_argument("--config", help="Path to config.yaml", default=None)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set logging level (default: WARNING)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # index command
    p_index = sub.add_parser("index", help="Index documentation from a repo folder")
    p_index.add_argument("repo", help="Path to documentation repository")
    p_index.set_defaults(func=cmd_index)

    # query command
    p_query = sub.add_parser("query", help="Ask a question about the documentation")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--repo", help="Path to documentation repository", required=True)
    p_query.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Show retrieved chunks without sending to LLM",
    )
    p_query.set_defaults(func=cmd_query)

    # eval command
    p_eval = sub.add_parser("eval", help="Run retrieval evaluation")
    p_eval.add_argument("--test-cases", help="Path to test_cases.json", required=True)
    p_eval.add_argument("--repo", help="Path to documentation repository", required=True)
    p_eval.set_defaults(func=cmd_eval)

    # db command
    p_db = sub.add_parser("db", help="Run database migrations")
    p_db.add_argument("action", choices=["upgrade", "downgrade"], help="Migration action")
    p_db.add_argument("revision", nargs="?", default=None, help="Target revision (default: head/−1)")
    p_db.set_defaults(func=cmd_db)

    # download-model command
    p_dl = sub.add_parser("download-model", help="Download the embedding model (run once with internet)")
    p_dl.set_defaults(func=cmd_download_model)

    # install-model command
    p_install = sub.add_parser(
        "install-model",
        help="Install the embedding model from a downloaded tar.gz file",
    )
    p_install.add_argument(
        "file",
        help="Path to the downloaded sentence-transformers-all-MiniLM-L6-v2.tar.gz file",
    )
    p_install.set_defaults(func=cmd_install_model)

    # serve command
    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--repo", help="Path to documentation repository (optional; falls back to config.yaml, or configure from Settings UI)", default=None)
    p_serve.add_argument("--host", help="Bind host", default=None)
    p_serve.add_argument("--port", help="Bind port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
