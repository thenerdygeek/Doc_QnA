"""FastEmbed wrapper for generating text embeddings.

The embedding model is cached in a **project-local** directory
(``data/models/``) so it can be copied between machines and works
fully offline after the first download.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Model cache ──────────────────────────────────────────────────────
# Project-local cache so the model can be copied between machines.
# Resolve relative to the package root (3 levels up from this file).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE_DIR = str(_PROJECT_ROOT / "data" / "models")


def get_cache_dir() -> str:
    """Return the model cache directory (project-local by default)."""
    return os.environ.get("FASTEMBED_CACHE_PATH", _DEFAULT_CACHE_DIR)


# ── Mapping: model name → GCS tar.gz URL and expected dir name ───────
_MODEL_GCS_INFO: dict[str, dict[str, str]] = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "url": "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
        "dir_name": "fast-all-MiniLM-L6-v2",
        "hf_dir_name": "models--qdrant--all-MiniLM-L6-v2-onnx",
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "dir_name": "fast-nomic-embed-text-v1.5",
        "hf_dir_name": "models--nomic-ai--nomic-embed-text-v1.5",
    },
}


def _cleanup_corrupt_hf_cache(cache_dir: str, model_name: str) -> None:
    """Remove corrupt HuggingFace cache (incomplete downloads).

    Fastembed tries the HF cache first. If a previous download was
    interrupted, the HF directory exists but model.onnx is missing.
    Removing it lets fastembed fall through to the GCS tar.gz layout.
    """
    info = _MODEL_GCS_INFO.get(model_name)
    if not info:
        return

    hf_dir = Path(cache_dir) / info["hf_dir_name"]
    if not hf_dir.exists():
        return

    # Check if model.onnx exists anywhere in the HF cache tree
    onnx_files = list(hf_dir.rglob("model.onnx"))
    if onnx_files:
        # Check file size — a valid model.onnx is > 10 MB
        for f in onnx_files:
            if f.stat().st_size > 10_000_000:
                return  # Valid cache, don't touch it

    # Corrupt / incomplete — remove it
    logger.info("Removing corrupt HF cache: %s", hf_dir)
    shutil.rmtree(hf_dir, ignore_errors=True)


def download_model_from_gcs(model_name: str, cache_dir: str | None = None) -> Path:
    """Download the embedding model from Google Cloud Storage.

    This bypasses HuggingFace entirely — more reliable on corporate
    networks since GCS URLs are less likely to be blocked by proxies.

    Returns the path to the extracted model directory.
    """
    import tarfile

    import requests

    cache = cache_dir or get_cache_dir()
    info = _MODEL_GCS_INFO.get(model_name)
    if not info or "url" not in info:
        raise ValueError(
            f"No GCS download URL available for model: {model_name}. "
            f"This model is downloaded automatically by FastEmbed from HuggingFace on first use. "
            f"Run: pip install fastembed && python -c \"from fastembed import TextEmbedding; TextEmbedding('{model_name}')\""
        )

    model_dir = Path(cache) / info["dir_name"]

    # Already downloaded?
    if model_dir.exists() and any(model_dir.glob("model.onnx")):
        logger.info("Model already cached at %s", model_dir)
        return model_dir

    os.makedirs(cache, exist_ok=True)
    tar_path = Path(cache) / f"{info['dir_name']}.tar.gz"

    print(f"  Downloading from: {info['url']}")
    print(f"  This is a one-time ~90 MB download...")

    # Stream download with progress
    resp = requests.get(info["url"], stream=True, timeout=300)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(tar_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                mb = downloaded / 1024 / 1024
                print(f"\r  Progress: {pct}% ({mb:.1f} MB)", end="", flush=True)

    print()  # newline after progress

    # Extract
    print("  Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=cache)

    # Clean up tar.gz
    tar_path.unlink(missing_ok=True)

    if not model_dir.exists():
        raise RuntimeError(f"Expected model directory not found after extraction: {model_dir}")

    return model_dir


# ── Lazy singleton ───────────────────────────────────────────────────
_model: object | None = None
_model_name: str = ""
_model_lock = threading.Lock()


def _has_local_model(cache_dir: str, model_name: str) -> bool:
    """Check if the model files already exist locally (GCS or HF layout)."""
    info = _MODEL_GCS_INFO.get(model_name)
    if not info:
        return False

    # Check GCS layout: fast-all-MiniLM-L6-v2/model.onnx
    gcs_onnx = Path(cache_dir) / info["dir_name"] / "model.onnx"
    if gcs_onnx.is_file() and gcs_onnx.stat().st_size > 10_000_000:
        return True

    # Check HF layout: models--qdrant--.../**/model.onnx
    hf_dir = Path(cache_dir) / info["hf_dir_name"]
    if hf_dir.exists():
        for f in hf_dir.rglob("model.onnx"):
            if f.stat().st_size > 10_000_000:
                return True

    return False


def _get_model(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> object:
    """Get or create the FastEmbed TextEmbedding model (lazy singleton)."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    with _model_lock:
        # Double-check after acquiring lock
        if _model is not None and _model_name == model_name:
            return _model

        cache_dir = get_cache_dir()

        # Clean up any corrupt HF cache before loading
        _cleanup_corrupt_hf_cache(cache_dir, model_name)

        # If model files exist locally, block HF from downloading again.
        # This prevents fastembed from ignoring the GCS layout and
        # starting a slow HuggingFace download.
        # Skip this guard when bundling (BUNDLE_MODE env var is set).
        if _has_local_model(cache_dir, model_name) and not os.environ.get("DOC_QA_BUNDLE_MODE"):
            os.environ["HF_HUB_OFFLINE"] = "1"

        from fastembed import TextEmbedding

        import time as _time
        logger.info("Loading embedding model: %s (cache: %s)", model_name, cache_dir)
        logger.warning("[PERF] HF_HUB_OFFLINE=%s TRANSFORMERS_OFFLINE=%s",
                    os.environ.get("HF_HUB_OFFLINE", "unset"),
                    os.environ.get("TRANSFORMERS_OFFLINE", "unset"))
        _t0 = _time.time()
        _model = TextEmbedding(model_name, cache_dir=cache_dir)
        _model_name = model_name
        logger.warning("[PERF] Model loaded in %.1fs", _time.time() - _t0)
        return _model


def get_embedding_dimension(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> int:
    """Return the embedding dimension for the given model.

    Known dimensions to avoid loading the model just for this:
    - all-MiniLM-L6-v2: 384
    - all-mpnet-base-v2: 768
    - nomic-embed-text-v1.5: 768
    """
    known = {
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
        "nomic-ai/nomic-embed-text-v1.5": 768,
    }
    if model_name in known:
        return known[model_name]
    # Fallback: embed a dummy string to get the dimension
    vecs = embed_texts(["test"], model_name=model_name)
    return len(vecs[0])


def embed_texts(
    texts: list[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    batch_size: int = 64,
) -> list[NDArray[np.float32]]:
    """Embed a list of texts into vectors.

    Args:
        texts: List of text strings to embed.
        model_name: FastEmbed model identifier.
        batch_size: Batch size for embedding (larger = faster but more memory).

    Returns:
        List of numpy arrays, each of shape (dimension,).
    """
    if not texts:
        return []

    model = _get_model(model_name)
    # FastEmbed's embed() returns a generator of numpy arrays
    embeddings = list(model.embed(texts, batch_size=batch_size))

    logger.debug("Embedded %d texts (dim=%d).", len(embeddings), len(embeddings[0]))
    return embeddings


def embed_query(
    query: str,
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
) -> NDArray[np.float32]:
    """Embed a single query string. Uses query_embed for asymmetric models."""
    model = _get_model(model_name)

    # Some models have a separate query embedding method
    if hasattr(model, "query_embed"):
        result = list(model.query_embed(query))
        return result[0]

    # Fallback to regular embed
    result = list(model.embed([query]))
    return result[0]
