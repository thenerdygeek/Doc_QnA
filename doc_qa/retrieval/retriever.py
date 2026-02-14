"""Hybrid retriever — vector + BM25 full-text search with RRF merging."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from doc_qa.indexing.embedder import embed_query

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (standard value from literature)
_RRF_K = 60


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with score and metadata."""

    text: str
    score: float
    chunk_id: str
    file_path: str
    file_type: str
    section_title: str
    section_level: int
    chunk_index: int
    vector: list[float] | None = None


class HybridRetriever:
    """Retrieves document chunks using vector search, BM25, or hybrid (RRF).

    Modes:
        - "vector": ANN search only
        - "fts": BM25 full-text search only
        - "hybrid": Both + Reciprocal Rank Fusion merge
    """

    def __init__(
        self,
        table: Any,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        mode: str = "hybrid",
    ) -> None:
        self._table = table
        self._embedding_model = embedding_model
        self._mode = mode

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Search for relevant chunks using the configured mode.

        Args:
            query: User query string.
            top_k: Number of results to return.
            min_score: Minimum score threshold.

        Returns:
            List of RetrievedChunk sorted by score descending.
        """
        if top_k <= 0:
            return []
        if self._mode == "vector":
            return self._vector_search(query, top_k, min_score)
        if self._mode == "fts":
            return self._fts_search(query, top_k, min_score)
        # hybrid
        return self._hybrid_search(query, top_k, min_score)

    def _vector_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """ANN vector search using query embedding."""
        query_vec = embed_query(query, model_name=self._embedding_model)

        results = (
            self._table.search(query_vec.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_arrow()
        )

        return self._arrow_to_chunks(results, min_score)

    def _fts_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """BM25 full-text search on the text column."""
        try:
            results = (
                self._table.search(query, query_type="fts")
                .limit(top_k)
                .to_arrow()
            )
            return self._arrow_to_chunks(results, min_score)
        except Exception:
            logger.warning("FTS search failed — falling back to vector.", exc_info=True)
            return self._vector_search(query, top_k, min_score)

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Hybrid search: vector + BM25 merged with Reciprocal Rank Fusion."""
        try:
            # LanceDB native hybrid: pass text query, it handles both vector + FTS
            results = (
                self._table.search(query, query_type="hybrid")
                .limit(top_k)
                .to_arrow()
            )
            return self._arrow_to_chunks(results, min_score)
        except Exception as e:
            logger.debug("Native hybrid search unavailable (%s) — using manual RRF.", e)
            query_vec = embed_query(query, model_name=self._embedding_model)
            return self._manual_rrf(query, query_vec, top_k, min_score)

    def _manual_rrf(
        self,
        query: str,
        query_vec: np.ndarray,
        top_k: int,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Manual RRF: run vector + FTS separately, merge with RRF scoring."""
        pool = top_k * 2  # fetch more candidates for merging

        # Vector results
        vec_results = (
            self._table.search(query_vec.tolist())
            .metric("cosine")
            .limit(pool)
            .to_arrow()
        )
        vec_chunks = self._arrow_to_chunks(vec_results, 0.0)

        # FTS results
        try:
            fts_results = (
                self._table.search(query, query_type="fts")
                .limit(pool)
                .to_arrow()
            )
            fts_chunks = self._arrow_to_chunks(fts_results, 0.0)
        except Exception:
            logger.warning("FTS unavailable for RRF — using vector only.")
            fts_chunks = []

        # RRF merge
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(vec_chunks):
            rrf = 1.0 / (rank + _RRF_K)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + rrf
            chunk_map[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(fts_chunks):
            rrf = 1.0 / (rank + _RRF_K)
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + rrf
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

        # Sort by RRF score (don't apply min_score — RRF scores are not comparable
        # to cosine similarity scores; they are rank-based, typically ~0.01-0.03)
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        merged: list[RetrievedChunk] = []
        for cid in sorted_ids[:top_k]:
            c = chunk_map[cid]
            c.score = scores[cid]
            merged.append(c)

        return merged

    @staticmethod
    def _arrow_to_chunks(
        arrow_table: Any,
        min_score: float,
    ) -> list[RetrievedChunk]:
        """Convert a PyArrow table of search results to RetrievedChunk list."""
        if arrow_table.num_rows == 0:
            return []

        col_names = set(arrow_table.column_names)

        # LanceDB returns _distance (lower = better) for vector search
        # and _score (higher = better) for FTS/hybrid
        has_distance = "_distance" in col_names
        has_score = "_score" in col_names

        chunks: list[RetrievedChunk] = []
        for i in range(arrow_table.num_rows):
            if has_score:
                score = float(arrow_table.column("_score")[i].as_py())
            elif has_distance:
                # Convert cosine distance to similarity: 1 - distance
                score = 1.0 - float(arrow_table.column("_distance")[i].as_py())
            else:
                score = 1.0

            if score < min_score:
                continue

            chunks.append(
                RetrievedChunk(
                    text=arrow_table.column("text")[i].as_py(),
                    score=score,
                    chunk_id=arrow_table.column("chunk_id")[i].as_py(),
                    file_path=arrow_table.column("file_path")[i].as_py(),
                    file_type=arrow_table.column("file_type")[i].as_py(),
                    section_title=arrow_table.column("section_title")[i].as_py(),
                    section_level=int(arrow_table.column("section_level")[i].as_py()),
                    chunk_index=int(arrow_table.column("chunk_index")[i].as_py()),
                    vector=arrow_table.column("vector")[i].as_py() if "vector" in col_names else None,
                )
            )

        return chunks
