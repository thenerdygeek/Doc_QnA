"""Tests for the FastEmbed wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from doc_qa.indexing.embedder import embed_query, embed_texts, get_embedding_dimension


class TestEmbedder:
    def test_embed_single_text(self) -> None:
        """Embedding a single text should return a vector of correct dimension."""
        vecs = embed_texts(["Hello world"])
        assert len(vecs) == 1
        assert isinstance(vecs[0], np.ndarray)
        assert vecs[0].shape == (768,)

    def test_embed_batch(self) -> None:
        """Embedding multiple texts should return one vector per text."""
        texts = ["First text", "Second text", "Third text"]
        vecs = embed_texts(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert v.shape == (768,)

    def test_embed_empty_list(self) -> None:
        """Embedding empty list should return empty list."""
        vecs = embed_texts([])
        assert vecs == []

    def test_similar_texts_have_higher_similarity(self) -> None:
        """Semantically similar texts should have higher cosine similarity."""
        vecs = embed_texts([
            "How does authentication work?",
            "What is the login flow?",
            "Recipe for chocolate cake",
        ])
        # Cosine similarity
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_related = cosine_sim(vecs[0], vecs[1])
        sim_unrelated = cosine_sim(vecs[0], vecs[2])
        assert sim_related > sim_unrelated

    def test_get_embedding_dimension(self) -> None:
        """Should return 768 for the default model (nomic-embed-text-v1.5)."""
        dim = get_embedding_dimension()
        assert dim == 768

    def test_embed_query(self) -> None:
        """embed_query should return a single vector."""
        vec = embed_query("What is authentication?")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (768,)
