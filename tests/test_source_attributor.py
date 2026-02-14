"""Tests for the source attributor module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from doc_qa.retrieval.retriever import RetrievedChunk
from doc_qa.verification.source_attributor import (
    Attribution,
    _cosine_similarity,
    _split_sentences,
    attribute_sources,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _chunk(text="chunk text", chunk_id="c1", score=0.8):
    return RetrievedChunk(
        text=text, score=score, chunk_id=chunk_id,
        file_path="test.md", file_type="md",
        section_title="Section", section_level=1, chunk_index=0,
    )


# ── Sentence splitting ──────────────────────────────────────────────


class TestSentenceSplitting:
    def test_basic_split(self):
        sentences = _split_sentences("Hello world. How are you? Fine!")
        assert len(sentences) == 3

    def test_newline_split(self):
        sentences = _split_sentences("First line.\nSecond line.")
        assert len(sentences) == 2

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence(self):
        sentences = _split_sentences("Just one sentence")
        assert len(sentences) == 1

    def test_whitespace_only(self):
        assert _split_sentences("   ") == []


# ── Cosine similarity ───────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0


# ── attribute_sources ────────────────────────────────────────────────


class TestAttributeSources:
    """Test attribute_sources with mocked embeddings."""

    def _mock_embed(self, texts):
        """Return deterministic embeddings based on text hash."""
        results = []
        for t in texts:
            # Simple hash-based vector for deterministic results
            h = hash(t) % 1000
            results.append([float(h), float(h + 1), float(h + 2)])
        return results

    def test_correct_source_assignment(self):
        """Each sentence should be assigned to the chunk it's most similar to."""
        chunks = [
            _chunk(text="OAuth tokens authenticate users.", chunk_id="c1"),
            _chunk(text="Docker containers run apps.", chunk_id="c2"),
        ]

        # Mock embeddings so that sentences about auth match chunk 1,
        # and sentences about docker match chunk 2
        def mock_embed(texts):
            results = []
            for t in texts:
                if "auth" in t.lower() or "oauth" in t.lower():
                    results.append([1.0, 0.0, 0.0])
                elif "docker" in t.lower() or "container" in t.lower():
                    results.append([0.0, 1.0, 0.0])
                else:
                    results.append([0.5, 0.5, 0.0])
            return results

        with patch("doc_qa.verification.source_attributor._embed_texts", side_effect=mock_embed):
            attrs = attribute_sources(
                "OAuth is used for auth. Docker runs the app.",
                chunks,
            )

        assert len(attrs) == 2
        # First sentence about OAuth → chunk 0 (auth chunk)
        assert attrs[0].source_index == 0
        # Second sentence about Docker → chunk 1 (docker chunk)
        assert attrs[1].source_index == 1

    def test_below_threshold_returns_minus_one(self):
        """Sentences below similarity threshold get source_index=-1."""
        chunks = [_chunk(text="Something completely different.", chunk_id="c1")]

        def mock_embed(texts):
            # Make sentence and chunk very dissimilar
            results = []
            for i, t in enumerate(texts):
                if i == 0:  # sentence
                    results.append([1.0, 0.0, 0.0])
                else:  # chunk
                    results.append([0.0, 1.0, 0.0])
            return results

        with patch("doc_qa.verification.source_attributor._embed_texts", side_effect=mock_embed):
            attrs = attribute_sources("Unrelated content.", chunks)

        assert len(attrs) == 1
        assert attrs[0].source_index == -1

    def test_empty_answer(self):
        chunks = [_chunk()]
        attrs = attribute_sources("", chunks)
        assert attrs == []

    def test_no_chunks(self):
        attrs = attribute_sources("Some answer text.", [])
        assert attrs == []

    def test_whitespace_answer(self):
        chunks = [_chunk()]
        attrs = attribute_sources("   ", chunks)
        assert attrs == []
