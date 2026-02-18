"""Tests for the feedback storage module."""
from __future__ import annotations

import os

import pytest

aiosqlite = pytest.importorskip("aiosqlite", reason="aiosqlite not installed")

from doc_qa.api.server import _QueryResultCache
from doc_qa.feedback.store import (
    FeedbackRecord,
    export_positive_feedback,
    get_feedback_stats,
    init_feedback_store,
    save_feedback,
)

# Use a temp directory that is guaranteed writable in sandbox mode.
_TEST_DB_DIR = "/tmp/claude/test_feedback"


@pytest.fixture(autouse=True)
async def _setup_db(tmp_path):
    """Create a fresh SQLite feedback database for each test."""
    db_path = str(tmp_path / "test_feedback.db")
    await init_feedback_store(db_path)
    yield db_path


class TestInitFeedbackStore:
    async def test_creates_table(self, _setup_db: str) -> None:
        """init_feedback_store should create the feedback table."""
        import aiosqlite

        async with aiosqlite.connect(_setup_db) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'"
            )
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] == "feedback"

    async def test_idempotent(self, _setup_db: str) -> None:
        """Calling init_feedback_store twice should not fail."""
        # Already initialized by fixture; calling again should be safe.
        await init_feedback_store(_setup_db)


class TestSaveFeedback:
    async def test_save_and_count(self, _setup_db: str) -> None:
        """save_feedback should insert a row into the feedback table."""
        record = FeedbackRecord(
            query_id="abc123",
            question="What is X?",
            answer="X is a thing.",
            chunks_used=["c1", "c2"],
            scores=[0.9, 0.8],
            confidence=0.85,
            verification_passed=True,
            rating=1,
            comment="Great answer",
        )
        await save_feedback(record)

        import aiosqlite

        async with aiosqlite.connect(_setup_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM feedback")
            count = (await cursor.fetchone())[0]
        assert count == 1

    async def test_save_multiple(self, _setup_db: str) -> None:
        """Saving multiple records should all persist."""
        for i in range(3):
            await save_feedback(
                FeedbackRecord(
                    query_id=f"q{i}",
                    question=f"Question {i}",
                    answer=f"Answer {i}",
                    rating=1 if i % 2 == 0 else -1,
                )
            )

        import aiosqlite

        async with aiosqlite.connect(_setup_db) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM feedback")
            count = (await cursor.fetchone())[0]
        assert count == 3

    async def test_save_with_none_verification(self, _setup_db: str) -> None:
        """verification_passed=None should store as NULL."""
        record = FeedbackRecord(
            query_id="q_none",
            question="Q",
            answer="A",
            verification_passed=None,
        )
        await save_feedback(record)

        import aiosqlite

        async with aiosqlite.connect(_setup_db) as db:
            cursor = await db.execute(
                "SELECT verification_passed FROM feedback WHERE query_id='q_none'"
            )
            row = await cursor.fetchone()
        assert row[0] is None

    async def test_save_with_false_verification(self, _setup_db: str) -> None:
        """verification_passed=False should store as 0."""
        record = FeedbackRecord(
            query_id="q_false",
            question="Q",
            answer="A",
            verification_passed=False,
        )
        await save_feedback(record)

        import aiosqlite

        async with aiosqlite.connect(_setup_db) as db:
            cursor = await db.execute(
                "SELECT verification_passed FROM feedback WHERE query_id='q_false'"
            )
            row = await cursor.fetchone()
        assert row[0] == 0


class TestGetFeedbackStats:
    async def test_empty_stats(self, _setup_db: str) -> None:
        """Stats on an empty database should return zeros."""
        stats = await get_feedback_stats()
        assert stats["total"] == 0
        assert stats["positive"] == 0
        assert stats["negative"] == 0
        assert stats["avg_confidence"] == 0.0
        assert stats["satisfaction_rate"] == 0.0

    async def test_stats_aggregation(self, _setup_db: str) -> None:
        """Stats should correctly aggregate ratings and confidence."""
        # 2 positive, 1 negative, 1 neutral
        records = [
            FeedbackRecord(query_id="a", question="Q1", answer="A1", rating=1, confidence=0.9),
            FeedbackRecord(query_id="b", question="Q2", answer="A2", rating=1, confidence=0.8),
            FeedbackRecord(query_id="c", question="Q3", answer="A3", rating=-1, confidence=0.3),
            FeedbackRecord(query_id="d", question="Q4", answer="A4", rating=0, confidence=0.5),
        ]
        for r in records:
            await save_feedback(r)

        stats = await get_feedback_stats()
        assert stats["total"] == 4
        assert stats["positive"] == 2
        assert stats["negative"] == 1
        assert stats["satisfaction_rate"] == 0.5  # 2/4
        # avg confidence = (0.9+0.8+0.3+0.5)/4 = 0.625
        assert stats["avg_confidence"] == 0.625


class TestExportPositiveFeedback:
    async def test_export_empty(self, _setup_db: str) -> None:
        """Export on empty DB should return empty list."""
        result = await export_positive_feedback()
        assert result == []

    async def test_export_only_positive(self, _setup_db: str) -> None:
        """Export should return only positively-rated entries."""
        await save_feedback(
            FeedbackRecord(query_id="pos", question="Good Q", answer="Good A", rating=1, confidence=0.9)
        )
        await save_feedback(
            FeedbackRecord(query_id="neg", question="Bad Q", answer="Bad A", rating=-1, confidence=0.2)
        )
        await save_feedback(
            FeedbackRecord(query_id="neu", question="Neutral Q", answer="Neutral A", rating=0)
        )

        result = await export_positive_feedback()
        assert len(result) == 1
        assert result[0]["question"] == "Good Q"

    async def test_export_respects_limit(self, _setup_db: str) -> None:
        """Export should respect the limit parameter."""
        for i in range(5):
            await save_feedback(
                FeedbackRecord(
                    query_id=f"p{i}", question=f"Q{i}", answer=f"A{i}", rating=1,
                )
            )

        result = await export_positive_feedback(limit=3)
        assert len(result) == 3

    async def test_export_contains_expected_fields(self, _setup_db: str) -> None:
        """Exported records should contain the expected fields."""
        await save_feedback(
            FeedbackRecord(
                query_id="test",
                question="Test Q",
                answer="Test A",
                chunks_used=["c1"],
                confidence=0.88,
                rating=1,
            )
        )
        result = await export_positive_feedback()
        assert len(result) == 1
        record = result[0]
        assert "question" in record
        assert "answer" in record
        assert "chunks_used" in record
        assert "confidence" in record


class TestQueryResultCache:
    """Tests for the in-memory _QueryResultCache used for feedback data capture."""

    def test_put_and_get(self) -> None:
        """Stored data is retrievable by query_id."""
        cache = _QueryResultCache(ttl=60, max_size=10)
        cache.put("q1", {"question": "What is X?", "answer": "X is Y."})
        result = cache.get("q1")
        assert result is not None
        assert result["question"] == "What is X?"

    def test_missing_key_returns_none(self) -> None:
        """Unknown query_id returns None."""
        cache = _QueryResultCache(ttl=60, max_size=10)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self) -> None:
        """Expired entries return None."""
        cache = _QueryResultCache(ttl=0, max_size=10)  # 0s TTL = immediate expiry
        cache.put("q1", {"question": "Q"})
        # TTL=0 means the entry expires immediately on the next get
        import time
        time.sleep(0.01)
        assert cache.get("q1") is None

    def test_max_size_eviction(self) -> None:
        """When cache is full, oldest entry is evicted."""
        cache = _QueryResultCache(ttl=600, max_size=2)
        cache.put("q1", {"n": 1})
        cache.put("q2", {"n": 2})
        cache.put("q3", {"n": 3})  # should evict q1
        assert cache.get("q1") is None
        assert cache.get("q2") is not None
        assert cache.get("q3") is not None

    def test_feedback_with_cache_populates_fields(self, _setup_db: str) -> None:
        """When cache has data, FeedbackRecord gets populated."""
        cache = _QueryResultCache(ttl=60, max_size=10)
        cache.put("qid_abc", {
            "question": "What is auth?",
            "answer": "Auth uses JWT.",
            "chunks_used": ["auth.md#0", "auth.md#1"],
            "scores": [0.95, 0.88],
            "confidence": 0.91,
            "verification_passed": True,
        })

        cached = cache.get("qid_abc")
        assert cached is not None

        record = FeedbackRecord(
            query_id="qid_abc",
            question=cached["question"],
            answer=cached["answer"],
            chunks_used=cached["chunks_used"],
            scores=cached["scores"],
            confidence=cached["confidence"],
            verification_passed=cached["verification_passed"],
            rating=1,
        )
        assert record.question == "What is auth?"
        assert record.answer == "Auth uses JWT."
        assert record.chunks_used == ["auth.md#0", "auth.md#1"]
        assert record.confidence == 0.91
