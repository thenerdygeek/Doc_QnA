"""SQLite-backed feedback storage for query quality tracking.

Stores user feedback (thumbs up/down) and query metadata so that:
- We can compute satisfaction rates and confidence trends.
- Positively-rated Q&A pairs can be exported as evaluation data.
- The feedback loop is fully async and does not block the query pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """A single feedback entry tied to a query."""

    query_id: str
    question: str
    answer: str
    chunks_used: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    confidence: float = 0.0
    verification_passed: bool | None = None
    rating: int = 0  # 1 = thumbs up, -1 = thumbs down, 0 = no rating
    comment: str = ""


_DB_PATH: str = ""


async def init_feedback_store(db_path: str = "./data/feedback.db") -> None:
    """Initialize the SQLite database and create the feedback table."""
    global _DB_PATH
    _DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                chunks_used TEXT DEFAULT '[]',
                scores TEXT DEFAULT '[]',
                confidence REAL DEFAULT 0.0,
                verification_passed INTEGER,
                rating INTEGER DEFAULT 0,
                comment TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def save_feedback(record: FeedbackRecord) -> None:
    """Persist a feedback record to the SQLite database."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(
            """INSERT INTO feedback
               (query_id, question, answer, chunks_used, scores,
                confidence, verification_passed, rating, comment)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.query_id,
                record.question,
                record.answer,
                json.dumps(record.chunks_used),
                json.dumps(record.scores),
                record.confidence,
                1 if record.verification_passed is True
                else (0 if record.verification_passed is False else None),
                record.rating,
                record.comment,
            ),
        )
        await db.commit()


async def get_feedback_stats() -> dict[str, Any]:
    """Return aggregated feedback statistics."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        total = (await (await db.execute("SELECT COUNT(*) FROM feedback")).fetchone())[0]
        positive = (
            await (await db.execute("SELECT COUNT(*) FROM feedback WHERE rating = 1")).fetchone()
        )[0]
        negative = (
            await (await db.execute("SELECT COUNT(*) FROM feedback WHERE rating = -1")).fetchone()
        )[0]
        avg_conf = (
            await (await db.execute("SELECT AVG(confidence) FROM feedback")).fetchone()
        )[0] or 0.0
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "avg_confidence": round(avg_conf, 3),
            "satisfaction_rate": round(positive / total, 3) if total > 0 else 0.0,
        }


async def get_chunk_feedback_scores() -> dict[str, float]:
    """Aggregate user feedback into per-chunk scores in [-1.0, 1.0].

    For each chunk that appears in rated feedback, computes::

        score = (positive_count - negative_count) / total_count

    Returns:
        Dict mapping chunk_id → score.  Chunks with no feedback are
        absent (treated as 0.0 by callers).
    """
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (
            await db.execute(
                "SELECT chunks_used, rating FROM feedback WHERE rating != 0"
            )
        ).fetchall()

    # Accumulate per-chunk tallies
    pos: dict[str, int] = {}
    neg: dict[str, int] = {}
    total: dict[str, int] = {}

    for row in rows:
        raw_chunks = row["chunks_used"]
        rating = int(row["rating"])
        try:
            chunk_ids = json.loads(raw_chunks) if isinstance(raw_chunks, str) else raw_chunks
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(chunk_ids, list):
            continue
        for cid in chunk_ids:
            if not isinstance(cid, str) or not cid:
                continue
            total[cid] = total.get(cid, 0) + 1
            if rating > 0:
                pos[cid] = pos.get(cid, 0) + 1
            elif rating < 0:
                neg[cid] = neg.get(cid, 0) + 1

    scores: dict[str, float] = {}
    for cid, t in total.items():
        p = pos.get(cid, 0)
        n = neg.get(cid, 0)
        scores[cid] = (p - n) / t

    return scores


async def export_positive_feedback(limit: int = 100) -> list[dict]:
    """Export positively-rated Q&A pairs for evaluation dataset building."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (
            await db.execute(
                "SELECT question, answer, chunks_used, confidence "
                "FROM feedback WHERE rating = 1 "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        ).fetchall()
        return [dict(r) for r in rows]
