"""SQLite-backed conversation storage.

Replaces PostgreSQL-based conversation persistence with a lightweight SQLite
store.  Pattern mirrors ``feedback/store.py``: module-level ``_DB_PATH``,
lazy ``aiosqlite`` imports, and plain ``async def`` functions.

Tables:
    conversations — id, title, created_at, updated_at
    messages      — id, conversation_id (FK), role, content, query_id,
                    metadata_json, created_at
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DB_PATH: str = ""


def _now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new UUID for primary keys."""
    return uuid.uuid4().hex


async def init_conversation_store(db_path: str = "./data/conversations.db") -> None:
    """Create the SQLite database and conversation/messages tables."""
    global _DB_PATH
    _DB_PATH = db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    import aiosqlite

    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                query_id TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv "
            "ON messages(conversation_id, created_at)"
        )
        await db.execute("PRAGMA foreign_keys = ON")
        await db.commit()
    logger.info("Conversation store initialized at %s", db_path)


async def create_conversation(title: str = "") -> dict[str, Any]:
    """Create a new conversation and return its dict representation."""
    import aiosqlite

    cid = _new_id()
    now = _now_iso()
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, title, now, now),
        )
        await db.commit()
    return {"id": cid, "title": title, "created_at": now, "updated_at": now}


async def list_conversations(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List conversations ordered by most recently updated."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        rows = await (
            await db.execute(
                "SELECT id, title, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        ).fetchall()
        return [dict(r) for r in rows]


async def get_conversation_with_messages(conversation_id: str) -> dict[str, Any] | None:
    """Fetch a conversation with all its messages, or None if not found."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        row = await (
            await db.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            )
        ).fetchone()
        if row is None:
            return None

        conv = dict(row)

        msg_rows = await (
            await db.execute(
                "SELECT id, conversation_id, role, content, query_id, metadata_json, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            )
        ).fetchall()

        messages = []
        for m in msg_rows:
            md = dict(m)
            # Parse metadata_json back to dict
            raw_meta = md.pop("metadata_json", None)
            md["metadata"] = json.loads(raw_meta) if raw_meta else None
            messages.append(md)

        conv["messages"] = messages
        return conv


async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    query_id: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Append a message to a conversation and touch updated_at."""
    import aiosqlite

    mid = _new_id()
    now = _now_iso()
    meta_json = json.dumps(metadata) if metadata else None

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, query_id, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mid, conversation_id, role, content, query_id, meta_json, now),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        await db.commit()

    return {
        "id": mid,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "query_id": query_id,
        "metadata": metadata,
        "created_at": now,
    }


async def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Rename a conversation. Returns True if found, False otherwise."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now_iso(), conversation_id),
        )
        await db.commit()
        return cursor.rowcount > 0


async def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation and cascade to its messages. Returns True if found."""
    import aiosqlite

    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        cursor = await db.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        await db.commit()
        return cursor.rowcount > 0
