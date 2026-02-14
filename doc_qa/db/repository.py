"""Async CRUD operations for conversations and messages."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy import delete, select, update
from sqlalchemy.orm import selectinload

from doc_qa.db.engine import get_session
from doc_qa.db.models import Conversation, DBMessage

logger = logging.getLogger(__name__)


async def create_conversation(
    title: str = "",
    user_id: uuid.UUID | None = None,
) -> dict:
    """Create a new conversation and return it as a dict."""
    async with get_session() as session:
        conv = Conversation(title=title, user_id=user_id)
        session.add(conv)
        await session.commit()
        await session.refresh(conv)
        return conv.to_dict()


async def list_conversations(
    user_id: uuid.UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List conversations ordered by most-recently updated."""
    async with get_session() as session:
        stmt = (
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if user_id is not None:
            stmt = stmt.where(Conversation.user_id == user_id)
        result = await session.execute(stmt)
        return [c.to_dict() for c in result.scalars().all()]


async def get_conversation_with_messages(conversation_id: str) -> dict | None:
    """Fetch a single conversation with all its messages."""
    async with get_session() as session:
        stmt = (
            select(Conversation)
            .where(Conversation.id == uuid.UUID(conversation_id))
            .options(selectinload(Conversation.messages))
        )
        result = await session.execute(stmt)
        conv = result.scalar_one_or_none()
        if conv is None:
            return None
        data = conv.to_dict()
        data["messages"] = [m.to_dict() for m in conv.messages]
        return data


async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict | None = None,
) -> dict:
    """Append a message to a conversation."""
    async with get_session() as session:
        msg = DBMessage(
            conversation_id=uuid.UUID(conversation_id),
            role=role,
            content=content,
            metadata_=metadata,
        )
        session.add(msg)
        # Touch the conversation's updated_at
        await session.execute(
            update(Conversation)
            .where(Conversation.id == uuid.UUID(conversation_id))
            .values(updated_at=Conversation.updated_at.default.arg)
        )
        await session.commit()
        await session.refresh(msg)
        return msg.to_dict()


async def update_conversation_title(conversation_id: str, title: str) -> bool:
    """Rename a conversation. Returns True if found."""
    async with get_session() as session:
        result = await session.execute(
            update(Conversation)
            .where(Conversation.id == uuid.UUID(conversation_id))
            .values(title=title)
        )
        await session.commit()
        return result.rowcount > 0


async def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation (cascades to messages). Returns True if found."""
    async with get_session() as session:
        result = await session.execute(
            delete(Conversation).where(Conversation.id == uuid.UUID(conversation_id))
        )
        await session.commit()
        return result.rowcount > 0
