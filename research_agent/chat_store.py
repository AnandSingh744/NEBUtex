"""Conversations and their messages, saved in PostgreSQL so chat history survives refreshes and restarts.

Each assistant message keeps a COPY of the passages it cited, so old answers still show their
evidence even after the documents are re-uploaded or deleted.
"""

import uuid

from psycopg.types.json import Jsonb

from research_agent.db import get_pool

TITLE_LENGTH = 60  # conversation titles come from the first question, shortened to this


class ChatStore:
    def __init__(self):
        self._pool = get_pool()

    def create_conversation(self, first_question: str) -> str:
        """Start a conversation titled after its first question. Returns its id."""
        title = " ".join(first_question.split())
        if len(title) > TITLE_LENGTH:
            title = title[: TITLE_LENGTH - 1].rstrip() + "…"
        with self._pool.connection() as conn:
            return str(conn.execute(
                "INSERT INTO conversations (title) VALUES (%s) RETURNING id", (title,)
            ).fetchone()[0])

    def list_conversations(self, limit: int = 30) -> list[dict]:
        """Most recently active first."""
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC LIMIT %s", (limit,)
            ).fetchall()
        return [{"id": str(r[0]), "title": r[1], "updated_at": r[2]} for r in rows]

    def add_message(
        self, conversation_id: str, role: str, content: str,
        status: str | None = None, citations: list[dict] | None = None,
    ) -> None:
        """Save one message and mark the conversation as just active, together (one transaction)."""
        with self._pool.connection() as conn, conn.transaction():
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, status, citations) VALUES (%s, %s, %s, %s, %s)",
                (conversation_id, role, content, status, Jsonb(citations or [])),
            )
            conn.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (conversation_id,))

    def get_messages(self, conversation_id: str) -> list[dict]:
        """All messages of a conversation, oldest first. Empty if the id is unknown or malformed."""
        if not _is_uuid(conversation_id):
            return []  # e.g. a hand-edited URL; don't send garbage to the database
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT role, content, status, citations FROM messages WHERE conversation_id = %s ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [{"role": r[0], "content": r[1], "status": r[2], "citations": r[3]} for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation; its messages go with it (ON DELETE CASCADE)."""
        if _is_uuid(conversation_id):
            with self._pool.connection() as conn:
                conn.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError, AttributeError):
        return False
