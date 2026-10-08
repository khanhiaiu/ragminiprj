"""Durable chat transcripts and independent working context for each session.

Each operation uses its own SQLite connection. Revision checks prevent concurrent
workers from silently overwriting a newer answer or a context reset.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class ConversationConflict(Exception):
    """The session changed while an answer was being generated."""


class ConversationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Bounded lock pool serializes requests within a process; SQLite revisions
        # also protect callers using multiple API processes.
        self._locks = [threading.RLock() for _ in range(64)]
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    context TEXT NOT NULL DEFAULT '{"turns": [], "summary": ""}',
                    revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS conversations_workspace
                    ON conversations(workspace_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    citations TEXT NOT NULL DEFAULT '[]',
                    sources TEXT NOT NULL DEFAULT '[]',
                    fallback INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, id);
            """)
            # Upgrade existing transcripts without losing messages or context.
            if "sources" not in {row[1] for row in db.execute("PRAGMA table_info(messages)")}:
                db.execute("ALTER TABLE messages ADD COLUMN sources TEXT NOT NULL DEFAULT '[]'")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def lock(self, session_id: str):
        return self._locks[hash(session_id) % len(self._locks)]

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat()

    def create(self, workspace_id: str, session_id: str | None = None) -> dict:
        session_id = session_id or uuid.uuid4().hex
        now = self._now()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT OR IGNORE INTO conversations(id, workspace_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, workspace_id, "Cuộc trò chuyện mới", now, now),
            )
        return self.get(workspace_id, session_id)

    def list(self, workspace_id: str) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count "
                "FROM conversations c LEFT JOIN messages m ON m.session_id=c.id "
                "WHERE c.workspace_id=? GROUP BY c.id ORDER BY c.updated_at DESC, c.id",
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, workspace_id: str, session_id: str) -> dict:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            row = db.execute(
                "SELECT * FROM conversations WHERE id=? AND workspace_id=?",
                (session_id, workspace_id),
            ).fetchone()
            if row is None:
                raise LookupError("Không tìm thấy cuộc trò chuyện.")
            messages = db.execute(
                "SELECT id, role, content, citations, sources, fallback, created_at FROM messages "
                "WHERE session_id=? ORDER BY id", (session_id,),
            ).fetchall()
        result = dict(row)
        result["context"] = json.loads(result["context"])
        result["messages"] = [
            {**dict(message), "citations": json.loads(message["citations"]),
             "sources": json.loads(message["sources"]),
             "fallback": bool(message["fallback"])} for message in messages
        ]
        return result

    def record_exchange(self, workspace_id: str, session_id: str, question: str,
                        result: dict, context: dict, revision: int) -> int:
        now = self._now()
        with closing(self._connect()) as db, db:
            changed = db.execute(
                "UPDATE conversations SET context=?, revision=revision+1, updated_at=?, "
                "title=CASE WHEN NOT EXISTS (SELECT 1 FROM messages WHERE session_id=conversations.id) "
                "THEN ? ELSE title END "
                "WHERE id=? AND workspace_id=? AND revision=?",
                (json.dumps(context, ensure_ascii=False), now, question.strip()[:80],
                 session_id, workspace_id, revision),
            ).rowcount
            if not changed:
                raise ConversationConflict("Phiên đã thay đổi. Vui lòng tải lại rồi thử lại.")
            db.execute(
                "INSERT INTO messages(session_id, role, content, created_at) VALUES (?, 'user', ?, ?)",
                (session_id, question, now),
            )
            assistant = db.execute(
                "INSERT INTO messages(session_id, role, content, citations, sources, fallback, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?, ?, ?)",
                (session_id, result["answer"], json.dumps(result.get("citations", []), ensure_ascii=False),
                 json.dumps(result.get("sources", []), ensure_ascii=False),
                 int(result.get("fallback", False)), now),
            )
            return assistant.lastrowid

    def delete(self, workspace_id: str, session_id: str) -> None:
        with self.lock(session_id), closing(self._connect()) as db, db:
            changed = db.execute("DELETE FROM conversations WHERE id=? AND workspace_id=?",
                                 (session_id, workspace_id)).rowcount
            if not changed:
                raise LookupError("Không tìm thấy cuộc trò chuyện.")

    def reset_context(self, workspace_id: str, session_id: str) -> dict:
        with self.lock(session_id), closing(self._connect()) as db, db:
            changed = db.execute(
                "UPDATE conversations SET context=?, revision=revision+1, updated_at=? "
                "WHERE id=? AND workspace_id=?",
                ('{"turns": [], "summary": ""}', self._now(), session_id, workspace_id),
            ).rowcount
            if not changed:
                raise LookupError("Không tìm thấy cuộc trò chuyện.")
        return self.get(workspace_id, session_id)
