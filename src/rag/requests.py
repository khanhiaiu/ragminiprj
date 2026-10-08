"""Persistent request telemetry, scoped to the same workspace as conversations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class RequestHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS request_history (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    model TEXT NOT NULL,
                    rerank INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    http_status INTEGER,
                    runtime_ms REAL,
                    ttft_ms REAL,
                    metrics TEXT NOT NULL DEFAULT '{}',
                    details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS requests_workspace_time
                    ON request_history(workspace_id, started_at DESC, id);
            """)

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def begin(self, *, workspace_id: str, session_id: str, question: str,
              model: str, rerank: bool) -> str:
        request_id = uuid.uuid4().hex
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO request_history "
                "(id, workspace_id, session_id, question, model, rerank, started_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (request_id, workspace_id, session_id, question, model, int(rerank),
                 datetime.now(timezone.utc).isoformat()),
            )
        return request_id

    def finish(self, request_id: str, *, status: str, http_status: int,
               metrics: dict, details: dict) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE request_history SET status=?, http_status=?, finished_at=?, "
                "runtime_ms=?, ttft_ms=?, metrics=?, details=? WHERE id=?",
                (status, http_status, datetime.now(timezone.utc).isoformat(),
                 metrics.get("runtime_ms"), metrics.get("ttft_ms"),
                 json.dumps(metrics, ensure_ascii=False, allow_nan=False),
                 json.dumps(details, ensure_ascii=False, allow_nan=False), request_id),
            )

    @staticmethod
    def _decode(row) -> dict:
        result = dict(row)
        result["rerank"] = bool(result["rerank"])
        result["metrics"] = json.loads(result["metrics"])
        if "details" in result:
            result["details"] = json.loads(result["details"])
        return result

    def list(self, workspace_id: str, *, limit: int = 50, offset: int = 0,
             status: str | None = None, session_id: str | None = None) -> dict:
        where = "workspace_id=?"
        values = [workspace_id]
        if status:
            where += " AND status=?"
            values.append(status)
        if session_id:
            where += " AND session_id=?"
            values.append(session_id)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            summary = dict(db.execute(
                "SELECT COUNT(*) AS total, "
                "COALESCE(SUM(status='success'), 0) AS success, "
                "COALESCE(SUM(status='fallback'), 0) AS fallback, "
                "COALESCE(SUM(status='error'), 0) AS error, "
                "COALESCE(SUM(status='running'), 0) AS running, "
                "AVG(runtime_ms) AS avg_runtime_ms, AVG(ttft_ms) AS avg_ttft_ms, "
                "COUNT(ttft_ms) AS ttft_samples "
                f"FROM request_history WHERE {where}", values,
            ).fetchone())
            rows = db.execute(
                "SELECT id, workspace_id, session_id, question, model, rerank, started_at, "
                "finished_at, status, http_status, runtime_ms, ttft_ms, metrics "
                f"FROM request_history WHERE {where} "
                "ORDER BY started_at DESC, id DESC LIMIT ? OFFSET ?",
                [*values, limit, offset],
            ).fetchall()
        return {"requests": [self._decode(row) for row in rows],
                "total": summary["total"], "summary": summary,
                "limit": limit, "offset": offset}

    def get(self, workspace_id: str, request_id: str) -> dict:
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT * FROM request_history WHERE id=? AND workspace_id=?",
                (request_id, workspace_id),
            ).fetchone()
        if row is None:
            raise LookupError("Không tìm thấy request.")
        return self._decode(row)
