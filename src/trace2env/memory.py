"""Episodic memory: what this session has already shown, searchable by the world-model agent.

Observed turns (from a trajectory prefix) and simulated turns are recorded in the session
database next to the structured state. The agent recalls them by lexical search (SQLite FTS5
when available, LIKE otherwise), so content created or revealed earlier in an episode can be
reproduced instead of guessed. Entries are evidence about the episode, never instructions.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from trace2env.models import MemoryEntry, utc_now

_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def fts_query(query: str) -> str:
    """Quote each token so shell-like text (``ls -la``) is a valid FTS5 expression."""
    tokens = [token.replace('"', "") for token in _TOKEN.findall(query)][:12]
    return " OR ".join(f'"{token}"' for token in tokens if token)


class EpisodicMemory:
    def __init__(self, database: str | Path):
        self.database = Path(database)
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, turn INTEGER NOT NULL, kind TEXT NOT NULL,
                    action_type TEXT NOT NULL, action_text TEXT NOT NULL, entry_json TEXT NOT NULL,
                    created TEXT NOT NULL
                );
                """
            )
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(action_text, observation, content='')"
                )
                self.fts = True
            except sqlite3.OperationalError:
                self.fts = False
            connection.commit()
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _action_text(entry: MemoryEntry) -> str:
        return f"{entry.action.type} " + json.dumps(entry.action.arguments, ensure_ascii=False, default=str)

    def record(self, entry: MemoryEntry) -> MemoryEntry:
        action_text = self._action_text(entry)
        connection = self._connect()
        try:
            cursor = connection.execute(
                "INSERT INTO memory(turn, kind, action_type, action_text, entry_json, created) VALUES (?, ?, ?, ?, ?, ?)",
                (entry.turn, entry.kind, entry.action.type, action_text, entry.model_dump_json(exclude={"id"}),
                 utc_now().isoformat()),
            )
            entry_id = int(cursor.lastrowid)
            if self.fts:
                connection.execute(
                    "INSERT INTO memory_fts(rowid, action_text, observation) VALUES (?, ?, ?)",
                    (entry_id, action_text, entry.observation),
                )
            connection.commit()
        finally:
            connection.close()
        return entry.model_copy(update={"id": entry_id})

    @staticmethod
    def _load(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry.model_validate({**json.loads(row["entry_json"]), "id": row["id"]})

    def count(self) -> int:
        connection = self._connect()
        try:
            return int(connection.execute("SELECT COUNT(*) FROM memory").fetchone()[0])
        finally:
            connection.close()

    def recent(self, limit: int = 5) -> list[MemoryEntry]:
        if limit <= 0:
            return []
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM memory ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [self._load(row) for row in reversed(rows)]
        finally:
            connection.close()

    def by_turn(self, turn: int) -> list[MemoryEntry]:
        """Every entry recorded for one turn number (observed before predicted), oldest first."""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM memory WHERE turn = ? ORDER BY CASE kind WHEN 'observed' THEN 0 ELSE 1 END, id",
                (int(turn),),
            ).fetchall()
            return [self._load(row) for row in rows]
        finally:
            connection.close()

    def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        if limit <= 0 or not query.strip():
            return []
        connection = self._connect()
        try:
            if self.fts:
                expression = fts_query(query)
                if not expression:
                    return []
                rows = connection.execute(
                    "SELECT m.* FROM memory_fts f JOIN memory m ON m.id = f.rowid "
                    "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expression, limit),
                ).fetchall()
            else:
                token = f"%{query.strip()}%"
                rows = connection.execute(
                    "SELECT * FROM memory WHERE action_text LIKE ? OR entry_json LIKE ? ORDER BY id DESC LIMIT ?",
                    (token, token, limit),
                ).fetchall()
            return [self._load(row) for row in rows]
        finally:
            connection.close()
