"""Transactional, per-session mutable workspace backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from trace2env.models import AuditRecord, EnvironmentState
from trace2env.storage import write_json, write_jsonl


class RevisionConflict(RuntimeError):
    pass


class SessionStore:
    def __init__(self, root: str | Path, initial_state: EnvironmentState | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "session.sqlite"
        self._initialize(initial_state or EnvironmentState())

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self, initial_state: EnvironmentState) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS current_state (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    revision INTEGER NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO current_state VALUES (1, ?, ?)",
                (initial_state.revision, initial_state.model_dump_json()),
            )
            connection.commit()
        finally:
            connection.close()
        self._refresh_mirrors()

    def load(self) -> EnvironmentState:
        connection = self._connect()
        try:
            row = connection.execute("SELECT state_json FROM current_state WHERE singleton = 1").fetchone()
            if row is None:
                raise RuntimeError("Session has no current state")
            return EnvironmentState.model_validate_json(row["state_json"])
        finally:
            connection.close()

    def commit(self, expected_revision: int, state: EnvironmentState, audit: AuditRecord) -> EnvironmentState:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT revision FROM current_state WHERE singleton = 1").fetchone()
            if row is None or row["revision"] != expected_revision:
                connection.rollback()
                raise RevisionConflict(
                    f"Expected revision {expected_revision}, found {None if row is None else row['revision']}"
                )
            committed = state.model_copy(deep=True)
            committed.revision = expected_revision + 1
            audit.committed = True
            audit.new_revision = committed.revision
            connection.execute(
                "UPDATE current_state SET revision = ?, state_json = ? WHERE singleton = 1",
                (committed.revision, committed.model_dump_json()),
            )
            connection.execute(
                "INSERT INTO audit(timestamp, record_json) VALUES (?, ?)",
                (audit.timestamp.isoformat(), audit.model_dump_json()),
            )
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()
        self._refresh_mirrors()
        return committed

    def record_failure(self, audit: AuditRecord) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO audit(timestamp, record_json) VALUES (?, ?)",
                (audit.timestamp.isoformat(), audit.model_dump_json()),
            )
            connection.commit()
        finally:
            connection.close()
        self._refresh_mirrors()

    def audit_records(self) -> list[AuditRecord]:
        connection = self._connect()
        try:
            rows = connection.execute("SELECT record_json FROM audit ORDER BY sequence").fetchall()
            return [AuditRecord.model_validate_json(row["record_json"]) for row in rows]
        finally:
            connection.close()

    def _refresh_mirrors(self) -> None:
        write_json(self.root / "state.json", self.load())
        write_jsonl(self.root / "audit.jsonl", self.audit_records())
