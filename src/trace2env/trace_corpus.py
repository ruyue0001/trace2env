"""Retrieval over raw construction traces: the ``agentic_raw_traces`` control for reconstructed knowledge.

A ``TraceCorpus`` indexes the action -> observation turns of the original construction episodes
(segmented deterministically, observations verbatim, nothing induced) so the world-model agent can
search and read them in place of rules, notes, demonstrations, and extracted evidence. Turn ids are
``trace:<episode>:<turn>`` and are citable like any other retrieved artifact.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.memory import fts_query
from trace2env.models import EventKind, NormalizedAction

__all__ = ["TraceCorpus", "action_query"]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _action_of(content: Any) -> tuple[str, dict[str, Any]]:
    """An action event's content is a normalized action dict (type + arguments) or free text."""
    if isinstance(content, dict) and isinstance(content.get("type"), str):
        arguments = content.get("arguments")
        return content["type"], dict(arguments) if isinstance(arguments, dict) else {}
    text = _text(content)
    return (text.split()[0] if text.split() else "action"), {"text": text}


def action_query(action: NormalizedAction | dict[str, Any]) -> str:
    """Search tokens for an action: its type, programs, and command words (the first dozen)."""
    data = action.model_dump(mode="json") if isinstance(action, NormalizedAction) else dict(action)
    arguments = data.get("arguments") or {}
    tokens: list[str] = [str(data.get("type", ""))]
    for key in ("program", "programs", "command", "commands", "argv", "name", "tool"):
        value = arguments.get(key)
        if isinstance(value, str):
            tokens.extend(value.split())
        elif isinstance(value, list):
            for item in value:
                tokens.extend(str(item).split())
    seen: list[str] = []
    for token in tokens:
        token = token.strip("'\"`;|&()<>")
        if token and token not in seen and not token.startswith("-"):
            seen.append(token)
    return " ".join(seen[:12])


class TraceCorpus:
    """SQLite/FTS5 index over raw trace turns; built once per run, read by the workspace tools."""

    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.fts = False
        with self._connect() as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
            self.fts = "turns_fts" in tables
            self.turns = connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0] if "turns" in tables else 0
            self.episodes = connection.execute("SELECT COUNT(DISTINCT episode) FROM turns").fetchone()[0] if "turns" in tables else 0

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    # ─── Building ─────────────────────────────────────────────────────────────

    @classmethod
    def build(cls, sources: str | Path | list[str | Path], database: str | Path, *, max_history_events: int = 20) -> "TraceCorpus":
        """Index every episode file under ``sources`` (a directory of trace files, or explicit files)."""
        paths: list[Path] = []
        for source in (sources if isinstance(sources, list) else [sources]):
            path = Path(source)
            if path.is_dir():
                paths.extend(sorted(p for p in path.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix in {".json", ".jsonl", ".txt"}))
            else:
                paths.append(path)
        database = Path(database)
        if database.exists():
            database.unlink()
        database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database)
        try:
            connection.executescript(
                """
                CREATE TABLE turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, episode TEXT NOT NULL, turn INTEGER NOT NULL,
                    action_type TEXT NOT NULL, action_json TEXT NOT NULL, action_text TEXT NOT NULL,
                    observation TEXT NOT NULL, source TEXT NOT NULL
                );
                CREATE INDEX turns_episode_idx ON turns(episode, turn);
                """
            )
            rows: list[tuple[Any, ...]] = []
            # One label per file: the task part of `<task>__<run>` names (the terminal traces, one file per task); when several
            # files share that part (the web episodes are `webarena__<task>__<run>`) the whole stem, so turns of different
            # episodes never share an id and `read` returns the turn that was retrieved.
            prefixes = [path.stem.split("__")[0] or path.stem for path in paths]
            labels = {path: (path.stem if prefixes.count(prefix) > 1 else prefix) for path, prefix in zip(paths, prefixes)}

            for path in paths:
                label = labels[path]
                for episode in load_raw_traces(path):
                    by_id = {event.id: event for event in episode.events}
                    for number, slice_ in enumerate(segment_transitions(episode, max_history_events=max_history_events), start=1):
                        actions = [by_id[i] for i in slice_.action_event_ids if i in by_id]
                        observations = [by_id[i] for i in slice_.observation_event_ids if i in by_id and by_id[i].kind != EventKind.ACTION]
                        if not actions:
                            continue
                        action_type, arguments = _action_of(actions[0].content)
                        action_json = json.dumps({"type": action_type, "arguments": arguments}, ensure_ascii=False, default=str)
                        turn = actions[0].metadata.get("turn") if isinstance(actions[0].metadata.get("turn"), int) else number
                        observation = "\n".join(_text(event.content) for event in observations)
                        rows.append((label, turn, action_type, action_json, action_query({"type": action_type, "arguments": arguments}),
                                     observation, path.name))
            connection.executemany(
                "INSERT INTO turns(episode, turn, action_type, action_json, action_text, observation, source) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            try:
                connection.execute("CREATE VIRTUAL TABLE turns_fts USING fts5(action_text, observation, content='turns', content_rowid='id')")
                connection.execute("INSERT INTO turns_fts(turns_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                pass  # FTS5 unavailable: search falls back to LIKE
            connection.commit()
        finally:
            connection.close()
        return cls(database)

    # ─── Reading ──────────────────────────────────────────────────────────────

    @staticmethod
    def identifier(row: sqlite3.Row | dict[str, Any]) -> str:
        return f"trace:{row['episode']}:{row['turn']}"

    def search(self, query: str, limit: int = 8, *, snippet_words: int = 32) -> list[dict[str, Any]]:
        """Turns whose action or observation matches; FTS5 rank with a snippet, else LIKE."""
        if not query.strip() or limit <= 0:
            return []
        with self._connect() as connection:
            rows: list[sqlite3.Row] = []
            expression = fts_query(query)
            if self.fts and expression:
                try:
                    rows = connection.execute(
                        "SELECT t.id, t.episode, t.turn, t.action_type, t.action_json, length(t.observation) AS observation_chars, "
                        f"snippet(turns_fts, 1, '[', ']', '…', {int(snippet_words)}) AS snippet "
                        "FROM turns_fts f JOIN turns t ON t.id = f.rowid WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?",
                        (expression, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                token = f"%{query.strip()}%"
                rows = connection.execute(
                    "SELECT id, episode, turn, action_type, action_json, length(observation) AS observation_chars, "
                    "substr(observation, 1, 240) AS snippet FROM turns WHERE action_text LIKE ? OR observation LIKE ? LIMIT ?",
                    (token, token, limit),
                ).fetchall()
        return [
            {"id": self.identifier(row), "episode": row["episode"], "turn": row["turn"], "action_type": row["action_type"],
             "action": json.loads(row["action_json"]), "observation_chars": row["observation_chars"], "snippet": row["snippet"]}
            for row in rows
        ]

    def read(self, episode: str, turn: int, *, offset: int = 0, length: int = 6000) -> dict[str, Any] | None:
        """One turn's action and an exact span of its observation, with its neighbours' actions for context."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM turns WHERE episode = ? AND turn = ?", (episode, turn)).fetchone()
            if row is None:
                return None
            neighbours = connection.execute(
                "SELECT turn, action_type FROM turns WHERE episode = ? AND turn IN (?, ?) ORDER BY turn", (episode, turn - 1, turn + 1)
            ).fetchall()
        text = row["observation"]
        offset = max(0, offset)
        length = max(1, length)
        return {
            "id": self.identifier(row), "episode": row["episode"], "turn": row["turn"],
            "action": json.loads(row["action_json"]),
            "total_chars": len(text), "offset": offset, "length": min(length, max(0, len(text) - offset)),
            "text": text[offset:offset + length],
            "next_offset": offset + length if offset + length < len(text) else None,
            "neighbours": [{"turn": item["turn"], "action_type": item["action_type"]} for item in neighbours],
        }

    def describe(self) -> dict[str, Any]:
        return {"database": str(self.database), "episodes": self.episodes, "turns": self.turns, "fts": self.fts}
