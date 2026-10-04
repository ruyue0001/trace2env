"""Read-only package repository and bounded retrieval interface for the runtime agent."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from trace2env.eligibility import rule_eligible

from trace2env.memory import fts_query
from trace2env.models import (
    ActionSchema,
    Demonstration,
    EnvironmentManifest,
    EnvironmentNote,
    EnvironmentState,
    Invariant,
    LocalTransitionEvidence,
    RenderContract,
    StateSchema,
    TransitionRule,
    TransitionSlice,
)
from trace2env.storage import load_models, read_json

KNOWLEDGE_KINDS = ["rule", "note", "demonstration", "evidence", "action"]


class PackageIntegrityError(RuntimeError):
    pass


class EnvironmentPackage:
    def __init__(self, root: str | Path, verify_hashes: bool = True):
        self.root = Path(root).resolve()
        self.manifest = EnvironmentManifest.model_validate(read_json(self.root / "manifest.json"))
        self.scope = dict(self.manifest.metadata.get("scope_context", {}))
        self.scope.setdefault("environment_id", self.manifest.environment_id)
        if self.manifest.metadata.get("tenant_scope") is not None:
            self.scope.setdefault("tenant", self.manifest.metadata["tenant_scope"])
        if verify_hashes:
            self.verify()
        self.action_schema = ActionSchema.model_validate(read_json(self.root / "action_schema.json"))
        self.state_schema = StateSchema.model_validate(read_json(self.root / "state_schema.json"))
        self.invariants = [Invariant.model_validate(value) for value in read_json(self.root / "invariants.json")]
        self.renderers = [
            RenderContract.model_validate(value) for value in read_json(self.root / "renderer" / "contracts.json")
        ]
        self.rules = load_models(self.root / "rules" / "index.jsonl", TransitionRule)
        self.demonstrations = load_models(
            self.root / "demonstrations" / "transitions.jsonl", Demonstration
        )
        # Packages compiled before notes existed simply have none.
        self.notes = load_models(self.root / "knowledge" / "notes.jsonl", EnvironmentNote)
        action_names = {action.name for action in self.action_schema.actions}
        renderer_ids = {renderer.id for renderer in self.renderers}
        if {rule.action_type for rule in self.rules} - action_names:
            raise PackageIntegrityError("Rules reference actions absent from action_schema.json")
        if {rule.renderer for rule in self.rules} - renderer_ids:
            raise PackageIntegrityError("Rules reference renderers absent from renderer/contracts.json")

    def verify(self) -> None:
        for relative, expected in self.manifest.files.items():
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise PackageIntegrityError(f"Manifest path escapes package: {relative}")
            if not path.is_file():
                raise PackageIntegrityError(f"Package file is missing: {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise PackageIntegrityError(f"Package file hash mismatch: {relative}")


class PackageInspector:
    """The runtime sees semantic query results, never unrestricted filesystem access."""

    def __init__(self, package: EnvironmentPackage):
        self.package = package

    def list_actions(self) -> list[dict[str, Any]]:
        return [action.model_dump(mode="json") for action in self.package.action_schema.actions]

    def inspect_action(self, action_type: str, limit: int | None = 12,
                       state: EnvironmentState | None = None) -> dict[str, Any]:
        specs = [
            action for action in self.package.action_schema.actions
            if action.name == action_type or action_type in action.aliases
        ]
        canonical = specs[0].name if specs else action_type
        rules = sorted(
            [
                rule for rule in self.package.rules
                if rule.action_type == canonical and rule_eligible(rule, self.package.scope, state,
                    minimum_confidence=self.package.manifest.metadata.get("min_rule_confidence", 0.0),
                    allow_state_unknown=state is None)
            ],
            key=lambda rule: (rule.priority, rule.confidence, rule.id),
            reverse=True,
        )[:limit]
        renderer_ids = {rule.renderer for rule in rules}
        renderers = [
            renderer for renderer in self.package.renderers
            if renderer.id in renderer_ids or canonical in renderer.action_types
        ]
        demonstrations = [
            demo for demo in self.package.demonstrations if demo.action.type == canonical
        ][:4]
        notes = [
            note for note in self.package.notes
            if note.status == "supported" and (not note.action_types or canonical in note.action_types)
        ][:limit or None]
        artifact_ids = [f"rule:{rule.id}" for rule in rules]
        artifact_ids += [f"renderer:{renderer.id}" for renderer in renderers]
        artifact_ids += [f"demo:{demo.id}" for demo in demonstrations]
        artifact_ids += [f"note:{note.id}" for note in notes]
        return {
            "canonical_action_type": canonical,
            "action_specs": [item.model_dump(mode="json") for item in specs],
            "rules": [item.model_dump(mode="json") for item in rules],
            "renderers": [item.model_dump(mode="json") for item in renderers],
            "demonstrations": [item.model_dump(mode="json") for item in demonstrations],
            "notes": [item.model_dump(mode="json") for item in notes],
            "invariants": [item.model_dump(mode="json") for item in self.package.invariants],
            "artifact_ids": artifact_ids,
        }

    # ─── Evidence tier: the raw turns behind the package, readable exactly (v3 read path) ─────

    def _evidence_index(self) -> dict[str, Any]:
        """Evidence records by id, their position in their episode, and each episode's turn sequence."""
        cached = getattr(self, "_evidence_cache", None)
        if cached is not None:
            return cached
        records = load_models(self.package.root / "evidence" / "local_transitions.jsonl", LocalTransitionEvidence)
        slices = load_models(self.package.root / "evidence" / "transitions.jsonl", TransitionSlice)
        order: dict[str, list[str]] = {}
        for item in slices:
            order.setdefault(item.episode_id, []).append(item.id)
        by_transition = {record.transition_id: record for record in records}
        sequences: dict[str, list[str]] = {}
        position: dict[str, tuple[str, int]] = {}
        for episode_id, transition_ids in order.items():
            sequence = [by_transition[t].id for t in transition_ids if t in by_transition]
            sequences[episode_id] = sequence
            for index, evidence_id in enumerate(sequence, start=1):
                position[evidence_id] = (episode_id, index)
        cached = {"by_id": {record.id: record for record in records}, "position": position, "sequences": sequences}
        self._evidence_cache = cached
        return cached

    def resolve_evidence_id(self, reference: str) -> str | None:
        """``evidence:<id>``, ``demo:<id>``, a bare evidence id, or a demonstration id -> the evidence record id."""
        ref = str(reference).strip()
        for prefix in ("evidence:", "demo:", "demonstration:"):
            if ref.startswith(prefix):
                ref = ref[len(prefix):]
        if ref.startswith("demo_"):
            ref = ref[len("demo_"):]
        return ref if ref in self._evidence_index()["by_id"] else None

    def evidence_summary(self, evidence_id: str, *, snippet: str | None = None) -> dict[str, Any] | None:
        index = self._evidence_index()
        record = index["by_id"].get(evidence_id)
        if record is None:
            return None
        episode_id, turn = index["position"].get(record.id, (record.episode_id, 0))
        return {
            "id": f"evidence:{record.id}", "episode": episode_id, "turn": turn, "action_type": record.action.type,
            "action_arguments": {key: value for key, value in record.action.arguments.items() if key != "keystrokes"},
            "observation_chars": len(record.observation_text),
            **({"snippet": snippet} if snippet is not None else {}),
            "read": f"read_evidence(id='evidence:{record.id}')",
        }

    def evidence_view(self, evidence_id: str, *, offset: int = 0, length: int = 6000) -> dict[str, Any] | None:
        """One raw turn: its action, an exact span of its observation, and the neighbouring turns' actions."""
        summary = self.evidence_summary(evidence_id)
        if summary is None:
            return None
        index = self._evidence_index()
        record = index["by_id"][evidence_id]
        episode_id, turn = summary["episode"], summary["turn"]
        sequence = index["sequences"].get(episode_id, [])
        neighbours = []
        for other_turn in (turn - 1, turn + 1):
            if turn and 1 <= other_turn <= len(sequence):
                other = index["by_id"][sequence[other_turn - 1]]
                neighbours.append({"id": f"evidence:{other.id}", "turn": other_turn, "action_type": other.action.type})
        text = record.observation_text
        offset = max(0, int(offset))
        length = max(1, int(length))
        summary.pop("read", None)
        return {
            **summary, "total_chars": len(text), "offset": offset, "length": min(length, max(0, len(text) - offset)),
            "text": text[offset:offset + length], "next_offset": offset + length if offset + length < len(text) else None,
            "neighbours": neighbours,
        }

    @staticmethod
    def observation_snippet(text: str, query: str, width: int = 240) -> str:
        """A window of the observation around the first query token it contains (the head when none matches)."""
        lowered = text.lower()
        tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_./-]+", query) if len(token) > 1]
        positions = [lowered.find(token) for token in tokens]
        positions = [position for position in positions if position >= 0]
        start = max(0, min(positions) - width // 3) if positions else 0
        window = text[start:start + width]
        return ("…" if start else "") + window + ("…" if start + width < len(text) else "")

    def search_evidence(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Full-text hits over the raw turns (evidence and demonstrations) as summaries with snippets, never clipped bodies."""
        hits = self.search_knowledge(query, kinds=["evidence", "demonstration"], limit=max(limit * 2, limit))
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        index = self._evidence_index()
        for hit in hits:
            evidence_id = self.resolve_evidence_id(hit["ref_id"])
            if evidence_id is None or evidence_id in seen:
                continue
            seen.add(evidence_id)
            record = index["by_id"][evidence_id]
            summary = self.evidence_summary(evidence_id, snippet=self.observation_snippet(record.observation_text, query))
            if summary is not None:
                results.append(summary)
            if len(results) >= limit:
                break
        return results

    def inspect_state(self, prefix: str | None = None) -> list[dict[str, Any]]:
        fields = self.package.state_schema.fields
        if prefix:
            fields = [field for field in fields if field.path.startswith(prefix)]
        return [field.model_dump(mode="json") for field in fields]

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Lexical discovery for unfamiliar actions; detailed reads remain typed."""
        index = self.package.root / "index.sqlite"
        connection = sqlite3.connect(f"file:{index.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        token = f"%{query}%"
        try:
            rows = connection.execute(
                "SELECT id, action_type, description, artifact_path FROM rules "
                "WHERE status = 'supported' AND (action_type LIKE ? OR description LIKE ?) "
                "ORDER BY confidence DESC, priority DESC LIMIT ?",
                (token, token, limit),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def search_knowledge(self, query: str, kinds: list[str] | None = None, limit: int = 8) -> list[dict[str, Any]]:
        """Full-text discovery across rules, notes, demonstrations, evidence, and actions.

        Uses the package's FTS5 index when the compiler could build one, a LIKE scan otherwise,
        and degrades to rule search for packages compiled before the knowledge catalog existed.
        """
        if not query.strip() or limit <= 0:
            return []
        selected = [kind for kind in (kinds or KNOWLEDGE_KINDS) if kind in KNOWLEDGE_KINDS] or KNOWLEDGE_KINDS
        placeholders = ",".join("?" for _ in selected)
        index = self.package.root / "index.sqlite"
        connection = sqlite3.connect(f"file:{index.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
            if "knowledge" not in tables:
                return [{"id": f"rule:{row['id']}", "kind": "rule", "ref_id": row["id"], "action_type": row["action_type"],
                         "snippet": row["description"], "text": row["description"]} for row in
                        [dict(item) for item in self.search(query, limit)]]
            expression = fts_query(query)
            rows: list[sqlite3.Row] = []
            if "knowledge_fts" in tables and expression:
                try:
                    rows = connection.execute(
                        "SELECT k.kind, k.ref_id, k.action_type, k.text, "
                        "snippet(knowledge_fts, 3, '[', ']', '…', 24) AS snippet "
                        f"FROM knowledge_fts f JOIN knowledge k ON k.id = f.rowid "
                        f"WHERE knowledge_fts MATCH ? AND k.kind IN ({placeholders}) ORDER BY rank LIMIT ?",
                        (expression, *selected, limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            if not rows:
                token = f"%{query.strip()}%"
                rows = connection.execute(
                    f"SELECT kind, ref_id, action_type, text, substr(text, 1, 200) AS snippet FROM knowledge "
                    f"WHERE text LIKE ? AND kind IN ({placeholders}) LIMIT ?",
                    (token, *selected, limit),
                ).fetchall()
            return [
                {"id": f"{row['kind']}:{row['ref_id']}", "kind": row["kind"], "ref_id": row["ref_id"],
                 "action_type": row["action_type"], "snippet": row["snippet"], "text": row["text"]}
                for row in rows
            ]
        finally:
            connection.close()

    def summary(self) -> dict[str, Any]:
        return {
            "manifest": self.package.manifest.model_dump(mode="json"),
            "actions": self.list_actions(),
            "state_field_count": len(self.package.state_schema.fields),
            "rule_count": len(self.package.rules),
            "invariant_count": len(self.package.invariants),
            "renderer_count": len(self.package.renderers),
            "note_count": len(self.package.notes),
        }
