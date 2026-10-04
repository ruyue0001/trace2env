"""Compile reconstructed artifacts into a versioned, inspectable environment package."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from trace2env.models import EnvironmentManifest, ReconstructionArtifacts, ReconstructionConfig, ReplayReport
from trace2env.eligibility import construction_scope, rule_eligible
from trace2env.validation import artifact_digest, config_digest, validate_artifacts
from trace2env.storage import atomic_write_text, write_json, write_jsonl


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return result or "unknown"


class EnvironmentCompiler:
    def __init__(self, config: ReconstructionConfig):
        self.config = config

    def compile(self, artifacts: ReconstructionArtifacts, output_dir: str | Path,
                validation_report: ReplayReport | None = None) -> Path:
        destination = Path(output_dir).resolve()
        if destination.exists():
            raise FileExistsError(f"Package destination already exists; choose a new version: {destination}")
        validate_artifacts(artifacts, require_provenance=self.config.construction_kind == "reconstructed")
        scope = construction_scope(self.config)
        digest = artifact_digest(artifacts)
        if validation_report is not None and (not validation_report.promotion_eligible
                or validation_report.artifact_digest != digest or not validation_report.cases
                or validation_report.config_digest != config_digest(self.config)
                or not all(case.passed for case in validation_report.cases) or validation_report.regressions
                or validation_report.uncovered_rule_ids):
            raise ValueError("Package promotion requires a passing report for these exact artifacts")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A fresh sibling is never visible at the destination until all writes succeed.
        # Failed staging directories remain available for diagnosis; existing packages are untouched.
        root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
        original = artifacts
        executable = [r for r in artifacts.rules if rule_eligible(r, scope, allow_state_unknown=True,
                      minimum_confidence=self.config.min_rule_confidence)]
        artifacts = artifacts.model_copy(update={"rules": executable})
        write_json(root / "construction" / "artifacts.json", original)
        write_json(root / "construction" / "config.json", self.config)
        write_jsonl(root / "candidates" / "rules.jsonl", original.rules)
        write_jsonl(root / "evidence" / "episodes.jsonl", original.episodes)
        write_jsonl(root / "evidence" / "transitions.jsonl", original.transitions)
        write_json(root / "evidence" / "sources.json", original.source_snapshots)
        if validation_report is not None:
            write_json(root / "validation" / "replay.json", validation_report)
        overview = (
            f"# {self.config.name}\n\n"
            f"{self.config.description or 'Environment recovered from execution traces.'}\n\n"
            "This package is an evidence-backed, read-only world-model knowledge base. "
            "Runtime session state is stored outside this directory.\n\n"
            f"- Environment ID: `{self.config.environment_id}`\n"
            f"- Domains: {', '.join(self.config.domains) or 'unspecified'}\n"
            f"- Source episodes: {len(artifacts.episodes)}\n"
            f"- Transition evidence records: {len(artifacts.evidence)}\n"
            f"- Supported rules: {len(artifacts.rules)}\n"
        )
        atomic_write_text(root / "overview.md", overview)
        write_json(root / "action_schema.json", artifacts.action_schema)
        write_json(root / "state_schema.json", artifacts.state_schema)
        write_json(root / "invariants.json", artifacts.invariants)
        write_jsonl(root / "rules" / "index.jsonl", artifacts.rules)
        write_jsonl(root / "evidence" / "local_transitions.jsonl", artifacts.evidence)
        write_jsonl(root / "demonstrations" / "transitions.jsonl", artifacts.demonstrations)
        write_jsonl(root / "knowledge" / "notes.jsonl", artifacts.notes)
        write_json(root / "renderer" / "contracts.json", artifacts.renderers)
        write_json(root / "exceptions" / "unresolved.json", artifacts.unresolved)
        write_json(root / "exceptions" / "rejected.json", original.rejected)

        grouped: dict[str, list[object]] = defaultdict(list)
        for rule in artifacts.rules:
            grouped[rule.action_type].append(rule)
        for action_type, rules in grouped.items():
            write_jsonl(root / "rules" / "by_action" / f"{_slug(action_type)}.jsonl", rules)

        self._build_index(root, artifacts)
        file_hashes = {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = EnvironmentManifest(
            schema_version="1.1",
            environment_id=self.config.environment_id,
            name=self.config.name,
            description=self.config.description,
            domains=self.config.domains,
            source_count=len(artifacts.episodes),
            transition_count=len(artifacts.transitions),
            files=file_hashes,
            metadata={
                "compiler": "trace2env",
                "reconstruction_model": self.config.model,
                "tenant_scope": self.config.tenant_scope,
                "unresolved_count": len(artifacts.unresolved),
                "scope_context": scope,
                "min_rule_confidence": self.config.min_rule_confidence,
                "construction_kind": self.config.construction_kind,
                "artifact_digest": digest,
                "config_digest": config_digest(self.config),
                "validation_status": "validated" if validation_report else (
                    "authored" if self.config.construction_kind == "authored" else "candidate"),
                **({"ablation": self.config.ablation} if self.config.ablation else {}),
            },
        )
        write_json(root / "manifest.json", manifest)
        if destination.exists():
            raise FileExistsError(f"Package destination appeared during compilation: {destination}")
        root.rename(destination)
        return destination

    @staticmethod
    def _build_index(root: Path, artifacts: ReconstructionArtifacts) -> None:
        index_path = root / "index.sqlite"
        if index_path.exists():
            index_path.unlink()
        connection = sqlite3.connect(index_path)
        try:
            connection.executescript(
                """
                CREATE TABLE rules (
                    id TEXT PRIMARY KEY, action_type TEXT NOT NULL, priority INTEGER NOT NULL,
                    confidence REAL NOT NULL, status TEXT NOT NULL, description TEXT NOT NULL,
                    artifact_path TEXT NOT NULL
                );
                CREATE INDEX rules_action_idx ON rules(action_type, priority DESC);
                CREATE TABLE demonstrations (
                    id TEXT PRIMARY KEY, action_type TEXT NOT NULL, observation TEXT NOT NULL,
                    artifact_path TEXT NOT NULL
                );
                CREATE INDEX demo_action_idx ON demonstrations(action_type);
                CREATE TABLE state_fields (
                    path TEXT PRIMARY KEY, type TEXT NOT NULL, description TEXT NOT NULL,
                    visibility TEXT NOT NULL
                );
                """
            )
            connection.executemany(
                "INSERT INTO rules VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        rule.id,
                        rule.action_type,
                        rule.priority,
                        rule.confidence,
                        rule.status,
                        rule.description,
                        f"rules/by_action/{_slug(rule.action_type)}.jsonl",
                    )
                    for rule in artifacts.rules
                ],
            )
            connection.executemany(
                "INSERT INTO demonstrations VALUES (?, ?, ?, ?)",
                [
                    (demo.id, demo.action.type, demo.observation, "demonstrations/transitions.jsonl")
                    for demo in artifacts.demonstrations
                ],
            )
            connection.executemany(
                "INSERT INTO state_fields VALUES (?, ?, ?, ?)",
                [(field.path, field.type, field.description, field.visibility) for field in artifacts.state_schema.fields],
            )
            # Knowledge catalog for the agent's full-text discovery: one row per artifact, FTS5 when available.
            connection.executescript(
                """
                CREATE TABLE knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, ref_id TEXT NOT NULL,
                    action_type TEXT NOT NULL, text TEXT NOT NULL, artifact_path TEXT NOT NULL
                );
                CREATE INDEX knowledge_kind_idx ON knowledge(kind, action_type);
                """
            )
            def action_text(action) -> str:
                return f"{action.type} " + json.dumps(action.arguments, ensure_ascii=False, default=str)
            rows = [
                ("action", spec.name, spec.name, f"{spec.name}: {spec.description} aliases: {', '.join(spec.aliases)}",
                 "action_schema.json")
                for spec in artifacts.action_schema.actions
            ]
            rows += [
                ("rule", rule.id, rule.action_type,
                 # Candidate rules (tentative/conflicted) stay searchable as hints; only supported ones execute.
                 f"[{rule.status}, confidence {rule.confidence:.2f}] {rule.description}\noutcome: {rule.outcome.value}; conditions: "
                 + json.dumps([c.model_dump(mode='json', exclude_none=True) for c in rule.conditions], ensure_ascii=False),
                 f"rules/by_action/{_slug(rule.action_type)}.jsonl")
                for rule in artifacts.rules
            ]
            rows += [("note", note.id, ",".join(note.action_types), f"{note.kind}: {note.statement}", "knowledge/notes.jsonl")
                     for note in artifacts.notes if note.status == "supported"]
            rows += [("demonstration", demo.id, demo.action.type, f"{action_text(demo.action)}\n{demo.observation}",
                      "demonstrations/transitions.jsonl") for demo in artifacts.demonstrations]
            rows += [("evidence", item.id, item.action.type, f"{action_text(item.action)}\n{item.observation_text}",
                      "evidence/local_transitions.jsonl") for item in artifacts.evidence]
            connection.executemany("INSERT INTO knowledge(kind, ref_id, action_type, text, artifact_path) VALUES (?, ?, ?, ?, ?)", rows)
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE knowledge_fts USING fts5(kind UNINDEXED, ref_id UNINDEXED, action_type, text, "
                    "content='knowledge', content_rowid='id')"
                )
                connection.execute("INSERT INTO knowledge_fts(knowledge_fts) VALUES ('rebuild')")
            except sqlite3.OperationalError:
                pass  # FTS5 unavailable: search_knowledge falls back to LIKE over the catalog
            connection.commit()
        finally:
            connection.close()
