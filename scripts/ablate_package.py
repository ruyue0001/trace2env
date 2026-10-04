#!/usr/bin/env python3
"""Derive an ablated package from a reconstructed one: keep the schemas, remove the reconstructed knowledge.

    python scripts/ablate_package.py SRC_PACKAGE --output DST_PACKAGE [--keep schemas]

``--keep schemas`` (the only mode so far) keeps the action schema and the state schema — what the
harness needs to validate actions, track state, and brief the agent — and removes rules, invariants,
renderer contracts, demonstrations, notes, extracted evidence, and the trace copies. The result
compiles through the ordinary compiler with ``construction_kind="ablated"`` and records its
provenance (source package, source manifest hash, kept and removed parts) in the manifest, so an
evaluation run can be traced back to exactly this derivation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trace2env.compiler import EnvironmentCompiler  # noqa: E402
from trace2env.models import ReconstructionArtifacts, ReconstructionConfig  # noqa: E402
from trace2env.package import EnvironmentPackage  # noqa: E402
from trace2env.storage import read_json  # noqa: E402

KEEP_MODES = {
    # Nothing reconstructed: the interface only (the agentic_schema_only control).
    "schemas": {
        "kept": ["action_schema", "state_schema"],
        "removed": ["rules", "invariants", "renderers", "demonstrations", "notes", "evidence",
                    "episodes", "transitions", "source_snapshots", "split_assignments", "unresolved", "rejected"],
    },
    # The example tier: raw turns (evidence) and demonstrations, with the episodes they cite; no induced structure.
    "examples": {
        "kept": ["action_schema", "state_schema", "evidence", "demonstrations", "episodes", "transitions", "source_snapshots"],
        "removed": ["rules", "invariants", "renderers", "notes", "split_assignments", "unresolved", "rejected"],
    },
    # The structure tier: rules, invariants, observation contracts, notes; no examples or raw turns.
    "structure": {
        "kept": ["action_schema", "state_schema", "rules", "invariants", "renderers", "notes"],
        "removed": ["evidence", "demonstrations", "episodes", "transitions", "source_snapshots", "split_assignments", "unresolved", "rejected"],
    },
}


def ablate(source: Path, output: Path, keep: str = "schemas") -> Path:
    package = EnvironmentPackage(source)  # verifies the source's hashes first
    artifacts = ReconstructionArtifacts.model_validate(read_json(source / "construction" / "artifacts.json"))
    config = ReconstructionConfig.model_validate(read_json(source / "construction" / "config.json"))
    mode = KEEP_MODES[keep]

    keep_traces = "episodes" in mode["kept"]

    def without_provenance(model):
        # Provenance points into trace material that this derivation removes; the artifact itself stays intact.
        if keep_traces or not hasattr(model, "provenance"):
            return model
        update = {"provenance": []}
        if hasattr(model, "counterexamples"):
            update["counterexamples"] = []
        return model.model_copy(update=update)

    def kept(name):
        items = [without_provenance(item) for item in getattr(artifacts, name)] if name in mode["kept"] else []
        if name == "demonstrations" and "rules" not in mode["kept"]:
            items = [item.model_copy(update={"rule_ids": []}) for item in items]  # the rules they pointed at are gone
        return items

    action_schema = artifacts.action_schema.model_copy(update={
        "actions": [without_provenance(spec) for spec in artifacts.action_schema.actions]})
    state_schema = artifacts.state_schema.model_copy(update={
        "fields": [without_provenance(field) for field in artifacts.state_schema.fields]})
    stripped = ReconstructionArtifacts(
        episodes=kept("episodes"), transitions=kept("transitions"), evidence=kept("evidence"),
        action_schema=action_schema, state_schema=state_schema,
        rules=kept("rules"), invariants=kept("invariants"), renderers=kept("renderers"),
        demonstrations=kept("demonstrations"), notes=kept("notes"),
        source_snapshots=dict(artifacts.source_snapshots) if "source_snapshots" in mode["kept"] else {},
        unresolved=[f"ablated: {name} removed" for name in mode["removed"]],
    )
    manifest_hash = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    ablated_config = config.model_copy(update={
        "construction_kind": "ablated",
        "ablation": {
            "mode": keep, "source_package": str(source.resolve()), "source_environment_id": package.manifest.environment_id,
            "source_manifest_sha256": manifest_hash, "kept": mode["kept"], "removed": mode["removed"],
            "source_counts": {"rules": len(artifacts.rules), "invariants": len(artifacts.invariants),
                              "renderers": len(artifacts.renderers), "demonstrations": len(artifacts.demonstrations),
                              "notes": len(artifacts.notes), "evidence": len(artifacts.evidence), "episodes": len(artifacts.episodes)},
        },
    })
    return EnvironmentCompiler(ablated_config).compile(stripped, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="A compiled reconstructed package directory")
    parser.add_argument("--output", required=True, help="Destination package directory (must not exist)")
    parser.add_argument("--keep", choices=sorted(KEEP_MODES), default="schemas")
    args = parser.parse_args(argv)
    destination = ablate(Path(args.source), Path(args.output), args.keep)
    result = EnvironmentPackage(destination)
    print(json.dumps({
        "package": str(destination), "construction_kind": result.manifest.metadata.get("construction_kind"),
        "ablation": result.manifest.metadata.get("ablation"),
        "actions": len(result.action_schema.actions), "state_fields": len(result.state_schema.fields),
        "rules": len(result.rules), "renderers": len(result.renderers), "demonstrations": len(result.demonstrations),
        "notes": len(result.notes), "invariants": len(result.invariants),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
