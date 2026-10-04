"""Apply explicit artifact edits as one candidate; promotion still requires replay."""
from __future__ import annotations

from trace2env.models import (ActionSpec, StateField, TransitionRule, Invariant, RenderContract,
                             Demonstration, EnvironmentNote, ReconstructionArtifacts, PackagePatch)
from trace2env.validation import artifact_digest


def apply_package_patch(artifacts: ReconstructionArtifacts, patch: PackagePatch) -> ReconstructionArtifacts:
    if artifact_digest(artifacts) != patch.base_artifact_digest:
        raise ValueError("Patch base fingerprint is stale")
    result = artifacts.model_copy(deep=True)
    targets = {
        "rules": (result.rules, TransitionRule, "id"),
        "invariants": (result.invariants, Invariant, "id"),
        "renderers": (result.renderers, RenderContract, "id"),
        "demonstrations": (result.demonstrations, Demonstration, "id"),
        "notes": (result.notes, EnvironmentNote, "id"),
        "actions": (result.action_schema.actions, ActionSpec, "name"),
        "state_fields": (result.state_schema.fields, StateField, "path"),
    }
    seen = set()
    for edit in patch.edits:
        identity = (edit.collection, edit.key)
        if identity in seen:
            raise ValueError("Patch contains conflicting edits to the same artifact")
        seen.add(identity)
        items, model, key = targets[edit.collection]
        index = next((i for i, item in enumerate(items) if getattr(item, key) == edit.key), None)
        if (edit.operation == "add") == (index is not None):
            raise ValueError("Patch add target exists or replace/remove target is missing")
        if edit.operation == "remove":
            if edit.value is not None:
                raise ValueError("Remove edits must not supply a value")
            items.pop(index)
        else:
            value = model.model_validate(edit.value)
            if getattr(value, key) != edit.key:
                raise ValueError("Patch value identity differs from its target")
            if index is None:
                items.append(value)
            else:
                items[index] = value
    return ReconstructionArtifacts.model_validate(result.model_dump(mode="python"))
