"""Remove two incomplete deterministic renderers from the Measurement candidate.

The induced go_to and focus_on rules cite state-dependent templates but do not
write the required state paths. Keep their evidence and demonstrations in the
new immutable package; the runtime agent handles those transitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace2env.compiler import EnvironmentCompiler
from trace2env.models import ArtifactEdit, PackagePatch, ReconstructionArtifacts, ReconstructionConfig
from trace2env.patching import apply_package_patch
from trace2env.validation import artifact_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    artifacts = ReconstructionArtifacts.model_validate(json.loads(
        (args.source / "construction/artifacts.json").read_text(encoding="utf-8")))
    config = ReconstructionConfig.model_validate(json.loads(
        (args.source / "construction/config.json").read_text(encoding="utf-8")))
    rules = {rule.id: rule for rule in artifacts.rules}
    renderers = {renderer.id: renderer for renderer in artifacts.renderers}
    demote = lambda key: rules[key].model_copy(update={"status": "tentative"}).model_dump(mode="json")
    no_template = lambda key: renderers[key].model_copy(update={"template": None, "required_fields": []}).model_dump(mode="json")
    patch = PackagePatch(
        base_artifact_digest=artifact_digest(artifacts),
        rationale=("The go_to and focus_on rules have no effects, but their exact "
                   "renderers require session.location and session.focused_object. "
                   "These fields cannot be derived by those rules from command-only arguments. "
                   "Demote the incomplete executable rules and disable their exact templates; retain all evidence."),
        failed_case_ids=["sciworld_624_turn_2_missing_render_field"],
        edits=[
            ArtifactEdit(collection="rules", key="go_to_reachable_destination_success", operation="replace",
                         value=demote("go_to_reachable_destination_success")),
            ArtifactEdit(collection="rules", key="focus_on_resolved_target_success", operation="replace",
                         value=demote("focus_on_resolved_target_success")),
            ArtifactEdit(collection="renderers", key="go_to.movement", operation="replace",
                         value=no_template("go_to.movement")),
            ArtifactEdit(collection="renderers", key="focus.confirmation", operation="replace",
                         value=no_template("focus.confirmation")),
        ],
    )
    repaired = apply_package_patch(artifacts, patch)
    EnvironmentCompiler(config).compile(repaired, args.output)
    args.output.with_suffix(".repair.json").write_text(
        patch.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
