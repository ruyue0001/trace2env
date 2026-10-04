"""Add a simulator-observed wait probe to a fresh ScienceWorld package.

This is a diagnostic package extension, not an automatic reconstruction of
all thermal dynamics. The probe uses construction variation 660 only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from collect_scienceworld_measurement import WORK, configure_java, normalize_action, write_json
from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.compiler import EnvironmentCompiler
from trace2env.models import (
    ActionSpec, ArgumentSpec, Confidence, Demonstration, EnvironmentNote,
    LocalTransitionEvidence, Outcome, ReconstructionArtifacts, ReconstructionConfig,
    SourceRef, SplitAssignment,
)
from trace2env.validation import validate_artifacts


def collect_probe(path: Path) -> None:
    from scienceworld import ScienceWorldEnv

    source = json.loads((WORK / "measurement-30-gold-v1/traces/sciworld_660.json").read_text())
    commands = [event["content"]["arguments"]["command"] for event in source["events"]
                if event["actor"] == "agent"][:18]
    configure_java()
    env = ScienceWorldEnv()
    try:
        env.load("measure-melting-point-known-substance", 0)
        observation, _, done, _ = env.step("look around")
        if done:
            raise RuntimeError("Probe ended during initial look")
        events = [{"actor": "environment", "kind": "observation", "content":
                   env.get_task_description() + "\n" + observation}]
        audit = []
        for command in [*commands, "wait", "use thermometer in inventory on chocolate",
                        "wait1", "use thermometer in inventory on chocolate"]:
            before = env.get_num_moves()
            observation, _, done, info = env.step(command)
            events.extend((
                {"actor": "agent", "kind": "action", "content": normalize_action(command)},
                {"actor": "environment", "kind": "observation", "content": observation},
            ))
            audit.append({"action": command, "moves_before": before,
                          "moves_after": env.get_num_moves(), "score": info["score"], "done": done})
            if done:
                raise RuntimeError(f"Probe ended early after {command!r}")
        write_json(path, {"episode_id": "sciworld_660_wait_probe", "events": events})
        write_json(path.with_suffix(".audit.json"), {"construction_item_id": 660, "turns": audit})
    finally:
        env.close()


def build_package(source: Path, probe: Path, output: Path) -> None:
    artifacts = ReconstructionArtifacts.model_validate(json.loads(
        (source / "construction/artifacts.json").read_text(encoding="utf-8")))
    config = ReconstructionConfig.model_validate(json.loads(
        (source / "construction/config.json").read_text(encoding="utf-8")))
    episode = load_raw_traces(probe)[0]
    episode.metadata["trajectory_group"] = "sciworld_660"
    if episode.source_id in artifacts.source_snapshots:
        raise ValueError("Probe source already present")
    transitions = segment_transitions(episode)
    by_type = {}
    for transition in transitions:
        action_event = next(event for event in episode.events if event.id == transition.action_event_ids[0])
        command = action_event.content["arguments"]["command"]
        if command in {"wait", "wait1"}:
            observation_event = next(event for event in episode.events
                                     if event.id == transition.observation_event_ids[0])
            ref = SourceRef(source_id=episode.source_id, episode_id=episode.id,
                            event_ids=[action_event.id, observation_event.id],
                            locator="observed action and immediate simulator response",
                            excerpt=f"{command} / {observation_event.content}")
            evidence = LocalTransitionEvidence(
                id=f"evidence_{transition.id}", episode_id=episode.id,
                transition_id=transition.id, action=normalize_action(command),
                outcome=Outcome.SUCCESS, observation_text=observation_event.content,
                confidence=Confidence(value=1.0, rationale="Direct simulator response."),
                provenance=[ref],
            )
            by_type[command] = (evidence, ref)
            artifacts.evidence.append(evidence)
            artifacts.action_schema.actions.append(ActionSpec(
                name=command,
                description=("Advance ten simulation iterations; active physical processes continue."
                             if command == "wait" else
                             "Advance one simulation iteration; active physical processes continue."),
                arguments={"command": ArgumentSpec(type="string", description="Literal command")},
                provenance=[ref],
            ))
            artifacts.demonstrations.append(Demonstration(
                id=f"demo_{transition.id}", action=normalize_action(command),
                observation=observation_event.content, outcome=Outcome.SUCCESS,
                provenance=[ref],
            ))
            artifacts.notes.append(EnvironmentNote(
                id=f"sciworld_{command}_iterations", kind="convention",
                statement=f"The simulator response to `{command}` is: {observation_event.content} "
                          "Time-dependent processes continue while the agent waits.",
                action_types=[command], confidence=1.0, status="supported", provenance=[ref],
            ))
    if set(by_type) != {"wait", "wait1"}:
        raise ValueError("Probe lacks wait and wait1 observations")
    thermometer = [(event.id, event.content) for event in episode.events
                   if event.actor.value == "environment" and isinstance(event.content, str)
                   and "thermometer measures a temperature of" in event.content]
    if len(thermometer) < 3:
        raise ValueError("Probe needs thermometer readings before and after both waits")
    readings = [int(re.search(r"temperature of (\d+) degrees", text).group(1))
                for _, text in thermometer[-3:]]
    heat_ref = SourceRef(source_id=episode.source_id, episode_id=episode.id,
                         event_ids=[event_id for event_id, _ in thermometer[-3:]],
                         locator="thermometer readings bracketing wait and wait1",
                         excerpt=f"Observed chocolate temperatures: {readings} °C")
    artifacts.notes.append(EnvironmentNote(
        id="sciworld_wait_heating_probe", kind="fact",
        statement=(f"In construction variation 660, with chocolate on an active stove, "
                   f"thermometer readings around `wait` and `wait1` were {readings} °C. "
                   "A wait can change a heated object's temperature substantially; this "
                   "example does not establish a universal heating rate."),
        action_types=["wait", "wait1", "use"], confidence=1.0,
        status="supported", provenance=[heat_ref],
    ))
    artifacts.episodes.append(episode)
    artifacts.transitions.extend(transitions)
    artifacts.source_snapshots[episode.source_id] = probe.read_text(encoding="utf-8")
    artifacts.split_assignments[episode.source_id] = SplitAssignment(
        split="train", trajectory_group="sciworld_660")
    validate_artifacts(artifacts)
    EnvironmentCompiler(config).compile(artifacts, output)
    write_json(output.with_suffix(".supplement.json"), {
        "source_package": str(source), "probe": str(probe),
        "probe_item_id": 660, "wait_readings_celsius": readings,
        "scope": "diagnostic package extension; no general thermal transition rule",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "build"))
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "collect":
        collect_probe(args.probe)
    else:
        if not args.source or not args.output:
            parser.error("build requires --source and --output")
        build_package(args.source, args.probe, args.output)


if __name__ == "__main__":
    main()
