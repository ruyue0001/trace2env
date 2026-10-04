#!/usr/bin/env python3
"""Prepare the curated EnvScaler turn set in the repository's AWB row format.

The source rows come from ``envscaler-export``.  The curated inputs/targets are
independently generated selections under ``work/envscaler/<env>``.  This script
joins them by task/turn, verifies every visible action and observation against
the curated copy, retains a deterministic number of construction-derived
few-shot examples, and writes trajectory-preserving shards for ``run_eval.sh``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"\*\*Action:\*\*\s*```json\s*(\{.*?\})\s*```", re.DOTALL)
OBSERVATION_PREFIX = "**Environment Observation:**\n"
FEW_SHOT_MARKER = "# Few-shot Examples"
EXAMPLE_RE = re.compile(r"(?ms)^## Example: .*?(?=^## Example: |\Z)")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def action_from_prompt(text: str) -> dict[str, Any]:
    match = ACTION_RE.search(text)
    if not match:
        raise ValueError("prompt has no JSON action block")
    return json.loads(match.group(1))


def observation_from_response(text: str) -> str:
    if not text.startswith(OBSERVATION_PREFIX):
        raise ValueError("response lacks Environment Observation marker")
    return text[len(OBSERVATION_PREFIX) :]


def retain_examples(system: str, count: int) -> tuple[str, list[str], int]:
    if FEW_SHOT_MARKER not in system:
        raise ValueError("source system prompt has no few-shot section")
    prefix, suffix = system.split(FEW_SHOT_MARKER, 1)
    header, separator, examples_text = suffix.partition("\n\nBelow are examples of interactions for the tools available in this environment:\n\n")
    if not separator or header.strip():
        raise ValueError("unrecognized few-shot section header")
    examples = [match.group(0).rstrip() for match in EXAMPLE_RE.finditer(examples_text)]
    if count < 1 or count > len(examples):
        raise ValueError(f"requested {count} examples from {len(examples)} available")
    if count == 1:
        indices = [0]
    else:
        indices = [round(index * (len(examples) - 1) / (count - 1)) for index in range(count)]
    kept = [examples[index] for index in indices]
    names = [block.splitlines()[0].removeprefix("## Example: ") for block in kept]
    rebuilt = (
        prefix
        + FEW_SHOT_MARKER
        + separator
        + "\n\n".join(kept)
        + "\n"
    )
    return rebuilt, names, len(examples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--source-rows", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--few-shot", type=int, default=3)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    inputs = read_jsonl(args.inputs)
    targets = {row["id"]: row["observation"] for row in read_jsonl(args.targets)}
    source_rows = read_jsonl(args.source_rows)
    source_index = {(row["id"], int(row["turn_idx"])): row for row in source_rows}
    if len(targets) != len(inputs):
        raise ValueError(f"input/target count mismatch: {len(inputs)} != {len(targets)}")

    selected: list[dict[str, Any]] = []
    example_names: list[str] | None = None
    source_example_count: int | None = None
    source_system_chars: int | None = None
    output_system_chars: int | None = None
    for item in inputs:
        env_id, task_id, turn_text = item["id"].split("/")
        if env_id != args.env:
            raise ValueError(f"unexpected environment in {item['id']}")
        row = source_index.get((f"envscaler-{task_id}", int(turn_text)))
        if row is None:
            raise ValueError(f"curated turn not found in exported rows: {item['id']}")

        visible = item["input"]
        if action_from_prompt(row["current_prompt"]) != visible["action"]:
            raise ValueError(f"current action mismatch: {item['id']}")
        history = visible["history"]
        if len(history) != int(turn_text) - 1:
            raise ValueError(f"curated history length mismatch: {item['id']}")
        for index, expected in enumerate(history):
            if action_from_prompt(row["prompt"][index]) != expected["action"]:
                raise ValueError(f"history action mismatch: {item['id']} turn {index + 1}")
            if observation_from_response(row["response"][index]) != expected["observation"]:
                raise ValueError(f"history observation mismatch: {item['id']} turn {index + 1}")
        if observation_from_response(row["response"][-1]) != targets[item["id"]]:
            raise ValueError(f"target mismatch: {item['id']}")

        system = row["system_str"]
        if "**Current State:**" in system or "# Current State" in system:
            raise ValueError(f"initial state found in prompt: {item['id']}")
        if re.search(r"(?m)^\s*(Returns|Constraints):\s*$", system):
            raise ValueError(f"answer-bearing description section remains: {item['id']}")
        reduced, names, available = retain_examples(system, args.few_shot)
        if example_names is None:
            example_names = names
            source_example_count = available
            source_system_chars = len(system)
            output_system_chars = len(reduced)
        elif names != example_names or available != source_example_count or reduced != selected[0]["system_str"]:
            raise ValueError("system prompt differs between source rows")
        copied = dict(row)
        copied["system_str"] = reduced
        selected.append(copied)

    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    trajectory_ids = sorted({row["id"] for row in selected})
    shards_dir = args.output / "shards"
    shards_dir.mkdir(exist_ok=True)
    for shard in range(args.shards):
        mine = set(trajectory_ids[shard :: args.shards])
        rows = [row for row in selected if row["id"] in mine]
        (shards_dir / f"shard{shard}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    manifest = {
        "environment": args.env,
        "source_rows": str(args.source_rows),
        "curated_inputs": str(args.inputs),
        "curated_targets": str(args.targets),
        "rows": len(selected),
        "trajectories": len(trajectory_ids),
        "few_shot": {
            "source_examples": source_example_count,
            "retained_examples": args.few_shot,
            "selection": "evenly_spaced_tool_definition_order",
            "tools": example_names,
        },
        "prompt": {
            "initial_database": False,
            "returns_and_constraints_sections": False,
            "source_system_chars": source_system_chars,
            "system_chars": output_system_chars,
        },
        "shards": args.shards,
    }
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
