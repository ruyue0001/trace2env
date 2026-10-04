#!/usr/bin/env python3
"""Select EnvScaler evaluation rows by current action and shard by trajectory."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_RE = re.compile(r"\*\*Action:\*\*\s*```json\s*(\{.*?\})\s*```", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def current_action(row: dict[str, Any]) -> dict[str, Any]:
    match = ACTION_RE.search(str(row.get("current_prompt") or ""))
    if not match:
        raise ValueError(f"row has no current JSON action: {row.get('id')} turn {row.get('turn_idx')}")
    return json.loads(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actions", nargs="+", required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()

    wanted = set(args.actions)
    source = read_jsonl(args.source)
    selected = [row for row in source if current_action(row).get("name") in wanted]
    counts = Counter(str(current_action(row).get("name")) for row in selected)
    missing = sorted(wanted - counts.keys())
    if missing:
        raise ValueError(f"requested actions have no rows: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8"
    )
    trajectory_ids = sorted({str(row["id"]) for row in selected})
    shards = args.output / "shards"
    shards.mkdir(exist_ok=True)
    for index in range(args.shards):
        mine = set(trajectory_ids[index :: args.shards])
        rows = [row for row in selected if str(row["id"]) in mine]
        (shards / f"shard{index}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    systems = {str(row.get("system_str") or "") for row in selected}
    if len(systems) != 1:
        raise ValueError(f"selected rows use {len(systems)} different system prompts")
    system = next(iter(systems))
    manifest = {
        "source": str(args.source),
        "rows": len(selected),
        "trajectories": len(trajectory_ids),
        "actions": dict(sorted(counts.items())),
        "shards": args.shards,
        "prompt_checks": {
            "initial_database_absent": "**Current State:**" not in system and "# Current State" not in system,
            "returns_and_constraints_absent": not bool(
                re.search(r"(?m)^\s*(Returns|Constraints):\s*$", system)
            ),
            "few_shot_examples": system.count("## Example: "),
        },
    }
    if not all(manifest["prompt_checks"].values()):
        raise ValueError(f"prompt check failed: {manifest['prompt_checks']}")
    (args.output / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
