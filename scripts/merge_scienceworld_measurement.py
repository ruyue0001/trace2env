"""Merge disjoint ScienceWorld Measurement evaluation slices with coverage checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_scienceworld_measurement import WORK, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("real", "prompt", "trace2env", "replay"), required=True)
    parser.add_argument("--parts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(40, 53), default=53)
    args = parser.parse_args()
    test = json.loads((WORK / "sciworld_test.json").read_text(encoding="utf-8"))
    measurement = json.loads((WORK / "measurement-30-gold-v1/selection.json").read_text(encoding="utf-8"))
    expected = measurement["evaluation_ids"][:args.limit]
    if len(expected) != args.limit:
        raise ValueError("Measurement evaluation selection is not 53 tasks")
    records: dict[int, dict] = {}
    settings = None
    for part in args.parts:
        selection = json.loads((part / "selection.json").read_text(encoding="utf-8"))
        if selection["mode"] != args.mode:
            raise ValueError(f"Mode mismatch in {part}")
        common = {k: v for k, v in selection.items() if k != "task_ids"}
        if settings is None:
            settings = common
        elif settings != common:
            raise ValueError(f"Evaluation settings differ in {part}")
        for item_id in selection["task_ids"]:
            if item_id in records:
                raise ValueError(f"Duplicate task ID {item_id}")
            file = part / "attempts" / f"sciworld_{item_id}.json"
            if file.is_file():
                record = json.loads(file.read_text(encoding="utf-8"))
                if record["item_id"] != item_id:
                    raise ValueError(f"Record ID mismatch in {file}")
                records[item_id] = record
    if set(records) != set(expected):
        raise ValueError(f"Task coverage mismatch: missing={set(expected)-set(records)}, extra={set(records)-set(expected)}")
    if len(test) != 200:
        raise ValueError("Published ScienceWorld test index changed")
    write_json(args.output / "selection.json", {**settings, "task_ids": expected,
                                                 "parts": [str(p) for p in args.parts]})
    for item_id in expected:
        write_json(args.output / "attempts" / f"sciworld_{item_id}.json", records[item_id])
    success_key = "wm_success" if args.mode in {"prompt", "trace2env"} else "success"
    errors = sum("error" in r["status"] or r["status"] == "source_error" for r in records.values())
    successes = sum(bool(r.get(success_key)) for r in records.values())
    summary = {"mode": args.mode, "tasks": args.limit, "finished": args.limit,
               "complete": errors == 0, "errors": errors,
               "successes": successes, "success_rate": successes / args.limit if errors == 0 else None,
               "observed_success_fraction": successes / args.limit,
               "total_turns": sum(r.get("steps", 0) for r in records.values())}
    write_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
