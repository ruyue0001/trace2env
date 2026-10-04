"""Combine disjoint ALFWorld Trace2Env evaluation slices for scoring and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from run_alfworld_prompting_wm import write_json


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True,
                        help="Disjoint completed evaluation slice directories")
    parser.add_argument("--retry", type=Path, action="append", default=[],
                        help="Fresh one-task retry directory replacing an errored record")
    parser.add_argument("--allow-errors", action="store_true",
                        help="Keep unresolved harness errors as task failures in the 100-task result")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = [(path, read_json(path / "selection.json")) for path in args.inputs]
    shards.sort(key=lambda item: item[1]["start_index"])
    common_keys = set(shards[0][1]) - {"task_ids", "start_index"}
    common = {key: shards[0][1][key] for key in common_keys}
    task_ids: list[int] = []
    sources: dict[int, Path] = {}
    source_selections: list[dict] = []
    next_index = 0
    for path, selection in shards:
        if {key: selection[key] for key in common_keys} != common:
            raise ValueError(f"Evaluation settings differ in {path}")
        ids = selection["task_ids"]
        if selection["start_index"] != next_index or ids != list(range(2420 + next_index, 2420 + next_index + len(ids))):
            raise ValueError(f"Noncontiguous or unexpected task IDs in {path}")
        for item_id in ids:
            record_path = path / "attempts" / f"alfworld_{item_id}.json"
            if not record_path.is_file():
                raise ValueError(f"Missing completed attempt: {record_path}")
            sources[item_id] = record_path
        task_ids.extend(ids)
        next_index += len(ids)
        source_selections.append({"path": str(path), "sha256": sha256(path / "selection.json")})
    if task_ids != list(range(2420, 2520)):
        raise ValueError("Expected exactly the 100 held-out ALFWorld IDs")
    first_attempts = [read_json(sources[item_id]) for item_id in task_ids]

    retried: list[int] = []
    for path in args.retry:
        selection = read_json(path / "selection.json")
        if {key: selection[key] for key in common_keys} != common or len(selection["task_ids"]) != 1:
            raise ValueError(f"Retry settings or task selection differ in {path}")
        item_id = selection["task_ids"][0]
        if item_id not in sources or item_id in retried:
            raise ValueError(f"Unexpected or duplicate retry for {item_id}")
        original = read_json(sources[item_id])
        if original["status"] != "api_or_harness_error":
            raise ValueError(f"Task {item_id} had no error to retry")
        replacement = path / "attempts" / f"alfworld_{item_id}.json"
        if not replacement.is_file():
            raise ValueError(f"Missing retry attempt: {replacement}")
        sources[item_id] = replacement
        retried.append(item_id)

    records = [read_json(sources[item_id]) for item_id in task_ids]
    for item_id, record in zip(task_ids, records):
        if record["item_id"] != item_id or record["steps"] != record["final_state_step"]:
            raise ValueError(f"Incomplete state for task {item_id}")
        if record["status"] != "api_or_harness_error" and record["final_memory_entries"] != record["steps"] + 1:
            raise ValueError(f"Incomplete episodic memory for task {item_id}")
    errors = [record["item_id"] for record in records if record["status"] == "api_or_harness_error"]
    if errors and not args.allow_errors:
        raise ValueError(f"Retry these API/harness errors in fresh sessions before merging: {errors}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh aggregate directory: {args.output}")

    for item_id in task_ids:
        target = args.output / "attempts" / f"alfworld_{item_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sources[item_id], target)
    write_json(args.output / "selection.json", {
        **common, "start_index": 0, "task_ids": task_ids,
        "source_selections": source_selections, "retried_ids": retried,
    })
    successes = sum(record["wm_success"] for record in records)
    first_errors = [record["item_id"] for record in first_attempts if record["status"] == "api_or_harness_error"]
    metrics = {"tasks": 100, "wm_successes": successes, "errors": errors,
               "total_turns": sum(record["steps"] for record in records),
               "wm_task_success_rate": successes / 100,
               "definition": "WM marker rate over all 100 tasks; unresolved harness errors count as failures",
               "first_attempt_wm_successes": sum(record["wm_success"] for record in first_attempts),
               "first_attempt_errors": first_errors,
               "first_attempt_wm_task_success_rate": sum(record["wm_success"] for record in first_attempts) / 100,
               "retried_ids": retried}
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
