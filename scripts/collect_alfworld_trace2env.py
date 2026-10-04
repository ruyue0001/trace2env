"""Select diverse held-out ALFWorld tasks, collect 40 real wins, export trace inputs.

Construction uses test-index rows 100:200 (IDs 2520-2619); the first 100 rows
remain evaluation-only. Selection reads task metadata and game-file hashes, not
trajectories or simulator ground truth. Collection uses the original ReAct agent
prompt and real ALFWorld wrapper from run_alfworld_real.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from run_alfworld_real import BASELINE, load_tasks, original_react_prompt, parse_react_action, run_attempt, write_json


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "exp-alfworld-real"
TEST_INDEX = WORK / "data" / "alfworld_test.json"
DATA_ROOT = WORK / "simulator" / "alfworld"
MAPPING = BASELINE / "AgentGym" / "agentenv-alfworld" / "configs" / "mappings_test.json"


def signature(task: dict[str, Any]) -> tuple[str, str, str]:
    """Goal family, target object, and destination/light, without instance suffix."""
    parts = task["task_type"].split("-")
    if len(parts) != 5:
        raise ValueError(f"Unexpected ALFWorld task type: {task['task_type']}")
    return parts[0], parts[1], parts[3]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_action(command: str) -> dict[str, Any]:
    """Stable ALFWorld text-command interface for construction and runtime."""
    command = command.strip()
    if command in {"look", "inventory", "help"}:
        return {"type": command, "arguments": {}, "raw": command}
    for prefix, action_type in (("go to ", "go_to"), ("open ", "open"),
                                ("close ", "close"), ("use ", "use"),
                                ("examine ", "examine")):
        if command.startswith(prefix) and command[len(prefix):].strip():
            return {"type": action_type,
                    "arguments": {"target": command[len(prefix):].strip()}, "raw": command}
    for pattern, action_type, names in (
        (r"take (.+?) from (.+)", "take", ("object", "receptacle")),
        (r"move (.+?) to (.+)", "move", ("object", "receptacle")),
        (r"put (.+?) into (.+)", "put_into", ("object", "container")),
        (r"put (.+?) in (.+)", "put_in", ("container", "receptacle")),
        (r"heat (.+?) with (.+)", "heat", ("object", "receptacle")),
        (r"cool (.+?) with (.+)", "cool", ("object", "receptacle")),
        (r"clean (.+?) with (.+)", "clean", ("object", "receptacle")),
        (r"slice (.+?) with (.+)", "slice", ("object", "knife")),
    ):
        match = re.fullmatch(pattern, command)
        if match:
            return {"type": action_type,
                    "arguments": dict(zip(names, match.groups())), "raw": command}
    return {"type": "invalid_command", "arguments": {"command": command}, "raw": command}


def choose_candidates(test_file: Path, data_root: Path) -> dict[str, Any]:
    evaluation = load_tasks(test_file, mapping_file=MAPPING, limit=100, start_index=0)
    construction = load_tasks(test_file, mapping_file=MAPPING, limit=100, start_index=100)
    if [task["item_id"] for task in evaluation] != list(range(2420, 2520)):
        raise ValueError("The frozen ALFWorld evaluation selection changed")
    if [task["item_id"] for task in construction] != list(range(2520, 2620)):
        raise ValueError("The ALFWorld construction pool changed")
    evaluation_signatures = {signature(task) for task in evaluation}
    seen_signatures: set[tuple[str, str, str]] = set()
    seen_games: set[str] = set()
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task in construction:
        sig = signature(task)
        game = data_root / "json_2.1.1" / "valid_train" / task["task_type"] / task["task_id"] / "game.tw-pddl"
        game_hash = file_sha256(game)
        reason = ("same_goal_object_destination_as_evaluation" if sig in evaluation_signatures else
                  "duplicate_construction_signature" if sig in seen_signatures else
                  "duplicate_game_file" if game_hash in seen_games else None)
        row = {**task, "signature": list(sig), "game_sha256": game_hash}
        if reason:
            excluded.append({**row, "reason": reason})
            continue
        candidates.append(row)
        seen_signatures.add(sig)
        seen_games.add(game_hash)
    if len(candidates) < 40:
        raise ValueError(f"Only {len(candidates)} distinct construction candidates remain")
    # Greedy ordering spreads early model calls across goal families, objects,
    # and destinations. It never examines successful trajectories or eval scores.
    ordered: list[dict[str, Any]] = []
    family_count: Counter[str] = Counter()
    object_count: Counter[str] = Counter()
    destination_count: Counter[str] = Counter()
    while candidates:
        picked = min(candidates, key=lambda row: (
            family_count[row["signature"][0]],
            object_count[row["signature"][1]],
            destination_count[row["signature"][2]],
            row["item_id"],
        ))
        candidates.remove(picked)
        ordered.append(picked)
        family_count[picked["signature"][0]] += 1
        object_count[picked["signature"][1]] += 1
        destination_count[picked["signature"][2]] += 1
    return {
        "protocol": "Word2World real ALFWorld ReAct, one attempt per distinct task",
        "evaluation_ids": [row["item_id"] for row in evaluation],
        "construction_pool_ids": [row["item_id"] for row in construction],
        "candidate_order": ordered,
        "excluded": excluded,
        "test_index_sha256": file_sha256(test_file),
        "mapping_sha256": file_sha256(MAPPING),
        "grammar_sha256": file_sha256(data_root / "logic" / "alfred.twl2"),
    }


def collect(selection: dict[str, Any], output: Path, *, data_root: Path,
            model: str, base_url: str, key_env: str, target: int,
            max_new_attempts: int | None = None) -> dict[str, Any]:
    intro, ack = original_react_prompt()
    config = {"selection_sha256": file_sha256(output / "selection.json"),
              "model": model, "base_url": base_url, "max_rounds": 50,
              "react_prompt_sha256": hashlib.sha256((intro + ack).encode("utf-8")).hexdigest()}
    config_path = output / "collection_config.json"
    if config_path.is_file() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Collection model or prompt changed; use a fresh output directory")
    write_json(config_path, config)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ValueError(f"Set {key_env} in the environment; never put a key in a command argument or file")
    from openai import OpenAI

    source = BASELINE / "AgentGym" / "agentenv-alfworld"
    os.environ["ALFWORLD_DATA"] = str(data_root.resolve())
    sys.path.insert(0, str(source))
    from agentenv_alfworld.env_wrapper import ALFWorld_Wrapper

    wrapper = ALFWorld_Wrapper(data_path=str(data_root.resolve()),
                               config_path=str(source / "configs" / "base_config.yaml"))
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)
    successes: list[int] = []
    attempted: list[int] = []
    errors: list[int] = []
    new_attempts = 0
    consecutive_api_errors = 0
    for row in selection["candidate_order"]:
        if len(successes) >= target:
            break
        item_id = row["item_id"]
        attempt_path = output / "attempts" / f"alfworld_{item_id}.json"
        prior = json.loads(attempt_path.read_text(encoding="utf-8")) if attempt_path.is_file() else None
        if prior is None or prior["status"] in {"api_error", "env_error"}:
            if max_new_attempts is not None and new_attempts >= max_new_attempts:
                break
            record = run_attempt(wrapper, client, task=row, attempt=1, intro=intro, ack=ack,
                                 model=model, max_rounds=50)
            for field in ("error", "cleanup_error"):
                if isinstance(record.get(field), str):
                    record[field] = record[field].replace(api_key, "[REDACTED]")
            write_json(attempt_path, record)
            new_attempts += 1
            print(f"{new_attempts} new / {len(attempted) + 1} considered; "
                  f"item={item_id} success={record['success']} status={record['status']}", flush=True)
        else:
            record = prior
        attempted.append(item_id)
        if record["success"] and record["reward"] in (1, 100):
            successes.append(item_id)
        if record["status"] in {"api_error", "env_error"}:
            errors.append(item_id)
        if record["status"] == "api_error":
            consecutive_api_errors += 1
            if consecutive_api_errors >= 3:
                print("Pausing collection after three consecutive API errors; rerun when connectivity returns.",
                      flush=True)
                break
            time.sleep(5 * consecutive_api_errors)
        else:
            consecutive_api_errors = 0
    summary = {"target_successes": target, "successes": len(successes),
               "success_ids": successes, "attempted_ids": attempted,
               "error_ids": errors, "new_attempts": new_attempts,
               "paused_for_api_errors": consecutive_api_errors >= 3,
               "selection_sha256": file_sha256(output / "selection.json"),
               "model": model, "base_url": base_url,
               "complete": len(successes) == target}
    write_json(output / "collection_summary.json", summary)
    return summary


def export_traces(selection: dict[str, Any], output: Path, target: int) -> dict[str, Any]:
    summary = json.loads((output / "collection_summary.json").read_text(encoding="utf-8"))
    if not summary["complete"] or summary["successes"] != target:
        raise ValueError(f"Need {target} complete real successes before exporting construction traces")
    by_id = {row["item_id"]: row for row in selection["candidate_order"]}
    assignments: dict[str, dict[str, str]] = {}
    exported: list[dict[str, Any]] = []
    for item_id in summary["success_ids"]:
        record = json.loads((output / "attempts" / f"alfworld_{item_id}.json").read_text(encoding="utf-8"))
        if not record["success"] or record["status"] != "done":
            raise ValueError(f"Selected attempt {item_id} is not a real success")
        messages = record["conversation"]
        actions = record["actions"]
        if len(messages) != 3 + 2 * len(actions):
            raise ValueError(f"Attempt {item_id} has misaligned actions and observations")
        events = [{"actor": "environment", "kind": "observation",
                   "content": messages[2]["content"]}]
        for turn, action in enumerate(actions):
            if action != parse_react_action(messages[3 + 2 * turn]["content"]):
                raise ValueError(f"Attempt {item_id} has an action/ReAct mismatch at turn {turn + 1}")
            events.extend(({"actor": "agent", "kind": "action", "content": normalize_action(action)},
                           {"actor": "environment", "kind": "observation",
                            "content": messages[4 + 2 * turn]["content"]}))
        trace = {"episode_id": f"alfworld_{item_id}", "events": events}
        trace_path = output / "traces" / f"alfworld_{item_id}.json"
        write_json(trace_path, trace)
        source_id = "src_" + file_sha256(trace_path)
        assignments[source_id] = {"split": "train", "trajectory_group": f"alfworld_{item_id}"}
        exported.append({"item_id": item_id, "signature": by_id[item_id]["signature"],
                         "source_id": source_id, "turns": len(actions),
                         "trace": str(trace_path.relative_to(output))})
    if len({tuple(row["signature"]) for row in exported}) != target:
        raise ValueError("Exported successful traces are not signature-unique")
    write_json(output / "split_manifest.json", {"assignments": assignments})
    manifest = {"traces": exported, "total_turns": sum(row["turns"] for row in exported),
                "selection_sha256": file_sha256(output / "selection.json"),
                "excluded_evaluation_ids": selection["evaluation_ids"]}
    write_json(output / "trace_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "collect", "export"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, default=TEST_INDEX)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--target-successes", type=int, default=40)
    parser.add_argument("--max-new-attempts", type=int)
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    args = parser.parse_args()
    if args.target_successes < 1:
        parser.error("target-successes must be positive")
    selection_path = args.output / "selection.json"
    expected = choose_candidates(args.test_file, args.data_root)
    if args.command == "select":
        if selection_path.is_file() and json.loads(selection_path.read_text(encoding="utf-8")) != expected:
            parser.error("Selection or source files changed; use a fresh output directory")
        write_json(selection_path, expected)
        print(json.dumps({"candidate_count": len(expected["candidate_order"]),
                          "excluded_count": len(expected["excluded"]),
                          "first_ids": [row["item_id"] for row in expected["candidate_order"][:10]]}, indent=2))
        return 0
    if not selection_path.is_file() or json.loads(selection_path.read_text(encoding="utf-8")) != expected:
        parser.error("Run select first; the frozen selection must match source files")
    if args.command == "collect":
        result = collect(expected, args.output, data_root=args.data_root, model=args.model,
                         base_url=args.base_url, key_env=args.api_key_env,
                         target=args.target_successes, max_new_attempts=args.max_new_attempts)
        print(json.dumps(result, indent=2))
        return 0 if result["complete"] else 1
    result = export_traces(expected, args.output, args.target_successes)
    print(json.dumps({"traces": len(result["traces"]), "total_turns": result["total_turns"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
