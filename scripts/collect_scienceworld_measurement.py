"""Collect 30 real ScienceWorld Measurement wins without evaluation leakage.

Construction paths are supplied by ScienceWorld's gold-path generator and are
replayed in the real simulator. Only visible task/observation/action text enters
Trace2Env episodes; score and gold-path provenance stay in the audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "exp-scienceworld"
EXCLUDED_TASK_IDS = {"5-1", "5-2", "9-1", "9-2", "9-3", "10-1", "10-2"}
MEASUREMENT_TASKS = (
    "use-thermometer",
    "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance",
)
ACTION_PREFIXES = (
    "look around", "look at", "look in", "go to", "pick up", "put down",
    "focus on", "wait1", "wait", "inventory", "task", "open", "close",
    "activate", "deactivate", "connect", "disconnect", "use", "read",
    "move", "pour", "dunk", "mix", "eat", "flush", "examine", "choose",
)


def configure_java() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    homes = sorted((WORK / "java").glob("*/Contents/Home"))
    if len(homes) != 1:
        raise RuntimeError("Set JAVA_HOME or install one portable JRE under work/exp-scienceworld/java")
    os.environ["JAVA_HOME"] = str(homes[0].resolve())
    os.environ["PATH"] = str((homes[0] / "bin").resolve()) + os.pathsep + os.environ.get("PATH", "")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = Path(file.name)
    os.replace(temporary, path)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_action(command: str) -> dict[str, Any]:
    raw = command.strip()
    lower = raw.lower()
    prefix = next((part for part in ACTION_PREFIXES if lower == part or lower.startswith(part + " ")), "invalid_command")
    return {"type": prefix.replace(" ", "_"),
            "arguments": {"command": raw}, "raw": raw}


def game_map(env: Any) -> dict[int, dict[str, Any]]:
    """Match Word2World's ID mapping exactly; task insertion order matters."""
    result = {}
    idx = 0
    for task_id, name in env.tasks.items():
        if task_id in EXCLUDED_TASK_IDS:
            continue
        for variation in range(env.get_max_variations(name)):
            result[idx] = {"item_id": idx, "task_id": task_id,
                           "task_name": name, "variation": variation}
            idx += 1
    if idx != 4639:
        raise RuntimeError(f"ScienceWorld variation counts changed: {idx} != 4639")
    return result


def load_test_ids(path: Path, mapping: dict[int, dict[str, Any]]) -> list[int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids = [int(row["item_id"].rsplit("_", 1)[1]) for row in rows]
    if len(ids) != 200 or len(set(ids)) != 200 or any(i not in mapping for i in ids):
        raise ValueError("The published ScienceWorld evaluation index changed")
    return ids


def spaced_order(numbers: list[int]) -> list[int]:
    """Visit the variation range broadly before nearby variants."""
    ordered = []
    lo, hi = 0, len(numbers) - 1
    while lo <= hi:
        ordered.append(numbers[lo])
        if lo != hi:
            ordered.append(numbers[hi])
        lo += 1
        hi -= 1
    return ordered


def select(env: Any, mapping: dict[int, dict[str, Any]], test_ids: list[int]) -> dict[str, Any]:
    test_set = set(test_ids)
    eval_rows = [mapping[i] for i in test_ids if mapping[i]["task_name"] in MEASUREMENT_TASKS]
    if len(eval_rows) != 53:
        raise ValueError(f"Expected 53 Measurement evaluation rows, got {len(eval_rows)}")
    reverse = {(row["task_name"], row["variation"]): i for i, row in mapping.items()}
    pools = {}
    native_test = {}
    for name in MEASUREMENT_TASKS:
        env.load(name, 0)
        train_variations = sorted(env.get_variations_train())
        native_test[name] = set(env.get_variations_test())
        pools[name] = [reverse[(name, v)] for v in spaced_order(train_variations)
                       if reverse[(name, v)] not in test_set]
    return {
        "protocol": "Word2World ScienceWorld task ID mapping; real simulator, no simplification",
        "evaluation_ids": [row["item_id"] for row in eval_rows],
        "evaluation_rows": eval_rows,
        "construction_pool": pools,
        "native_test_split_rows": sum(row["variation"] in native_test[row["task_name"]]
                                      for row in eval_rows),
        "source_test_sha256": digest((WORK / "sciworld_test.json").read_bytes()),
        "selection_rule": "10 simulator-scored wins per Measurement task type; unique task descriptions and gold action sequences within each type",
    }


def collect(env: Any, mapping: dict[int, dict[str, Any]], selection: dict[str, Any],
            output: Path, per_type: int) -> dict[str, Any]:
    goals = defaultdict(list)
    seen_descriptions = set()
    seen_actions = set()
    attempted = []
    test_set = set(selection["evaluation_ids"])
    for name in MEASUREMENT_TASKS:
        for item_id in selection["construction_pool"][name]:
            if len(goals[name]) >= per_type:
                break
            if item_id in test_set:
                raise RuntimeError(f"Construction item {item_id} is in evaluation")
            row = mapping[item_id]
            try:
                env.load(name, row["variation"], "", generateGoldPath=True)
                description = env.get_task_description()
                actions = env.get_gold_action_sequence()
                description_key = (name, re.sub(r"\s+", " ", description.strip().lower()))
                actions_key = (name, tuple(actions))
                if description_key in seen_descriptions or actions_key in seen_actions:
                    attempted.append({**row, "status": "duplicate_description_or_path"})
                    continue
                initial, _, _, initial_info = env.step("look around")
                events = [{"actor": "environment", "kind": "observation",
                           "content": description + "\n" + initial}]
                steps = []
                for action in actions:
                    observation, _, done, info = env.step(action)
                    events.extend(({"actor": "agent", "kind": "action",
                                    "content": normalize_action(action)},
                                   {"actor": "environment", "kind": "observation",
                                    "content": observation}))
                    steps.append({"action": action, "observation": observation,
                                  "score": info["score"], "done": done})
                    if done:
                        break
                score = steps[-1]["score"] if steps else initial_info["score"]
                success = score == 100
                audit = {**row, "status": "success" if success else "not_success",
                         "score": score, "steps": len(steps), "gold_path_length": len(actions),
                         "task_description": description}
                attempted.append(audit)
                if not success:
                    continue
                trace = {"episode_id": f"sciworld_{item_id}", "events": events}
                trace_path = output / "traces" / f"sciworld_{item_id}.json"
                write_json(trace_path, trace)
                write_json(output / "attempts" / f"sciworld_{item_id}.json",
                           {"provenance": "simulator-generated gold path replayed in real simulator",
                            "item_id": item_id, "task_name": name, "variation": row["variation"],
                            "score": score, "success": True, "initial_observation": events[0]["content"],
                            "turns": steps})
                seen_descriptions.add(description_key)
                seen_actions.add(actions_key)
                goals[name].append({**audit, "trace": str(trace_path.relative_to(output)),
                                    "sha256": digest(trace_path.read_bytes())})
                print(f"selected {name} #{len(goals[name])}/{per_type}: {item_id}, {len(steps)} turns", flush=True)
            except Exception as exc:
                attempted.append({**row, "status": "env_error", "error": f"{type(exc).__name__}: {exc}"})
                print(f"error {item_id}: {type(exc).__name__}: {exc}", flush=True)
    selected = [row for name in MEASUREMENT_TASKS for row in goals[name]]
    assignments = {"src_" + row["sha256"]: {"split": "train", "trajectory_group": f"sciworld_{row['item_id']}"}
                   for row in selected}
    write_json(output / "split_manifest.json", {"assignments": assignments})
    manifest = {"selection_sha256": digest((output / "selection.json").read_bytes()),
                "provenance": "ScienceWorld gold paths, verified by real simulator score 100",
                "excluded_evaluation_ids": selection["evaluation_ids"],
                "target": per_type * len(MEASUREMENT_TASKS), "complete": len(selected) == per_type * len(MEASUREMENT_TASKS),
                "traces": selected, "attempts": attempted,
                "total_turns": sum(row["steps"] for row in selected)}
    write_json(output / "trace_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "collect"))
    parser.add_argument("--output", type=Path, default=WORK / "measurement-30-gold-v1")
    parser.add_argument("--per-type", type=int, default=10)
    args = parser.parse_args()
    if args.per_type < 1:
        parser.error("per-type must be positive")
    configure_java()
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv()
    try:
        mapping = game_map(env)
        ids = load_test_ids(WORK / "sciworld_test.json", mapping)
        selection = select(env, mapping, ids)
        selection_path = args.output / "selection.json"
        if selection_path.is_file() and json.loads(selection_path.read_text()) != selection:
            parser.error("Existing selection differs; use a fresh output directory")
        write_json(selection_path, selection)
        if args.command == "select":
            print(json.dumps({"test": len(selection["evaluation_ids"]),
                              "native_test_rows": selection["native_test_split_rows"],
                              "construction_pool": {k: len(v) for k, v in selection["construction_pool"].items()}}, indent=2))
            return 0
        manifest = collect(env, mapping, selection, args.output, args.per_type)
        print(json.dumps({"complete": manifest["complete"], "traces": len(manifest["traces"]),
                          "turns": manifest["total_turns"]}, indent=2))
        return 0 if manifest["complete"] else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
