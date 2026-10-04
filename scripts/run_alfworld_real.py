"""Word2World-compatible ALFWorld real-environment task-agent evaluation.

Uses the original ALFWorld ReAct prompt, test index, game mapping, and simulator
wrapper. Unlike Word2World's APIAgent, model failures are bounded and recorded
instead of retried forever. Each attempt resets a fresh real environment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "Word2World"
MAPPING = BASELINE / "AgentGym" / "agentenv-alfworld" / "configs" / "mappings_test.json"
ADAPTER = BASELINE / "AgentGym" / "agentenv" / "agentenv" / "envs" / "alfworld.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_tasks(test_file: Path, mapping_file: Path = MAPPING, limit: int = 100,
               start_index: int = 0) -> list[dict[str, Any]]:
    rows = json.loads(test_file.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    if (not isinstance(rows, list) or not isinstance(mapping, list) or limit < 1 or
            start_index < 0 or len(rows) < start_index + limit):
        raise ValueError("Invalid ALFWorld test index, mapping, or task limit")
    by_id = {int(row["item_id"]): row for row in mapping}
    if len(by_id) != len(mapping):
        raise ValueError("Duplicate ALFWorld mapping item IDs")
    selected = []
    for row in rows[start_index:start_index + limit]:
        item_id = int(row["item_id"].rsplit("_", 1)[-1])
        if item_id not in by_id:
            raise ValueError(f"Test item {item_id} is missing from the mapping")
        selected.append({"item_id": item_id, "task_type": by_id[item_id]["task_type"],
                         "task_id": by_id[item_id]["task_id"]})
    if len({row["item_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate selected test items")
    return selected


def original_react_prompt(adapter_file: Path = ADAPTER) -> tuple[str, str]:
    """Read, rather than retype, AgentGym's exact ALFWorld ReAct conversation start."""
    tree = ast.parse(adapter_file.read_text(encoding="utf-8"), filename=str(adapter_file))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AlfWorldAdapter")
    assignment = next(node for node in cls.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == "conversation_start_dict"
                          for target in node.targets))
    table = assignment.value
    assert isinstance(table, ast.Dict)
    pair = next(value for key, value in zip(table.keys, table.values)
                if isinstance(key, ast.Attribute) and key.attr == "REACT")
    assert isinstance(pair, ast.Tuple)
    messages = [ast.literal_eval(call.args[0]) for call in pair.elts]
    return messages[0]["value"], messages[1]["value"]


def parse_react_action(text: str) -> str:
    """ALFWorld branch of AgentGym BaseAdapter.parse_react."""
    parts = text.rsplit("Action:", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def summarize(records: list[dict[str, Any]], task_ids: list[int], attempts_per_task: int) -> dict[str, Any]:
    by_task = {task_id: {} for task_id in task_ids}
    for row in records:
        by_task[row["item_id"]][row["attempt"]] = row
    complete = all(set(items) == set(range(1, attempts_per_task + 1)) for items in by_task.values())
    error_count = sum(row["status"] in {"api_error", "env_error"} for row in records)
    pass_at_k = {}
    for k in range(1, attempts_per_task + 1):
        pass_at_k[str(k)] = sum(any(by_task[task_id].get(j, {}).get("success", False)
                                   for j in range(1, k + 1)) for task_id in task_ids) / len(task_ids)
    return {"tasks": len(task_ids), "attempts_per_task": attempts_per_task,
            "attempts_finished": len(records), "complete": complete,
            "api_or_env_errors": error_count, "valid_final_metric": complete and error_count == 0,
            "pass_at_k": pass_at_k,
            "real_task_success_rate": pass_at_k["1"] if complete and error_count == 0 else None,
            "definition": "pass@k = fraction of selected tasks solved in at least one of their first k fresh real-environment attempts"}


def run_attempt(wrapper: Any, client: Any, *, task: dict[str, Any], attempt: int,
                intro: str, ack: str, model: str, max_rounds: int) -> dict[str, Any]:
    started = time.monotonic()
    conversation: list[dict[str, str]] = [{"role": "user", "content": intro},
                                          {"role": "assistant", "content": ack}]
    actions: list[str] = []
    reward = 0.0
    done = False
    status = "max_rounds"
    error = None
    cleanup_error = None
    environment_id = None
    try:
        created = wrapper.create()
        if "error" in created:
            raise RuntimeError(created["error"])
        environment_id = created["id"]
        initial = wrapper.reset(environment_id, task["item_id"], "Text")
        if "error" in initial:
            raise RuntimeError(initial["error"])
        latest_observation = initial["observation"]
        latest_actions = initial["available_actions"]
        conversation.append({"role": "user", "content": latest_observation +
                             "\nAVAILABLE ACTIONS: " + ",".join(latest_actions)})
        for _ in range(max_rounds):
            try:
                response = client.chat.completions.create(model=model, messages=conversation)
                generated = response.choices[0].message.content
                if not isinstance(generated, str) or not generated.strip():
                    raise RuntimeError("Model returned no action text")
            except Exception as exc:  # noqa: BLE001 - one failed call must not loop forever
                status, error = "api_error", f"{type(exc).__name__}: {exc}"
                break
            conversation.append({"role": "assistant", "content": generated})
            action = parse_react_action(generated)
            actions.append(action)
            if not action:
                observation = ("Invalid Action.\n\n" + latest_observation +
                               "\nAVAILABLE ACTIONS: " + ",".join(latest_actions))
            else:
                step = wrapper.step(environment_id, action)
                if "error" in step:
                    raise RuntimeError(step["error"])
                observation = step["observation"]
                latest_observation = observation
                latest_actions = step["available_actions"]
                reward = float(step["reward"])
                done = bool(step["done"])
            conversation.append({"role": "user", "content": observation})
            if done:
                status = "done"
                break
    except Exception as exc:  # noqa: BLE001 - preserve partial attempt and continue others
        status, error = "env_error", f"{type(exc).__name__}: {exc}"
    finally:
        if environment_id is not None:
            # The bundled wrapper's close() rejects already-done games. Close its
            # TextWorld instance directly and remove the ID from its destructor list.
            with wrapper._lock:
                instance = wrapper.env_init.pop(environment_id, None)
                wrapper.env.pop(environment_id, None)
                wrapper.info.pop(environment_id, None)
                if environment_id in wrapper.ls:
                    wrapper.ls.remove(environment_id)
            if instance is not None:
                try:
                    instance.close()
                except Exception as exc:  # noqa: BLE001 - cleanup does not change the reward
                    cleanup_error = f"{type(exc).__name__}: {exc}"
    return {"item_id": task["item_id"], "task_id": task["task_id"],
            "task_type": task["task_type"], "attempt": attempt, "model": model,
            "reward": reward, "success": reward in (1.0, 100.0), "done": done,
            "status": status, "error": error, "cleanup_error": cleanup_error, "actions": actions,
            "conversation": conversation, "elapsed_seconds": time.monotonic() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing json_2.1.1/ and logic/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--base-url", help="OpenAI-compatible Chat Completions endpoint")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based offset in Word2World test ordering")
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Keep at 1: TextWorld is not thread-safe; parallelize with separate processes")
    parser.add_argument("--dry-run", action="store_true", help="Validate selection/data and write manifest without model calls")
    args = parser.parse_args()
    if min(args.limit, args.attempts, args.max_rounds, args.max_workers) < 1:
        parser.error("limit, attempts, max-rounds, and max-workers must be positive")
    if args.max_workers != 1:
        parser.error("ALFWorld/TextWorld is not thread-safe; use separate processes for parallelism")
    tasks = load_tasks(args.test_file, limit=args.limit, start_index=args.start_index)
    for task in tasks:
        path = args.data_root / "json_2.1.1" / "valid_train" / task["task_type"] / task["task_id"] / "game.tw-pddl"
        if not path.is_file():
            parser.error(f"Missing ALFWorld game file: {path}")
    grammar_path = args.data_root / "logic" / "alfred.twl2"
    if not grammar_path.is_file():
        parser.error("Missing Word2World ALFWorld logic/alfred.twl2")
    grammar = grammar_path.read_text(encoding="utf-8")
    if 'template :: "move {o} to {r}"' not in grammar or "action help" not in grammar:
        parser.error("ALFWorld grammar lacks Word2World's move/help modifications")
    intro, ack = original_react_prompt()
    manifest = {"split": "test/valid_train", "selection": "Word2World eval rows by offset",
                "start_index": args.start_index,
                "tasks": tasks, "attempts_per_task": args.attempts, "max_rounds": args.max_rounds,
                "model": args.model, "prompt_source": str(ADAPTER), "data_root": str(args.data_root),
                "test_index_sha256": hashlib.sha256(args.test_file.read_bytes()).hexdigest(),
                "mapping_sha256": hashlib.sha256(MAPPING.read_bytes()).hexdigest(),
                "grammar_sha256": hashlib.sha256(grammar_path.read_bytes()).hexdigest(),
                "react_prompt_sha256": hashlib.sha256((intro + ack).encode("utf-8")).hexdigest()}
    manifest_path = args.output / "selection.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("Existing output selection/configuration differs; use a fresh output directory")
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"selected": len(tasks), "first_item": tasks[0], "last_item": tasks[-1]}, indent=2))
        return 0
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Set {args.api_key_env} in the environment; do not pass API keys on the command line")
    from openai import OpenAI

    source = BASELINE / "AgentGym" / "agentenv-alfworld"
    # ALFWorld imports initialize their data directory before the wrapper is
    # constructed; keep that initialization inside the task-specific workdir.
    os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
    sys.path.insert(0, str(source))
    from agentenv_alfworld.env_wrapper import ALFWorld_Wrapper

    wrapper = ALFWorld_Wrapper(data_path=str(args.data_root.resolve()),
                               config_path=str(source / "configs" / "base_config.yaml"))
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=180, max_retries=2)
    pending = []
    records = []
    for task in tasks:
        for attempt in range(1, args.attempts + 1):
            path = args.output / "attempts" / f"alfworld_{task['item_id']}_attempt_{attempt:02d}.json"
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["status"] in {"api_error", "env_error"}:
                    pending.append((task, attempt, path))
                else:
                    records.append(previous)
            else:
                pending.append((task, attempt, path))
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_attempt, wrapper, client, task=task, attempt=attempt,
                               intro=intro, ack=ack, model=args.model, max_rounds=args.max_rounds): path
                   for task, attempt, path in pending}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            write_json(futures[future], record)
            records.append(record)
            write_json(args.output / "metrics.json", summarize(records, [task["item_id"] for task in tasks], args.attempts))
            print(f"{index}/{len(pending)} item={record['item_id']} attempt={record['attempt']} "
                  f"success={record['success']} status={record['status']}", flush=True)
    summary = summarize(records, [task["item_id"] for task in tasks], args.attempts)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid_final_metric"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
