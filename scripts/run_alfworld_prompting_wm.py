"""Prompt-only Word2World ALFWorld rollouts on a frozen task selection.

The task agent and world model use separate chat histories. Every world-model
request contains its initial context and every preceding action/observation pair.
The source real-environment records supply only the initial agent observation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
DEFAULT_WORK = ROOT / "work" / "exp-alfworld-real"
WM_CONTEXT_SOURCE = BASELINE / "scripts" / "collect_init_context" / "collect_wm_instruct_alfworld.py"
# These command forms come from the bundled ALFWorld TextWorld grammar. In
# particular, ordinary placement and toggling do not use their English verbs.
COMMAND_REFERENCE = """ALFWorld commands have exact syntax. Do not interpret a natural-language paraphrase as an action:
- Move to a receptacle: `go to <receptacle>`.
- Pick up an object: `take <object> from <receptacle>`.
- Place an ordinary held object on/in a receptacle: `move <object> to <receptacle>`. `put <object> on <receptacle>` is not a command.
- Operate a toggleable object: `use <object>`. `toggle <object>` and `turn on <object>` are not commands.
- Other commands: `open <receptacle>`, `close <receptacle>`, `heat <object> with <receptacle>`, `cool <object> with <receptacle>`, `clean <object> with <receptacle>`, `slice <object> with <knife>`, `examine <object or receptacle>`, `look`, `inventory`, and `help`.
- The separate `put <object> into <container object>` and `put <held container object> in <receptacle>` forms apply only to their specific container-object actions.
For any command that does not match a valid grammar form, or whose preconditions are unmet, leave state unchanged and reply exactly `Nothing happens.` Never emit a success marker for it."""

WM_INSTRUCTION = """You are the ALFWorld text environment. The task agent sends you one action per user message. Simulate that action and return only the resulting environment observation, in ALFWorld's style. Do not choose the agent's next action or include your reasoning.

Maintain one persistent world state. Use the initial environment facts below and every prior action and observation in this conversation. Track locations, receptacle contents, open or closed objects, inventory, and changes from heating, cooling, cleaning, and using objects. A repeated look or inventory action must reflect the current state.

{command_reference}

Append the exact marker ` [SUCCESS]` to an observation only if the task objective has been completed in your simulated state. Never append it for a partial objective. Do not invent a new task or skip required actions.

# Environment Information (Only visible to Assistant)

{environment_description}

# User Environment Information (Displayed to User)

{initial_observation}
"""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def description_builder():
    spec = importlib.util.spec_from_file_location("word2world_wm_context", WM_CONTEXT_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {WM_CONTEXT_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.get_alfworld_description


def verify_command_reference(grammar_path: Path) -> None:
    """Fail before a live run if the simulator no longer uses these commands."""
    grammar = grammar_path.read_text(encoding="utf-8")
    required = ('template :: "go to [',
                'template :: "take {o} from {r}";',
                'template :: "move {o} to {r}";',
                'template :: "use {o}";',
                'template :: "open {r}";',
                'template :: "close {r}";',
                'template :: "heat {o} with {r}";',
                'template :: "cool {o} with {r}";',
                'template :: "clean {o} with {r}";',
                'template :: "slice {co} with {ko}";',
                'template :: "examine {o}";',
                'template :: "look";',
                'template :: "inventory";',
                'template :: "help";',
                'template :: "put {o} into {outero}";',
                'template :: "put {outero} in {r}";')
    missing = [form for form in required if form not in grammar]
    unexpected = [form for form in ('template :: "put {o} on {r}";',
                                    'template :: "toggle {o}";',
                                    'template :: "turn on {o}";') if form in grammar]
    if missing or unexpected:
        raise ValueError(f"ALFWorld command reference disagrees with {grammar_path}: "
                         f"missing={missing}, unexpected={unexpected}")


def parse_action(response: str) -> str:
    parts = response.rsplit("Action:", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def split_success_marker(response: str) -> tuple[str, bool]:
    # Match Word2World's WorldModel.done implementation.
    if " [SUCCESS]" in response:
        return response.split(" [SUCCESS]", 1)[0], True
    return response, False


def initial_contexts(work_root: Path, data_root: Path, limit: int) -> list[dict[str, Any]]:
    verify_command_reference(data_root / "logic" / "alfred.twl2")
    aggregate = work_root / "real-openrouter-gpt56sol-k1-aggregate"
    selection = json.loads((aggregate / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((aggregate / "attempt_index.json").read_text(encoding="utf-8"))
    if limit < 1 or limit > len(selection["tasks"]):
        raise ValueError("Task limit exceeds the frozen real-environment selection")
    by_id = {int(row["item_id"]): row for row in index}
    build_description = description_builder()
    contexts = []
    for task in selection["tasks"][:limit]:
        item_id = int(task["item_id"])
        reference = json.loads((work_root / by_id[item_id]["record"]).read_text(encoding="utf-8"))
        initial_agent_messages = reference["conversation"][:3]
        if ([row["role"] for row in initial_agent_messages] != ["user", "assistant", "user"] or
                reference["item_id"] != item_id):
            raise ValueError(f"Invalid initial agent conversation for {item_id}")
        trajectory = (data_root / "json_2.1.1" / "valid_train" / task["task_type"] /
                      task["task_id"] / "traj_data.json")
        environment_description = build_description(trajectory)
        wm_system = WM_INSTRUCTION.format(command_reference=COMMAND_REFERENCE,
                                          environment_description=environment_description,
                                          initial_observation=initial_agent_messages[-1]["content"])
        contexts.append({"task": task, "agent_initial": initial_agent_messages,
                         "wm_initial": [{"role": "system", "content": wm_system}],
                         "agent_initial_sha256": digest(initial_agent_messages),
                         "wm_initial_sha256": digest(wm_system)})
    return contexts


def run_task(context: dict[str, Any], client: Any, *, agent_model: str, wm_model: str,
             max_steps: int) -> dict[str, Any]:
    agent_history = [dict(row) for row in context["agent_initial"]]
    wm_history = [dict(row) for row in context["wm_initial"]]
    turns: list[dict[str, Any]] = []
    status = "max_steps"
    error = None
    started = time.monotonic()
    for turn_number in range(1, max_steps + 1):
        try:
            agent_response = client.chat.completions.create(model=agent_model, messages=agent_history)
            react = agent_response.choices[0].message.content
            if not isinstance(react, str) or not react.strip():
                raise RuntimeError("Task agent returned empty text")
            action = parse_action(react)
            agent_history.append({"role": "assistant", "content": react})

            wm_history.append({"role": "user", "content": action})
            wm_request = [dict(row) for row in wm_history]
            wm_response = client.chat.completions.create(model=wm_model, messages=wm_request)
            raw_observation = wm_response.choices[0].message.content
            if not isinstance(raw_observation, str) or not raw_observation.strip():
                raise RuntimeError("World model returned empty text")
            observation, done = split_success_marker(raw_observation)
            wm_history.append({"role": "assistant", "content": observation})
            agent_history.append({"role": "user", "content": observation})
            turns.append({"turn": turn_number, "react": react, "action": action,
                          "wm_raw_observation": raw_observation, "observation": observation,
                          "wm_success_marker": done,
                          "wm_request_message_count": len(wm_request),
                          "wm_request_sha256": digest(wm_request),
                          "wm_request_characters": sum(len(row["content"]) for row in wm_request)})
            if done:
                status = "wm_success"
                break
        except Exception as exc:  # preserve the partial transcript for inspection/resume
            status = "api_error"
            error = f"{type(exc).__name__}: {exc}"
            # Provider exceptions can contain request details. Never persist a key.
            for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
                secret = os.environ.get(name)
                if secret:
                    error = error.replace(secret, "[REDACTED]")
            break
    task = context["task"]
    return {"item_id": task["item_id"], "task_id": task["task_id"],
            "task_type": task["task_type"], "status": status,
            "wm_success": status == "wm_success", "error": error,
            "steps": len(turns), "turns": turns, "agent_history": agent_history,
            "wm_history": wm_history, "elapsed_seconds": time.monotonic() - started}


def summarize(records: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    complete = len(records) == expected and len({row["item_id"] for row in records}) == expected
    errors = sum(row["status"] == "api_error" for row in records)
    successes = sum(row["wm_success"] for row in records)
    return {"tasks": expected, "finished": len(records), "complete": complete,
            "api_errors": errors, "valid_final_metric": complete and errors == 0,
            "wm_successes": successes, "wm_task_success_rate": successes / expected if complete and errors == 0 else None,
            "total_turns": sum(row["steps"] for row in records),
            "definition": "Word2World WM Task Success Rate: fraction of tasks whose prompted world model emitted [SUCCESS] by the step cap"}


def audit_history(record: dict[str, Any]) -> None:
    """Confirm every logged WM request is the prefix of the persistent history."""
    history = record["wm_history"]
    for turn in record["turns"]:
        count = turn["wm_request_message_count"]
        assert count == 2 * turn["turn"]
        assert history[count - 1] == {"role": "user", "content": turn["action"]}
        assert digest(history[:count]) == turn["wm_request_sha256"]
        assert history[count] == {"role": "assistant", "content": turn["observation"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_WORK / "simulator" / "alfworld")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--agent-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--wm-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.limit, args.max_steps, args.max_workers) < 1:
        parser.error("limit, max-steps, and max-workers must be positive")
    contexts = initial_contexts(args.work_root, args.data_root, args.limit)
    manifest = {"task_ids": [row["task"]["item_id"] for row in contexts],
                "source": "first three agent messages of frozen real ALFWorld attempts",
                "wm_context_source": str(WM_CONTEXT_SOURCE),
                "wm_instruction_sha256": digest(WM_INSTRUCTION),
                "command_reference_sha256": digest(COMMAND_REFERENCE),
                "command_grammar_sha256": hashlib.sha256(
                    (args.data_root / "logic" / "alfred.twl2").read_bytes()).hexdigest(),
                "agent_initial_sha256": [row["agent_initial_sha256"] for row in contexts],
                "wm_initial_sha256": [row["wm_initial_sha256"] for row in contexts],
                "agent_model": args.agent_model, "wm_model": args.wm_model,
                "base_url": args.base_url, "max_steps": args.max_steps}
    manifest_path = args.output / "selection.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("Output selection/configuration differs; use a fresh output directory")
    write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps({"selected": len(contexts), "first": contexts[0]["task"],
                          "last": contexts[-1]["task"]}, indent=2))
        return 0
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Set {args.api_key_env} in the environment; do not pass it on the command line")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=180, max_retries=2)
    records = []
    pending = []
    for context in contexts:
        path = args.output / "attempts" / f"alfworld_{context['task']['item_id']}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            audit_history(record)
            if record["status"] != "api_error":
                records.append(record)
                continue
        pending.append((context, path))
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(run_task, context, client, agent_model=args.agent_model,
                               wm_model=args.wm_model, max_steps=args.max_steps): path
                   for context, path in pending}
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            audit_history(record)
            write_json(futures[future], record)
            records.append(record)
            write_json(args.output / "metrics.json", summarize(records, len(contexts)))
            print(f"{index}/{len(pending)} item={record['item_id']} steps={record['steps']} "
                  f"status={record['status']}", flush=True)
    metrics = summarize(records, len(contexts))
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["valid_final_metric"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
