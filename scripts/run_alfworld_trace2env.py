"""Interact a GPT task agent with a persistent Trace2Env ALFWorld session.

The caller sends only the current normalized action to RuntimeHarness.step.
Trace2Env loads the prior state and episodic memory from the same session SQLite
database; the task agent keeps its own separate conversation history. The
initial observation and Word2World privileged initial layout are seeded once.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

from collect_alfworld_trace2env import normalize_action, signature
from run_alfworld_prompting_wm import DEFAULT_WORK, description_builder, split_success_marker, write_json


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "Word2World"
PROTOCOL = """Play the ALFWorld text environment for the task agent's current command. The task-specific initial facts and observation are in this session's turn-0 memory and state. Use the package, current state, and episodic memory to continue the same episode. Exact command syntax matters; invalid commands leave state unchanged and return `Nothing happens.` Append ` [SUCCESS]` only when the household task goal is complete. Return only the environment observation plus that marker when applicable."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_contexts(work_root: Path, data_root: Path) -> list[dict[str, Any]]:
    aggregate = work_root / "real-openrouter-gpt56sol-k1-aggregate"
    selection = json.loads((aggregate / "selection.json").read_text(encoding="utf-8"))
    index = json.loads((aggregate / "attempt_index.json").read_text(encoding="utf-8"))
    by_id = {int(row["item_id"]): row for row in index}
    # The released context module imports tqdm solely for its command-line
    # main. Keep this read-only reuse working in the project venv without
    # installing that unrelated progress-bar dependency.
    shimmed_tqdm = importlib.util.find_spec("tqdm") is None
    if shimmed_tqdm:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
        sys.modules["tqdm"] = tqdm_module
    try:
        build_description = description_builder()
    finally:
        if shimmed_tqdm:
            sys.modules.pop("tqdm", None)
    contexts = []
    for task in selection["tasks"]:
        item_id = int(task["item_id"])
        prior = json.loads((work_root / by_id[item_id]["record"]).read_text(encoding="utf-8"))
        agent_initial = prior["conversation"][:3]
        if ([row["role"] for row in agent_initial] != ["user", "assistant", "user"] or
                prior["item_id"] != item_id):
            raise ValueError(f"Bad initial task-agent context for {item_id}")
        trajectory = (data_root / "json_2.1.1" / "valid_train" / task["task_type"] /
                      task["task_id"] / "traj_data.json")
        contexts.append({"task": task, "agent_initial": agent_initial,
                         "initial_layout": build_description(trajectory)})
    return contexts


def seed_session(harness: Any, context: dict[str, Any]) -> None:
    """Record initial information once; never resend the interaction prefix."""
    from trace2env.models import MemoryEntry, NormalizedAction

    if harness.session.load().step != 0 or harness.memory.count() != 0:
        raise ValueError("A fresh ALFWorld episode must start with empty state and memory")
    initial_observation = context["agent_initial"][2]["content"]
    initial_layout = context["initial_layout"]
    harness.memory.record(MemoryEntry(
        turn=0, kind="observed", action=NormalizedAction(type="session.start"),
        observation=("# Word2World task-specific initial environment facts\n" + initial_layout +
                     "\n# Initial task-agent observation\n" + initial_observation),
        metadata={"source": "Word2World privileged initialization; no action has occurred"},
    ))


def initial_state(context: dict[str, Any]):
    from trace2env.models import EnvironmentState

    initial_observation = context["agent_initial"][2]["content"]
    goal = re.search(r"Your task is to:\s*(.+)", initial_observation)
    return EnvironmentState(
        world={"initial_layout": context["initial_layout"]},
        session={"goal": goal.group(1).strip() if goal else ""},
        surface={"initial_observation": initial_observation},
    )


def assert_memory_contiguous(harness: Any, *, expected_observation: str | None = None) -> None:
    """Catch a missing write after state commit before relying on current-turn-only input."""
    state = harness.session.load()
    count = harness.memory.count()
    if count != state.step + 1:  # one turn-0 seed, then one predicted entry per committed step
        raise RuntimeError(f"ALFWorld session memory gap: step={state.step}, entries={count}")
    seed = harness.memory.by_turn(0)
    latest = harness.memory.by_turn(state.step)
    if len(seed) != 1 or seed[0].kind != "observed":
        raise RuntimeError("ALFWorld session is missing its turn-0 observation")
    if state.step and (len(latest) != 1 or latest[0].kind != "predicted"):
        raise RuntimeError(f"ALFWorld session is missing predicted turn {state.step}")
    if expected_observation is not None and latest[0].observation != expected_observation:
        raise RuntimeError(f"ALFWorld turn {state.step} memory differs from the committed observation")


def run_one(context: dict[str, Any], package_dir: Path, output: Path, client: Any,
            *, task_model: str, wm_model: str, max_steps: int,
            api_key: str) -> dict[str, Any]:
    from trace2env.llm import OpenAIToolLLM
    from trace2env.models import NormalizedAction, StepRequest
    from trace2env.runtime import DEFAULT_FEATURES, RuntimeHarness

    task = context["task"]
    item_id = task["item_id"]
    session_dir = output / "sessions" / f"alfworld_{item_id}"
    if session_dir.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite a partial session: {session_dir}")
    harness = RuntimeHarness(
        package_dir, session_dir,
        # GPT-5-family Chat Completions reject the legacy max_tokens parameter.
        agent_llm=OpenAIToolLLM(model=wm_model, client=client, max_tokens=None),
        initial_state=initial_state(context), allow_unvalidated=True,
        trust_policy="schema_checked", effect_rejection="keep_observation",
        fast_path="confident", max_tool_calls=8, features=set(DEFAULT_FEATURES),
        memory_recent=12, environment_prompt_chars=None,  # the recorded runs passed PROTOCOL whole
    )
    seed_session(harness, context)
    assert_memory_contiguous(harness)
    agent_history = [dict(row) for row in context["agent_initial"]]
    turns: list[dict[str, Any]] = []
    status = "max_steps"
    error = None
    for turn_number in range(1, max_steps + 1):
        try:
            response = client.chat.completions.create(model=task_model, messages=agent_history)
            react = response.choices[0].message.content
            if not isinstance(react, str) or not react.strip():
                raise RuntimeError("Task agent returned no ReAct text")
            action = react.rsplit("Action:", 1)[-1].strip() if "Action:" in react else ""
            agent_history.append({"role": "assistant", "content": react})
            normalized = NormalizedAction.model_validate(normalize_action(action))
            assert_memory_contiguous(harness)
            # This request has no prior-turn payload. The fixed protocol is
            # static environment instruction; history comes from session memory.
            result = harness.step(StepRequest(action=normalized,
                                              metadata={"environment_prompt": PROTOCOL}))
            assert_memory_contiguous(harness, expected_observation=result.observation)
            observation, done = split_success_marker(result.observation)
            agent_history.append({"role": "user", "content": observation})
            turns.append({"turn": turn_number, "react": react, "action": action,
                          "normalized_action": normalized.model_dump(mode="json"),
                          "observation": observation, "wm_raw_observation": result.observation,
                          "wm_success_marker": done, "route": result.route,
                          "state_revision": result.state.revision,
                          "memory_entry_count": harness.memory.count(),
                          "tool_calls": result.tool_calls})
            if done:
                status = "wm_success"
                break
        except Exception as exc:
            status = "api_or_harness_error"
            error = f"{type(exc).__name__}: {exc}".replace(api_key, "[REDACTED]")
            break
    return {"item_id": item_id, "task_type": task["task_type"], "task_id": task["task_id"],
            "status": status, "wm_success": status == "wm_success", "error": error,
            "steps": len(turns), "turns": turns, "agent_history": agent_history,
            "session_dir": str(session_dir),
            "final_state_step": harness.session.load().step,
            "final_memory_entries": harness.memory.count()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_WORK / "simulator" / "alfworld")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0,
                        help="Zero-based offset in the 100 held-out Word2World tasks")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--task-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--wm-model", default="openai/gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.start_index < 0 or args.start_index + args.limit > 100 or args.max_steps < 1:
        parser.error("start-index and limit must select 1..100 held-out tasks; max-steps must be positive")
    contexts = initial_contexts(args.work_root, args.data_root)[args.start_index:args.start_index + args.limit]
    trace_manifest = json.loads((args.work_root / "trace2env-40-success-v1" / "trace_manifest.json").read_text())
    train_signatures = {tuple(row["signature"]) for row in trace_manifest["traces"]}
    if any(signature(ctx["task"]) in train_signatures for ctx in contexts):
        parser.error("Construction/evaluation goal signatures overlap")
    selection = {"task_ids": [ctx["task"]["item_id"] for ctx in contexts],
                 "start_index": args.start_index,
                 "package_manifest_sha256": sha256(args.package / "manifest.json"),
                 "trace_manifest_sha256": sha256(args.work_root / "trace2env-40-success-v1" / "trace_manifest.json"),
                 "protocol_sha256": hashlib.sha256(PROTOCOL.encode("utf-8")).hexdigest(),
                 "task_model": args.task_model, "wm_model": args.wm_model,
                 "base_url": args.base_url, "max_steps": args.max_steps,
                 "communication": "current action plus fixed protocol; prior turns loaded from per-session episodic memory"}
    selection_path = args.output / "selection.json"
    if selection_path.is_file() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        parser.error("Existing selection or package differs; use a fresh output directory")
    write_json(selection_path, selection)
    if args.dry_run:
        print(json.dumps({"selected": len(contexts), "first": contexts[0]["task"],
                          "last": contexts[-1]["task"]}, indent=2))
        return 0
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Set {args.api_key_env} in the environment")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=180, max_retries=2)
    records = []
    for index, context in enumerate(contexts, 1):
        item_id = context["task"]["item_id"]
        path = args.output / "attempts" / f"alfworld_{item_id}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
        else:
            record = run_one(context, args.package, args.output, client,
                             task_model=args.task_model, wm_model=args.wm_model,
                             max_steps=args.max_steps, api_key=api_key)
            write_json(path, record)
        records.append(record)
        if index % 10 == 0 or index == len(contexts):
            print(f"{index}/{len(contexts)}; WM successes={sum(r['wm_success'] for r in records)}; "
                  f"errors={sum(r['status'] == 'api_or_harness_error' for r in records)}", flush=True)
    summary = {"tasks": len(contexts), "wm_successes": sum(r["wm_success"] for r in records),
               "errors": [r["item_id"] for r in records if r["status"] == "api_or_harness_error"],
               "total_turns": sum(r["steps"] for r in records),
               "wm_task_success_rate": (sum(r["wm_success"] for r in records) / len(contexts)
                                        if all(r["status"] != "api_or_harness_error" for r in records) else None)}
    write_json(args.output / "metrics.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
