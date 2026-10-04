"""Evaluate one GPT task agent on real and modeled ScienceWorld Measurement tasks.

The published Word2World test IDs and ReAct prompt are reused. The prompting
world model starts from Word2World's privileged initial room description.
Trace2Env receives the same description once, then only the current action;
its session database carries prior turns. Replay scores model actions in real
ScienceWorld. Every command writes resumable per-episode records.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from collect_scienceworld_measurement import (
    MEASUREMENT_TASKS, ROOT, WORK, configure_java, game_map, load_test_ids,
    normalize_action, write_json,
)


ADAPTER = ROOT / "baseline/Word2World/AgentGym/agentenv/agentenv/envs/sciworld.py"
PROMPT_PROTOCOL = (
    "You play the ScienceWorld environment for the task agent. For each user action, "
    "return only the observation that the environment would show after that action. "
    "Track object locations, inventory, open doors, temperatures, phase changes, "
    "and task progress across turns. An invalid or impossible action must not change "
    "the world. Do not choose the task agent's next action. Append ` [SUCCESS]` "
    "to the observation only when the stated task is fully completed."
)
TRACE_PROTOCOL = (
    "Play the ScienceWorld Measurement text environment for the task agent's "
    "current command. Task-specific initial facts and the initial observation "
    "were seeded once in session state and turn-0 memory. Use the package, "
    "current state, and episodic memory to continue this same episode. "
    "Exact object referents and command syntax matter. If a hidden value cannot "
    "be recovered from evidence, make your best prediction and submit the "
    "transition instead of repeating tool searches. Append ` [SUCCESS]` "
    "only when the stated task is fully complete. Return only the observation "
    "and optional marker."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def react_prompt() -> tuple[str, str]:
    """Read the original AgentGym ScienceWorld ReAct messages, not a rewrite."""
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"), filename=str(ADAPTER))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SciWorldAdapter")
    assignment = next(n for n in cls.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "conversation_start_dict" for t in n.targets))
    table = assignment.value
    pair = next(v for k, v in zip(table.keys, table.values)
                if isinstance(k, ast.Attribute) and k.attr == "REACT")
    messages = [ast.literal_eval(call.args[0]) for call in pair.elts]
    return messages[0]["value"], messages[1]["value"]


def parse_action(react: str) -> str:
    parts = react.rsplit("Action:", 1)
    if len(parts) != 2:
        return ""
    # ScienceWorld accepts one command per turn. Model spillover after a newline
    # is commentary, not part of the environment action.
    return next((line.strip() for line in parts[1].splitlines() if line.strip()), "")


def split_marker(text: str) -> tuple[str, bool]:
    if " [SUCCESS]" in text:
        return text.split(" [SUCCESS]", 1)[0], True
    return text, False


def select_measurement(env: Any) -> list[dict[str, Any]]:
    mapping = game_map(env)
    ids = load_test_ids(WORK / "sciworld_test.json", mapping)
    rows = [mapping[i] for i in ids if mapping[i]["task_name"] in MEASUREMENT_TASKS]
    if len(rows) != 53 or len({x["item_id"] for x in rows}) != 53:
        raise ValueError("Expected 53 unique Word2World Measurement test rows")
    return rows


def real_initial(env: Any, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    env.load(row["task_name"], row["variation"])
    task_description = env.get_task_description()
    observation, _, done, info = env.step("look around")
    if done:
        raise RuntimeError(f"Episode {row['item_id']} finished during initial look")
    return task_description + "\n" + observation, info


def privileged_description(env: Any, row: dict[str, Any]) -> str:
    """Match the fields and formatting of Word2World's SciWorld WM initializer."""
    env.load(row["task_name"], row["variation"], "teleportAction")
    env.step("look around")
    goal = env.get_goal_progress() or ""
    actions = env.get_possible_actions() or []
    parts = ["\n=== Goal Progress ===\n", str(goal) + "\n",
             "\n=== Possible Actions ===\n"]
    parts.extend(" - " + str(action) + "\n" for action in actions)
    parts.append("\n=== Per-Room Observations ===\n")
    tree = env.getObjectTree() or {}
    rooms = [item.get("name") for item in (tree.get("contents") or {}).values()
             if isinstance(item, dict)]
    for room in rooms:
        if not room:
            continue
        env.step(f"teleport {room}")
        seen = env.look()
        objects = env.get_possible_objects() or []
        parts.append(f"== Room: {room} ==\n")
        parts.append(seen.strip() + "\n")
        parts.append("Possible Objects:" + (" " + ", ".join(objects) if objects else " (none)") + "\n\n")
    return "".join(parts)


def context_file(row: dict[str, Any]) -> Path:
    return WORK / "contexts" / f"sciworld_{row['item_id']}.json"


def make_contexts(env: Any, rows: list[dict[str, Any]]) -> None:
    intro, ack = react_prompt()
    for n, row in enumerate(rows, 1):
        path = context_file(row)
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing["task"] != row or existing["react_prompt_sha256"] != hashlib.sha256(
                    (intro + ack).encode()).hexdigest():
                raise ValueError(f"Context mismatch at {path}")
            continue
        initial, _ = real_initial(env, row)
        description = privileged_description(env, row)
        system = ("# Environment Information (Only visible to Assistant)\n\n" +
                  description + "\n# User Environment Information (Displayed to User)\n\n" +
                  initial + "\n\n# Simulation Protocol\n" + PROMPT_PROTOCOL)
        write_json(path, {"task": row, "agent_initial": [
            {"role": "user", "content": intro},
            {"role": "assistant", "content": ack},
            {"role": "user", "content": initial}],
            "wm_system": system, "initial_description": description,
            "react_prompt_sha256": hashlib.sha256((intro + ack).encode()).hexdigest()})
        print(f"context {n}/{len(rows)}: {row['item_id']}", flush=True)


def load_context(row: dict[str, Any]) -> dict[str, Any]:
    path = context_file(row)
    if not path.is_file():
        raise FileNotFoundError(f"Run contexts first: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["task"] != row:
        raise ValueError(f"Context task mismatch: {path}")
    return value


def safe_error(exc: Exception, key: str | None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    return message.replace(key, "[REDACTED]") if key else message


def task_action(client: Any, model: str, history: list[dict[str, str]]) -> tuple[str, str]:
    response = client.chat.completions.create(model=model, messages=history)
    react = response.choices[0].message.content
    if not isinstance(react, str) or not react.strip():
        raise RuntimeError("Task agent returned empty ReAct text")
    return react, parse_action(react)


def run_real_one(env: Any, context: dict[str, Any], client: Any, model: str,
                 max_steps: int, key: str) -> dict[str, Any]:
    row = context["task"]
    started = time.monotonic()
    history = [dict(x) for x in context["agent_initial"]]
    turns = []
    status, error, score = "max_steps", None, 0
    context_match = None
    try:
        initial, _ = real_initial(env, row)
        context_match = initial == history[2]["content"]
        history[2] = {"role": "user", "content": initial}
        latest = initial
        for turn in range(1, max_steps + 1):
            react, action = task_action(client, model, history)
            history.append({"role": "assistant", "content": react})
            if action:
                observation, _, done, info = env.step(action)
                score = info["score"]
            else:
                observation, done = "Invalid Action.\n\n" + latest, False
            latest = observation
            history.append({"role": "user", "content": observation})
            turns.append({"turn": turn, "react": react, "action": action,
                          "observation": observation, "score": score, "done": done})
            if done:
                status = "done"
                break
    except Exception as exc:
        status, error = "api_or_env_error", safe_error(exc, key)
    return {"item_id": row["item_id"], "task": row, "status": status,
            "success": score == 100, "score": score, "error": error,
            "frozen_initial_context_match": context_match,
            "steps": len(turns), "turns": turns, "agent_history": history,
            "elapsed_seconds": time.monotonic() - started}


def run_prompt_one(context: dict[str, Any], client: Any, model: str,
                   max_steps: int, key: str) -> dict[str, Any]:
    row = context["task"]
    started = time.monotonic()
    agent = [dict(x) for x in context["agent_initial"]]
    wm = [{"role": "system", "content": context["wm_system"]}]
    turns = []
    status, error = "max_steps", None
    for turn in range(1, max_steps + 1):
        try:
            react, action = task_action(client, model, agent)
            agent.append({"role": "assistant", "content": react})
            wm.append({"role": "user", "content": action})
            raw = client.chat.completions.create(model=model, messages=wm).choices[0].message.content
            if not isinstance(raw, str) or not raw.strip():
                raise RuntimeError("Prompted world model returned empty observation")
            observation, done = split_marker(raw)
            wm.append({"role": "assistant", "content": observation})
            agent.append({"role": "user", "content": observation})
            turns.append({"turn": turn, "react": react, "action": action,
                          "wm_raw_observation": raw, "observation": observation,
                          "wm_success_marker": done, "wm_request_message_count": len(wm) - 1})
            if done:
                status = "wm_success"
                break
        except Exception as exc:
            status, error = "api_error", safe_error(exc, key)
            break
    return {"item_id": row["item_id"], "task": row, "status": status,
            "wm_success": status == "wm_success", "error": error,
            "steps": len(turns), "turns": turns, "agent_history": agent,
            "wm_history": wm, "elapsed_seconds": time.monotonic() - started}


def seed_trace2env(harness: Any, context: dict[str, Any]) -> None:
    from trace2env.models import MemoryEntry, NormalizedAction
    if harness.session.load().step != 0 or harness.memory.count() != 0:
        raise ValueError("Trace2Env session must be fresh")
    harness.memory.record(MemoryEntry(
        turn=0, kind="observed", action=NormalizedAction(type="session.start"),
        observation=("# Word2World initial environment facts\n" + context["initial_description"] +
                     "\n# Initial task-agent observation\n" + context["agent_initial"][2]["content"]),
        metadata={"source": "Word2World privileged initialization; no action has occurred"},
    ))


def assert_memory(harness: Any, expected: str | None = None) -> None:
    state = harness.session.load()
    if harness.memory.count() != state.step + 1:
        raise RuntimeError("Trace2Env episodic memory is not contiguous")
    seed = harness.memory.by_turn(0)
    latest = harness.memory.by_turn(state.step)
    if len(seed) != 1 or seed[0].kind != "observed":
        raise RuntimeError("Trace2Env turn-0 memory missing")
    if state.step and (len(latest) != 1 or latest[0].kind != "predicted"):
        raise RuntimeError(f"Trace2Env turn {state.step} missing")
    if expected is not None and latest[0].observation != expected:
        raise RuntimeError("Trace2Env last memory observation differs from returned observation")


def run_trace_one(context: dict[str, Any], output: Path, package: Path,
                  client: Any, model: str, max_steps: int, key: str,
                  resume_record: dict[str, Any] | None = None) -> dict[str, Any]:
    from trace2env.agent import AgentProtocolError
    from trace2env.llm import OpenAIToolLLM
    from trace2env.models import EnvironmentState, NormalizedAction, StepRequest
    from trace2env.runtime import DEFAULT_FEATURES, RuntimeHarness

    row = context["task"]
    started = time.monotonic()
    session = output / "sessions" / f"sciworld_{row['item_id']}"
    if resume_record is None and session.exists():
        raise FileExistsError(f"Refusing to reuse partial Trace2Env session {session}")
    if resume_record is not None and not session.exists():
        raise FileNotFoundError(f"Missing Trace2Env session for resume: {session}")
    initial = context["agent_initial"][2]["content"]
    goal_match = re.search(r"Your task is to\s*(.+)", initial)
    room_match = re.search(r"This room is called the ([^.\n]+)\.", initial)
    initial_state = EnvironmentState(
        world={"initial_description": context["initial_description"]},
        session={"goal": goal_match.group(1).strip() if goal_match else initial.splitlines()[0],
                 **({"location": room_match.group(1).strip()} if room_match else {})},
        surface={"initial_observation": initial},
    )
    harness = RuntimeHarness(
        package, session, agent_llm=OpenAIToolLLM(model=model, client=client, max_tokens=None),
        initial_state=initial_state, allow_unvalidated=True, trust_policy="schema_checked",
        effect_rejection="keep_observation", fast_path="confident", max_tool_calls=24,
        features=set(DEFAULT_FEATURES), memory_recent=12,
        environment_prompt_chars=None,  # the recorded runs passed TRACE_PROTOCOL whole
    )
    if resume_record is None:
        seed_trace2env(harness, context)
    assert_memory(harness)
    agent = [dict(x) for x in (resume_record["agent_history"] if resume_record else context["agent_initial"])]
    turns = list(resume_record["turns"]) if resume_record else []
    if harness.session.load().step != len(turns):
        raise RuntimeError("Resume record and SQLite session disagree on committed turns")
    pending_react = None
    if resume_record is not None and agent[-1]["role"] == "assistant":
        pending_react = agent[-1]["content"]
    status, error = "max_steps", None
    for turn in range(len(turns) + 1, max_steps + 1):
        try:
            if pending_react is not None:
                react, action = pending_react, parse_action(pending_react)
                pending_react = None
            else:
                react, action = task_action(client, model, agent)
                agent.append({"role": "assistant", "content": react})
            normalized = NormalizedAction.model_validate(normalize_action(action))
            assert_memory(harness)
            retries = 0
            while True:
                try:
                    result = harness.step(StepRequest(
                        action=normalized, metadata={"environment_prompt": TRACE_PROTOCOL}))
                    break
                except AgentProtocolError as exc:
                    if "exceeded its tool budget" not in str(exc) or retries >= 2:
                        raise
                    # A failed tool loop has not committed the transition. Retry
                    # from the same SQLite state and memory, never from a new turn.
                    assert_memory(harness)
                    if harness.session.load().step != turn - 1:
                        raise RuntimeError("Failed Trace2Env step changed session state") from exc
                    retries += 1
            assert_memory(harness, expected=result.observation)
            observation, done = split_marker(result.observation)
            agent.append({"role": "user", "content": observation})
            turns.append({"turn": turn, "react": react, "action": action,
                          "normalized_action": normalized.model_dump(mode="json"),
                          "wm_raw_observation": result.observation, "observation": observation,
                          "wm_success_marker": done, "route": result.route,
                          "state_revision": result.state.revision,
                          "memory_entry_count": harness.memory.count(),
                          "wm_step_retries": retries,
                          "tool_calls": result.tool_calls})
            if done:
                status = "wm_success"
                break
        except Exception as exc:
            status, error = "api_or_harness_error", safe_error(exc, key)
            break
    return {"item_id": row["item_id"], "task": row, "status": status,
            "wm_success": status == "wm_success", "error": error,
            "steps": len(turns), "turns": turns, "agent_history": agent,
            "session_dir": str(session), "final_state_step": harness.session.load().step,
            "final_memory_entries": harness.memory.count(),
            "elapsed_seconds": (resume_record.get("elapsed_seconds", 0) if resume_record else 0)
                               + time.monotonic() - started,
            "resumed_from_api_error": bool(resume_record)}


def replay_one(env: Any, row: dict[str, Any], source: dict[str, Any], max_steps: int) -> dict[str, Any]:
    initial, _ = real_initial(env, row)
    turns = []
    score, status, error = 0, "replay_complete", None
    if source["status"] in {"api_error", "api_or_harness_error"}:
        return {"item_id": row["item_id"], "status": "source_error",
                "success": False, "score": 0, "steps": 0, "turns": [], "error": source.get("error")}
    try:
        if len(source["turns"]) > max_steps:
            raise ValueError("Source exceeds frozen turn cap")
        for turn in source["turns"]:
            action = turn["action"]
            if action:
                observation, _, done, info = env.step(action)
                score = info["score"]
            else:
                observation, done = "Invalid Action.\n\n" + initial, False
            turns.append({"turn": turn["turn"], "action": action,
                          "real_observation": observation, "score": score, "done": done})
            if done:
                break
    except Exception as exc:
        status, error = "env_error", f"{type(exc).__name__}: {exc}"
    return {"item_id": row["item_id"], "status": status, "success": score == 100,
            "score": score, "steps": len(turns), "turns": turns, "error": error}


def summarize(mode: str, rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    records = []
    for row in rows:
        path = output / "attempts" / f"sciworld_{row['item_id']}.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
    key = "wm_success" if mode in {"prompt", "trace2env"} else "success"
    errors = sum("error" in record["status"] or record["status"] == "source_error" for record in records)
    complete = len(records) == len(rows) and errors == 0
    successes = sum(bool(record.get(key)) for record in records)
    summary = {"mode": mode, "tasks": len(rows), "finished": len(records),
               "complete": complete, "errors": errors, "successes": successes,
               "success_rate": successes / len(rows) if complete else None,
               "observed_success_fraction": successes / len(records) if records else None,
               "total_turns": sum(record.get("steps", 0) for record in records),
               "definition": ("model success marker" if key == "wm_success" else
                              "real ScienceWorld score 100")}
    write_json(output / "metrics.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("contexts", "real", "prompt", "trace2env", "replay"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path, help="Prompt/Trace2Env output for replay")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=53)
    parser.add_argument("--item-ids", type=int, nargs="+",
                        help="Run exactly these frozen Measurement item IDs, in the given order")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--resume-errors", action="store_true",
                        help="Resume only API connection failures from intact Trace2Env SQLite sessions")
    args = parser.parse_args()
    if args.start_index < 0 or args.limit < 1 or (not args.item_ids and args.start_index + args.limit > 53) or args.max_steps < 1:
        parser.error("Select 1..53 Measurement test rows and a positive turn cap")
    if args.item_ids and len(set(args.item_ids)) != len(args.item_ids):
        parser.error("--item-ids must be unique")
    if args.mode != "contexts" and not args.output:
        parser.error("--output is required")
    if args.mode == "trace2env" and not args.package:
        parser.error("--package is required for Trace2Env")
    if args.mode == "replay" and not args.source:
        parser.error("--source is required for replay")
    if args.resume_errors and args.mode != "trace2env":
        parser.error("--resume-errors applies only to Trace2Env")
    env = None
    if args.mode in {"contexts", "real", "replay"}:
        configure_java()
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv()
    try:
        if env is None:
            selection = json.loads((WORK / "measurement-30-gold-v1/selection.json").read_text(encoding="utf-8"))
            all_rows = selection["evaluation_rows"]
            if len(all_rows) != 53 or [row["item_id"] for row in all_rows] != selection["evaluation_ids"]:
                raise ValueError("Frozen Measurement evaluation map is incomplete")
        else:
            all_rows = select_measurement(env)
        if args.item_ids:
            by_id = {row["item_id"]: row for row in all_rows}
            missing = set(args.item_ids) - by_id.keys()
            if missing:
                parser.error(f"Unknown frozen Measurement item IDs: {sorted(missing)}")
            rows = [by_id[item_id] for item_id in args.item_ids]
        else:
            rows = all_rows[args.start_index:args.start_index + args.limit]
        if args.mode == "contexts":
            make_contexts(env, rows)
            return 0
        config = {"mode": args.mode, "task_ids": [r["item_id"] for r in rows],
                  "test_sha256": sha256(WORK / "sciworld_test.json"),
                  "adapter_sha256": sha256(ADAPTER), "max_steps": args.max_steps,
                  "model": args.model, "base_url": args.base_url,
                  "package_manifest_sha256": sha256(args.package / "manifest.json") if args.package else None,
                  "source": str(args.source) if args.source else None,
                  "max_tool_calls": 24 if args.mode == "trace2env" else None,
                  "tool_budget_retries": 2 if args.mode == "trace2env" else None,
                  "trace_protocol_sha256": hashlib.sha256(TRACE_PROTOCOL.encode()).hexdigest() if args.mode == "trace2env" else None,
                  "protocol": "current action only; session SQLite supplies Trace2Env memory" if args.mode == "trace2env" else None}
        selection_path = args.output / "selection.json"
        if selection_path.is_file() and json.loads(selection_path.read_text(encoding="utf-8")) != config:
            parser.error("Existing output selection differs; choose a fresh directory")
        write_json(selection_path, config)
        key = os.environ.get(args.api_key_env) or ""
        client = None
        if args.mode in {"real", "prompt", "trace2env"}:
            if not key:
                parser.error(f"Set {args.api_key_env}")
            from openai import OpenAI
            client = OpenAI(api_key=key, base_url=args.base_url, timeout=180, max_retries=2)
        for n, row in enumerate(rows, 1):
            path = args.output / "attempts" / f"sciworld_{row['item_id']}.json"
            resume_record = None
            if path.is_file():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if "error" not in previous["status"]:
                    continue
                if args.mode == "trace2env":
                    if not args.resume_errors or previous.get("error", "").split(":", 1)[0] != "APIConnectionError":
                        continue
                    resume_record = previous
            if args.mode == "real":
                record = run_real_one(env, load_context(row), client, args.model, args.max_steps, key)
            elif args.mode == "prompt":
                record = run_prompt_one(load_context(row), client, args.model, args.max_steps, key)
            elif args.mode == "trace2env":
                record = run_trace_one(load_context(row), args.output, args.package,
                                       client, args.model, args.max_steps, key, resume_record)
            else:
                source_path = args.source / "attempts" / f"sciworld_{row['item_id']}.json"
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing source rollout {source_path}")
                source = json.loads(source_path.read_text(encoding="utf-8"))
                record = replay_one(env, row, source, args.max_steps)
            write_json(path, record)
            score = record.get("score", "-")
            print(f"{args.mode} {n}/{len(rows)} item={row['item_id']} status={record['status']} "
                  f"success={record.get('success', record.get('wm_success'))} score={score} "
                  f"steps={record['steps']}", flush=True)
            summarize(args.mode, rows, args.output)
        print(json.dumps(summarize(args.mode, rows, args.output), indent=2))
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
