"""Closed-loop task-agent/world-model rollouts with evaluation kept outside the loop.

Word2World's ``interact_with_world_model/run.py`` alternates a ReAct task agent and
its world model. Here the world-model side is a persistent ``RuntimeHarness`` session;
the task agent never receives evaluator state, checklist code, or oracle observations.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from trace2env.envscaler import tool_definitions, world_model_system_prompt
from trace2env.envscaler_oracle import EnvironmentOracle
from trace2env.llm import AgentLLM, StructuredLLM
from trace2env.models import EnvironmentState, NormalizedAction, StepRequest
from trace2env.runtime import DEFAULT_FEATURES, InvalidAction, RuntimeHarness
from trace2env.storage import read_json, write_json


class TaskDecision(BaseModel):
    kind: Literal["action", "finish"]
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


def _task_system(tools: list[dict[str, Any]]) -> str:
    return (
        "You are the task agent interacting with a simulated environment. Complete the user's task "
        "by issuing one tool action at a time. Use only the listed tools and their documented arguments. "
        "The tool responses may be imperfect; use them as your observations, not as proof of task success. "
        "Return kind='action' with name and arguments for the next call, or kind='finish' when you are done. "
        "Do not invent additional tool names.\n\nTools:\n"
        + json.dumps(tools, ensure_ascii=False, indent=2)
    )


def _task_user(task: str, turns: list[dict[str, Any]]) -> str:
    return "Task:\n" + task + "\n\nInteraction so far:\n" + json.dumps(turns, ensure_ascii=False, indent=2)


def _session_name(task_id: str, attempt: int) -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}-attempt-{attempt:03d}"


def aggregate_pass_at_k(attempts: list[dict[str, Any]], task_ids: list[str], n: int) -> dict[str, Any]:
    """Observed best-of-k success, not the unbiased code-generation estimator C(n-c,k)."""
    by_task = {task_id: [] for task_id in task_ids}
    for row in attempts:
        by_task[row["task_id"]].append(bool(row["success"]))
    if any(len(values) != n for values in by_task.values()):
        raise ValueError("Every task needs exactly n completed attempt records")
    return {
        "tasks": len(task_ids),
        "attempts_per_task": n,
        "attempt_success_rate": sum(sum(values) for values in by_task.values()) / (len(task_ids) * n),
        "pass_at_k": {str(k): sum(any(values[:k]) for values in by_task.values()) / len(task_ids)
                      for k in range(1, n + 1)},
        "definition": "pass@k is the fraction of tasks with at least one oracle-verified success in the first k fresh attempts",
    }


def load_scenarios(path: str | Path, env_id: str, task_ids: set[str] | None = None) -> list[dict[str, Any]]:
    rows = read_json(Path(path))
    if not isinstance(rows, list):
        raise ValueError("EnvScaler scenarios must be a JSON list")
    selected = [row for row in rows if row.get("env_id") == env_id and
                (task_ids is None or row.get("task_id") in task_ids)]
    found = [row["task_id"] for row in selected]
    if len(found) != len(set(found)) or not selected or (task_ids is not None and set(found) != task_ids):
        raise ValueError("Task selection is empty, duplicated, or includes unknown task IDs")
    for row in selected:
        if not isinstance(row.get("task"), str) or not isinstance(row.get("init_config"), dict) or not row.get("checklist_with_func"):
            raise ValueError(f"Incomplete scenario {row.get('task_id')}")
    return selected


def evaluate_envscaler_attempt(
    scenario: dict[str, Any], actions: list[dict[str, Any]], metadata: dict[str, Any], *, seed: int = 0,
) -> dict[str, Any]:
    """Evaluation-side only: replay actions on source and run the official state checks.

    ``metadata`` and ``checklist_with_func`` are trusted local evaluation assets. Neither
    their code nor the resulting state/observations is fed into the task or world-model calls.
    """
    oracle = EnvironmentOracle(metadata, seed=seed)
    state = copy.deepcopy(scenario["init_config"])
    oracle_errors = []
    for index, action in enumerate(actions, 1):
        result = oracle.step(state, action)
        if result["raised"]:
            oracle_errors.append({"step": index, "error": result["raised"]})
        state = result["state"]
    checks = []
    for item in scenario["checklist_with_func"]:
        namespace: dict[str, Any] = {}
        try:
            exec(compile(item["check_func"], "<EnvScaler evaluation checklist>", "exec"), namespace)  # noqa: S102
            passed = bool(namespace["check_func"](copy.deepcopy(state)))
            error = None
        except Exception as exc:  # noqa: BLE001 - a broken check cannot certify success
            passed, error = False, f"{type(exc).__name__}: {exc}"
        checks.append({"item": item.get("check_item", ""), "passed": passed, "error": error})
    count = sum(item["passed"] for item in checks)
    return {"success": count == len(checks), "replayed_steps": len(actions), "oracle_errors": oracle_errors,
            "checklist_passed": count, "checklist_total": len(checks), "checks": checks}


class LongHorizonRunner:
    """Run task-agent actions against a fresh persistent world-model session per attempt."""

    def __init__(
        self, *, package_dir: str | Path, metadata: dict[str, Any], task_llm: StructuredLLM,
        world_agent_llm: AgentLLM, output_dir: str | Path, attempts_per_task: int = 1,
        max_steps: int = 50, allow_unvalidated: bool = False, trust_policy: str = "rules_only",
        fast_path: str = "confident", max_tool_calls: int = 8, features: set[str] | None = None,
        harness_factory: Callable[[Path], Any] | None = None,
        evaluator: Callable[[dict[str, Any], list[dict[str, Any]], dict[str, Any]], dict[str, Any]] = evaluate_envscaler_attempt,
    ):
        if attempts_per_task < 1 or max_steps < 1:
            raise ValueError("attempts_per_task and max_steps must be positive")
        self.package_dir = Path(package_dir)
        self.metadata = metadata
        self.task_llm = task_llm
        self.world_agent_llm = world_agent_llm
        self.output_dir = Path(output_dir)
        self.attempts_per_task = attempts_per_task
        self.max_steps = max_steps
        self.evaluator = evaluator
        self.tools = tool_definitions(metadata)
        self.tool_names = {item["name"] for item in self.tools}
        self.world_prompt = world_model_system_prompt(self.tools)
        self.task_system = _task_system(self.tools)
        self.harness_factory = harness_factory or (lambda session_dir: RuntimeHarness(
            self.package_dir, session_dir, agent_llm=self.world_agent_llm,
            initial_state=EnvironmentState(), allow_unvalidated=allow_unvalidated,
            trust_policy=trust_policy, fast_path=fast_path, max_tool_calls=max_tool_calls,
            features=set(DEFAULT_FEATURES) if features is None else features,
            environment_prompt_chars=None,  # the tool interface sits at the tail of the world-model prompt: never cut it
        ))

    def _attempt(self, scenario: dict[str, Any], attempt: int) -> dict[str, Any]:
        session_dir = self.output_dir / "sessions" / _session_name(scenario["task_id"], attempt)
        if session_dir.exists():
            raise FileExistsError(f"Refusing to reuse a previous attempt session: {session_dir}")
        harness = self.harness_factory(session_dir)
        turns: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        stop_reason = "max_steps"
        error = None
        for _ in range(self.max_steps):
            try:
                decision = self.task_llm.complete(system=self.task_system,
                    user=_task_user(scenario["task"], turns), response_model=TaskDecision, role="task_agent")
                if decision.kind == "finish":
                    stop_reason = "finish"
                    break
                if not decision.name:
                    raise ValueError("Task-agent action has an empty name")
                action = {"name": decision.name, "arguments": decision.arguments}
                actions.append(action)
                if decision.name not in self.tool_names:
                    turns.append({"action": action, "observation": f"Unknown tool: {decision.name}",
                                  "route": "invalid_action"})
                    continue
                try:
                    result = harness.step(StepRequest(action=NormalizedAction(type=decision.name,
                        arguments=decision.arguments), metadata={"environment_prompt": self.world_prompt}))
                except InvalidAction as exc:
                    turns.append({"action": action, "observation": f"Invalid arguments: {exc}",
                                  "route": "invalid_action"})
                    continue
                turns.append({"action": action, "observation": result.observation, "route": result.route})
            except Exception as exc:  # noqa: BLE001 - preserve partial rollout for audit
                stop_reason = "error"
                error = f"{type(exc).__name__}: {exc}"
                break
        evaluation = self.evaluator(scenario, actions, self.metadata)
        return {"task_id": scenario["task_id"], "attempt": attempt, "session_dir": str(session_dir),
                "stop_reason": stop_reason, "error": error, "actions": actions, "turns": turns,
                "evaluation": evaluation, "success": bool(evaluation["success"]) and stop_reason != "error"}

    def run(self, scenarios: list[dict[str, Any]]) -> dict[str, Any]:
        if not scenarios:
            raise ValueError("No scenarios selected")
        task_ids = [item["task_id"] for item in scenarios]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Duplicate task IDs")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for scenario in scenarios:
            for attempt in range(1, self.attempts_per_task + 1):
                record = self._attempt(scenario, attempt)
                write_json(self.output_dir / "attempts" / (_session_name(scenario["task_id"], attempt) + ".json"), record)
                records.append(record)
        summary = aggregate_pass_at_k(records, task_ids, self.attempts_per_task)
        summary.update({"environment_id": self.metadata["env_id"], "max_steps": self.max_steps,
                        "package": str(self.package_dir), "attempts_file_dir": str(self.output_dir / "attempts")})
        write_json(self.output_dir / "summary.json", summary)
        return summary
