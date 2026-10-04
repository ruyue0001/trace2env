"""ATIF adapter: Harbor / Terminus-2 trajectories → Trace2Env episodes and benchmark rows.

ATIF (Agent Trajectory Interchange Format) is the trajectory format written by the Harbor
harness (``trajectory.json`` per trial). For Terminus-2, every ``agent`` step carries one
``bash_command`` tool call per keystroke batch entry and one observation holding the terminal
screen after the batch; the first ``user`` step holds the task instruction and the initial
screen; summarization appears as ``system``/``user`` steps and continuation files. This module
turns those trajectories into the same episode JSON that ``awb-export`` writes (so
``reconstruct`` ingests them unchanged) and, optionally, into AgentWorldBench-format records
so held-out collected trajectories can be scored with the official protocol.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from trace2env.agentworld import RESPONSE_MARKER, load_system_prompt_template, split_of, terminal_keystrokes_action
from trace2env.models import SplitAssignment, SplitManifest
from trace2env.storage import write_json

COMMAND_TOOL = "bash_command"
COMPLETE_TOOL = "mark_task_complete"
INSTRUCTION_HEADER = "Task Description:\n"
STATE_HEADER = "\n\nCurrent terminal state:\n"
# The instruction block AgentWorldBench appends to the evaluated turn's prompt.
BENCHMARK_SUFFIX = (
    "\n\nPlease analyze the input above and predict the realistic output/response that the environment would produce."
    "\n\nRequirements:\n- Be precise and accurate in your prediction\n- Consider edge cases and error conditions"
    "\n- Maintain state consistency with previous interactions in this session\n- Match the expected output format"
    "\n\nFirst, think step by step to explain your reasoning. Then, provide the simulated environment observation "
    "wrapped strictly within the <predicted_observation></predicted_observation> tags."
)


def load_trajectory(path: str | Path) -> dict[str, Any]:
    """Load a trajectory and splice in its continuation files (``continued_trajectory_ref``)."""
    path = Path(path)
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    steps = list(trajectory.get("steps", []))
    seen = {path.resolve()}
    current = trajectory
    while current.get("continued_trajectory_ref"):
        next_path = (path.parent / current["continued_trajectory_ref"]).resolve()
        if next_path in seen or not next_path.is_file():
            break
        seen.add(next_path)
        current = json.loads(next_path.read_text(encoding="utf-8"))
        steps.extend(current.get("steps", []))
    return {**trajectory, "steps": steps, "continuation_files": len(seen) - 1}


def initial_context(trajectory: dict[str, Any]) -> tuple[str, str | None]:
    """Task instruction and initial terminal screen from the first user step, when present."""
    for step in trajectory.get("steps", []):
        if step.get("source") != "user":
            continue
        message = step.get("message") if isinstance(step.get("message"), str) else ""
        instruction, state = message, None
        if INSTRUCTION_HEADER in message:
            instruction = message.split(INSTRUCTION_HEADER, 1)[1]
        if STATE_HEADER in instruction:
            instruction, state = instruction.split(STATE_HEADER, 1)
            state = strip_screen_wrapper(state)
        return instruction.strip(), state
    return "", None


SCREEN_HEADERS = ("New Terminal Output:", "Current Terminal Screen:", "**Current Terminal Screen:**")
WARNING_HEADER = "Previous response had warnings:"


def strip_screen_wrapper(text: str) -> str:
    """Return the raw terminal screen: Terminus prefixes it with a header and, sometimes, parser warnings."""
    text = text.strip()
    if text.startswith(WARNING_HEADER):
        for header in SCREEN_HEADERS:
            index = text.find(header)
            if index >= 0:
                text = text[index:]
                break
        else:
            text = text.split("\n\n", 1)[1] if "\n\n" in text else text
    for header in SCREEN_HEADERS:
        if text.startswith(header):
            text = text[len(header):]
            break
    return text.strip("\n")


def _observation_text(step: dict[str, Any]) -> str:
    results = (step.get("observation") or {}).get("results") or []
    parts = []
    for result in results:
        content = result.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return strip_screen_wrapper("\n".join(part for part in parts if part))


def terminus_turns(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Environment interactions only: each agent step with keystroke commands and its screen."""
    turns: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        if step.get("source") != "agent" or step.get("is_copied_context"):
            continue
        calls = step.get("tool_calls") or []
        commands = [
            {"keystrokes": str(call["arguments"].get("keystrokes", "")),
             "duration": float(call["arguments"].get("duration", 1.0) or 0.0)}
            for call in calls if call.get("function_name") == COMMAND_TOOL
        ]
        completed = any(call.get("function_name") == COMPLETE_TOOL for call in calls)
        if not commands:
            if completed:
                turns.append({"step_id": step.get("step_id"), "commands": [], "observation": _observation_text(step),
                              "task_complete": True, "message": step.get("message") or ""})
            continue  # parse errors, summarization answers, and other non-environment steps
        turns.append({
            "step_id": step.get("step_id"),
            "commands": commands,
            "observation": _observation_text(step),
            "task_complete": completed,
            "message": step.get("message") if isinstance(step.get("message"), str) else "",
            "timestamp": step.get("timestamp"),
        })
    return turns


def trajectory_metrics(trajectory: dict[str, Any]) -> dict[str, Any]:
    final = trajectory.get("final_metrics") or {}
    steps = trajectory.get("steps", [])
    summed = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "llm_steps": 0}
    for step in steps:
        metrics = step.get("metrics") or {}
        if not metrics or step.get("is_copied_context"):
            continue  # copied context repeats earlier steps; counting it would double-bill them
        summed["llm_steps"] += 1
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            summed[key] += int(metrics.get(key) or 0)
        summed["cost_usd"] += float(metrics.get("cost_usd") or 0.0)
    return {**summed, "final_metrics": final, "steps": len(steps)}


def trajectory_episode(path: str | Path, *, task: str, trial: str, split: str | None = None,
                       outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Terminus-2 trajectory as an episode JSON that ``load_raw_traces`` ingests unchanged."""
    trajectory = load_trajectory(path)
    instruction, initial_state = initial_context(trajectory)
    turns = [turn for turn in terminus_turns(trajectory) if turn["commands"]]
    events: list[dict[str, Any]] = []
    if instruction:
        events.append({"actor": "user", "kind": "message", "content": instruction,
                       "metadata": {"turn": 0, "section": "task_instruction"}})
    if initial_state:
        events.append({"actor": "environment", "kind": "state", "content": initial_state,
                       "metadata": {"turn": 0, "section": "current_state"}})
    for index, turn in enumerate(turns, start=1):
        action = terminal_keystrokes_action(turn["commands"])
        events.append({"actor": "agent", "kind": "action", "content": action.model_dump(mode="json"),
                       "metadata": {"turn": index, "step_id": turn["step_id"],
                                    "raw_action": json.dumps(turn["commands"], ensure_ascii=False),
                                    "agent_message": turn["message"][:2000]}})
        events.append({"actor": "environment", "kind": "observation", "content": turn["observation"],
                       "metadata": {"turn": index, "step_id": turn["step_id"]}})
    group = f"tb2:{task}"
    return {
        "episode_id": f"tb2:{task}:{trial}",
        "task": "terminal",
        "benchmark_task": task,
        "trajectory_id": f"{task}:{trial}",
        "trajectory_group": group,
        "split": split or split_of(group),
        "agent": trajectory.get("agent"),
        "instruction": instruction,
        "turns": len(turns),
        "task_complete": any(turn["task_complete"] for turn in terminus_turns(trajectory)),
        "outcome": outcome,
        "metrics": trajectory_metrics(trajectory),
        "source_path": str(Path(path).resolve()),
        "events": events,
    }


def trial_directory(trajectory_path: Path) -> Path:
    """Harbor writes ``<job>/<task>__<id>/agent/trajectory.json``; the trial is the parent of ``agent``."""
    parent = trajectory_path.parent
    return parent.parent if parent.name == "agent" else parent


def find_trajectories(roots: Iterable[str | Path]) -> list[tuple[Path, str, str]]:
    """``(trajectory.json, task, trial)`` for every trial under the given Harbor job directories."""
    found: list[tuple[Path, str, str]] = []
    for root in roots:
        for path in sorted(Path(root).rglob("trajectory.json")):
            trial = trial_directory(path).name
            task = trial.rsplit("__", 1)[0] if "__" in trial else trial
            found.append((path, task, trial))
    return found


def trial_outcome(trajectory_path: Path) -> dict[str, Any] | None:
    """Verifier reward and failure info from the trial's own ``result.json`` (absent while it runs)."""
    result_path = trial_directory(trajectory_path) / "result.json"
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    verifier = result.get("verifier_result") or {}
    rewards = verifier.get("rewards")
    return {
        "reward": rewards.get("reward") if isinstance(rewards, dict) else verifier.get("reward"),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
        "task_name": result.get("task_name"),
        "trial_name": result.get("trial_name"),
        "finished": True,
    }


def export_trajectories(roots: Iterable[str | Path], output_dir: str | Path, *, split: str | None = None
                        ) -> tuple[list[Path], SplitManifest, list[dict[str, Any]]]:
    """Write one episode file per trial plus a split manifest; returns paths, manifest, and per-episode summaries."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    assignments: dict[str, SplitAssignment] = {}
    summaries: list[dict[str, Any]] = []
    for trajectory_path, task, trial in find_trajectories(roots):
        episode = trajectory_episode(trajectory_path, task=task, trial=trial, outcome=trial_outcome(trajectory_path))
        if split is not None and episode["split"] != split:
            continue
        if not episode["turns"]:
            summaries.append({"task": task, "trial": trial, "turns": 0, "skipped": "no environment turns"})
            continue
        path = output / f"{task}__{trial}.json"
        write_json(path, episode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assignments[f"src_{digest}"] = SplitAssignment(split=episode["split"], trajectory_group=episode["trajectory_group"])
        paths.append(path)
        summaries.append({"task": task, "trial": trial, "turns": episode["turns"], "split": episode["split"],
                          "reward": (episode.get("outcome") or {}).get("reward"), "cost_usd": episode["metrics"]["cost_usd"],
                          "prompt_tokens": episode["metrics"]["prompt_tokens"], "path": str(path)})
    return paths, SplitManifest(assignments=assignments), summaries


# ─── Benchmark-format rows ────────────────────────────────────────────────────

def _turn_prompt(index: int, commands: list[dict[str, Any]], initial_state: str | None) -> str:
    state = f"**Current State:**\n{initial_state}\n\n" if index == 1 and initial_state else ""
    body = json.dumps(commands, ensure_ascii=False, indent=2)
    return f"### Turn {index}\n{state}**Action:**\n```json\n{body}\n```"


def episode_rows(episode: dict[str, Any], *, turns_per_trajectory: int | None = None,
                 system_prompt: str | None = None) -> list[dict[str, Any]]:
    """AgentWorldBench-format records (prompt/response lists through the evaluated turn) for an episode."""
    actions = [event for event in episode["events"] if event["kind"] == "action"]
    observations = {event["metadata"]["turn"]: event["content"] for event in episode["events"] if event["kind"] == "observation"}
    initial_state = next((event["content"] for event in episode["events"] if event["kind"] == "state"), None)
    prompts, responses = [], []
    for event in actions:
        turn = event["metadata"]["turn"]
        commands = json.loads(event["metadata"]["raw_action"])
        prompts.append(_turn_prompt(turn, commands, initial_state))
        responses.append(f"{RESPONSE_MARKER}\n{observations.get(turn, '')}")
    total = len(prompts)
    if not total:
        return []
    if turns_per_trajectory and turns_per_trajectory < total:
        chosen = sorted({max(1, round(i * total / turns_per_trajectory)) for i in range(1, turns_per_trajectory + 1)})
    else:
        chosen = list(range(1, total + 1))
    system = system_prompt if system_prompt is not None else load_system_prompt_template("terminal")
    rows = []
    for turn in chosen:
        rows.append({
            "task": "terminal",
            "id": episode["trajectory_id"],
            "prompt": prompts[:turn - 1] + [prompts[turn - 1] + BENCHMARK_SUFFIX],
            "response": responses[:turn],
            "current_prompt": prompts[turn - 1],
            "system_str": system,
            "turn_idx": turn,
            "total_turns": total,
            "benchmark_task": episode.get("benchmark_task"),
            "split": episode.get("split"),
        })
    return rows
