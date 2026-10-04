"""EnvScaler adapter: synthesized tool-environment rollouts → Trace2Env episodes, ground truth, and benchmark rows.

An EnvScaler environment is a programmatic sandbox: an in-memory database (record collections keyed
by id) behind a set of Python tool functions. One rollout JSON records one task: ``env_id``,
``task_id``, the verifier's verdict (``reward``, checklist counts), ``init_state`` / ``final_state``,
and a ``trajectory`` whose steps carry the tool call (``action``), the text the task agent received
(``observation``), and — unlike terminal or MCP traces — the simulator's own ground truth: the full
database before and after the step, their diff, and an outcome label (``sigma_t``), repeated under
the aliases ``S_t`` / ``A_t`` / ``O_tplus1`` / ``S_tplus1`` / ``Delta_t``. Environment definitions
(``<env>_metadata.json`` with the tool schemas, ``<env>_scenarios.json`` with the task texts) sit
beside the rollouts.

Episodes have the shape ``awb-export`` / ``atif-export`` / ``mcp-export`` write (a ``user``
instruction event, then alternating ``action`` / ``observation`` events), so ``reconstruct`` ingests
them unchanged, and they hold only what crossed the tool interface. Two rules follow from the data:

* The observation is the recorded ``observation`` string, verbatim. ``observation_raw`` /
  ``O_tplus1`` is *not* a faithful copy: it holds references to live records, so a later step's
  write shows up in an earlier step's parsed observation (210 of the 1795 tool calls of the first
  collection). It is never used as evidence; the export summary only counts the affected steps.
* Ground truth (databases, diffs, outcome labels) never enters an episode file, because a compiled
  package embeds its episode files verbatim. It goes to a separate directory on request, for
  evaluation only.

The closing ``chat_with_user`` call is the task agent talking to its user, not an environment
interaction; it is dropped and recorded as ``task_complete``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from trace2env.agentworld import RESPONSE_MARKER, load_system_prompt_template, split_of
from trace2env.atif import BENCHMARK_SUFFIX
from trace2env.mcp_traces import render_tool_definitions
from trace2env.models import NormalizedAction, SplitAssignment, SplitManifest
from trace2env.storage import read_json, write_json

SOURCE = "envscaler"
# Harness-level tools of the task agent (not functions of the environment class).
AGENT_SIDE_TOOLS = ("chat_with_user",)
# Some terminated rollouts omit ``chat_with_user`` and record the collector's completion sentinel
# against the environment-tool name from the pending action instead.  It is not a tool response and
# must not become evidence (or an evaluation target).
COLLECTOR_COMPLETION_OBSERVATIONS = ("Task Completed",)
# EnvScaler's outcome label per step -> the environment-level outcome Trace2Env uses.
OUTCOMES = {"success-effect": "success", "success-noop": "success", "rejected": "failure"}
# State entries that are not record collections. Every environment class keeps its constructor argument
# (``self.init_config = init_config``) and no tool reads it; a state snapshot shows it as ``{}`` or as an
# inert second copy of the initial database that goes stale at the first write.
CONFIG_KEYS = ("init_config",)
_ANSWER_BEARING_DESCRIPTION_SECTIONS = {"return", "returns", "constraints"}
_DESCRIPTION_SECTION = re.compile(r"^([A-Za-z][A-Za-z0-9 _/-]*):(?:\s.*)?$")
_RENDERED_DESCRIPTION = re.compile(
    r"(#### \*\*Description\*\*:\n)(.*?)(\n\n#### \*\*Parameters\*\*:)", re.DOTALL
)


def evaluation_tool_description(description: Any) -> str:
    """Remove answer-bearing return and constraint sections from an EnvScaler tool docstring.

    Evaluation keeps the tool's summary, argument documentation, and any unrelated sections. The source
    collection's descriptions commonly quote exact success/error payloads and implementation conditions under
    top-level ``Returns:`` and ``Constraints:`` headings; exposing those sections makes held-out prediction a
    response-template lookup rather than a test of behavior learned from traces.
    """
    kept: list[str] = []
    dropping = False
    for line in str(description or "").splitlines():
        heading = _DESCRIPTION_SECTION.match(line)
        if heading:
            dropping = heading.group(1).strip().lower() in _ANSWER_BEARING_DESCRIPTION_SECTIONS
            if dropping:
                continue
        if not dropping:
            kept.append(line.rstrip())
    return "\n".join(kept).strip()


def sanitize_evaluation_system_prompt(system_prompt: str) -> str:
    """Apply ``evaluation_tool_description`` to rendered EnvScaler tool blocks.

    This supports already-exported rows as well as prompts supplied explicitly to ``episode_rows``. It only
    touches text between the standard Description and Parameters headings, so the shared MCP instructions,
    parameter schemas, history, and few-shot examples remain unchanged.
    """
    return _RENDERED_DESCRIPTION.sub(
        lambda match: match.group(1) + evaluation_tool_description(match.group(2)) + match.group(3),
        system_prompt,
    )


def is_rollout(value: Any) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("trajectory"), list)
            and isinstance(value.get("env_id"), str) and isinstance(value.get("task_id"), str))


def iter_rollouts(roots: Iterable[str | Path]) -> Iterator[tuple[Path, dict[str, Any]]]:
    """``(path, rollout)`` for the given files and for every rollout under the given directories.

    A directory is searched recursively and may hold other JSON (environment definitions, a selection
    manifest); those are skipped. A file named explicitly must be a rollout.
    """
    for root in roots:
        root = Path(root)
        if root.is_dir():
            for path in sorted(path for path in root.rglob("*.json") if not path.name.startswith(".")):
                value = read_json(path)
                if is_rollout(value):
                    yield path, value
        else:
            value = read_json(root)
            if not is_rollout(value):
                raise ValueError(f"{root} is not an EnvScaler rollout (expected env_id, task_id, and a trajectory list)")
            yield root, value


# ─── Environment definitions ──────────────────────────────────────────────────

def tool_definitions(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvScaler evaluation definitions, without answer-bearing return/constraint docstring sections."""
    definitions = []
    for tool in metadata.get("tools") or []:
        function = tool.get("function") or {}
        definitions.append({"name": str(function.get("name", "")),
                            "description": evaluation_tool_description(function.get("description", "")),
                            "parameters": function.get("parameters")})
    return definitions


_CLOCK_READS = {"time", "time_ns", "localtime", "gmtime", "ctime", "asctime"}  # functions of the ``time`` module


def generated_shapes(source: str) -> list[str] | None:
    """Which generated values an environment's source can put into an observation: ``uuid`` when it draws ids
    (``uuid4`` / ``uuid1``), the clock shapes (``stamp`` / ``date`` / ``epoch``) when it reads the clock. ``None`` when
    the source cannot be parsed (then nothing is known and the scorer keeps every shape).

    The scorer accepts any value of a generated shape where the truth holds one the model was never shown. Without
    this list a *computed* date (a registration deadline derived from a session's start) would be accepted the same
    way; an environment that never reads the clock has no generated dates, so its dates are compared exactly.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    from_time = {alias.asname or alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module == "time"
                 for alias in node.names}
    shapes: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ("uuid4", "uuid1"):
            shapes.add("uuid")
        elif leaf in ("now", "utcnow", "today") or (name.startswith("time.") and leaf in _CLOCK_READS) or name in from_time & _CLOCK_READS:
            shapes.update(("stamp", "date", "epoch"))
    return sorted(shapes)


def load_environment(env_defs: str | Path | None, env_id: str) -> dict[str, Any]:
    """Tool definitions, task instructions, and the dataset's own description of an environment (empty when absent).

    ``generated_shapes`` is evaluation-side (it is read off the environment source): it goes into benchmark rows for
    the scorer, never into an episode or a description.
    """
    environment: dict[str, Any] = {"env_id": env_id, "summary": "", "introduction": "", "tools": None, "instructions": {},
                                   "generated_shapes": None}
    if env_defs is None:
        return environment
    metadata_path = Path(env_defs) / f"{env_id}_metadata.json"
    scenarios_path = Path(env_defs) / f"{env_id}_scenarios.json"
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        environment.update(summary=str(metadata.get("environment_summary", "")).strip(),
                           introduction=str(metadata.get("environment_introduction", "")).strip(),
                           tools=tool_definitions(metadata),
                           generated_shapes=generated_shapes(str(metadata.get("env_class_code") or "")))
    if scenarios_path.is_file():
        environment["instructions"] = {str(item.get("task_id")): str(item.get("task", ""))
                                       for item in read_json(scenarios_path) if isinstance(item, dict)}
    return environment


# ─── Rollout → episode ────────────────────────────────────────────────────────

def _harness_error(step: dict[str, Any]) -> str | None:
    """The exception line of a loader traceback (no machine paths); the agent saw only ``observation``."""
    lines = [line for line in str(step.get("error") or "").splitlines() if line.strip()]
    return lines[-1].strip() if lines else None


def _collector_completion(step: dict[str, Any]) -> bool:
    observation = step.get("observation")
    return isinstance(observation, str) and observation.strip() in COLLECTOR_COMPLETION_OBSERVATIONS


def rollout_turns(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """Environment interactions in order: every step whose action is a tool of the environment."""
    turns: list[dict[str, Any]] = []
    for position, step in enumerate(rollout.get("trajectory") or [], start=1):
        action = step.get("action") or {}
        name = str(action.get("name", ""))
        if name in AGENT_SIDE_TOOLS or _collector_completion(step):
            continue
        arguments = action.get("arguments")
        observation, raw = step.get("observation"), step.get("observation_raw")
        turns.append({
            "step": step.get("step", position),
            "name": name,
            "action": NormalizedAction(type=name, arguments=arguments if isinstance(arguments, dict) else {}, raw=action),
            # The text the agent received, verbatim (the recorder already stringified the tool's return value).
            "output": observation if isinstance(observation, str) else str(observation),
            "harness_error": _harness_error(step),
            "stale_raw": isinstance(raw, dict) and str(raw) != observation,
            "truth": {"outcome_label": step.get("sigma_t", step.get("outcome")), "obs_success": step.get("obs_success"),
                      "n_delta": step.get("n_delta"), "state_diff": step.get("state_diff"), "error": step.get("error")},
        })
    return turns


def _events(instruction: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if instruction:
        events.append({"actor": "user", "kind": "message", "content": instruction,
                       "metadata": {"turn": 0, "section": "task_instruction"}})
    for index, turn in enumerate(turns, start=1):
        events.append({"actor": "agent", "kind": "action", "content": turn["action"].model_dump(mode="json"),
                       "metadata": {"turn": index, "step": turn["step"], "tool_name": turn["name"]}})
        metadata = {"turn": index, "step": turn["step"]}
        if turn["harness_error"]:
            metadata["harness_error"] = turn["harness_error"]
        events.append({"actor": "environment", "kind": "observation", "content": turn["output"], "metadata": metadata})
    return events


def _episode(path: Path, rollout: dict[str, Any], turns: list[dict[str, Any]], split: str | None,
             environment: dict[str, Any]) -> dict[str, Any]:
    env_id, task = str(rollout["env_id"]), str(rollout["task_id"])
    instruction = environment["instructions"].get(task, "")
    group = f"{SOURCE}:{task}"
    return {
        "episode_id": group,
        "task": "mcp",  # tool-call layout and judge of the AgentWorldBench mcp domain
        "source": SOURCE,
        # ``env_id`` on purpose: ``environment_id`` is a construction-scope selector in ``reconstruct``.
        "env_id": env_id,
        "benchmark_task": task,
        "trajectory_id": f"{SOURCE}-{task}",  # no colon: the runner names session directories after it
        "trajectory_group": group,
        "split": split or split_of(group),
        "agent": rollout.get("model"),
        "instruction": instruction,
        "turns": len(turns),
        "task_complete": any(str((step.get("action") or {}).get("name")) in AGENT_SIDE_TOOLS
                             or _collector_completion(step)
                             for step in rollout.get("trajectory") or []),
        "outcome": {key: rollout.get(key) for key in ("reward", "label", "checklist_pass", "checklist_total",
                                                       "terminated", "truncated")},
        "metrics": {key: rollout.get(key) for key in ("steps", "elapsed_s", "attempts_used", "backend")},
        "tool_names": sorted({turn["name"] for turn in turns}),
        "tool_definitions": None,
        "source_path": str(path.resolve()),
        "events": _events(instruction, turns),
    }


def rollout_episode(path: str | Path, *, split: str | None = None, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """An EnvScaler rollout as an episode JSON that ``load_raw_traces`` ingests unchanged; no ground truth inside."""
    rollout = read_json(Path(path))
    if not is_rollout(rollout):
        raise ValueError(f"{path} is not an EnvScaler rollout (expected env_id, task_id, and a trajectory list)")
    return _episode(Path(path), rollout, rollout_turns(rollout), split, environment or load_environment(None, str(rollout["env_id"])))


def _ground_truth(rollout: dict[str, Any], turns: list[dict[str, Any]], episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode["episode_id"],
        "trajectory_id": episode["trajectory_id"],
        "env_id": episode["env_id"],
        "benchmark_task": episode["benchmark_task"],
        "split": episode["split"],
        "source_path": episode["source_path"],
        "initial_state": rollout.get("init_state"),
        "final_state": rollout.get("final_state"),
        "turns": [{"turn": index, "step": turn["step"], "tool": turn["name"],
                   "outcome": OUTCOMES.get(str(turn["truth"]["outcome_label"]), "unknown"), **turn["truth"]}
                  for index, turn in enumerate(turns, start=1)],
    }


def ground_truth(path: str | Path, episode: dict[str, Any]) -> dict[str, Any]:
    """The simulator-side record of an episode, indexed by the episode's turns: the initial and final database,
    and per turn the outcome label, the database diff, and any loader error. Evaluation only."""
    rollout = read_json(Path(path))
    return _ground_truth(rollout, rollout_turns(rollout), episode)


def export_rollouts(roots: Iterable[str | Path], output_dir: str | Path, *, env_defs: str | Path | None = None,
                    split: str | None = None, ground_truth_dir: str | Path | None = None
                    ) -> tuple[list[Path], SplitManifest, list[dict[str, Any]]]:
    """Write one episode per rollout plus the split manifest; tool definitions go to ``tools/<env>.json``.

    ``split`` assigns every exported rollout to that split (the caller decides which directory plays
    which role); without it the trajectory-group hash decides, as for the other exporters. With
    ``ground_truth_dir`` each episode also gets a ground-truth file there. That directory may be a
    subdirectory of the output (episode discovery is not recursive) but not the output itself, where
    every JSON file is taken for an episode.
    """
    output = Path(output_dir)
    truth_dir = Path(ground_truth_dir) if ground_truth_dir is not None else None
    if truth_dir is not None and truth_dir.resolve() == output.resolve():
        raise ValueError("ground truth must not be written into the episode directory: its files would be ingested as traces")
    output.mkdir(parents=True, exist_ok=True)
    environments: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    assignments: dict[str, SplitAssignment] = {}
    summaries: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for source, rollout in iter_rollouts(roots):
        env_id = str(rollout["env_id"])
        if env_id not in environments:
            environments[env_id] = load_environment(env_defs, env_id)
        environment = environments[env_id]
        turns = rollout_turns(rollout)
        episode = _episode(source, rollout, turns, split, environment)
        task = episode["benchmark_task"]
        if task in seen:
            raise ValueError(f"Task {task} was supplied twice ({seen[task]} and {source}); export one rollout per task")
        seen[task] = source
        if not turns:
            summaries.append({"env_id": env_id, "task": task, "turns": 0, "skipped": "no environment turns"})
            continue
        if environment["tools"] is not None:
            tools_path = output / "tools" / f"{env_id}.json"
            if not tools_path.exists():
                write_json(tools_path, environment["tools"])
            episode["tool_definitions"] = str(tools_path.relative_to(output))
        path = output / f"{task}.json"
        write_json(path, episode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assignments[f"src_{digest}"] = SplitAssignment(split=episode["split"], trajectory_group=episode["trajectory_group"])
        paths.append(path)
        if truth_dir is not None:
            write_json(truth_dir / f"{task}.json", _ground_truth(rollout, turns, episode))
        labels = [str(turn["truth"]["outcome_label"]) for turn in turns]
        summaries.append({"env_id": env_id, "task": task, "split": episode["split"], "group": episode["trajectory_group"],
                          "turns": len(turns), "reward": rollout.get("reward"), "label": rollout.get("label"),
                          "effect_turns": labels.count("success-effect"), "rejected_turns": labels.count("rejected"),
                          "harness_errors": sum(1 for turn in turns if turn["harness_error"]),
                          "stale_raw_observations": sum(1 for turn in turns if turn["stale_raw"]),
                          "instruction": bool(episode["instruction"]), "tools": len(episode["tool_names"]), "path": str(path)})
    return paths, SplitManifest(assignments=assignments), summaries


# ─── Domain description draft ─────────────────────────────────────────────────

def database(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """The record collections of a state snapshot, without the environment's bookkeeping entries."""
    return None if state is None else {key: value for key, value in state.items() if key not in CONFIG_KEYS}


def state_layout(states: Iterable[dict[str, Any] | None]) -> dict[str, list[str]]:
    """Record collections of the environment database and the field names their records carry, in first-seen order."""
    layout: dict[str, list[str]] = {}
    for state in states:
        for collection, records in (database(state) or {}).items():
            fields = layout.setdefault(str(collection), [])
            for record in (records.values() if isinstance(records, dict) else []):
                for key in (record if isinstance(record, dict) else {}):
                    if key not in fields:
                        fields.append(str(key))
    return layout


def domain_description(environment: dict[str, Any], layout: dict[str, list[str]], tool_names: Iterable[str]) -> str:
    """A draft of the domain description ``reconstruct`` takes: what the environment is, how actions were
    normalized, and a suggested state vocabulary mirroring the database layout.

    It uses the dataset's own summary and introduction, the tool names, and the collection and field names
    of the given databases. It deliberately leaves out what reconstruction is meant to recover: the tools'
    docstrings, the environment's constraint list, and its source code.
    """
    env_id = environment["env_id"]
    lines = [f"Environment: {environment['summary'] or env_id} (EnvScaler `{env_id}`). "
             f"{' '.join(environment['introduction'].split())}".rstrip(),
             "",
             "A task agent operates the system through tool calls, one at a time. It never sees the database behind "
             "the tools; it only issues calls and reads what they return. Each episode is one agent run on one task, "
             "and every task starts from its own database contents.",
             "",
             "Actions. One action = one tool call: the action type is the tool name exactly as recorded and the "
             "arguments are the call's JSON arguments. Tools seen in these traces: "
             + ", ".join(f"`{name}`" for name in sorted(set(tool_names))) + ". The agent's closing message to its user "
             "(`chat_with_user`) is not an environment interaction and is not part of the traces.",
             "",
             "Observations. The tool's return value, recorded verbatim as the text the agent received: the Python "
             "literal form of a dict (single-quoted strings, `True` / `False` / `None`) with a boolean `success` "
             "key. Keep that surface form exactly; which other keys appear, and the wording of messages and errors, "
             "is tool behavior to be learned from the traces.",
             "",
             "State that matters. The environment is a small database: one map per record collection, keyed by "
             "record id, each record an object with the fields listed below. Use exactly these paths:"]
    for collection, fields in layout.items():
        shape = "{" + ", ".join(fields) + "}" if fields else "object (no records observed)"
        lines.append(f"- world.{collection} (object): record id -> {shape}")
    lines += ["A write tool changes single records: address one as world.<collection>.<record id> (or a field below it) "
              "and use op set for a new record or a changed field, op delete for a removed record, op merge to change "
              "several fields of one record; never replace a whole collection. A read tool returns records or "
              "fields of the current database and changes nothing. A rejected call (`'success': False`) changes nothing.",
              "",
              "Scope. Record ids, names, and values are facts about one task's database, not universal behavior. "
              "What is shared across episodes is each tool's behavior: which arguments it needs, the checks it makes "
              "and the error it returns when one fails, which records and fields it writes, how it words "
              "confirmations, and what a later read returns after an earlier write in the same episode. Ids of "
              "records created during an episode are generated by the environment and differ between runs."]
    return "\n".join(lines) + "\n"


# ─── Which turns are evaluated ────────────────────────────────────────────────

UNIQUE_LEVELS = ("episode", "environment", "answer")
REPEATED_CALL_RULES = ("last", "same-answer")


def unique_turns(episodes: Iterable[dict[str, Any]], level: str = "environment", repeated: str = "last"
                 ) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """The turns of one environment's held-out episodes that get an evaluation row, and why the others do not.

    Only the *evaluated* turn is filtered: a kept row still carries the full recorded history, dropped turns
    included. Criteria, applied in this order, each recorded per dropped turn:

    * ``loader_artifact`` — the call named a tool the environment does not have; the recorded ``None`` is the
      collector's answer, not the environment's.
    * ``repeated_action`` (every level) — the same tool call (name and arguments) occurs again later in the
      episode. ``repeated="last"`` keeps only the last occurrence. In the collection that discards mostly
      unique data: 34 of the 40 dropped occurrences have a *different* answer than the kept one (13 are
      rejections of a call that succeeds once the state has changed, 21 are reads before a write).
      ``repeated="same-answer"`` drops an occurrence only when a later occurrence of the call returns the same
      answer, so every distinct (call, answer) of an episode keeps a row. Either way, a kept row whose answer an
      earlier occurrence already gave is listed under ``kept_with_answer_in_own_history`` (its target can be
      copied from its own history).
    * ``same_call_and_answer_elsewhere`` (``environment``, ``answer``) — another held-out episode of the
      environment has the same call with the same answer: one is kept.
    * ``same_tool_and_answer_elsewhere`` (``answer``) — the same tool returns the identical answer text
      somewhere else in the environment (a fixed confirmation or error), whatever the arguments: one is kept.
      This is a much stronger cut: it removes rows whose inputs differ.

    "One is kept" means the occurrence with the longest history (largest turn), then the last task in name
    order, so the choice is deterministic. Returns ``{trajectory_id: kept turns}`` and the report.
    """
    if level not in UNIQUE_LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {UNIQUE_LEVELS}")
    if repeated not in REPEATED_CALL_RULES:
        raise ValueError(f"unknown rule for repeated calls {repeated!r}; expected one of {REPEATED_CALL_RULES}")
    episodes = list(episodes)
    if len({episode.get("env_id") for episode in episodes}) > 1:
        raise ValueError("unique_turns compares the episodes of one environment; pass them per environment")
    turns: list[dict[str, Any]] = []
    for episode in episodes:
        observations = {event["metadata"]["turn"]: event for event in episode["events"] if event["kind"] == "observation"}
        for event in (item for item in episode["events"] if item["kind"] == "action"):
            turn = event["metadata"]["turn"]
            observation = observations.get(turn) or {"content": "", "metadata": {}}
            turns.append({"trajectory_id": episode["trajectory_id"], "benchmark_task": episode.get("benchmark_task"), "turn": turn,
                          "tool": event["content"]["type"], "answer": observation["content"],
                          "call": json.dumps({"name": event["content"]["type"], "arguments": event["content"].get("arguments") or {}},
                                             sort_keys=True, ensure_ascii=False),
                          "artifact": bool(observation["metadata"].get("harness_error"))})
    dropped: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []

    def drop(item: dict[str, Any], reason: str, kept: dict[str, Any] | None = None) -> None:
        dropped.append({"benchmark_task": item["benchmark_task"], "turn": item["turn"], "tool": item["tool"], "reason": reason,
                        **({"kept": {"benchmark_task": kept["benchmark_task"], "turn": kept["turn"]}} if kept else {})})

    remaining = []
    for item in turns:
        if item["artifact"]:
            drop(item, "loader_artifact")
        else:
            remaining.append(item)
    by_call: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for item in remaining:
        # One group per call, or per call and answer: within a group only the last occurrence is kept.
        group = (item["trajectory_id"], item["call"]) + ((item["answer"],) if repeated == "same-answer" else ())
        by_call.setdefault(group, []).append(item)
    remaining = []
    for occurrences in by_call.values():
        last = max(occurrences, key=lambda item: item["turn"])
        for item in occurrences:
            if item is not last:
                drop(item, "repeated_action", last)
        if any(item is not last and item["answer"] == last["answer"] for item in occurrences):
            copied.append({"benchmark_task": last["benchmark_task"], "turn": last["turn"], "tool": last["tool"]})
        remaining.append(last)

    def keep_one(items: list[dict[str, Any]], key, reason: str) -> list[dict[str, Any]]:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for item in items:
            groups.setdefault(key(item), []).append(item)
        kept_items = []
        for group in groups.values():
            best = max(group, key=lambda item: (item["turn"], str(item["benchmark_task"])))
            for item in group:
                if item is not best:
                    drop(item, reason, best)
            kept_items.append(best)
        return kept_items

    if level in ("environment", "answer"):
        remaining = keep_one(remaining, lambda item: (item["call"], item["answer"]), "same_call_and_answer_elsewhere")
    if level == "answer":
        remaining = keep_one(remaining, lambda item: (item["tool"], item["answer"]), "same_tool_and_answer_elsewhere")
    kept: dict[str, list[int]] = {episode["trajectory_id"]: [] for episode in episodes}
    for item in remaining:
        kept[item["trajectory_id"]].append(item["turn"])
    kept = {trajectory: sorted(items) for trajectory, items in kept.items()}
    reasons: dict[str, int] = {}
    for item in dropped:
        reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    kept_tasks = {(item["benchmark_task"], item["turn"]) for item in remaining}
    report = {"level": level, "repeated_calls": repeated, "env_id": episodes[0].get("env_id") if episodes else None,
              "turns": len(turns), "kept": len(remaining),
              "dropped_by_reason": reasons,
              "kept_with_answer_in_own_history": sorted((item for item in copied if (item["benchmark_task"], item["turn"]) in kept_tasks),
                                                        key=lambda item: (str(item["benchmark_task"]), item["turn"])),
              "dropped": sorted(dropped, key=lambda item: (str(item["benchmark_task"]), item["turn"]))}
    return kept, report


# ─── Benchmark-format rows ────────────────────────────────────────────────────

def _turn_prompt(index: int, action: dict[str, Any], initial_state: dict[str, Any] | None) -> str:
    """The benchmark's turn prompt: every turn shows its action; an opt-in ablation may show the
    episode's initial database under ``**Current State:**`` before the first action."""
    state = ""
    if index == 1 and initial_state is not None:
        state = f"**Current State:**\n{json.dumps(database(initial_state), ensure_ascii=False)}\n\n"
    body = json.dumps({"name": action.get("type"), "arguments": action.get("arguments") or {}}, ensure_ascii=False, indent=2)
    return f"### Turn {index}\n{state}**Action:**\n```json\n{body}\n```"


def episode_rows(episode: dict[str, Any], *, tool_definitions: list[dict[str, Any]] | None = None,
                 initial_state: dict[str, Any] | None = None, turns_per_trajectory: int | None = None,
                 system_prompt: str | None = None, examples: list[dict[str, Any]] | None = None,
                 turns: Iterable[int] | None = None, generated_shapes: list[str] | None = None) -> list[dict[str, Any]]:
    """AgentWorldBench-format records for an exported episode (mcp layout, official instruction suffix).

    By default the state is latent, as in the benchmark's own MCP rows. When ``initial_state`` is
    explicitly supplied, its record collections (``database``) appear in the first turn's
    ``**Current State:**`` section as an initialization-visible ablation.
    ``examples`` (``few_shot_examples`` of construction episodes) fill the system prompt's few-shot section,
    which every official mcp record has. ``turns`` restricts which turns are evaluated (``unique_turns``); the
    history of a row is always the full recorded history. ``generated_shapes`` (``load_environment``) tells the
    scorer which generated values the environment can produce.
    """
    prompts, responses = recorded_turns(episode, initial_state)
    total = len(prompts)
    if not total:
        return []
    candidates = list(range(1, total + 1)) if turns is None else sorted(turn for turn in set(turns) if 1 <= turn <= total)
    if turns_per_trajectory and turns_per_trajectory < len(candidates):
        picks = sorted({max(1, round(i * len(candidates) / turns_per_trajectory)) for i in range(1, turns_per_trajectory + 1)})
        chosen = [candidates[pick - 1] for pick in picks]
    else:
        chosen = candidates
    if system_prompt is None:
        system_prompt = world_model_system_prompt(tool_definitions, episode.get("tool_names", []), examples)
    else:
        system_prompt = sanitize_evaluation_system_prompt(system_prompt)
    return [benchmark_row(episode, prompts, responses, turn, system_prompt, generated_shapes=generated_shapes) for turn in chosen]


def recorded_turns(episode: dict[str, Any], initial_state: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """The episode's turn prompts and marked responses, in order (the teacher-forced history of every row)."""
    actions = [event for event in episode["events"] if event["kind"] == "action"]
    observations = {event["metadata"]["turn"]: event["content"] for event in episode["events"] if event["kind"] == "observation"}
    prompts = [_turn_prompt(event["metadata"]["turn"], event["content"], initial_state) for event in actions]
    responses = [f"{RESPONSE_MARKER}\n{observations.get(event['metadata']['turn'], '')}" for event in actions]
    return prompts, responses


FEW_SHOT_HEADER = ("\n---\n\n# Few-shot Examples\n\nBelow are examples of interactions for the tools available in this "
                   "environment:\n\n")


def few_shot_examples(episodes: Iterable[dict[str, Any]], tool_definitions: list[dict[str, Any]] | None = None
                      ) -> list[dict[str, Any]]:
    """One recorded interaction per tool, as the benchmark's mcp records carry in their system prompt.

    Every official mcp record ends its system prompt with a ``# Few-shot Examples`` section: one real
    action / observation pair for each tool that has one (a median of 15 per record, 70% of the defined tools),
    in the order of the tool definitions. Here the pair is the first call of the tool in the given episodes, in
    the order given, whatever its outcome. The episodes must be construction episodes (``split: train``): an
    example taken from a held-out task would show the evaluated environment instance to the model (in the
    official records 35 of 286 evaluated turns appear verbatim among their own examples; that is not copied).
    """
    found: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        if episode.get("split") != "train":
            raise ValueError(f"few-shot examples come from construction episodes only; {episode.get('episode_id')} is {episode.get('split')!r}")
        observations = {event["metadata"]["turn"]: event["content"] for event in episode["events"] if event["kind"] == "observation"}
        for event in episode["events"]:
            if event["kind"] != "action" or event["content"]["type"] in found:
                continue
            found[event["content"]["type"]] = {"name": event["content"]["type"], "arguments": event["content"].get("arguments") or {},
                                               "observation": observations.get(event["metadata"]["turn"], ""),
                                               "episode_id": episode.get("episode_id")}
    order = [definition["name"] for definition in tool_definitions or []]
    return sorted(found.values(), key=lambda item: (order.index(item["name"]) if item["name"] in order else len(order), item["name"]))


def limit_few_shot_examples(examples: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Deterministically retain at most ``limit`` examples across the ordered tool list.

    Even spacing avoids making a small-example ablation depend only on the tools that happen to
    occur first in the environment definition. ``None`` preserves the benchmark-compatible full
    set, while zero removes the few-shot section entirely.
    """
    if limit is None or limit >= len(examples):
        return list(examples)
    if limit < 0:
        raise ValueError("few-shot limit must be nonnegative")
    if limit == 0:
        return []
    if limit == 1:
        return [examples[0]]
    indexes = [round(i * (len(examples) - 1) / (limit - 1)) for i in range(limit)]
    return [examples[index] for index in indexes]


def render_few_shot_examples(examples: list[dict[str, Any]]) -> str:
    """The ``# Few-shot Examples`` section in the layout of the official mcp records (empty without examples)."""
    if not examples:
        return ""
    blocks = [f"## Example: {item['name']}\n**Action:**\n```json\n"
              + json.dumps({"name": item["name"], "arguments": item["arguments"]}, ensure_ascii=False, indent=2)
              + f"\n```\n{RESPONSE_MARKER}\n{item['observation']}\n\n" for item in examples]
    return FEW_SHOT_HEADER + "".join(blocks)


def world_model_system_prompt(tool_definitions: list[dict[str, Any]] | None, tool_names: Iterable[str] = (),
                              examples: list[dict[str, Any]] | None = None) -> str:
    """The official mcp world-model system prompt with the environment's tool definitions and few-shot examples."""
    template = load_system_prompt_template("mcp")
    system_prompt = template.replace("{tool_definitions}", render_tool_definitions(tool_definitions, tool_names))
    if examples:
        return system_prompt.replace("{demonstrations}", render_few_shot_examples(examples))
    return system_prompt.replace("{demonstrations}", "").rstrip() + "\n"


def benchmark_row(episode: dict[str, Any], prompts: list[str], responses: list[str], turn: int, system_prompt: str, *,
                  current: tuple[str, str] | None = None, generated_shapes: list[str] | None = None) -> dict[str, Any]:
    """One record evaluating ``turn`` after the recorded turns before it. ``current`` = ``(prompt, response)``
    replaces the recorded turn with another action at the same point (a probe); the history stays recorded."""
    prompt, response = current if current is not None else (prompts[turn - 1], responses[turn - 1])
    return {
        "task": "mcp",
        "id": episode["trajectory_id"],
        "prompt": prompts[:turn - 1] + [prompt + BENCHMARK_SUFFIX],
        "response": responses[:turn - 1] + [response],
        "current_prompt": prompt,
        "system_str": system_prompt,
        "turn_idx": turn,
        "total_turns": len(prompts),
        "benchmark_task": episode.get("benchmark_task"),
        "source": episode.get("source"),
        "env_id": episode.get("env_id"),
        "split": episode.get("split"),
        # Evaluation-side, never shown to a model: which generated values the scorer may accept by shape.
        **({} if generated_shapes is None else {"generated_shapes": list(generated_shapes)}),
    }
