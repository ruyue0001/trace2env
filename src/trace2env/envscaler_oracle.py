"""EnvScaler oracle: the environment's own source as ground truth for counterfactual probe turns. Evaluation only.

Recorded EnvScaler rollouts solved their tasks, so they walk the environment's success paths: the four
collected environments have 149 rejection branches in code and the held-out rollouts reach 16 of them,
while most recorded observations can be copied from the prompt (the database, a docstring, an earlier
turn). A turn-wise test on those turns alone is near its ceiling for a strong model. The environment
definitions include the executable class, so harder turns with exact ground truth are cheap: at a
recorded state of a held-out rollout, run a different action (a recorded call with one argument
perturbed, or another recorded call of the same episode) through the real code.

``EnvironmentOracle`` **executes Python source taken from an environment definition file**
(``env_class_code``). Run it only on definitions you trust; the collected ones import ``typing``,
``uuid``, ``datetime``, and ``time`` and touch no file, process, or network. It is the answer key: it
serves evaluation (probe rows, their ground truth) and nothing here may feed reconstruction.

Two safeguards. ``replay_check`` re-executes every recorded step from its recorded state and compares
observation and next state with the recording; ``trace2env envscaler-probe`` generates nothing unless
every step is reproduced (``reproduces``). A probe whose execution raises is discarded: what the agent
would have received then depends on the collector's loader, which the recordings show only once.
Values the environment generates (``uuid4`` ids, clocks) come from a seeded source so an export is
reproducible; they stay unpredictable for a simulator and ``envscaler_scoring`` treats them so. The
element order of a list an environment builds from a ``set`` follows the interpreter's hash seed; it
is compared without order.
"""

from __future__ import annotations

import ast
import copy
import datetime as _datetime
import json
import random
import re
import time as _time
import uuid as _uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from trace2env.agentworld import RESPONSE_MARKER
from trace2env.envscaler import (OUTCOMES, _turn_prompt, benchmark_row, database, recorded_turns, rollout_turns, tool_definitions,
                                 world_model_system_prompt)
from trace2env.envscaler_scoring import kind_of
from trace2env.models import NormalizedAction

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CLOCK = _datetime.datetime(2026, 1, 1, 12, 0, 0)  # what the environment's clock reads during probe generation
ENUM_FIELD_VALUES = 12  # a field with at most this many distinct values in a database is treated as enum-like
# Probe selection prefers what the recorded turns lack: rejections, then calls that change the database, then the rest.
LABEL_ORDER = {"rejected": 0, "success-effect": 1, "success-noop": 2}
# Among candidates that reach the same outcome, a call an agent could plausibly make (a recorded call at
# another moment, a valid id or value in the wrong place) is preferred to a malformed one.
OPERATOR_ORDER = {"replayed_call": 0, "other_record": 0, "other_value": 0, "flipped": 0, "missing_record": 1, "missing_element": 1,
                  "invalid_value": 1, "repeated_element": 1, "zero": 2, "negative": 2, "empty_list": 2, "empty": 3}


# ─── Executing the environment ────────────────────────────────────────────────

class _SeededUuid:
    def __init__(self, seed: int):
        self._random = random.Random(seed)

    def uuid4(self) -> _uuid.UUID:
        return _uuid.UUID(int=self._random.getrandbits(128), version=4)

    def __getattr__(self, name: str) -> Any:
        return getattr(_uuid, name)


class _FixedDatetime(_datetime.datetime):
    @classmethod
    def now(cls, tz: Any = None) -> "_FixedDatetime":
        return cls(CLOCK.year, CLOCK.month, CLOCK.day, CLOCK.hour, CLOCK.minute, CLOCK.second, tzinfo=tz)

    @classmethod
    def utcnow(cls) -> "_FixedDatetime":
        return cls.now()

    @classmethod
    def today(cls) -> "_FixedDatetime":
        return cls.now()


class _FixedTime:
    def time(self) -> float:
        return CLOCK.replace(tzinfo=_datetime.timezone.utc).timestamp()

    def __getattr__(self, name: str) -> Any:
        return getattr(_time, name)


class _ModuleShim:
    """A module with some attributes replaced (``import datetime`` style access to the fixed clock)."""

    def __init__(self, module: Any, **replaced: Any):
        self._module, self._replaced = module, replaced

    def __getattr__(self, name: str) -> Any:
        return self._replaced[name] if name in self._replaced else getattr(self._module, name)


class EnvironmentOracle:
    """The environment class of one definition file, stepped from explicit states (one action, no episode)."""

    def __init__(self, metadata: dict[str, Any], *, seed: int = 0):
        namespace: dict[str, Any] = {"__name__": f"envscaler_{metadata['env_id']}"}
        exec(compile(metadata["env_class_code"], f"<{metadata['env_id']} env_class_code>", "exec"), namespace)  # noqa: S102
        # Generated values: replace what the source imported, inside its own namespace only, whichever
        # import style it used (``import uuid`` / ``from uuid import uuid4``, and likewise for the clocks).
        seeded, clock = _SeededUuid(seed), _FixedTime()
        replacements = {_uuid: seeded, _uuid.uuid4: seeded.uuid4, _datetime.datetime: _FixedDatetime, _time: clock,
                        _time.time: clock.time}
        for name, value in list(namespace.items()):
            if value is _datetime:
                namespace[name] = _ModuleShim(_datetime, datetime=_FixedDatetime)
            elif any(value is original for original in replacements):
                namespace[name] = next(new for original, new in replacements.items() if value is original)
        self.env_id = str(metadata["env_id"])
        self.environment_class = namespace[metadata["env_class_name"]]
        self.tools = {definition["name"]: definition for definition in tool_definitions(metadata)}

    def step(self, state: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call on a copy of ``state``: the observation text the agent would receive, the next
        state, and the exception line when the call raised (then no observation is claimed)."""
        environment = self.environment_class({})
        for key, value in copy.deepcopy(state).items():
            setattr(environment, key, value)
        try:
            result = getattr(environment, str(action.get("name")))(**(action.get("arguments") or {}))
        except Exception as exc:  # noqa: BLE001 - any failure of the environment code disqualifies the step
            return {"observation": None, "state": state, "raised": f"{type(exc).__name__}: {exc}"}
        after = json.loads(json.dumps({key: getattr(environment, key) for key in state}, default=str))
        return {"observation": str(result), "state": after, "raised": None}


def state_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """The collection's diff format (``changed`` / ``added`` / ``removed`` at the first differing level)."""
    diff: dict[str, Any] = {}
    for key in list(before) + [key for key in after if key not in before]:
        if key not in after:
            diff[key] = {"removed": before[key]}
        elif key not in before:
            diff[key] = {"added": after[key]}
        elif before[key] != after[key]:
            if isinstance(before[key], dict) and isinstance(after[key], dict):
                diff[key] = state_diff(before[key], after[key])
            else:
                diff[key] = {"changed": {"old": before[key], "new": after[key]}}
    return diff


def canonical(value: Any) -> Any:
    """A value with generated parts neutralized: uuid and clock text masked, scalar lists unordered, and
    records under a generated key compared by content."""
    if isinstance(value, dict):
        # Keys too: a created record sits under its generated id (a uuid, or ``<user>:<group>:<epoch>``).
        items = [(canonical(key), canonical(item)) for key, item in value.items()]
        return sorted((str(key), json.dumps(item, sort_keys=True, default=str)) for key, item in items)
    if isinstance(value, list):
        items = [canonical(item) for item in value]
        scalars = all(not isinstance(item, (dict, list)) for item in value)
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str)) if scalars else items
    if isinstance(value, str):
        return re.sub(r"\d{4}-\d{2}-\d{2}T[\d:.]+|\b1[6-9]\d{8}(?:\.\d+)?\b", "<clock>", UUID.sub("<uuid>", value))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1.6e9:
        return "<clock>"
    return value


def replay_check(rollouts: Iterable[dict[str, Any]], oracle: EnvironmentOracle) -> dict[str, int]:
    """Re-execute every recorded tool call from its recorded state and compare with the recording."""
    counts: Counter[str] = Counter()
    for rollout in rollouts:
        for step in rollout.get("trajectory") or []:
            action = step.get("action") or {}
            if action.get("name") not in oracle.tools:
                continue  # the agent's own tools, and calls to tools the environment does not have
            counts["steps"] += 1
            result = oracle.step(step["state_before"], action)
            if result["raised"]:
                counts["raised"] += 1
                continue
            if result["observation"] == step.get("observation"):
                counts["observation_identical"] += 1
            elif canonical(result["observation"]) == canonical(step.get("observation")):
                counts["observation_identical_up_to_generated_values"] += 1
            else:
                counts["observation_differs"] += 1
            if canonical(result["state"]) == canonical(step.get("state_after")):
                counts["state_identical_up_to_generated_values"] += 1
            else:
                counts["state_differs"] += 1
    return dict(counts)


def reproduces(check: dict[str, int]) -> bool:
    return bool(check.get("steps")) and not any(check.get(key) for key in ("raised", "observation_differs", "state_differs"))


# ─── Probe actions ────────────────────────────────────────────────────────────

def _missing_like(value: str, taken: Iterable[str]) -> str:
    """An id of the same shape that names no record."""
    taken = set(taken)
    match = re.match(r"^(.*?)(\d+)$", value)
    if match:
        candidate = f"{match.group(1)}{int(match.group(2)) + 900}"
    elif UUID.fullmatch(value):
        candidate = value[:-4] + "0000"
    else:
        candidate = value + "-x"
    while candidate in taken or candidate == value:
        candidate += "9"
    return candidate


def perturbations(state: dict[str, Any], action: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """``(operator, action)`` variants of a recorded call with exactly one argument changed.

    An argument naming a record gets another record of that collection and an id that names none; an
    argument carrying a value of an enum-like field gets that field's other values and an unknown one;
    strings also go empty; lists go empty, repeat an element, and gain an unknown element; booleans flip;
    numbers become 0 and -1.
    """
    arguments = action.get("arguments") or {}
    collections = {name: records for name, records in (database(state) or {}).items() if isinstance(records, dict)}
    field_values: dict[str, set[str]] = {}
    for records in collections.values():
        for record in records.values():
            for field, item in (record.items() if isinstance(record, dict) else []):
                if isinstance(item, str):
                    field_values.setdefault(field, set()).add(item)
    variants: list[tuple[str, dict[str, Any]]] = []
    for name, value in arguments.items():
        changed: list[tuple[str, Any]] = []
        if isinstance(value, str):
            homes = [records for records in collections.values() if value in records]
            for records in homes:
                changed += [("other_record", key) for key in sorted(key for key in records if key != value)[:2]]
                changed.append(("missing_record", _missing_like(value, records)))
            if not homes:
                for field in sorted(field_values):
                    values = field_values[field]
                    if value in values and len(values) <= ENUM_FIELD_VALUES:
                        changed += [("other_value", item) for item in sorted(values - {value})[:3]]
                        changed.append(("invalid_value", "invalid_value"))
            changed.append(("empty", ""))
        elif isinstance(value, bool):
            changed.append(("flipped", not value))
        elif isinstance(value, (int, float)):
            changed += [("zero", 0), ("negative", -1)]
        elif isinstance(value, list):
            changed += [("empty_list", []), ("repeated_element", value + value[:1])]
            if all(isinstance(item, str) for item in value):
                changed.append(("missing_element", value + ["MISSING-ID"]))
        for operator, replacement in changed:
            if replacement != value:
                variants.append((operator, {"name": action["name"], "arguments": {**arguments, name: replacement}}))
    return variants


def outcome_signature(observation: Any, action: dict[str, Any], state: dict[str, Any]) -> tuple[str, str, str]:
    """``(tool, kind, wording)`` of an outcome: the response shape (``envscaler_scoring.kind_of``: data / message /
    error / other) and its message or error text with argument values, record ids, and quoted values blanked."""
    if not isinstance(observation, dict):
        return str(action["name"]), "other", ""
    kind = kind_of(observation)
    text = str(observation.get("error") or observation.get("message") or "")
    values: set[str] = {key for records in (database(state) or {}).values() if isinstance(records, dict) for key in records}

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)
        elif item not in (None, ""):
            values.add(str(item))

    collect(action.get("arguments"))
    for value in sorted(values, key=len, reverse=True):
        if len(value) >= 2:
            text = text.replace(value, "{}")
    text = re.sub(r"\[[^\]]*\]", "{}", re.sub(r"'[^']*'", "{}", UUID.sub("{}", text)))
    return str(action["name"]), kind, re.sub(r"\{\}(?:\s*\{\})+", "{}", text).strip()


def literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def probe_rows(rollout: dict[str, Any], episode: dict[str, Any], oracle: EnvironmentOracle, *, system_prompt: str,
               initial_state: dict[str, Any] | None = None, per_trajectory: int = 10,
               coverage: Counter | None = None, generated_shapes: list[str] | None = None
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Counterfactual rows for one held-out rollout, and their ground truth.

    At every recorded turn the candidates are the perturbations of that turn's call and the other
    recorded calls of the episode, run from the recorded state before the turn. Selection is greedy and
    deterministic: the outcome signature (tool, kind, wording) chosen least often so far in this
    environment (``coverage``, shared across rollouts) wins, rejections before calls that change the database
    before the rest (by the true effect of the call, not by the shape of its response), a
    plausible call before a malformed one (``OPERATOR_ORDER``), at most one row per signature and turn.
    The history of a probe row is the recorded history.
    """
    coverage = coverage if coverage is not None else Counter()
    steps = [step for step in rollout.get("trajectory") or [] if (step.get("action") or {}).get("name") in oracle.tools]
    turns = rollout_turns(rollout)
    if len(turns) != episode["turns"]:
        raise ValueError(f"episode {episode['episode_id']} does not match its rollout")
    by_step = {turn["step"]: index for index, turn in enumerate(turns, start=1)}
    recorded = [step["action"] for step in steps]
    candidates: list[dict[str, Any]] = []
    for step in steps:
        turn = by_step[step.get("step")]
        pool = perturbations(step["state_before"], step["action"]) + [("replayed_call", action) for action in recorded]
        seen: set[str] = {json.dumps(step["action"], sort_keys=True)}
        for operator, action in pool:
            key = json.dumps(action, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            result = oracle.step(step["state_before"], action)
            if result["raised"]:
                continue
            signature = outcome_signature(literal(result["observation"]), action, step["state_before"])
            diff = state_diff(step["state_before"], result["state"])
            label = "rejected" if signature[1] == "error" else "success-effect" if diff else "success-noop"
            candidates.append({"turn": turn, "operator": operator, "action": action, "key": key, "signature": signature,
                               "observation": result["observation"], "state_diff": diff, "label": label})
    chosen: list[dict[str, Any]] = []
    while candidates and len(chosen) < per_trajectory:
        candidates.sort(key=lambda item: (coverage[item["signature"]], LABEL_ORDER[item["label"]],
                                          OPERATOR_ORDER.get(item["operator"], 1), item["turn"], item["key"]))
        best = candidates.pop(0)
        coverage[best["signature"]] += 1
        chosen.append(best)
        candidates = [item for item in candidates if (item["turn"], item["signature"]) != (best["turn"], best["signature"])]
    prompts, responses = recorded_turns(episode, initial_state)
    rows, truth = [], []
    for number, probe in enumerate(sorted(chosen, key=lambda item: (item["turn"], item["key"])), start=1):
        action = NormalizedAction(type=probe["action"]["name"], arguments=probe["action"].get("arguments") or {})
        prompt = _turn_prompt(probe["turn"], action.model_dump(mode="json"), initial_state)
        row = benchmark_row(episode, prompts, responses, probe["turn"], system_prompt,
                            current=(prompt, f"{RESPONSE_MARKER}\n{probe['observation']}"), generated_shapes=generated_shapes)
        row["probe"] = f"{episode['benchmark_task']}:t{probe['turn']:02d}:p{number}"
        rows.append(row)
        label = probe["label"]
        truth.append({"probe": row["probe"], "trajectory_id": episode["trajectory_id"], "turn": probe["turn"],
                      "operator": probe["operator"], "action": probe["action"], "observation": probe["observation"],
                      "outcome": OUTCOMES[label], "outcome_label": label, "state_diff": probe["state_diff"],
                      "signature": {"tool": probe["signature"][0], "kind": probe["signature"][1], "wording": probe["signature"][2]}})
    return rows, truth


def load_metadata(env_defs: str | Path, env_id: str) -> dict[str, Any]:
    path = Path(env_defs) / f"{env_id}_metadata.json"
    if not path.is_file():
        raise ValueError(f"no environment definition at {path}; the oracle needs the environment's source")
    return json.loads(path.read_text(encoding="utf-8"))


def system_prompt_for(oracle: EnvironmentOracle, examples: list[dict[str, Any]] | None = None) -> str:
    return world_model_system_prompt(list(oracle.tools.values()), examples=examples)
