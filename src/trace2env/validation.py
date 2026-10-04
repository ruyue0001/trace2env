"""Deterministic artifact checks; these establish structure, not causal truth."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from trace2env.engine import ACTION_TOKEN, MISSING, get_path

STATE_TOKEN = re.compile(r"\{state_(?:before|after)\.")  # template-only tokens; never valid inside an effect value
from trace2env.models import ActionSchema, ActionSpec, ReconstructionArtifacts, SourceRef, SplitManifest, StateMutation, StateSchema

TYPES = {
    "string": lambda x: isinstance(x, str),
    "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
    "boolean": lambda x: isinstance(x, bool), "object": lambda x: isinstance(x, dict),
    "array": lambda x: isinstance(x, list), "any": lambda x: True,
}


def artifact_digest(artifacts: ReconstructionArtifacts) -> str:
    return hashlib.sha256(json.dumps(artifacts.model_dump(mode="json"), sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def config_digest(config) -> str:
    return hashlib.sha256(json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")).hexdigest()


def unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")


def validate_source_ref(ref: SourceRef, episodes: dict, *, require_event: bool = False) -> None:
    matching = [ep for ep in episodes.values() if ep.source_id == ref.source_id]
    if not matching:
        raise ValueError(f"Unknown provenance source: {ref.source_id}")
    if ref.episode_id is not None:
        matching = [ep for ep in matching if ep.id == ref.episode_id]
    if not matching:
        raise ValueError(f"Unknown provenance episode: {ref.episode_id}")
    if require_event and (ref.episode_id is None or not ref.event_ids):
        raise ValueError("Executable evidence requires an episode and event anchors")
    if ref.event_ids and not any(set(ref.event_ids) <= {e.id for e in ep.events} for ep in matching):
        raise ValueError("Provenance references unknown or cross-episode events")


def action_aliases(schema: ActionSchema) -> dict[str, str]:
    aliases = {}
    for action in schema.actions:
        for name in [action.name, *action.aliases]:
            if name in aliases and aliases[name] != action.name:
                raise ValueError(f"Ambiguous action alias: {name}")
            aliases[name] = action.name
    return aliases


def state_path_aliases(schema: StateSchema) -> dict[str, str]:
    """Alias path -> canonical path for every declared state field (a field is its own alias)."""
    aliases: dict[str, str] = {}
    for field in schema.fields:
        for name in [field.path, *field.aliases]:
            if name in aliases and aliases[name] != field.path:
                raise ValueError(f"Ambiguous state path alias: {name}")
            aliases[name] = field.path
    return aliases


def canonical_state_path(path: str, aliases: dict[str, str]) -> str:
    """Rewrite a path through its longest aliased prefix (``world.git.x`` -> ``world.repositories.x``)."""
    if path in aliases:
        return aliases[path]
    parts = path.split(".")
    for length in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:length])
        if prefix in aliases:
            return aliases[prefix] + "." + ".".join(parts[length:])
    return path


def argument_type(path: str, action: ActionSpec | None) -> str:
    name, _, tail = path.partition(".")
    if action is None or name not in action.arguments:
        raise ValueError(f"Undeclared action argument: {path}")
    spec = action.arguments[name]
    if tail and spec.type not in {"object", "array", "any"}:
        raise ValueError(f"Cannot traverse argument: {path}")
    return "any" if tail else spec.type


def state_field(path: str, schema: StateSchema, action: ActionSpec | None = None):
    for token in ACTION_TOKEN.findall(path):
        argument_type(token, action)
    normalized = ACTION_TOKEN.sub("__key__", path)
    if "{" in normalized or "}" in normalized or any(not part for part in normalized.split(".")):
        raise ValueError(f"Invalid state path: {path}")
    candidates = [f for f in schema.fields if f.path == normalized or (
        normalized.startswith(f.path + ".") and f.type in {"object", "array", "any"})]
    if not candidates:
        raise ValueError(f"Undeclared state path: {path}")
    return max(candidates, key=lambda f: len(f.path))


def value_type(value: Any, action: ActionSpec | None) -> str:
    if isinstance(value, dict):
        if "$action_arg" in value:
            if set(value) != {"$action_arg"}:
                raise ValueError("Dynamic argument values cannot contain extra fields")
            return argument_type(str(value["$action_arg"]), action)
        if "$state_path" in value:
            raise ValueError("State-value interpolation is not implemented")
        for key, item in value.items():
            for token in ACTION_TOKEN.findall(str(key)):  # dynamic map keys, e.g. a file path named by the action
                argument_type(token, action)
            value_type(item, action)
        return "object"
    if isinstance(value, list):
        for item in value:
            value_type(item, action)
        return "array"
    if isinstance(value, str):
        if STATE_TOKEN.search(value):
            # Template tokens belong in observation templates; an effect cannot read state values.
            raise ValueError("State-value interpolation is not implemented")
        for token in ACTION_TOKEN.findall(value):
            argument_type(token, action)
        full = ACTION_TOKEN.fullmatch(value)
        return argument_type(full.group(1), action) if full else "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "null"


def validate_mutations(mutations: list[StateMutation], schema: StateSchema, action: ActionSpec | None) -> None:
    for mutation in mutations:
        field = state_field(mutation.path, schema, action)
        normalized = ACTION_TOKEN.sub("__key__", mutation.path)
        protected = [f for f in schema.fields if not f.mutable and (
            normalized == f.path or normalized.startswith(f.path + ".") or f.path.startswith(normalized + "."))]
        if not field.mutable or protected:
            raise ValueError(f"Effect modifies immutable state: {mutation.path}")
        if mutation.op == "schedule":
            if mutation.delay_steps is None:
                raise ValueError("Scheduled effects require delay_steps")
            nested = mutation.value.get("effects") if isinstance(mutation.value, dict) else None
            if nested is not None:
                validate_mutations([StateMutation.model_validate(x) for x in nested], schema, action)
            else:
                validate_mutations([StateMutation(op="set", path=mutation.path, value=mutation.value)], schema, action)
            continue
        if mutation.op == "delete":
            continue
        actual = value_type(mutation.value, action)
        exact = field.path == mutation.path
        expected = field.type if exact else "any"
        if mutation.op in {"merge", "remove"}:
            if expected not in {"object", "any"}:
                raise ValueError(f"{mutation.op} requires object state: {mutation.path}")
            if mutation.op == "merge" and actual not in {"object", "any"}:
                raise ValueError(f"merge requires an object value: {mutation.path}")
            if mutation.op == "remove" and not (isinstance(mutation.value, str) or (
                    isinstance(mutation.value, list) and all(isinstance(key, str) for key in mutation.value))):
                raise ValueError(f"remove requires a key or a list of keys: {mutation.path}")
            continue
        if mutation.op in {"increment", "decrement"}:
            if expected not in {"integer", "number", "any"} or actual not in {"integer", "number", "any"}:
                raise ValueError(f"Non-numeric arithmetic effect: {mutation.path}")
            if expected == "integer" and actual == "number":
                raise ValueError(f"Fractional effect on integer state: {mutation.path}")
        elif mutation.op == "append":
            if expected not in {"array", "any"}:
                raise ValueError(f"Append requires array state: {mutation.path}")
        elif expected != "any" and actual != "any" and actual != expected and not (expected == "number" and actual == "integer"):
            raise ValueError(f"Effect type {actual} does not match {expected}: {mutation.path}")


def _render_path_issue(path: str, schema: StateSchema, action: ActionSpec | None) -> str | None:
    try:
        if path.startswith("state_before.") or path.startswith("state_after."):
            state_field(path.split(".", 1)[1], schema, action)
        elif path.startswith("action.arguments."):
            argument_type(path[len("action.arguments."):], action)
        elif path not in {"action.type", "outcome"}:
            return f"unsupported renderer reference: {path}"
    except ValueError as exc:
        return str(exc)
    return None


def rule_issues(rule, actions: dict[str, ActionSpec], schema: StateSchema, renderer_ids: set[str]) -> list[str]:
    """Static problems with one rule against the schema; empty means the rule is well-formed."""
    issues: list[str] = []
    action = actions.get(rule.action_type)
    if action is None:
        return [f"unknown action {rule.action_type!r}"]
    if rule.renderer not in renderer_ids:
        issues.append(f"unknown renderer {rule.renderer!r}")
    try:
        validate_mutations(rule.effects, schema, action)
    except ValueError as exc:
        issues.append(f"effects: {exc}")
    for condition in rule.conditions:
        for operand in (condition.left, condition.right):
            if operand is None:
                continue
            try:
                if operand.path is not None:
                    state_field(operand.path, schema, action)
                if operand.action_arg is not None:
                    argument_type(operand.action_arg, action)
            except ValueError as exc:
                issues.append(f"condition: {exc}")
    for path in re.findall(r"\{([^{}]+)\}", rule.observation_template or ""):
        problem = _render_path_issue(path, schema, action)
        if problem:
            issues.append(f"template: {problem}")
    return issues


def invalid_render_references(renderer, actions: dict[str, ActionSpec], schema: StateSchema, rules
                              ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """``(required_fields, template paths)`` of a contract that are not renderable references.

    Renderable references are ``state_before.*`` / ``state_after.*`` paths of the schema,
    ``action.arguments.<declared>``, ``action.type``, and ``outcome``. Induction often writes prose
    labels ("total row count") or invented ``outcome.*`` paths instead; the caller drops those and
    keeps the contract's instructions rather than quarantining the whole contract.
    """
    related = set(renderer.action_types) | {rule.action_type for rule in rules if rule.renderer == renderer.id}
    known = [actions[action_type] for action_type in related if action_type in actions]

    def problem(path: str) -> str | None:
        for action in known:
            issue = _render_path_issue(path, schema, action)
            if issue:
                return issue
        return None

    fields = [(path, issue) for path in renderer.required_fields if (issue := problem(path))]
    template = [(path, issue) for path in re.findall(r"\{([^{}]+)\}", renderer.template or "") if (issue := problem(path))]
    return fields, template


def renderer_issues(renderer, actions: dict[str, ActionSpec], schema: StateSchema, rules) -> list[str]:
    issues: list[str] = []
    related = set(renderer.action_types) | {rule.action_type for rule in rules if rule.renderer == renderer.id}
    for action_type in related:
        action = actions.get(action_type)
        if action is None:
            issues.append(f"unknown action {action_type!r}")
            continue
        for path in renderer.required_fields + re.findall(r"\{([^{}]+)\}", renderer.template or ""):
            problem = _render_path_issue(path, schema, action)
            if problem:
                issues.append(problem)
    return issues


def repair_provenance(refs: list[SourceRef], episodes: list, unresolved: list[str], owner: str) -> list[SourceRef]:
    """Re-anchor citations to the episodes their event ids actually belong to; drop what cannot be resolved.

    Induction models cite event ids they saw in evidence but sometimes attach them to the wrong
    episode or source, or invent ids. Every repair or drop is recorded in ``unresolved``.
    """
    event_episode = {event.id: episode for episode in episodes for event in episode.events}
    episodes_by_id = {episode.id: episode for episode in episodes}
    repaired: list[SourceRef] = []
    for ref in refs:
        known = [event_id for event_id in ref.event_ids if event_id in event_episode]
        dropped = [event_id for event_id in ref.event_ids if event_id not in event_episode]
        if dropped:
            unresolved.append(f"{owner}: dropped unknown event ids {dropped[:5]}")
        if not known:
            episode = episodes_by_id.get(ref.episode_id or "")
            if episode is not None and not ref.event_ids:
                repaired.append(ref.model_copy(update={"source_id": episode.source_id}))
            elif not ref.event_ids and any(ep.source_id == ref.source_id for ep in episodes):
                repaired.append(ref)
            else:
                unresolved.append(f"{owner}: dropped citation that resolves to no known episode or events")
            continue
        grouped: dict[str, list[str]] = {}
        for event_id in known:
            grouped.setdefault(event_episode[event_id].id, []).append(event_id)
        for episode_id, event_ids in grouped.items():
            episode = episodes_by_id[episode_id]
            if ref.episode_id != episode_id or ref.source_id != episode.source_id:
                unresolved.append(f"{owner}: re-anchored {len(event_ids)} event citation(s) to episode {episode_id}")
            repaired.append(SourceRef(source_id=episode.source_id, episode_id=episode_id, event_ids=event_ids,
                                      locator=ref.locator, excerpt=ref.excerpt))
    return repaired


def rule_has_observed_support(rule, evidence: list, transitions: dict) -> bool:
    """A supported rule needs an unambiguous, non-concurrent observed transition of its action among its citations."""
    for item in evidence:
        transition = transitions.get(item.transition_id)
        if transition is None or item.action.type != rule.action_type:
            continue
        if transition.alignment == "ambiguous" or transition.concurrent_action_event_ids:
            continue
        if any(ref.episode_id == item.episode_id and set(ref.event_ids) & set(transition.observation_event_ids)
               for ref in rule.provenance):
            return True
    return False


def validate_artifacts(artifacts: ReconstructionArtifacts, *, require_provenance: bool = True) -> None:
    # Revalidate even when a caller used model_copy(update=...) to assemble a candidate.
    ReconstructionArtifacts.model_validate(artifacts.model_dump(mode="python"))
    SplitManifest(assignments=artifacts.split_assignments)
    if artifacts.split_assignments:
        for ep in artifacts.episodes:
            assignment = artifacts.split_assignments.get(ep.source_id)
            if assignment is None or assignment.split != "train" or ep.metadata.get("trajectory_group") != assignment.trajectory_group:
                raise ValueError("Construction episode violates its split manifest")
    for values, label in [(artifacts.episodes, "episode IDs"), (artifacts.transitions, "transition IDs"),
                          (artifacts.evidence, "evidence IDs"), (artifacts.renderers, "renderer IDs"),
                          (artifacts.invariants, "invariant IDs"), (artifacts.demonstrations, "demonstration IDs"),
                          (artifacts.notes, "note IDs")]:
        unique([x.id for x in values], label)
    unique([a.name for a in artifacts.action_schema.actions], "action names")
    unique([f.path for f in artifacts.state_schema.fields], "state paths")
    action_aliases(artifacts.action_schema)
    actions = {a.name: a for a in artifacts.action_schema.actions}
    episodes = {e.id: e for e in artifacts.episodes}
    transitions = {t.id: t for t in artifacts.transitions}
    all_events = [event.id for ep in artifacts.episodes for event in ep.events]
    unique(all_events, "event IDs")
    for source_id, text in artifacts.source_snapshots.items():
        if source_id != "src_" + hashlib.sha256(text.encode("utf-8")).hexdigest():
            raise ValueError("Source snapshot digest mismatch")
    if require_provenance:
        if not artifacts.evidence or not artifacts.episodes:
            raise ValueError("Reconstructed packages require trace evidence")
        if {ep.source_id for ep in artifacts.episodes} - artifacts.source_snapshots.keys():
            raise ValueError("Reconstructed package is missing raw source snapshots")
    for t in artifacts.transitions:
        ep = episodes.get(t.episode_id)
        if ep is None:
            raise ValueError("Transition references unknown episode")
        ids = t.history_event_ids + t.action_event_ids + t.observation_event_ids + t.delayed_observation_event_ids + t.concurrent_action_event_ids
        if set(ids) - {e.id for e in ep.events}:
            raise ValueError("Transition references unknown events")
    for item in artifacts.evidence:
        t = transitions.get(item.transition_id)
        if t is None or item.episode_id != t.episode_id:
            raise ValueError("Evidence references an unknown or mismatched transition")
        if item.action.type not in actions:
            raise ValueError("Evidence action is absent from the action schema")
    def refs(items, required=False):
        if required and not items:
            raise ValueError("Reconstructed executable claim has no provenance")
        for ref in items:
            validate_source_ref(ref, episodes, require_event=required)
    def condition(c, action):
        for operand in (c.left, c.right):
            if operand is None:
                continue
            if operand.path is not None:
                state_field(operand.path, artifacts.state_schema, action)
            if operand.action_arg is not None:
                argument_type(operand.action_arg, action)
    def render_path(path, action):
        if path.startswith("state_before.") or path.startswith("state_after."):
            state_field(path.split(".", 1)[1], artifacts.state_schema, action)
        elif path.startswith("action.arguments."):
            argument_type(path[len("action.arguments."):], action)
        elif path not in {"action.type", "outcome"}:
            raise ValueError(f"Unsupported renderer reference: {path}")
    for field in artifacts.state_schema.fields:
        if field.default is not None and not TYPES[field.type](field.default):
            raise ValueError(f"Invalid default type for {field.path}")
        refs(field.provenance)
    for action in actions.values():
        refs(action.provenance)
    for item in artifacts.evidence:
        refs(item.provenance, require_provenance)
        for fact in item.preconditions + item.observation_facts:
            refs(fact.provenance)
    for rule in artifacts.rules:
        refs(rule.provenance, require_provenance and rule.status == "supported")
        refs(rule.counterexamples)
        if require_provenance and rule.status == "supported":
            if not rule_has_observed_support(rule, artifacts.evidence, transitions):
                raise ValueError(f"Rule {rule.id} lacks an unambiguous observed transition for its action")
        action = actions[rule.action_type]
        validate_mutations(rule.effects, artifacts.state_schema, action)
        for c in rule.conditions:
            condition(c, action)
        for path in re.findall(r"\{([^{}]+)\}", rule.observation_template or ""):
            render_path(path, action)
    for inv in artifacts.invariants:
        refs(inv.provenance, require_provenance)
        condition(inv.condition, None)
    for renderer in artifacts.renderers:
        refs(renderer.provenance)
        related = set(renderer.action_types) | {r.action_type for r in artifacts.rules if r.renderer == renderer.id}
        if related - actions.keys():
            raise ValueError("Renderer references an unknown action")
        for action_type in related:
            for path in renderer.required_fields + re.findall(r"\{([^{}]+)\}", renderer.template or ""):
                render_path(path, actions[action_type])
    for demo in artifacts.demonstrations:
        refs(demo.provenance)
        if demo.action.type not in actions or set(demo.rule_ids) - {r.id for r in artifacts.rules}:
            raise ValueError("Demonstration references unknown action or rules")
    for note in artifacts.notes:
        refs(note.provenance, require_provenance and note.status == "supported")
        if set(note.action_types) - actions.keys():
            raise ValueError(f"Note {note.id} references an unknown action")


def validate_state_types(state, schema: StateSchema) -> None:
    data = state.model_dump(mode="python")
    for field in schema.fields:
        value = get_path(data, field.path, MISSING)
        if value is not MISSING and not TYPES[field.type](value):
            raise ValueError(f"State type mismatch at {field.path}")
