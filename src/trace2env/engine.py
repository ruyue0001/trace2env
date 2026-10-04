"""Deterministic rule evaluation, state patching, and invariant checking."""

from __future__ import annotations

import copy
import re
from typing import Any
from uuid import uuid4
from trace2env.eligibility import rule_eligible

from trace2env.models import (
    Condition,
    EnvironmentState,
    Invariant,
    NormalizedAction,
    Operand,
    PendingEvent,
    StateMutation,
    TransitionPlan,
    TransitionRule,
    TrustPolicy,
    VerificationIssue,
    VerificationResult,
)


MISSING = object()
_RAISE_MISSING = object()


def get_path(value: Any, path: str, default: Any = _RAISE_MISSING) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            if default is _RAISE_MISSING:
                raise KeyError(path)
            return default
    return current


def set_path(value: dict[str, Any], path: str, new_value: Any, create: bool = True) -> None:
    parts = path.split(".")
    current = value
    for part in parts[:-1]:
        child = current.get(part)
        if child is None and create:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(f"Cannot traverse non-object at {part} in {path}")
        current = child
    current[parts[-1]] = new_value


def delete_path(value: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


class UnresolvableOperand(KeyError):
    """A condition operand names an action argument the action does not carry (``{action.arguments.index}`` for a
    click without ``index``): the condition cannot hold, and the rule cannot apply."""


def resolve_operand(operand: Operand, state: EnvironmentState, action: NormalizedAction) -> Any:
    if operand.path is not None:
        try:
            path = re.sub(
                r"\{action\.arguments\.([^{}]+)\}",
                lambda match: str(get_path(action.arguments, match.group(1))),
                operand.path,
            )
        except KeyError as exc:
            raise UnresolvableOperand(str(exc)) from exc
        return get_path(state.model_dump(mode="python"), path, MISSING)
    if operand.action_arg is not None:
        return get_path(action.arguments, operand.action_arg, MISSING)
    return operand.literal


def evaluate_condition(condition: Condition, state: EnvironmentState, action: NormalizedAction) -> bool:
    try:
        return _evaluate_condition(condition, state, action)
    except UnresolvableOperand:
        return False  # a condition about an argument this action lacks never holds (found live: the android tracker crashed)


def _evaluate_condition(condition: Condition, state: EnvironmentState, action: NormalizedAction) -> bool:
    left = resolve_operand(condition.left, state, action)
    if condition.op == "exists":
        return left is not MISSING
    if condition.op == "not_exists":
        return left is MISSING
    if left is MISSING or condition.right is None:
        return False
    right = resolve_operand(condition.right, state, action)
    if right is MISSING:
        return False
    operations = {
        "eq": lambda: left == right,
        "ne": lambda: left != right,
        "gt": lambda: left > right,
        "gte": lambda: left >= right,
        "lt": lambda: left < right,
        "lte": lambda: left <= right,
        "contains": lambda: right in left,
        "in": lambda: left in right,
    }
    try:
        return bool(operations[condition.op]())
    except (TypeError, KeyError):
        return False


ACTION_TOKEN = re.compile(r"\{action\.arguments\.([^{}]+)\}")


def resolve_dynamic(value: Any, action: NormalizedAction) -> Any:
    if isinstance(value, dict) and set(value) == {"$action_arg"}:
        return get_path(action.arguments, str(value["$action_arg"]))
    if isinstance(value, dict) and set(value) == {"$state_path"}:
        return value
    if isinstance(value, dict):
        # Keys may carry tokens too: a file map entry keyed by the path the action names.
        return {_resolve_key(key, action): resolve_dynamic(item, action) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_dynamic(item, action) for item in value]
    if isinstance(value, str):
        full = ACTION_TOKEN.fullmatch(value)
        if full:
            return get_path(action.arguments, full.group(1))
        return ACTION_TOKEN.sub(lambda match: str(get_path(action.arguments, match.group(1))), value)
    return value


def _resolve_key(key: Any, action: NormalizedAction) -> Any:
    if isinstance(key, str) and ACTION_TOKEN.search(key):
        return ACTION_TOKEN.sub(lambda match: str(get_path(action.arguments, match.group(1))), key)
    return key


def resolve_mutation(mutation: StateMutation, action: NormalizedAction) -> StateMutation:
    path = ACTION_TOKEN.sub(lambda match: str(get_path(action.arguments, match.group(1))), mutation.path)
    return mutation.model_copy(update={"path": path, "value": resolve_dynamic(mutation.value, action)})


def apply_mutations(
    state: EnvironmentState,
    mutations: list[StateMutation],
    *,
    action: NormalizedAction | None = None,
) -> EnvironmentState:
    result = state.model_copy(deep=True)
    data = result.model_dump(mode="python")
    for original in mutations:
        mutation = resolve_mutation(original, action) if action else original
        if mutation.op == "schedule":
            if mutation.delay_steps is None:
                raise ValueError("schedule mutation requires delay_steps")
            if isinstance(mutation.value, dict) and "effects" in mutation.value:
                effects = [StateMutation.model_validate(item) for item in mutation.value["effects"]]
            else:
                effects = [StateMutation(op="set", path=mutation.path, value=mutation.value)]
            pending = PendingEvent(
                id=f"pending_{uuid4().hex}",
                due_step=result.step + mutation.delay_steps,
                effects=effects,
            )
            data.setdefault("pending_events", []).append(pending.model_dump(mode="python"))
            continue

        current = get_path(data, mutation.path, MISSING)
        if mutation.op in {"set", "create"}:
            set_path(data, mutation.path, copy.deepcopy(mutation.value))
        elif mutation.op == "delete":
            delete_path(data, mutation.path)
        elif mutation.op in {"increment", "decrement"}:
            if current is MISSING:
                raise ValueError(f"Cannot {mutation.op} missing path {mutation.path}")
            try:
                updated = current + mutation.value if mutation.op == "increment" else current - mutation.value
            except TypeError as exc:
                raise ValueError(f"Cannot {mutation.op} non-numeric path {mutation.path}: {exc}") from exc
            set_path(data, mutation.path, updated)
        elif mutation.op == "append":
            if current is MISSING:
                set_path(data, mutation.path, [copy.deepcopy(mutation.value)])
            elif isinstance(current, list):
                current.append(copy.deepcopy(mutation.value))
            else:
                raise ValueError(f"Cannot append to non-list path {mutation.path}")
        elif mutation.op == "merge":
            if not isinstance(mutation.value, dict):
                raise ValueError(f"merge requires an object value at {mutation.path}")
            if current is MISSING:
                set_path(data, mutation.path, copy.deepcopy(mutation.value))
            elif isinstance(current, dict):
                current.update(copy.deepcopy(mutation.value))
            else:
                raise ValueError(f"Cannot merge into non-object path {mutation.path}")
        elif mutation.op == "remove":
            if current is MISSING:
                continue
            if not isinstance(current, dict):
                raise ValueError(f"Cannot remove keys from non-object path {mutation.path}")
            keys = mutation.value if isinstance(mutation.value, list) else [mutation.value]
            for key in keys:
                current.pop(str(key), None)
    return EnvironmentState.model_validate(data)


def process_due_events(state: EnvironmentState, target_step: int) -> tuple[EnvironmentState, list[str]]:
    result = state.model_copy(deep=True)
    due = [event for event in result.pending_events if event.due_step <= target_step]
    remaining = [event for event in result.pending_events if event.due_step > target_step]
    result.pending_events = remaining
    for event in sorted(due, key=lambda item: (item.due_step, item.id)):
        result = apply_mutations(result, event.effects)
    return result, [event.id for event in due]


class AmbiguousRuleError(RuntimeError):
    pass


def select_rule(
    rules: list[TransitionRule], state: EnvironmentState, action: NormalizedAction,
    scope: dict[str, str] | None = None,
) -> TransitionRule | None:
    candidates: dict[str, tuple[TransitionRule, TransitionPlan]] = {}
    for rule in rules:
        if rule.action_type != action.type or not rule_eligible(rule, scope, state):
            continue
        if not all(evaluate_condition(condition, state, action) for condition in rule.conditions):
            continue
        try:
            plan = deterministic_plan(rule, action)
        except (KeyError, ValueError):
            # The rule's effects reference arguments this action does not carry; it cannot apply.
            continue
        candidates[rule.id] = (rule, plan)
    if not candidates:
        return None
    priority = max(rule.priority for rule, _ in candidates.values())
    finalists = [(rule, plan) for rule, plan in candidates.values() if rule.priority == priority]
    signatures = {plan.model_copy(update={"rule_ids": []}).model_dump_json() for _, plan in finalists}
    if len(signatures) > 1:
        raise AmbiguousRuleError(
            "Incompatible rules apply at the same priority: " + ", ".join(rule.id for rule, _ in finalists)
        )
    return max((rule for rule, _ in finalists), key=lambda rule: (rule.confidence, rule.id))


def deterministic_plan(rule: TransitionRule, action: NormalizedAction) -> TransitionPlan:
    return TransitionPlan(
        action=action,
        rule_ids=[rule.id],
        effects=[resolve_mutation(effect, action) for effect in rule.effects],
        outcome=rule.outcome,
        renderer_id=rule.renderer,
        observation_template=rule.observation_template,
    )


def check_invariants(
    invariants: list[Invariant], state: EnvironmentState, action: NormalizedAction
) -> VerificationResult:
    issues: list[VerificationIssue] = []
    for invariant in invariants:
        if not evaluate_condition(invariant.condition, state, action):
            issues.append(
                VerificationIssue(
                    code=f"invariant:{invariant.id}",
                    message=invariant.description,
                    severity=invariant.severity,
                )
            )
    return VerificationResult(
        accepted=not any(issue.severity == "error" for issue in issues),
        issues=issues,
        confidence=1.0 if not issues else 0.7,
    )


def verify_rule_support(
    plan: TransitionPlan, rules: list[TransitionRule], state_before: EnvironmentState,
    scope: dict[str, str] | None = None, *, trust_policy: TrustPolicy = "rules_only",
) -> list[VerificationIssue]:
    if not plan.rule_ids:
        if plan.effects and trust_policy == "schema_checked":
            # Graded trust: the effects already passed schema, mutability, type, and invariant
            # checks. They are accepted as a model proposal and flagged for the audit record.
            return [
                VerificationIssue(
                    code="model_proposed_effects",
                    message="The plan cites no reconstructed rule; its effects were accepted under the "
                            "schema_checked trust policy after schema, type, and invariant checks.",
                    severity="warning",
                )
            ]
        return [
            VerificationIssue(
                code="unsupported_plan",
                message="The plan cites no reconstructed rule; it requires agentic fallback judgment.",
                severity="error" if plan.effects else "warning",
            )
        ]
    by_id = {rule.id: rule for rule in rules}
    missing = [rule_id for rule_id in plan.rule_ids if rule_id not in by_id]
    if missing:
        return [
            VerificationIssue(
                code="unknown_rule",
                message=f"Plan cites rules that were not retrieved: {', '.join(missing)}",
                severity="error",
            )
        ]
    cited = [by_id[rule_id] for rule_id in plan.rule_ids]
    if len(plan.rule_ids) != len(set(plan.rule_ids)):
        return [VerificationIssue(code="duplicate_rule", message="A plan cannot apply the same rule twice")]
    if any(rule.action_type != plan.action.type or not rule_eligible(rule, scope, state_before) for rule in cited):
        return [VerificationIssue(code="ineligible_rule", message="A cited rule is not eligible for this action and scope.")]
    if any(
        not all(evaluate_condition(condition, state_before, plan.action) for condition in rule.conditions)
        for rule in cited
    ):
        return [
            VerificationIssue(
                code="inapplicable_rule",
                message="At least one cited rule does not apply to the pre-transition state.",
                severity="error",
            )
        ]
    if any(rule.outcome != plan.outcome for rule in cited):
        return [
            VerificationIssue(
                code="unsupported_outcome",
                message="The planned outcome differs from every cited rule outcome.",
                severity="error",
            )
        ]
    supported = {
        mutation.model_dump_json(exclude_none=True)
        for rule_id in plan.rule_ids
        for mutation in [resolve_mutation(item, plan.action) for item in by_id[rule_id].effects]
    }
    unsupported = [
        effect for effect in plan.effects if effect.model_dump_json(exclude_none=True) not in supported
    ]
    if unsupported:
        return [
            VerificationIssue(
                code="unsupported_effect",
                message="Plan contains effects absent from its cited rules.",
                severity="error",
            )
        ]
    actual = [effect.model_dump_json(exclude_none=True) for effect in plan.effects]
    expected = [resolve_mutation(effect, plan.action).model_dump_json(exclude_none=True)
                for rule in cited for effect in rule.effects]
    if actual != expected:
        return [VerificationIssue(code="effect_sequence_mismatch", message="Plan must preserve its rules' effect order and multiplicity.")]
    if any(rule.renderer != plan.renderer_id or rule.observation_template != plan.observation_template for rule in cited):
        return [VerificationIssue(code="unsupported_rendering", message="Plan changed the rendering specified by its rules.")]
    return []


TEMPLATE_TOKEN = re.compile(r"\{([a-zA-Z0-9_.]+)\}")


def render_template(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = get_path(context, match.group(1), MISSING)
        if value is MISSING:
            raise KeyError(f"Template references missing value {match.group(1)}")
        return str(value)

    return TEMPLATE_TOKEN.sub(replace, template)
