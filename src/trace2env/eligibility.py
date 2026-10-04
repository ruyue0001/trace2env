"""Shared, fail-closed eligibility for compilation and execution."""
from __future__ import annotations

from typing import Any
from trace2env.models import EnvironmentState, ReconstructionConfig, TransitionRule

SCOPE_ALIASES = {"tenant_scope": "tenant", "version": "environment_version"}


def construction_scope(config: ReconstructionConfig) -> dict[str, str]:
    context = {SCOPE_ALIASES.get(key, key): value for key, value in config.scope.items()}
    reserved = {"environment_id": config.environment_id}
    if config.tenant_scope is not None:
        reserved["tenant"] = config.tenant_scope
    if config.environment_version is not None:
        reserved["environment_version"] = config.environment_version
    for key, value in reserved.items():
        if key in context and context[key] != value:
            raise ValueError(f"Conflicting construction scope: {key}")
        context[key] = value
    return context


def rule_eligible(rule: TransitionRule, context: dict[str, str] | None = None,
                  state: EnvironmentState | None = None, *, minimum_confidence: float = 0.0,
                  allow_state_unknown: bool = False) -> bool:
    if rule.status != "supported" or rule.confidence < minimum_confidence:
        return False
    for key, expected in rule.scope.items():
        key = SCOPE_ALIASES.get(key, key)
        if key.split(".", 1)[0] in {"world", "session", "surface", "epistemic"}:
            if state is None:
                if allow_state_unknown:
                    continue
                return False
            value: Any = state.model_dump(mode="python")
            for part in key.split("."):
                if not isinstance(value, dict) or part not in value:
                    return False
                value = value[part]
            if str(value) != expected:
                return False
        elif (context or {}).get(key) != expected:
            return False
    return True
