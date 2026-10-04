"""Matched observational contrasts. Differences are hypotheses, never causal labels."""
from __future__ import annotations

import json
from trace2env.models import LocalTransitionEvidence


def _facts(item: LocalTransitionEvidence) -> dict:
    return {f"{fact.subject}:{fact.predicate}": fact.value for fact in item.preconditions
            if fact.status.value == "observed" and fact.provenance}


def build_contrasts(evidence: list[LocalTransitionEvidence], episode_groups: dict[str, str],
                    episode_scopes: dict[str, dict], *, limit_per_group: int = 32) -> list[dict]:
    groups = {}
    for item in sorted(evidence, key=lambda e: (-e.confidence.value, e.id)):
        scope = episode_scopes.get(item.episode_id, {})
        key = json.dumps([item.action.type, item.action.arguments, scope], sort_keys=True)
        groups.setdefault(key, []).append(item)
    contrasts = []
    for items in groups.values():
        successes = [item for item in items if item.outcome.value == "success"]
        failures = [item for item in items if item.outcome.value == "failure"]
        retained = 0
        for positive in successes:
            # One strongest independent counterexample per success, under a fixed group budget.
            negative = next((item for item in failures if episode_groups.get(item.episode_id, item.episode_id)
                             != episode_groups.get(positive.episode_id, positive.episode_id)), None)
            if negative is None:
                continue
            before_positive, before_negative = _facts(positive), _facts(negative)
            shared = {key: value for key, value in before_positive.items()
                      if key in before_negative and before_negative[key] == value}
            differing = {key: {"success": before_positive[key], "failure": before_negative[key]}
                         for key in before_positive.keys() & before_negative.keys()
                         if before_positive[key] != before_negative[key]}
            contrasts.append({"action_type": positive.action.type, "arguments": positive.action.arguments,
                "scope": episode_scopes.get(positive.episode_id, {}),
                "success_evidence_id": positive.id, "failure_evidence_id": negative.id,
                "episode_groups": [episode_groups.get(x.episode_id, x.episode_id) for x in [positive, negative]],
                "shared_observed_preconditions": shared, "differing_observed_preconditions": differing,
                "unmatched_state_fields": sorted(before_positive.keys() ^ before_negative.keys()),
                "interpretation": "Same action, arguments, and declared scope; hidden state remains uncontrolled."})
            retained += 1
            if retained >= limit_per_group:
                break
    return contrasts
