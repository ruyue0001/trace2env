"""Run-time state reconstruction from observed interactions.

A benchmark record or a live caller supplies a trajectory prefix: the actions a task agent took
and the observations the real environment returned, but no explicit state. ``StateTracker``
advances an ``EnvironmentState`` through such turns. A turn explained by an applicable rule
whose rendered observation matches the observed one is applied deterministically; every other
turn is batched into a bounded ``track_state`` model call whose typed mutations are checked
against the state schema before they touch the state. Tracking follows observed reality: an
invariant contradicted by an observation is recorded as a diagnostic about the package, not
enforced against the evidence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from trace2env.engine import (
    AmbiguousRuleError,
    apply_mutations,
    check_invariants,
    deterministic_plan,
    process_due_events,
    render_template,
    select_rule,
)
from trace2env.eligibility import rule_eligible
from trace2env.llm import ProviderRefusal, StructuredLLM, StructuredOutputError
from trace2env.memory import EpisodicMemory
from trace2env.models import (
    EnvironmentState,
    MemoryEntry,
    NormalizedAction,
    ObservedTurn,
    StateMutation,
    StateTrackingResult,
    StateTrackingStep,
    TransitionPlan,
)
from trace2env.package import EnvironmentPackage
from trace2env.prompts import STATE_TRACKER, STATE_TRACKER_EXACT, STATE_TRACKER_WAIT
from trace2env.transcript import initial_prompt_state
from trace2env.validation import validate_mutations, validate_state_types
from trace2env.workspace import clip_text, compact_action

# Same default bound as runtime.ENVIRONMENT_PROMPT_CHARS, kept local so neither module imports the other.
ENVIRONMENT_PROMPT_CHARS = 24000
_WHITESPACE = re.compile(r"\s+")


def normalize_observation(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


class StateTracker:
    def __init__(
        self,
        package: EnvironmentPackage,
        llm: StructuredLLM | None = None,
        *,
        window: int = 8,
        observation_chars: int = 4000,
        environment_prompt: str = "",
        memory: EpisodicMemory | None = None,
        features: set[str] | None = None,
        disabled: bool = False,
        environment_prompt_chars: int | None = ENVIRONMENT_PROMPT_CHARS,
    ):
        self.package = package
        self.llm = llm
        self.memory = memory
        # ``disabled``: the trace2env_no_state control — every observed turn is remembered (history stays
        # available) but no rule and no model ever changes the state.
        self.disabled = disabled
        self.window = max(1, window)
        self.observation_chars = observation_chars
        # Caller-owned environment context, bounded by default (the setting of every reported AgentWorldBench
        # run). It may end with tool definitions or examples; ``environment_prompt_chars=None`` keeps it whole
        # (the long-horizon runs).
        self.environment_prompt = environment_prompt if environment_prompt_chars is None else environment_prompt[:environment_prompt_chars]
        # "wait": a rule without effects and without a template explains nothing and never replaces
        # model tracking; "compact": static schema/environment text goes into the (cacheable) system
        # message as a compact field list; "unknown": exact observed values are kept verbatim.
        self.features = set(features or ())
        self.aliases = {
            alias: spec.name for spec in package.action_schema.actions for alias in [spec.name, *spec.aliases]
        }
        self.minimum_confidence = float(package.manifest.metadata.get("min_rule_confidence", 0.0))
        self._carried: list[str] = []

    def canonical(self, action: NormalizedAction) -> NormalizedAction:
        return action.model_copy(update={"type": self.aliases.get(action.type, action.type)})

    def advance(
        self, state: EnvironmentState, turns: list[ObservedTurn]
    ) -> tuple[EnvironmentState, list[StateTrackingStep]]:
        steps: list[StateTrackingStep] = []
        pending: list[ObservedTurn] = []
        if self.disabled:
            for observed in turns:
                self._remember(observed.model_copy(update={"action": self.canonical(observed.action)}), [], [])
            return state, [StateTrackingStep(turns=[t.turn for t in turns], route="skipped",
                                             issues=["state tracking disabled; turns recorded in memory only"])]
        for observed in turns:
            turn = observed.model_copy(update={"action": self.canonical(observed.action)})
            rule = self._applicable_rule(state, turn)
            if rule is not None and pending:
                # Pending model-explained turns precede this one; refresh the state before trusting the rule.
                state, step = self._flush(state, pending)
                steps.append(step)
                pending = []
                rule = self._applicable_rule(state, turn)
            if rule is not None:
                outcome = self._apply_rule(state, rule, turn)
                if outcome is not None:
                    state, step = outcome
                    steps.append(step)
                    continue
            pending.append(turn)
            if len(pending) >= self.window:
                state, step = self._flush(state, pending)
                steps.append(step)
                pending = []
        if pending:
            state, step = self._flush(state, pending)
            steps.append(step)
        return state, steps

    def _eligible_rules(self, state: EnvironmentState) -> list:
        return [
            rule for rule in self.package.rules
            if rule_eligible(rule, self.package.scope, state, minimum_confidence=self.minimum_confidence)
        ]

    def _applicable_rule(self, state: EnvironmentState, turn: ObservedTurn):
        working, _ = process_due_events(state, state.step + 1)
        try:
            return select_rule(self._eligible_rules(working), working, turn.action, self.package.scope)
        except AmbiguousRuleError as exc:
            self._carried.append(f"turn {turn.turn}: {exc}")
            return None

    def _render_context(self, before: EnvironmentState, after: EnvironmentState, plan: TransitionPlan) -> dict[str, Any]:
        return {
            "action": plan.action.model_dump(mode="python"),
            "state_before": before.model_dump(mode="python"),
            "state_after": after.model_dump(mode="python"),
            "outcome": plan.outcome.value,
        }

    def _apply_rule(
        self, state: EnvironmentState, rule, turn: ObservedTurn
    ) -> tuple[EnvironmentState, StateTrackingStep] | None:
        working, _ = process_due_events(state, state.step + 1)
        working.step = state.step + 1
        if "wait" in self.features and not rule.effects:
            contract_template = next((item.template for item in self.package.renderers if item.id == rule.renderer), None)
            if not (rule.observation_template or contract_template):
                # Nothing to apply and nothing to check against the observation: the observed turn still
                # has to be explained by the model, or its effects (a finished process, new files) are lost.
                self._carried.append(f"turn {turn.turn}: rule {rule.id} has no effects and no template; tracked by the model instead")
                return None
        plan = deterministic_plan(rule, turn.action)
        spec = next((item for item in self.package.action_schema.actions if item.name == turn.action.type), None)
        try:
            validate_mutations(plan.effects, self.package.state_schema, spec)
            candidate = apply_mutations(working, plan.effects, action=turn.action)
            validate_state_types(candidate, self.package.state_schema)
        except (ValueError, KeyError) as exc:
            self._carried.append(f"turn {turn.turn}: rule {rule.id} could not be applied: {exc}")
            return None
        contract = next((item for item in self.package.renderers if item.id == plan.renderer_id), None)
        template = plan.observation_template or (contract.template if contract else None)
        if template:
            try:
                rendered = render_template(template, self._render_context(state, candidate, plan))
            except KeyError as exc:
                self._carried.append(f"turn {turn.turn}: rule {rule.id} template failed: {exc}")
                return None
            if normalize_observation(rendered) != normalize_observation(turn.observation):
                # The rule disagrees with what the environment actually returned; do not trust it here.
                self._carried.append(f"turn {turn.turn}: rule {rule.id} predicted a different observation than observed")
                return None
        issues, self._carried = self._carried, []
        verification = check_invariants(self.package.invariants, candidate, turn.action)
        issues += [f"{issue.code}: {issue.message}" for issue in verification.issues if issue.severity == "error"]
        self._remember(turn, plan.effects, [f"rule:{rule.id}"])
        step = StateTrackingStep(turns=[turn.turn], route="rule", rule_id=rule.id, applied=len(plan.effects), issues=issues)
        return candidate, step

    def _remember(self, turn: ObservedTurn, effects: list[StateMutation], citations: list[str]) -> None:
        if self.memory is not None:
            self.memory.record(MemoryEntry(turn=turn.turn, kind="observed", action=turn.action,
                                           observation=turn.observation, effects=effects, citations=citations))

    def system_prompt(self) -> str:
        """The tracker's system text; with ``compact`` it also carries the static schema and environment description."""
        system = STATE_TRACKER
        if "wait" in self.features:
            system += "\n\n" + STATE_TRACKER_WAIT
        if "unknown" in self.features:
            system += "\n\n" + STATE_TRACKER_EXACT
        if "compact" in self.features:
            lines = [f"- {field.path} ({field.type}{'' if field.mutable else ', immutable'}): {field.description[:120]}"
                     for field in self.package.state_schema.fields]
            system += "\n\n# Declared state fields\n" + "\n".join(lines)
            if self.environment_prompt:
                system += "\n\n# Environment description (context only)\n" + self.environment_prompt
        return system

    def _tracking_request(self, working: EnvironmentState, pending: list[ObservedTurn]) -> tuple[str, dict[str, Any]]:
        turns = [
            {
                "turn": turn.turn,
                "action": compact_action(turn.action) if "compact" in self.features else turn.action.model_dump(mode="json"),
                "observation": clip_text(turn.observation, self.observation_chars),
            }
            for turn in pending
        ]
        instruction = ("Return the typed mutations, on declared state paths only, that reflect what these "
                       "observations establish about the environment after the last listed turn.")
        if "compact" in self.features:
            return self.system_prompt(), {"current_state": working.model_dump(mode="json"), "observed_turns": turns,
                                          "instruction": instruction}
        return self.system_prompt(), {
            "environment": self.environment_prompt,
            "state_schema": [field.model_dump(mode="json") for field in self.package.state_schema.fields],
            "current_state": working.model_dump(mode="json"),
            "observed_turns": turns,
            "instruction": instruction,
        }

    def track_initial_prompt(self, state: EnvironmentState, observation: str, action: NormalizedAction) -> tuple[EnvironmentState, StateTrackingStep] | None:
        """Track an initial screen that is just an idle prompt without a model call (``compact``).

        The working directory and the idle terminal mode are set when the schema declares them;
        the prompt line itself is kept in episodic memory, where the agent and the transcript
        scaffold read the user/host from.
        """
        if self.disabled:
            return None
        if "compact" not in self.features:
            return None
        parsed = initial_prompt_state(observation)
        if parsed is None:
            return None
        declared = {field.path: field for field in self.package.state_schema.fields}
        mutations = []
        if "session.cwd" in declared and declared["session.cwd"].type in {"string", "any"}:
            mutations.append(StateMutation(op="set", path="session.cwd", value=parsed["cwd"]))
        if "surface.mode" in declared and declared["surface.mode"].type in {"string", "any"}:
            mutations.append(StateMutation(op="set", path="surface.mode", value="prompt"))
        working, _ = process_due_events(state, state.step + 1)
        working.step = state.step + 1
        try:
            validate_mutations(mutations, self.package.state_schema, None)
            working = apply_mutations(working, mutations)
            validate_state_types(working, self.package.state_schema)
        except (ValueError, KeyError):
            return None
        turn = ObservedTurn(turn=0, action=action, observation=observation)
        self._remember(turn, mutations, ["prompt-line"])
        return working, StateTrackingStep(turns=[0], route="rule", rule_id="initial-prompt-line", applied=len(mutations))

    def _flush(self, state: EnvironmentState, pending: list[ObservedTurn]) -> tuple[EnvironmentState, StateTrackingStep]:
        issues, self._carried = self._carried, []
        turns = [turn.turn for turn in pending]
        working, _ = process_due_events(state, state.step + len(pending))
        working.step = state.step + len(pending)
        if self.llm is None:
            issues.append("no tracking model configured; state left unchanged for these turns")
            for turn in pending:
                self._remember(turn, [], [])
            return working, StateTrackingStep(turns=turns, route="skipped", issues=issues)
        system, payload = self._tracking_request(working, pending)
        try:
            result = self.llm.complete(
                system=system,
                user=json.dumps(payload, ensure_ascii=False, default=str),
                response_model=StateTrackingResult,
                role="track_state",
            )
        except RuntimeError as exc:
            # One handler for every model-side failure, because ProviderRefusal and StructuredOutputError are
            # RuntimeErrors too and a re-raise from a sibling clause would escape the whole try statement.
            truncated = "cut off at the output limit" in str(exc)
            if isinstance(exc, ProviderRefusal) or (isinstance(exc, StructuredOutputError) and not truncated):
                # A content-policy refusal, or an answer that never met the contract, is deterministic for this
                # window: the turns stay in memory (history is unaffected), the state is left unchanged, and the
                # gap is recorded instead of ending the trajectory.
                reason = "refused by the provider" if isinstance(exc, ProviderRefusal) else "answer invalid after repair attempts"
                issues.append(f"turns {turns[0]}-{turns[-1]}: track_state {reason} ({str(exc)[:120]}); state left unchanged")
                for turn in pending:
                    self._remember(turn, [], [])
                return working, StateTrackingStep(turns=turns, route="skipped", issues=issues)
            if not truncated:
                raise
            # The model tried to write more mutations than fit (typically whole file contents). Track
            # the window in halves so the rest of the trajectory keeps a state; a single turn that
            # still overflows is skipped with the state left unchanged, and the gap is recorded.
            if len(pending) > 1:
                middle = len(pending) // 2
                self._carried = issues + [f"turns {turns[0]}-{turns[-1]}: track_state output exceeded the limit; tracked in halves"]
                state, first = self._flush(state, pending[:middle])
                state, second = self._flush(state, pending[middle:])
                return state, StateTrackingStep(
                    turns=turns, route="model", applied=first.applied + second.applied,
                    dropped=first.dropped + second.dropped, issues=first.issues + second.issues)
            issues.append(f"turn {turns[0]}: track_state output exceeded the limit; state left unchanged for this turn")
            for turn in pending:
                self._remember(turn, [], [])
            return working, StateTrackingStep(turns=turns, route="skipped", issues=issues)
        accepted: list[StateMutation] = []
        dropped: list[str] = []
        for mutation in result.mutations:
            try:
                # No action spec: tracker mutations must carry literal values on declared paths.
                validate_mutations([mutation], self.package.state_schema, None)
                candidate = apply_mutations(working, [mutation])
                validate_state_types(candidate, self.package.state_schema)
            except (ValueError, KeyError) as exc:
                dropped.append(f"{mutation.op} {mutation.path}: {exc}")
                continue
            working = candidate
            accepted.append(mutation)
        issues += [f"uncertain: {path}" for path in result.uncertain_paths]
        verification = check_invariants(self.package.invariants, working, pending[-1].action)
        issues += [f"{issue.code}: {issue.message}" for issue in verification.issues if issue.severity == "error"]
        for turn in pending:
            # The window's mutations are attributed to its last observed turn.
            self._remember(turn, accepted if turn is pending[-1] else [], [])
        return working, StateTrackingStep(turns=turns, route="model", applied=len(accepted), dropped=dropped, issues=issues)
