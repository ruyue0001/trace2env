"""Agentic world-model harness: the agent operates the workspace; the harness verifies and commits.

One step is ``Harness(package, session, action) -> observation, next state``. The world-model
agent decides the transition through workspace tools (knowledge, state, episodic memory, rule
dry-runs) and submits typed effects, the observation, and citations. The harness owns
everything the agent must not: action validation, effect checking, invariant and rule-support
verification, rendering from templates, atomic commit, audit, and memory. A confident,
templated rule may short-circuit the agent (``fast_path``) because that is cheaper and exactly
reproducible; the policy is explicit and recorded as the step's route.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from trace2env.agent import AgentLoop
from trace2env.engine import (
    MISSING,
    TEMPLATE_TOKEN,
    apply_mutations,
    check_invariants,
    deterministic_plan,
    evaluate_condition,
    get_path,
    process_due_events,
    render_template,
    select_rule,
    verify_rule_support,
)
from trace2env.llm import AgentLLM, StructuredAgentLLM, StructuredLLM
from trace2env.knowledge_gate import documented_format, KnowledgeGate, action_text, build_episode_context
from trace2env.memory import EpisodicMemory
from trace2env.models import (
    SUBMIT_TOOL_NAME,
    AuditRecord,
    EnvironmentState,
    FastPathPolicy,
    MemoryEntry,
    RenderContract,
    RenderedObservation,
    SingleShotPrediction,
    StepRequest,
    StepResult,
    TransitionPlan,
    TransitionRule,
    TransitionSubmission,
    TrustPolicy,
    VerificationIssue,
    VerificationResult,
)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.prompts import (
    RUNTIME_RENDERER,
    RUNTIME_VERIFIER,
    WORLD_MODEL_AGENT,
    WORLD_MODEL_AGENT_EVIDENCE,
    WORLD_MODEL_AGENT_HISTORY,
    WORLD_MODEL_AGENT_KNOWLEDGE_GATE,
    WORLD_MODEL_AGENT_NO_KNOWLEDGE_TOOLS,
    WORLD_MODEL_AGENT_NO_PACKAGE,
    WORLD_MODEL_AGENT_NO_STATE,
    WORLD_MODEL_AGENT_SCAFFOLD,
    WORLD_MODEL_AGENT_OFFICIAL_INPUT,
    WORLD_MODEL_AGENT_TRACES,
    WORLD_MODEL_AGENT_UNKNOWN,
    WORLD_MODEL_AGENT_WAIT,
    WORLD_MODEL_SINGLE_SHOT,
    WORLD_MODEL_SINGLE_SHOT_EVIDENCE,
)
from trace2env.session import SessionStore
from trace2env.trace_corpus import TraceCorpus, action_query
from trace2env.transcript import build_scaffold, last_prompt, prompt_signature, wait_hint
from trace2env.validation import validate_mutations, validate_state_types
from trace2env.eligibility import rule_eligible
from trace2env.workspace import ABSENCE_MEANS_UNKNOWN, WorkspaceTools, clip_span, clip_value, compact_action, state_summary

# Online-harness features (see docs/RUNTIME_HARNESS.md): each is separable so its effect can be measured.
HARNESS_FEATURES = ("history", "wait", "compact", "scaffold", "unknown", "evidence")
# The subset evaluation of 2026-09-19 (work/exp-v1_20_r=1/awb/diag) selected history/wait/compact/unknown; the
# full-set run of 2026-09-20 (harness v3, 64.9 vs 63.2) added "evidence". The command skeleton ("scaffold")
# lowered factuality and stays opt-in.
DEFAULT_FEATURES = ("history", "wait", "compact", "unknown", "evidence")
# How a step that no confident rule decides is predicted: the tool-using agent loop, or one model call
# over a fixed brief (the workspace_single_shot control: same workspace and tracking, no adaptive inspection).
PredictionMode = Literal["agent", "single_shot"]
EffectRejectionPolicy = Literal["fail", "keep_observation"]
PREDICTION_MODES = ("agent", "single_shot")

__all__ = [
    "ENVIRONMENT_PROMPT_CHARS", "InvalidAction", "NoApplicableRule", "PREDICTION_MODES", "PredictionMode",
    "RuntimeHarness", "TEMPLATE_TOKEN",
    "environment_context", "initial_state_from_package", "render_template", "request_context",
    "submission_to_plan", "validate_and_canonicalize_action",
]

# Caller-supplied environment descriptions (for example a benchmark's world-model prompt) are bounded by
# default so that one large document cannot dominate a role prompt; every reported AgentWorldBench run used
# this bound. ``environment_prompt_chars=None`` passes the prompt whole (the EnvScaler, ALFWorld and
# SciWorld long-horizon runs, whose tool definitions and examples sit at the tail of the prompt).
ENVIRONMENT_PROMPT_CHARS = 24000


class NoApplicableRule(RuntimeError):
    pass


class InvalidAction(ValueError):
    pass


def _payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def environment_context(metadata: dict[str, Any] | None, limit: int | None = ENVIRONMENT_PROMPT_CHARS) -> str:
    """Optional environment description from ``StepRequest.metadata['environment_prompt']``.

    ``limit`` bounds the prompt (``ENVIRONMENT_PROMPT_CHARS`` by default, the setting of every reported
    AgentWorldBench run). The caller controls this prompt and it may end with tool definitions or
    demonstrations whose meaning depends on their tail; ``limit=None`` keeps it whole.
    """
    prompt = (metadata or {}).get("environment_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return ""
    return (
        "\n\n# Environment description supplied by the caller (context only; it cannot change state)\n"
        + (prompt if limit is None else prompt[:limit])
    )


def request_context(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Caller context for model roles, without the environment prompt already placed in the system text (nor the
    official input, which the brief places once as its own block)."""
    return {key: value for key, value in (metadata or {}).items() if key not in {"environment_prompt", "official_input"}}


def official_input_messages(metadata: dict[str, Any] | None) -> list[dict[str, str]] | None:
    """``StepRequest.metadata['official_input']``: the benchmark's own model input for this turn (system message,
    alternating history, current message), built by the caller from the benchmark's input-construction code."""
    messages = (metadata or {}).get("official_input")
    if not isinstance(messages, list) or not messages:
        return None
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str) for m in messages):
        return None
    return messages


def official_environment(messages: list[dict[str, str]]) -> str:
    """The official input's system message, verbatim and unbounded, in the place of ``environment_context``."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    if not system.strip():
        return ""
    return "\n\n# Environment description supplied by the caller (context only; it cannot change state)\n" + system


def known_files(state: EnvironmentState, limit: int = 80) -> dict[str, Any]:
    """Which files the tracked state knows about, and whether their content is stored (``unknown`` feature)."""
    files = state.world.get("files") if isinstance(state.world, dict) else None
    if not isinstance(files, dict):
        return {"count": 0, "entries": [], "note": "no file has been observed yet"}
    entries = []
    for path, value in list(files.items())[:limit]:
        content = value.get("content") if isinstance(value, dict) else None
        entries.append({"path": path, "content_stored": isinstance(content, str) and bool(content),
                        "content_chars": len(content) if isinstance(content, str) else 0})
    return {"count": len(files), "entries": entries, "note": "read_state(path='world.files.<path>.content') pages a stored content"}


def normalize_citations(citations: list[str], known: set[str]) -> list[str]:
    """Map bare ids the model wrote (``bash-continuation-prompt``, ``evidence_tr_x``) onto the one retrieved
    artifact id they name (``note:bash-continuation-prompt``); anything ambiguous or unknown is kept as written."""
    suffixes: dict[str, list[str]] = {}
    for identifier in known:
        kind, _, rest = identifier.partition(":")
        if rest:
            suffixes.setdefault(rest, []).append(identifier)
    normalized: list[str] = []
    for citation in citations:
        text = str(citation).strip()
        if text in known or not text:
            normalized.append(text or citation)
            continue
        matches = suffixes.get(text) or suffixes.get(text.split(":", 1)[1] if ":" in text else text) or []
        normalized.append(matches[0] if len(matches) == 1 else text)
    return list(dict.fromkeys(normalized))


def last_usage(llm: Any) -> dict[str, float] | None:
    """Token usage of the most recent call on a (possibly wrapped) structured provider, when it reports one."""
    seen = 0
    while llm is not None and seen < 8:
        usage = getattr(llm, "last_usage", None)
        if isinstance(usage, dict):
            return usage
        llm = getattr(llm, "inner", None)
        seen += 1
    return None


def initial_state_from_package(package: EnvironmentPackage) -> EnvironmentState:
    data: dict[str, Any] = {
        "revision": 0,
        "step": 0,
        "world": {},
        "session": {},
        "surface": {},
        "epistemic": {},
        "pending_events": [],
    }
    for field in package.state_schema.fields:
        if field.default is None:
            continue
        parts = field.path.split(".")
        current = data
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = field.default
    return EnvironmentState.model_validate(data)


def submission_to_plan(
    submission: TransitionSubmission, action, rules_by_id: dict[str, TransitionRule]
) -> TransitionPlan:
    """Turn the agent's submission into the harness's typed plan; cited rules fix the rendering contract."""
    cited = [rules_by_id[rule_id] for rule_id in submission.rule_ids if rule_id in rules_by_id]
    renderer_id = cited[0].renderer if cited else "default"
    template = cited[0].observation_template if len(cited) == 1 else None
    citations = list(dict.fromkeys([*submission.citations, *(f"rule:{rule_id}" for rule_id in submission.rule_ids)]))
    return TransitionPlan(
        action=action,
        rule_ids=submission.rule_ids,
        effects=submission.effects,
        outcome=submission.outcome,
        renderer_id=renderer_id,
        observation_template=template,
        uncertainty=submission.uncertainty,
        observation=submission.observation,
        citations=citations,
        rationale=submission.rationale,
    )


class RuntimeHarness:
    def __init__(
        self,
        package_dir: str | Path,
        session_dir: str | Path,
        *,
        planner_llm: StructuredLLM | None = None,
        verifier_llm: StructuredLLM | None = None,
        renderer_llm: StructuredLLM | None = None,
        agent_llm: AgentLLM | None = None,
        verify_package: bool = True,
        max_tool_calls: int = 8,
        allow_unvalidated: bool = False,
        initial_state: EnvironmentState | None = None,
        trust_policy: TrustPolicy = "rules_only",
        fast_path: FastPathPolicy = "confident",
        fast_path_confidence: float = 0.9,
        package: EnvironmentPackage | None = None,
        memory_recent: int = 4,
        text_limit: int = 1500,
        features: set[str] | None = None,
        history_turn_chars: int = 6000,
        history_total_chars: int = 60000,
        history_full: bool = False,
        official_input: bool = False,
        knowledge_gate: bool = False,
        knowledge_gate_mode: str = "full",
        knowledge_judge_llm: StructuredLLM | None = None,
        knowledge_names: str = "paths",
        prediction_mode: PredictionMode = "agent",
        single_shot_llm: StructuredLLM | None = None,
        trace_corpus: TraceCorpus | None = None,
        state_tracking: bool = True,
        package_knowledge: bool = True,
        knowledge_tools: bool = True,
        effect_rejection: EffectRejectionPolicy = "fail",
        environment_prompt_chars: int | None = ENVIRONMENT_PROMPT_CHARS,
    ):
        self.package = package if package is not None else EnvironmentPackage(package_dir, verify_hashes=verify_package)
        if self.package.manifest.metadata.get("validation_status") == "candidate" and not allow_unvalidated:
            raise ValueError("Package is an unvalidated candidate; run replay validation or explicitly allow candidate simulation")
        if (self.package.manifest.metadata.get("construction_kind") in ("reconstructed", "ablated")
                and initial_state is None and not (Path(session_dir) / "session.sqlite").exists()):
            raise ValueError("A new reconstructed session requires explicit initial state; schema defaults are not episode evidence")
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"Unknown prediction mode {prediction_mode!r}; known: {PREDICTION_MODES}")
        self.prediction_mode: PredictionMode = prediction_mode
        self.single_shot_llm = single_shot_llm
        self.trace_corpus = trace_corpus
        # False for the trace2env_no_state control: no state tools, no state in the brief, no effects, no rule
        # route (rule conditions read a state that nothing maintains).
        self.state_tracking = state_tracking
        # False for the harness_only control: the agent sees nothing from the package — no action specs, rules,
        # contracts, demonstrations, notes, evidence, or knowledge tools — only the environment prompt, the
        # episode's history, and the memory tools. The action is not canonicalized through package aliases either.
        self.package_knowledge = package_knowledge
        # False for the schema_only_hv3 control: the schemas, state tracking, memory tools, and loop stay, but the
        # knowledge tools (list_actions, inspect_action, search_knowledge, read_evidence) and similar-turn retrieval go.
        self.knowledge_tools = knowledge_tools
        # "keep_observation": when a model-proposed transition's effects fail validation or verification, commit the
        # step with the effects dropped and the agent's observation kept (state unchanged, warning recorded) instead of
        # failing the step; "fail" (default) keeps the fail-closed behaviour.
        self.effect_rejection: EffectRejectionPolicy = effect_rejection
        # Bound on metadata["environment_prompt"] in the agent and renderer system text (None = whole prompt).
        self.environment_prompt_chars = environment_prompt_chars
        self.inspector = PackageInspector(self.package)
        self.session = SessionStore(session_dir, initial_state if initial_state is not None else initial_state_from_package(self.package))
        self.memory = EpisodicMemory(self.session.database)
        # A StructuredLLM planner is driven through typed AgentTurn decisions; native tool calling is preferred.
        self.agent_llm: AgentLLM | None = agent_llm if agent_llm is not None else (
            StructuredAgentLLM(planner_llm) if planner_llm is not None else None)
        self.verifier_llm = verifier_llm
        self.renderer_llm = renderer_llm
        self.max_tool_calls = max_tool_calls
        self.trust_policy: TrustPolicy = trust_policy
        self.fast_path: FastPathPolicy = fast_path
        self.fast_path_confidence = fast_path_confidence
        self.memory_recent = memory_recent
        self.text_limit = text_limit
        unknown = set(features or ()) - set(HARNESS_FEATURES)
        if unknown:
            raise ValueError(f"Unknown harness features: {sorted(unknown)}; known: {HARNESS_FEATURES}")
        self.features = set(features or ())
        self.history_turn_chars = history_turn_chars
        # Harness v4 information parity: the brief carries every observed turn of the session verbatim, ignoring
        # the history window and the character budgets (which still size evidence pages), so the world model sees
        # exactly the transcript the prompting layout sees.
        self.history_full = history_full
        # Harness v5.1 information preservation: the caller's official model input (metadata["official_input"], built
        # by the benchmark's own input-construction code) is placed once, verbatim and unclipped: its system message in
        # the system text instead of environment_context, its turn messages as the brief's first block instead of the
        # budgeted recent_memory views. Loop, tools, tracking, package access, and verification are unchanged.
        self.official_input = official_input
        # Harness v5.2: applicability-gated package knowledge. Every evidence turn, demonstration, and note shown to the
        # agent (brief and tools) gets a label, provenance, reason, and disposition; foreign concrete values are masked
        # in the model-facing view; contradicted notes are dropped; the brief says when the gate abstains. Off = v5.1.
        self.knowledge_gate = knowledge_gate
        self.knowledge_gate_mode = knowledge_gate_mode  # "full" (v5.2) or "format_only" (pre-analysis ablation)
        self.knowledge_judge_llm = knowledge_judge_llm  # v5.3: the applicability judge (propose, then verify anchors)
        self.knowledge_names = knowledge_names  # v5.3.2: "pages" = page identity as the supporting signal (web); "paths" = v5.3.1
        self.history_total_chars = history_total_chars

    # ─── One action ───────────────────────────────────────────────────────────

    def step(self, request: StepRequest) -> StepResult:
        before = self.session.load()
        audit = AuditRecord(prior_revision=before.revision, request=request)
        try:
            working, due_ids = process_due_events(before, before.step + 1)
            working.step = before.step + 1
            retrieval = self.inspector.inspect_action(request.action.type, state=working)
            if not retrieval["action_specs"]:
                retrieval["discovery"] = self.inspector.search(request.action.type)
            action = validate_and_canonicalize_action(request.action, retrieval) if self.package_knowledge else request.action
            effective_request = request.model_copy(update={"action": action})
            rules = [TransitionRule.model_validate(item) for item in retrieval["rules"]]
            plan, route, tools = self._decide(effective_request, working, retrieval, rules)
            if plan.action.type != action.type or plan.action.arguments != action.arguments:
                raise ValueError("Planner changed the requested action")
            audit.route = route
            audit.tool_calls = list(tools.calls) if tools is not None else []
            audit.retrieved_artifacts = list(dict.fromkeys(retrieval["artifact_ids"]))
            audit.plan = plan
            audit.citations = list(plan.citations)
            action_spec = next((spec for spec in self.package.action_schema.actions if spec.name == plan.action.type), None)
            try:
                validate_mutations(plan.effects, self.package.state_schema, action_spec)
                candidate = apply_mutations(working, plan.effects, action=plan.action)
                validate_state_types(candidate, self.package.state_schema)
                verification = self._verify(working, candidate, plan, retrieval, rules, tools)
                if not verification.accepted:
                    raise RuntimeError("Transition verification failed: " + "; ".join(i.message for i in verification.issues))
            except (ValueError, RuntimeError, KeyError, TypeError) as exc:
                if (self.effect_rejection != "keep_observation" or route not in {"agent", "single_shot"}
                        or plan.observation is None):
                    raise
                # The observation is the agent's prediction; the effects were the part that failed. Keep the former,
                # drop the latter, leave the state untouched, and say so in the audit.
                plan = plan.model_copy(update={"effects": [], "rule_ids": [], "observation_template": None})
                audit.plan = plan
                candidate = working.model_copy(deep=True)
                verification = VerificationResult(accepted=True, issues=[VerificationIssue(
                    code="effects_rejected", message=f"{type(exc).__name__}: {exc}"[:400], severity="warning")])
            if due_ids:
                verification.issues.append(
                    VerificationIssue(
                        code="due_events_applied",
                        message=f"Applied due events: {', '.join(due_ids)}",
                        severity="warning",
                    )
                )
            audit.verification = verification
            observation = self._render(before, candidate, plan, retrieval, request.metadata)
            audit.observation = observation
            committed = self.session.commit(before.revision, candidate, audit)
            self.memory.record(MemoryEntry(turn=committed.step, kind="predicted", action=plan.action,
                                           observation=observation, effects=plan.effects, citations=plan.citations))
            return StepResult(
                observation=observation,
                outcome=plan.outcome,
                state=committed,
                plan=plan,
                verification=verification,
                retrieved_artifacts=audit.retrieved_artifacts,
                route=route,
                tool_calls=sum(1 for call in audit.tool_calls if call.get("tool") not in {SUBMIT_TOOL_NAME, "_usage"}),
                citations=plan.citations,
            )
        except BaseException as exc:
            audit.error = str(exc)
            self.session.record_failure(audit)
            raise

    # ─── Deciding the transition ──────────────────────────────────────────────

    def _expose_rule(self, rule: TransitionRule, retrieval: dict[str, Any], rules: list[TransitionRule]) -> None:
        if rule.id not in {item.id for item in rules}:
            rules.append(rule)
            retrieval["rules"].append(rule.model_dump(mode="json"))
            retrieval["artifact_ids"].append(f"rule:{rule.id}")
        if rule.renderer not in {item["id"] for item in retrieval["renderers"]}:
            contract = next((item for item in self.package.renderers if item.id == rule.renderer), None)
            if contract is not None:
                retrieval["renderers"].append(contract.model_dump(mode="json"))
                retrieval["artifact_ids"].append(f"renderer:{contract.id}")

    def _fast_path_applies(self, rule: TransitionRule) -> bool:
        if self.fast_path == "always":
            return True
        if self.fast_path == "never":
            return False
        contract = next((item for item in self.package.renderers if item.id == rule.renderer), None)
        templated = bool(rule.observation_template or (contract and contract.template))
        return templated and rule.confidence >= self.fast_path_confidence

    def _decide(
        self,
        request: StepRequest,
        state: EnvironmentState,
        retrieval: dict[str, Any],
        rules: list[TransitionRule],
    ) -> tuple[TransitionPlan, str, WorkspaceTools | None]:
        # Check all eligible rules before bounded retrieval can hide an overlap or a
        # lower-ranked applicable branch. Only the chosen rule needs to enter context.
        candidates = [r for r in self.package.rules if rule_eligible(r, self.package.scope, state,
            minimum_confidence=self.package.manifest.metadata.get("min_rule_confidence", 0.0))]
        if "wait" in self.features:
            # A rule without effects and without a template cannot produce a transition; it must not
            # count as "the applicable rule" (that would route a wait to a plan that predicts nothing).
            templated = {contract.id for contract in self.package.renderers if contract.template}
            candidates = [r for r in candidates if r.effects or r.observation_template or r.renderer in templated]
        rule = (select_rule(candidates, state, request.action, self.package.scope)
                if self.state_tracking and self.package_knowledge else None)
        model_available = self.single_shot_llm is not None if self.prediction_mode == "single_shot" else self.agent_llm is not None
        if rule is not None:
            self._expose_rule(rule, retrieval, rules)
            if not model_available or (self._fast_path_applies(rule) and not self._format_documented(request)):
                return deterministic_plan(rule, request.action), "rule", None
        elif not model_available:
            raise NoApplicableRule(
                f"No reconstructed rule applies to action {request.action.type!r}; configure a world-model agent for it"
            )
        if self.prediction_mode == "single_shot":
            return self._single_shot_decide(request, state, retrieval, rules, rule)
        return self._agent_decide(request, state, retrieval, rules, rule)

    def _workspace(self, state: EnvironmentState, request: StepRequest) -> WorkspaceTools:
        return WorkspaceTools(
            self.package, self.inspector, self.memory, state, request.action, text_limit=self.text_limit,
            minimum_confidence=float(self.package.manifest.metadata.get("min_rule_confidence", 0.0)),
            features=set(self.features), brief_action_type=request.action.type, turn_chars=self.history_turn_chars,
            trace_corpus=self.trace_corpus, state_tools=self.state_tracking,
            knowledge_tools=self.package_knowledge and self.knowledge_tools,
            gate=self._gate(state, request) if self.knowledge_gate and self.package_knowledge else None,
            withhold_templates=self._format_documented(request) and self.package_knowledge,
        )

    def _format_documented(self, request: StepRequest, action_type: str | None = None) -> bool:
        """Option B: the official system message documents this action's observation format with an example."""
        if not (self.knowledge_gate and self.official_input):
            return False
        official = official_input_messages(request.metadata)
        system = official[0]["content"] if official and official[0].get("role") == "system" else ""
        return documented_format(system, action_type or request.action.type)

    def _gate(self, state: EnvironmentState, request: StepRequest) -> KnowledgeGate:
        """The v5.2 gate for this step: the episode's own transcript (official messages when present, else the
        remembered observations) and, conservatively, its tracked state define what is known here."""
        official = official_input_messages(request.metadata)
        if official:
            # The history only: the current message (the action to predict) is not part of what the episode showed;
            # its current-state block does tell the v5.3.2 gate which page the episode is on.
            texts = [m["content"] for m in official[:-1] if m["role"] != "system"]
            current_text = official[-1]["content"]
        else:
            texts = [f"{action_text(entry.action)}\n{entry.observation}" for entry in self.memory.recent(1_000_000)]
            current = (request.metadata or {}).get("current_input")
            current_text = current if isinstance(current, str) else None
        context = build_episode_context(texts, state if self.state_tracking else None, current_text=current_text)
        return KnowledgeGate(self.inspector, context, request.action,
                             mode=self.knowledge_gate_mode, judge_llm=self.knowledge_judge_llm,
                             format_documented=self._format_documented(request), names=self.knowledge_names)

    def _system_text(self, request: StepRequest, *, single_shot: bool = False) -> str:
        features = self.features
        official = official_input_messages(request.metadata) if self.official_input else None
        system = (WORLD_MODEL_SINGLE_SHOT if single_shot else WORLD_MODEL_AGENT) + (
            official_environment(official) if official else environment_context(request.metadata, self.environment_prompt_chars))
        if not single_shot:
            if self.trust_policy == "schema_checked":
                system += ("\n\nTrust policy: schema_checked. When no rule explains the action you may propose literal "
                           "effects on declared, mutable state paths; they are accepted if they pass type and invariant checks.")
            else:
                system += ("\n\nTrust policy: rules_only. Effects are accepted only when a cited rule supports them "
                           "verbatim; otherwise submit no effects and describe the outcome in the observation.")
            if "history" in features and not official:  # the official block replaces the budgeted recent_memory view
                system += "\n\n" + WORLD_MODEL_AGENT_HISTORY
            if self.trace_corpus is not None:
                system += "\n\n" + WORLD_MODEL_AGENT_TRACES
            if "evidence" in features:
                system += "\n\n" + WORLD_MODEL_AGENT_EVIDENCE
        elif "evidence" in features:
            system += "\n\n" + WORLD_MODEL_SINGLE_SHOT_EVIDENCE
        if "unknown" in features and self.state_tracking:
            system += "\n\n" + WORLD_MODEL_AGENT_UNKNOWN
        if not self.state_tracking:
            system += "\n\n" + WORLD_MODEL_AGENT_NO_STATE
        if not self.package_knowledge:
            system += "\n\n" + WORLD_MODEL_AGENT_NO_PACKAGE
        elif not self.knowledge_tools:
            system += "\n\n" + WORLD_MODEL_AGENT_NO_KNOWLEDGE_TOOLS
        if "scaffold" in features:
            system += "\n\n" + WORLD_MODEL_AGENT_SCAFFOLD
        if "wait" in features or "scaffold" in features:
            system += "\n\n" + WORLD_MODEL_AGENT_WAIT
        if official:
            system += "\n\n" + WORLD_MODEL_AGENT_OFFICIAL_INPUT
        if self.knowledge_gate and self.package_knowledge:
            system += "\n\n" + WORLD_MODEL_AGENT_KNOWLEDGE_GATE
        return system

    def _recent_entries(self, request: StepRequest) -> list[MemoryEntry]:
        if self.history_full or (self.official_input and official_input_messages(request.metadata)):
            return self.memory.recent(1_000_000)  # every observed turn, oldest first
        recent = request.metadata.get("history_window", self.memory_recent)
        recent = self.memory_recent if not isinstance(recent, int) else recent
        return self.memory.recent(recent)

    def _brief(
        self,
        request: StepRequest,
        state: EnvironmentState,
        retrieval: dict[str, Any],
        rules: list[TransitionRule],
        applicable: TransitionRule | None,
        tools: WorkspaceTools,
        entries: list[MemoryEntry],
    ) -> dict[str, Any]:
        """The agent's first user message: action, context, state, recent turns, and the action's retrieved knowledge."""
        features = self.features
        demonstrations = clip_value(retrieval["demonstrations"][:2], self.text_limit // 2)
        notes = retrieval.get("notes", [])
        if tools.gate is not None:
            notes = tools._gate_notes(notes)  # v5.2: contradicted notes dropped, others labelled
        if "evidence" in features:
            # v3 read path: a demonstration is a raw turn; show it as a summary the agent can page exactly.
            demonstrations = [view for view in (tools.demo_view(demo) for demo in retrieval["demonstrations"][:2]) if not view.get("withheld")]
        elif "compact" in features:
            demonstrations = [{**demo, "action": compact_action(demo["action"]), "provenance": []} for demo in demonstrations]
            notes = [{key: note[key] for key in ("id", "kind", "statement", "action_types", "confidence") if key in note}
                     for note in notes]
        official = official_input_messages(request.metadata) if self.official_input else None
        brief: dict[str, Any] = {
            "action": compact_action(request.action) if "compact" in features else request.action.model_dump(mode="json"),
            "context": request_context(request.metadata),
            "state_summary": state_summary(state, text_limit=self.text_limit // 8),
            "recent_memory": [] if official else self._history_views(entries, tools),
            "retrieved": {
                "action_specs": retrieval["action_specs"],
                "rules": [
                    {"id": rule.id, "description": rule.description, "outcome": rule.outcome.value,
                     "confidence": rule.confidence,
                     "applies_now": all(evaluate_condition(c, state, request.action) for c in rule.conditions)}
                    for rule in rules
                ],
                "applicable_rule": applicable.id if applicable else None,
                "invariants": retrieval["invariants"],
                "notes": notes,
                "renderers": [
                    {"id": item["id"], "content_type": item["content_type"], "instructions": item["instructions"],
                     "required_fields": item["required_fields"], "examples": clip_value(item["examples"][:3], self.text_limit // 2)}
                    for item in retrieval["renderers"]
                ],
                "demonstrations": demonstrations,
            },
            "budget": {"tool_calls": self.max_tool_calls},
        }
        if not self.state_tracking:
            brief.pop("state_summary", None)
            brief["state"] = "not tracked in this session (no state tools); infer the environment from recent turns, memory, and knowledge"
        if "compact" in features and self.state_tracking:
            brief["state_fields"] = [{"path": field.path, "type": field.type} for field in self.package.state_schema.fields]
        if "unknown" in features and self.state_tracking:
            brief["state_semantics"] = ABSENCE_MEANS_UNKNOWN
            brief["known_files"] = known_files(state)
        if "scaffold" in features or "wait" in features:
            # The command skeleton needs the scaffold feature; the pending-work hint for waits and key
            # presses is part of wait handling and is included whenever either feature is on.
            scaffold = self._scaffold(request.action, entries, state)
            if scaffold is not None and ("scaffold" in features or "skeleton" not in scaffold):
                brief["transcript_scaffold"] = scaffold
        if self.trace_corpus is not None:
            # Raw-trace control: the turns of other episodes whose action resembles this one stand in for
            # the action's rules, notes, and demonstrations (which such a package does not carry).
            brief["trace_hits"] = tools.trace_hits(action_query(request.action), 5)
            brief["trace_corpus"] = {"episodes": self.trace_corpus.episodes, "turns": self.trace_corpus.turns}
        if "evidence" in features and self.package_knowledge and self.knowledge_tools:
            # v3: the package's own raw turns whose action resembles this one, as summaries with read pointers.
            brief["similar_turns"] = tools.evidence_hits(action_query(request.action), 5)
        if tools.gate is not None:
            # v5.2: what the gate decided about everything the brief shows, and whether it abstains.
            brief["package_applicability"] = tools.gate.brief_notice()
        if not self.package_knowledge:
            brief.pop("retrieved", None)
            brief.pop("state_fields", None)
            brief["package"] = "none attached (harness_only): no actions, rules, contracts, demonstrations, notes, or evidence"
        if official:
            brief = {"official_input": self._official_block(official, entries), **{k: v for k, v in brief.items() if k != "recent_memory"}}
        return brief

    @staticmethod
    def _official_block(official: list[dict[str, str]], entries: list[MemoryEntry]) -> dict[str, Any]:
        """The official input's turn messages, verbatim and in order, each observation labelled with its memory id."""
        turn_ids = {entry.turn: entry.id for entry in entries if entry.kind == "observed"}
        messages: list[dict[str, Any]] = []
        turn = 0
        for message in official:
            if message["role"] == "system":
                continue  # placed once in the system text
            item: dict[str, Any] = {"role": message["role"], "content": message["content"]}
            if message["role"] == "assistant":
                turn += 1
                if turn in turn_ids:
                    item["turn"] = turn
                    item["memory_id"] = f"memory:{turn_ids[turn]}"
            messages.append(item)
        return {
            "description": ("Verbatim official input for this turn: the earlier turns (user: '### Turn k' with the action; "
                            "assistant: '**Environment Observation:**' with the real screen) and, last, the current turn's "
                            "message with the action to predict. The environment description is in the system text."),
            "messages": messages,
        }

    def _agent_decide(
        self,
        request: StepRequest,
        state: EnvironmentState,
        retrieval: dict[str, Any],
        rules: list[TransitionRule],
        applicable: TransitionRule | None,
    ) -> tuple[TransitionPlan, str, WorkspaceTools]:
        assert self.agent_llm is not None
        features = self.features
        tools = self._workspace(state, request)
        system = self._system_text(request)
        entries = self._recent_entries(request)
        brief = self._brief(request, state, retrieval, rules, applicable, tools, entries)
        user = _payload(brief)
        loop = AgentLoop(self.agent_llm, tools, system=system, max_tool_calls=self.max_tool_calls)
        submission, trace, usage = loop.run(user)
        if "wait" in features:
            # A rule with no effects and no template cannot support effects; citing it is evidence, not an
            # application, so it moves from rule_ids to citations instead of making the step unverifiable.
            templated = {contract.id for contract in self.package.renderers if contract.template}
            vacuous = {rule.id for rule in [*rules, *tools.discovered_rules.values()]
                       if not rule.effects and not rule.observation_template and rule.renderer not in templated}
            if any(rule_id in vacuous for rule_id in submission.rule_ids):
                submission = submission.model_copy(update={
                    "rule_ids": [rule_id for rule_id in submission.rule_ids if rule_id not in vacuous],
                    "citations": list(dict.fromkeys([*submission.citations,
                                                     *(f"rule:{rule_id}" for rule_id in submission.rule_ids if rule_id in vacuous)])),
                })
        if not self.state_tracking and submission.effects:
            submission = submission.model_copy(update={"effects": [], "rule_ids": []})  # nothing maintains state to apply them to
        tools.calls = trace
        if usage.get("model_calls"):
            tools.calls.append({"tool": "_usage", "arguments": usage})
        if tools.gate is not None:
            tools.calls.append({"tool": "_knowledge_gate", "arguments": tools.gate.summary()})
        for rule_id, discovered in tools.discovered_rules.items():
            if rule_id not in {item.id for item in rules}:
                self._expose_rule(discovered, retrieval, rules)
        for entry in entries:
            # Recent memory was shown in the brief, so the agent may legitimately cite it.
            tools.retrieved_ids.add(f"memory:{entry.id}")
        retrieval["artifact_ids"].extend(sorted(tools.retrieved_ids - set(retrieval["artifact_ids"])))
        known = tools.retrieved_ids | set(retrieval["artifact_ids"]) | {f"rule:{rule.id}" for rule in rules}
        submission = submission.model_copy(update={"citations": normalize_citations(submission.citations, known)})
        plan = submission_to_plan(submission, request.action, {rule.id: rule for rule in rules})
        for rule in rules:
            if rule.id in plan.rule_ids:
                self._expose_rule(rule, retrieval, rules)
        return plan, "agent", tools

    def _single_shot_decide(
        self,
        request: StepRequest,
        state: EnvironmentState,
        retrieval: dict[str, Any],
        rules: list[TransitionRule],
        applicable: TransitionRule | None,
    ) -> tuple[TransitionPlan, str, WorkspaceTools]:
        """One model call over the brief plus a fixed retrieval procedure; no tools, no adaptive inspection.

        The fixed procedure is what the agent's first tool calls usually do: full-text knowledge search
        for the action's command words, recall of earlier turns matching them, and a dry run of the
        applicable rule. Everything shown is registered so the prediction's citations can be checked.
        """
        assert self.single_shot_llm is not None
        tools = self._workspace(state, request)
        system = self._system_text(request, single_shot=True)
        entries = self._recent_entries(request)
        brief = self._brief(request, state, retrieval, rules, applicable, tools, entries)
        brief.pop("budget", None)
        query = action_query(request.action)
        shown = {entry.id for entry in entries}
        evidence_pages: list[dict[str, Any]] = []
        if "evidence" in self.features and query:
            # v3: the three most similar raw turns, first page verbatim, instead of clipped evidence hits.
            for hit in tools.evidence_hits(query, 3):
                page = tools.inspector.evidence_view(hit["id"].split(":", 1)[1], length=self.history_turn_chars)
                if page is not None:
                    evidence_pages.append(page)
            knowledge = tools.inspector.search_knowledge(query, kinds=["rule", "note", "action"], limit=8)
        else:
            knowledge = tools.inspector.search_knowledge(query, limit=8) if query else []
        tools.retrieved_ids.update(item["id"] for item in knowledge)
        recalled = [entry for entry in self.memory.search(query, 6) if entry.id not in shown][:3] if query else []
        brief["fixed_retrieval"] = {
            "query": query,
            "knowledge_hits": [{key: item[key] for key in ("id", "kind", "action_type")} | {"text": clip_value(item["text"], self.text_limit)}
                               for item in knowledge],
            "recalled_turns": [tools._memory_view(entry, full=False) for entry in recalled],
            "applicable_rule_dry_run": tools._apply_rule({"rule_id": applicable.id}) if applicable is not None else None,
        }
        if "evidence" in self.features:
            brief["fixed_retrieval"]["similar_turns"] = evidence_pages
            brief.pop("similar_turns", None)
        for entry in entries:
            tools.retrieved_ids.add(f"memory:{entry.id}")
        prediction = self.single_shot_llm.complete(
            system=system, user=_payload(brief), response_model=SingleShotPrediction, role="runtime_single_shot")
        usage = last_usage(self.single_shot_llm) or {}
        tools.calls = [{"tool": "_usage", "arguments": {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                                                        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                                                        "model_calls": 1}}]
        if tools.gate is not None:
            tools.calls.append({"tool": "_knowledge_gate", "arguments": tools.gate.summary()})
        retrieval["artifact_ids"].extend(sorted(tools.retrieved_ids - set(retrieval["artifact_ids"])))
        known = tools.retrieved_ids | set(retrieval["artifact_ids"]) | {f"rule:{rule.id}" for rule in rules}
        plan = TransitionPlan(
            action=request.action, outcome=prediction.outcome, observation=prediction.observation,
            uncertainty=prediction.uncertainty, citations=normalize_citations(prediction.citations, known),
            rationale=prediction.rationale,
        )
        return plan, "single_shot", tools

    def _history_views(self, entries: list[MemoryEntry], tools: WorkspaceTools) -> list[dict[str, Any]]:
        """The brief's view of recent turns: verbatim within a byte budget (``history``), else short clips."""
        if "history" not in self.features:
            return [
                {"turn": entry.turn, "kind": entry.kind, "action": entry.action.model_dump(mode="json", exclude={"raw"}),
                 "observation": clip_value(entry.observation, self.text_limit // 2), "id": f"memory:{entry.id}"}
                for entry in entries
            ]
        if self.history_full:
            views_full = []
            for entry in entries:
                view = tools._memory_view(entry, full=False)
                view["observation"] = entry.observation  # complete, no budget, no paging marker
                views_full.append(view)
            return views_full
        views: list[dict[str, Any]] = []
        used = 0
        for entry in reversed(entries):  # newest first: it keeps its full text; older turns give way
            view = tools._memory_view(entry, full=False)
            if used + len(view["observation"]) > self.history_total_chars:
                view["observation"] = clip_span(entry.observation, max(400, self.history_turn_chars // 6), turn=entry.turn)
            used += len(view["observation"])
            views.append(view)
        return list(reversed(views))

    def _scaffold(self, action, entries: list[MemoryEntry], state: EnvironmentState) -> dict[str, Any] | None:
        """Deterministic transcript skeleton from the last observed prompt line and the tracked cwd."""
        # The prompt's user/host is the signature most recent captures agree on (robust to one garbled
        # line); the cwd comes from the last capture when it ended idle at a prompt, else from state.
        signature = prompt_signature([entry.observation for entry in entries[-8:]])
        last = last_prompt(entries[-1].observation) if entries else None
        idle = bool(last and last.get("idle_at_end") == "True")
        prompt = dict(signature) if signature else None
        if prompt and last and idle:
            prompt["cwd"] = last["cwd"]
        cwd = prompt["cwd"] if (prompt and idle) else state.session.get("cwd") or (prompt or {}).get("cwd")
        typed = "".join(str(e.get("keystrokes", "")) for e in (action.arguments.get("keystrokes") or []) if isinstance(e, dict))
        if not typed.strip() or action.type == "keys" or "keys" in action.arguments or (entries and not idle):
            # A wait, a key press, or keystrokes typed into a program that still owns the terminal: the
            # capture continues whatever the previous capture left pending, so no shell skeleton applies.
            hint = wait_hint(entries[-1].observation if entries else "")
            hint["last_prompt"] = (prompt or {}).get("prompt")
            if typed.strip():
                hint["keys_sent"] = typed[:2000]
                hint["guidance"] += (" The keystrokes above go to whatever owns the terminal: a shell echoes them after its "
                                     "prompt, a program consumes them (C-c interrupts it), a prompt for input takes them as the answer.")
            return hint
        scaffold = build_scaffold(action.arguments, prompt, cwd if isinstance(cwd, str) else None)
        return scaffold

    # ─── Verification and rendering ───────────────────────────────────────────

    def _verify(
        self,
        before: EnvironmentState,
        after: EnvironmentState,
        plan: TransitionPlan,
        retrieval: dict[str, Any],
        rules: list[TransitionRule],
        tools: WorkspaceTools | None = None,
    ) -> VerificationResult:
        result = check_invariants(self.package.invariants, after, plan.action)
        # A violation that already held before the step is a package diagnostic, not this transition's fault.
        preexisting = {issue.code for issue in check_invariants(self.package.invariants, before, plan.action).issues}
        for issue in result.issues:
            if issue.code in preexisting and issue.severity == "error":
                issue.severity = "warning"
                issue.code = issue.code.replace("invariant:", "invariant_preexisting:", 1)
                issue.message = "Already violated before this transition: " + issue.message
        result.accepted = not any(issue.severity == "error" for issue in result.issues)
        if not self.state_tracking:
            result.issues.append(VerificationIssue(code="state_tracking_disabled", severity="warning",
                                                   message="State tracking is disabled for this session; effects are not applied"))
        result.issues.extend(verify_rule_support(plan, rules, before, self.package.scope, trust_policy=self.trust_policy))
        render_context = {
            "action": plan.action.model_dump(mode="python"),
            "state_before": before.model_dump(mode="python"),
            "state_after": after.model_dump(mode="python"),
            "outcome": plan.outcome.value,
        }
        contracts = [RenderContract.model_validate(item) for item in retrieval["renderers"]]
        contract = next((item for item in contracts if item.id == plan.renderer_id), None)
        if contract and (plan.observation_template or contract.template or plan.observation is None):
            for path in contract.required_fields:
                if get_path(render_context, path, MISSING) is MISSING:
                    result.issues.append(
                        VerificationIssue(
                            code="missing_render_field",
                            message=f"Renderer {contract.id} requires absent field {path}",
                            severity="error",
                        )
                    )
        if tools is not None and plan.citations:
            known = tools.retrieved_ids | set(retrieval["artifact_ids"]) | {f"rule:{rule.id}" for rule in rules}
            unknown = [citation for citation in plan.citations if citation not in known]
            if unknown:
                result.issues.append(
                    VerificationIssue(
                        code="uncited_reference",
                        message="Citations were not retrieved during this step: " + ", ".join(unknown),
                        severity="warning",
                    )
                )
        if any(issue.severity == "error" for issue in result.issues):
            result.accepted = False
        if result.accepted and self.verifier_llm is not None:
            independent = self.verifier_llm.complete(
                system=RUNTIME_VERIFIER,
                user=_payload(
                    {
                        "state_before": before.model_dump(mode="json"),
                        "state_after": after.model_dump(mode="json"),
                        "plan": plan.model_dump(mode="json"),
                        "retrieved_package_artifacts": retrieval,
                        "deterministic_checks": result.model_dump(mode="json"),
                    }
                ),
                response_model=VerificationResult,
                role="runtime_verify",
            )
            result.accepted = result.accepted and independent.accepted
            result.confidence = min(result.confidence, independent.confidence)
            result.issues.extend(independent.issues)
        return result

    def _render(
        self,
        before: EnvironmentState,
        after: EnvironmentState,
        plan: TransitionPlan,
        retrieval: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        contracts = [RenderContract.model_validate(item) for item in retrieval["renderers"]]
        contract = next((item for item in contracts if item.id == plan.renderer_id), None)
        template = plan.observation_template or (contract.template if contract else None)
        if template and self.knowledge_gate and self.official_input:
            official = official_input_messages(metadata)
            system = official[0]["content"] if official and official[0].get("role") == "system" else ""
            if documented_format(system, plan.action.type):
                template = None  # option B: the official example's format is authoritative; the agent's observation stands
        if template:
            # A cited rule's template is exact, reproducible knowledge; it outranks free-form rendering.
            context = {
                "action": plan.action.model_dump(mode="python"),
                "state_before": before.model_dump(mode="python"),
                "state_after": after.model_dump(mode="python"),
                "outcome": plan.outcome.value,
            }
            return render_template(template, context)
        if plan.observation is not None:
            return plan.observation
        if self.renderer_llm is None:
            return json.dumps(
                {"outcome": plan.outcome.value, "state": after.model_dump(mode="json")},
                ensure_ascii=False,
            )
        rendered = self.renderer_llm.complete(
            system=RUNTIME_RENDERER + environment_context(metadata, self.environment_prompt_chars),
            user=_payload(
                {
                    "state_before": before.model_dump(mode="json"),
                    "state_after": after.model_dump(mode="json"),
                    "verified_plan": plan.model_dump(mode="json"),
                    "render_contracts": retrieval["renderers"],
                    "demonstrations": retrieval["demonstrations"],
                    "context": request_context(metadata),
                }
            ),
            response_model=RenderedObservation,
            role="runtime_render",
        )
        return rendered.observation


TYPE_CHECKS = {
    "string": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "object": lambda value: isinstance(value, dict),
    "array": lambda value: isinstance(value, list),
    "any": lambda value: True,
}


def validate_and_canonicalize_action(action, retrieval: dict[str, Any]):
    specs = retrieval["action_specs"]
    if not specs:
        return action
    spec = specs[0]
    for name, argument_spec in spec.get("arguments", {}).items():
        if argument_spec.get("required") and name not in action.arguments:
            raise InvalidAction(f"Missing required argument {name!r} for action {spec['name']!r}")
        if name not in action.arguments:
            continue
        value = action.arguments[name]
        expected = argument_spec.get("type", "any")
        if not TYPE_CHECKS[expected](value):
            raise InvalidAction(f"Argument {name!r} must be {expected}, got {type(value).__name__}")
        allowed = argument_spec.get("enum")
        if allowed is not None and value not in allowed:
            raise InvalidAction(f"Argument {name!r} must be one of {allowed!r}")
    return action.model_copy(update={"type": retrieval["canonical_action_type"]})
