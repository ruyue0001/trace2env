"""Typed intermediate representation shared by reconstruction and runtime."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Actor(str, Enum):
    USER = "user"
    AGENT = "agent"
    ENVIRONMENT = "environment"
    TOOL = "tool"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class EventKind(str, Enum):
    MESSAGE = "message"
    ACTION = "action"
    OBSERVATION = "observation"
    STATE = "state"
    ERROR = "error"
    META = "meta"


class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    HYPOTHESIZED = "hypothesized"
    CONFLICTED = "conflicted"


class Outcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Confidence(StrictModel):
    value: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class SourceRef(StrictModel):
    source_id: str
    episode_id: str | None = None
    event_ids: list[str] = Field(default_factory=list)
    locator: str | None = None
    excerpt: str | None = None


class RawEvent(StrictModel):
    id: str
    actor: Actor = Actor.UNKNOWN
    kind: EventKind = EventKind.MESSAGE
    content: Any
    timestamp: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_payload: Any = None


class Episode(StrictModel):
    id: str
    source_id: str
    events: list[RawEvent]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransitionSlice(StrictModel):
    id: str
    episode_id: str
    history_event_ids: list[str] = Field(default_factory=list)
    action_event_ids: list[str]
    observation_event_ids: list[str]
    delayed_observation_event_ids: list[str] = Field(default_factory=list)
    concurrent_action_event_ids: list[str] = Field(default_factory=list)
    alignment: Literal["correlated", "sequential", "ambiguous"] = "sequential"
    ambiguities: list[str] = Field(default_factory=list)


class EventAnnotation(StrictModel):
    source_event_id: str
    actor: Actor
    kind: EventKind
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class TraceAnnotations(StrictModel):
    annotations: list[EventAnnotation]


class NormalizedAction(StrictModel):
    type: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw: Any = None


class Fact(StrictModel):
    subject: str
    predicate: str
    value: Any
    status: EpistemicStatus = EpistemicStatus.OBSERVED
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    provenance: list[SourceRef] = Field(default_factory=list)


class StateMutation(StrictModel):
    # ``merge`` updates entries of an object-valued path with the keys of an object value and ``remove``
    # deletes the named key(s) from it: this is how maps keyed by strings that contain dots (file paths,
    # URLs, identifiers) are edited, since dotted state paths cannot address such keys.
    op: Literal["set", "delete", "increment", "decrement", "append", "create", "schedule", "merge", "remove"]
    path: str
    value: Any = None
    delay_steps: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def valid_state_path(cls, value: str) -> str:
        if value.split(".", 1)[0] not in {"world", "session", "surface", "epistemic"}:
            raise ValueError("mutation path must start with world, session, surface, or epistemic")
        return value


class LocalTransitionEvidence(StrictModel):
    id: str
    episode_id: str
    transition_id: str
    preconditions: list[Fact] = Field(default_factory=list)
    action: NormalizedAction
    outcome: Outcome = Outcome.UNKNOWN
    mutations: list[StateMutation] = Field(default_factory=list)
    observation_facts: list[Fact] = Field(default_factory=list)
    # Filled from the source events by the reconstructor; the extractor is told to leave it empty.
    observation_text: str = ""
    latent_variables: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    confidence: Confidence
    provenance: list[SourceRef] = Field(default_factory=list)


class ArgumentSpec(StrictModel):
    type: Literal["string", "integer", "number", "boolean", "object", "array", "any"] = "any"
    required: bool = False
    description: str = ""
    enum: list[Any] | None = None


class ActionSpec(StrictModel):
    name: str
    description: str = ""
    arguments: dict[str, ArgumentSpec] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)


class ActionSchema(StrictModel):
    actions: list[ActionSpec]


class StateField(StrictModel):
    path: str
    type: Literal["string", "integer", "number", "boolean", "object", "array", "any"] = "any"
    description: str = ""
    default: Any = None
    mutable: bool = True
    visibility: Literal["world", "session", "surface", "epistemic"] = "world"
    # Other paths the evidence used for this same field; evidence is rewritten to the canonical path.
    aliases: list[str] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def namespace_matches_visibility(self) -> "StateField":
        namespace = self.path.split(".", 1)[0]
        if namespace not in {"world", "session", "surface", "epistemic"}:
            raise ValueError("state path must start with world, session, surface, or epistemic")
        if namespace != self.visibility:
            raise ValueError(f"state path namespace {namespace!r} differs from visibility {self.visibility!r}")
        return self


class StateSchema(StrictModel):
    fields: list[StateField]


class Operand(StrictModel):
    path: str | None = None
    action_arg: str | None = None
    literal: Any = None

    @model_validator(mode="after")
    def one_source(self) -> "Operand":
        # When neither reference is selected, the operand is a literal (including JSON null).
        if self.path is not None and self.action_arg is not None:
            raise ValueError("operand must define exactly one of path, action_arg, or literal")
        return self


class Condition(StrictModel):
    left: Operand
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "exists", "not_exists", "contains", "in"]
    right: Operand | None = None


class TransitionRule(StrictModel):
    id: str
    action_type: str
    description: str
    priority: int = 0
    conditions: list[Condition] = Field(default_factory=list)
    effects: list[StateMutation] = Field(default_factory=list)
    outcome: Outcome = Outcome.SUCCESS
    renderer: str = "default"
    observation_template: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: Literal["supported", "tentative", "conflicted", "deprecated"] = "supported"
    provenance: list[SourceRef] = Field(default_factory=list)
    counterexamples: list[SourceRef] = Field(default_factory=list)
    scope: dict[str, str] = Field(default_factory=dict)

    @field_validator("scope", mode="before")
    @classmethod
    def scalar_scope_values(cls, value: Any) -> Any:
        # Scope is matched by string equality, so numbers and booleans from a model are kept as their text.
        if isinstance(value, dict):
            return {str(key): item if isinstance(item, str) else json.dumps(item) if isinstance(item, (dict, list))
                    else str(item) for key, item in value.items()}
        return value


class Invariant(StrictModel):
    id: str
    description: str
    condition: Condition
    severity: Literal["error", "warning"] = "error"
    provenance: list[SourceRef] = Field(default_factory=list)


class RenderContract(StrictModel):
    id: str
    action_types: list[str] = Field(default_factory=list)
    content_type: str = "text/plain"
    template: str | None = None
    required_fields: list[str] = Field(default_factory=list)
    instructions: str = ""
    examples: list[str] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)


class Demonstration(StrictModel):
    id: str
    action: NormalizedAction
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)
    observation: str
    rule_ids: list[str] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN


class EnvironmentNote(StrictModel):
    """Reconstructed knowledge that is not a transition rule: conventions, formats, constraints, facts."""

    id: str
    kind: Literal["convention", "format", "constraint", "fact", "concept"] = "fact"
    statement: str
    action_types: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: Literal["supported", "tentative", "conflicted", "deprecated"] = "supported"
    provenance: list[SourceRef] = Field(default_factory=list)


class EnvironmentManifest(StrictModel):
    schema_version: str = "1.0"
    environment_id: str
    name: str
    description: str
    version: str = "0.1.0"
    created_at: datetime = Field(default_factory=utc_now)
    domains: list[str] = Field(default_factory=list)
    source_count: int = 0
    transition_count: int = 0
    files: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PendingEvent(StrictModel):
    id: str
    due_step: int
    effects: list[StateMutation]
    source_rule_id: str | None = None


class EnvironmentState(StrictModel):
    revision: int = 0
    step: int = 0
    world: dict[str, Any] = Field(default_factory=dict)
    session: dict[str, Any] = Field(default_factory=dict)
    surface: dict[str, Any] = Field(default_factory=dict)
    epistemic: dict[str, Any] = Field(default_factory=dict)
    pending_events: list[PendingEvent] = Field(default_factory=list)


class TransitionPlan(StrictModel):
    action: NormalizedAction
    rule_ids: list[str] = Field(default_factory=list)
    precondition_notes: list[str] = Field(default_factory=list)
    effects: list[StateMutation] = Field(default_factory=list)
    outcome: Outcome
    renderer_id: str = "default"
    observation_template: str | None = None
    uncertainty: list[str] = Field(default_factory=list)
    # Agent-produced plans carry their rendering and the workspace artifacts they relied on.
    observation: str | None = None
    citations: list[str] = Field(default_factory=list)
    rationale: str = ""


SUBMIT_TOOL_NAME = "submit_transition"


class TransitionSubmission(StrictModel):
    """The world-model agent's final answer: typed effects, the observation, and its provenance."""

    effects: list[StateMutation] = Field(default_factory=list, description="Typed state changes on declared paths")
    outcome: Outcome = Field(description="success, failure, partial, or unknown as the environment would classify it")
    observation: str | None = Field(default=None, description="The exact observation text the environment returns")
    rule_ids: list[str] = Field(default_factory=list, description="Rules whose effects are applied verbatim")
    citations: list[str] = Field(default_factory=list, description="Identifiers of retrieved artifacts relied on, e.g. rule:x, note:y, memory:3")
    uncertainty: list[str] = Field(default_factory=list, description="What could not be established from the workspace")
    rationale: str = Field(default="", description="One or two sentences on why this is what the environment does")


class SingleShotPrediction(StrictModel):
    """One-call prediction (``prediction_mode="single_shot"``): the observation from a fixed brief, no tools."""

    observation: str = Field(description="The exact observation text the environment returns")
    outcome: Outcome = Field(default=Outcome.SUCCESS, description="success, failure, partial, or unknown")
    citations: list[str] = Field(default_factory=list, description="Identifiers from the brief relied on, e.g. rule:x, memory:3")
    uncertainty: list[str] = Field(default_factory=list, description="What could not be established from the brief")
    rationale: str = Field(default="", description="One or two sentences on why this is what the environment does")


class AgentTurn(StrictModel):
    """One decision of the world-model agent when driven through structured calls instead of native tools."""

    tool: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final: TransitionSubmission | None = None
    reason: str = ""

    @model_validator(mode="after")
    def one_decision(self) -> "AgentTurn":
        if (self.tool is None) == (self.final is None):
            raise ValueError("an agent turn is exactly one tool call or one final submission")
        return self


class ToolCall(StrictModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentReply(StrictModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


class VerificationIssue(StrictModel):
    code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class VerificationResult(StrictModel):
    accepted: bool
    issues: list[VerificationIssue] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ApplicabilityVerdict(StrictModel):
    """The knowledge judge's verdict on one package item (harness v5.3): a label, the concrete anchors in this
    episode that tie the item to the current task, and a short reason. ``supporting`` is honoured only when at least one
    anchor is verified against the episode's own transcript or current action."""

    id: str
    label: Literal["supporting", "format_only", "contradicted"]
    anchors: list[str] = Field(default_factory=list)
    reason: str = ""


class ApplicabilityJudgement(StrictModel):
    items: list[ApplicabilityVerdict] = Field(default_factory=list)


class RenderedObservation(StrictModel):
    observation: str


class StepRequest(StrictModel):
    action: NormalizedAction
    metadata: dict[str, Any] = Field(default_factory=dict)


class StepResult(StrictModel):
    observation: str
    outcome: Outcome
    state: EnvironmentState
    plan: TransitionPlan
    verification: VerificationResult
    retrieved_artifacts: list[str] = Field(default_factory=list)
    route: Literal["rule", "agent", "single_shot"] = "rule"
    tool_calls: int = 0
    citations: list[str] = Field(default_factory=list)


class AuditRecord(StrictModel):
    timestamp: datetime = Field(default_factory=utc_now)
    prior_revision: int
    new_revision: int | None = None
    request: StepRequest
    plan: TransitionPlan | None = None
    verification: VerificationResult | None = None
    observation: str | None = None
    committed: bool = False
    retrieved_artifacts: list[str] = Field(default_factory=list)
    route: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    error: str | None = None


class ReconstructionConfig(StrictModel):
    environment_id: str
    name: str
    description: str = ""
    domains: list[str] = Field(default_factory=list)
    model: str = "gpt-5.6-sol"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "medium"
    max_history_events: int = Field(default=20, ge=0)
    # Character limits for the event text shown to the evidence extractor (0 = unlimited). Clipped text
    # keeps its head and tail with an explicit omission marker; the recorded observation_text is never clipped.
    extraction_history_chars: int = Field(default=2000, ge=0)
    extraction_event_chars: int = Field(default=40000, ge=0)
    # Character limits for evidence shown to the induction stages (schema, rules, renderers, notes):
    # string leaves of mutation/fact values and the observation text are clipped head-and-tail so the
    # stages reason over shapes and formats, not over whole files. 0 = unlimited.
    induction_value_chars: int = Field(default=400, ge=0)
    induction_observation_chars: int = Field(default=2000, ge=0)
    induction_batch_size: int = Field(default=100, ge=2)
    min_rule_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    tenant_scope: str | None = None
    environment_version: str | None = None
    scope: dict[str, str] = Field(default_factory=dict)
    # "ablated": derived from a reconstructed package with its semantic knowledge removed (an experiment
    # control); it keeps the reconstructed package's runtime gates but needs no trace evidence.
    construction_kind: Literal["reconstructed", "authored", "ablated"] = "reconstructed"
    ablation: dict[str, Any] | None = None  # provenance of an ablated package: source, hash, kept/removed parts
    max_prompt_bytes: int = Field(default=600000, ge=4096)


class SplitAssignment(StrictModel):
    split: Literal["train", "validation", "test"]
    trajectory_group: str


class SplitManifest(StrictModel):
    assignments: dict[str, SplitAssignment]

    @model_validator(mode="after")
    def grouped_splits(self) -> "SplitManifest":
        groups: dict[str, str] = {}
        for assignment in self.assignments.values():
            if groups.get(assignment.trajectory_group, assignment.split) != assignment.split:
                raise ValueError("A trajectory group cannot cross data splits")
            groups[assignment.trajectory_group] = assignment.split
        return self


class ReconstructionArtifacts(StrictModel):
    episodes: list[Episode]
    transitions: list[TransitionSlice]
    evidence: list[LocalTransitionEvidence]
    action_schema: ActionSchema
    state_schema: StateSchema
    rules: list[TransitionRule]
    invariants: list[Invariant] = Field(default_factory=list)
    renderers: list[RenderContract]
    demonstrations: list[Demonstration] = Field(default_factory=list)
    notes: list[EnvironmentNote] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    # Induced artifacts that violated the schema and were quarantined at compile time, kept verbatim for audit.
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    source_snapshots: dict[str, str] = Field(default_factory=dict)
    # Source digest -> declared split and independent trajectory family.
    split_assignments: dict[str, "SplitAssignment"] = Field(default_factory=dict)

    @model_validator(mode="after")
    def references_are_closed(self) -> "ReconstructionArtifacts":
        action_names = {action.name for action in self.action_schema.actions}
        renderer_ids = {renderer.id for renderer in self.renderers}
        rule_ids = [rule.id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("transition rule ids must be unique")
        unknown_actions = {rule.action_type for rule in self.rules} - action_names
        if unknown_actions:
            raise ValueError(f"rules reference unknown actions: {sorted(unknown_actions)}")
        unknown_renderers = {rule.renderer for rule in self.rules} - renderer_ids
        if unknown_renderers:
            raise ValueError(f"rules reference unknown renderers: {sorted(unknown_renderers)}")
        return self


class SchemaInductionResult(StrictModel):
    action_schema: ActionSchema
    state_schema: StateSchema
    unresolved: list[str] = Field(default_factory=list)


class RuleInductionResult(StrictModel):
    rules: list[TransitionRule]
    invariants: list[Invariant] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class RenderInductionResult(StrictModel):
    renderers: list[RenderContract]
    unresolved: list[str] = Field(default_factory=list)


class NoteInductionResult(StrictModel):
    notes: list[EnvironmentNote] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class ReplayStep(StrictModel):
    action: NormalizedAction
    expected_observation: str
    observation_match: Literal["exact", "json"] = "exact"
    expected_outcome: Outcome | None = None
    # Flat state-path projections. Only explicitly supplied facts are scored.
    expected_state: dict[str, Any] = Field(default_factory=dict)


class ReplayCase(StrictModel):
    id: str
    trajectory_group: str
    source_ids: list[str] = Field(min_length=1)
    split: Literal["train", "validation", "test"]
    initial_state: EnvironmentState
    initial_state_complete: bool = False
    known_absent_paths: list[str] = Field(default_factory=list)
    steps: list[ReplayStep] = Field(min_length=1)


class ReplayCaseResult(StrictModel):
    case_id: str
    trajectory_group: str
    passed: bool
    attempted_steps: int
    expected_steps: int
    errors: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)


class ReplayReport(StrictModel):
    artifact_digest: str
    config_digest: str
    cases: list[ReplayCaseResult]
    total_cases: int
    passed_cases: int
    total_steps: int
    attempted_steps: int
    baseline_passed_cases: int | None = None
    regressions: list[str] = Field(default_factory=list)
    promotion_eligible: bool = False
    reasons: list[str] = Field(default_factory=list)
    uncovered_rule_ids: list[str] = Field(default_factory=list)


class ArtifactEdit(StrictModel):
    collection: Literal["rules", "invariants", "renderers", "demonstrations", "notes", "actions", "state_fields"]
    key: str
    operation: Literal["add", "replace", "remove"]
    value: dict[str, Any] | None = None


class PackagePatch(StrictModel):
    base_artifact_digest: str
    rationale: str
    failed_case_ids: list[str] = Field(min_length=1)
    edits: list[ArtifactEdit] = Field(min_length=1)


TrustPolicy = Literal["rules_only", "schema_checked"]
FastPathPolicy = Literal["confident", "always", "never"]


class MemoryEntry(StrictModel):
    """One episodic-memory record: a turn this session has observed or simulated."""

    id: int | None = None
    turn: int = Field(ge=0)
    kind: Literal["observed", "predicted"]
    action: NormalizedAction
    observation: str
    effects: list[StateMutation] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservedTurn(StrictModel):
    """One recorded interaction: the task agent's action and the observation the real environment returned."""

    turn: int = Field(ge=0)
    action: NormalizedAction
    observation: str
    prompt: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateTrackingResult(StrictModel):
    """Response of the ``track_state`` role: typed mutations explained by a window of observed turns."""

    mutations: list[StateMutation] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    uncertain_paths: list[str] = Field(default_factory=list)


class StateTrackingStep(StrictModel):
    turns: list[int]
    route: Literal["rule", "model", "skipped"]
    rule_id: str | None = None
    applied: int = 0
    dropped: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class AgentWorldCase(StrictModel):
    """One AgentWorldBench record: the turn to predict plus its teacher-forced history."""

    case_id: str
    task: str
    trajectory_id: str
    turn_idx: int = Field(ge=1)
    total_turns: int | None = None
    system_prompt: str = ""
    prompts: list[str] = Field(min_length=1)
    responses: list[str] = Field(min_length=1)
    current_prompt: str
    action: NormalizedAction | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def history_covers_turn(self) -> "AgentWorldCase":
        if len(self.prompts) < self.turn_idx or len(self.responses) < self.turn_idx:
            raise ValueError("prompt and response lists must cover every turn through turn_idx")
        return self

    @property
    def ground_truth(self) -> str:
        return self.responses[self.turn_idx - 1]
