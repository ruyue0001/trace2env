"""Offline evidence-to-package reconstruction pipeline."""

from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable

from trace2env.adapters import annotate_episode, events_for_slice, load_raw_traces, segment_transitions
from trace2env.induction import BoundedInduction
from trace2env.validation import (action_aliases, canonical_state_path, invalid_render_references, renderer_issues, repair_provenance,
                                  rule_has_observed_support, rule_issues, state_field, state_path_aliases,
                                  validate_source_ref, value_type)
from trace2env.engine import set_path
from trace2env.contrasts import build_contrasts
from trace2env.eligibility import construction_scope
from trace2env.compiler import EnvironmentCompiler
from trace2env.llm import ProviderRefusal, StructuredLLM, StructuredOutputError
from trace2env.models import (
    ActionSchema,
    ArgumentSpec,
    Confidence,
    Demonstration,
    EnvironmentNote,
    Episode,
    Invariant,
    LocalTransitionEvidence,
    NormalizedAction,
    NoteInductionResult,
    Outcome,
    RawEvent,
    ReconstructionArtifacts,
    ReconstructionConfig,
    RenderContract,
    RenderInductionResult,
    RuleInductionResult,
    SchemaInductionResult,
    ActionSpec,
    SourceRef,
    StateField,
    StateSchema,
    TransitionRule,
    TransitionSlice,
    TraceAnnotations,
    SplitManifest,
)
from trace2env.prompts import (
    LOCAL_EVIDENCE_EXTRACTOR,
    NOTE_INDUCER,
    RENDERER_INDUCER,
    RULE_FALSIFIER,
    RULE_INDUCER,
    SCHEMA_INDUCER,
    TRACE_NORMALIZER,
)
from trace2env.storage import write_json, write_jsonl


def _json_payload(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")  # type: ignore[union-attr]
    return json.dumps(value, ensure_ascii=False, indent=2)


DESCRIPTION_CHARS = 4000
# Event metadata that repeats the event content or carries the task agent's own commentary; neither is
# environment evidence, so the extractor does not see it.
EXCLUDED_EVENT_METADATA = {"raw_action", "agent_message", "raw_timestamp"}


def clip_text(text: str, limit: int) -> tuple[str, bool]:
    """Head-and-tail clip with an explicit marker; ``limit`` 0 disables clipping."""
    if not limit or len(text) <= limit:
        return text, False
    head, tail = text[: limit // 2], text[-(limit - limit // 2):]
    return f"{head}\n... [{len(text) - limit} characters omitted] ...\n{tail}", True


def clip_leaves(value: Any, limit: int) -> tuple[Any, bool]:
    """Clip every string leaf of a JSON value; containers are rebuilt, nothing else changes."""
    if isinstance(value, str):
        return clip_text(value, limit)
    if isinstance(value, dict):
        clipped = False
        out = {}
        for key, item in value.items():
            out[key], flag = clip_leaves(item, limit)
            clipped = clipped or flag
        return out, clipped
    if isinstance(value, list):
        items = [clip_leaves(item, limit) for item in value]
        return [item for item, _ in items], any(flag for _, flag in items)
    return value, False


def event_view(event: RawEvent, limit: int) -> dict:
    """What the evidence extractor sees of one event: identity, actor/kind, (clipped) content, plain metadata."""
    content = event.content
    clipped = False
    if isinstance(content, dict) and "raw" in content and "arguments" in content:
        # A normalized action repeats its keystrokes under ``raw``; the arguments already hold them.
        content = {key: item for key, item in content.items() if key != "raw"}
    if isinstance(content, str):
        content, clipped = clip_text(content, limit)
    elif isinstance(content, (dict, list)):
        content, clipped = clip_leaves(content, limit)
    view = {
        "id": event.id,
        "actor": event.actor.value,
        "kind": event.kind.value,
        "content": content,
        "metadata": {key: value for key, value in event.metadata.items() if key not in EXCLUDED_EVENT_METADATA},
    }
    if event.timestamp is not None:
        view["timestamp"] = event.timestamp.isoformat()
    if clipped:
        view["clipped"] = True
    return view


def canonical_evidence(evidence: list[LocalTransitionEvidence], schema: SchemaInductionResult) -> list[LocalTransitionEvidence]:
    """Evidence rewritten to the induced vocabulary: action types through action aliases, mutation paths
    and fact subjects through state-field aliases. Returns copies; the extracted records stay as written."""
    actions = action_aliases(schema.action_schema)
    paths = state_path_aliases(schema.state_schema)
    canonical: list[LocalTransitionEvidence] = []
    for item in evidence:
        copy = item.model_copy(deep=True)
        copy.action = copy.action.model_copy(update={"type": actions.get(copy.action.type, copy.action.type)})
        for mutation in copy.mutations:
            mutation.path = canonical_state_path(mutation.path, paths)
        for fact in copy.preconditions + copy.observation_facts:
            fact.subject = canonical_state_path(fact.subject, paths)
        canonical.append(copy)
    return canonical


def declare_observed_actions(schema: SchemaInductionResult, evidence: list[LocalTransitionEvidence]) -> list[str]:
    """Every action type the evidence records is declared in the action schema.

    The schema inducer merges and describes the actions it sees, but on small corpora it can leave a rare
    type out (one ``python`` turn among fifty ``shell`` turns), and a package cannot compile with evidence
    whose action is undeclared. Missing types are added with an empty argument list (``declare_observed_arguments``
    fills the arguments afterwards), a description saying they were not described by the inducer, and the
    evidence records as provenance; each addition is reported. Deterministic, no model call.
    """
    aliases = action_aliases(schema.action_schema)
    declared = {spec.name for spec in schema.action_schema.actions}
    observed: dict[str, list[LocalTransitionEvidence]] = {}
    for item in evidence:
        name = aliases.get(item.action.type, item.action.type)
        if name not in declared:
            observed.setdefault(name, []).append(item)
    notes: list[str] = []
    for name, items in sorted(observed.items()):
        provenance = [ref for item in items[:5] for ref in item.provenance][:10]
        schema.action_schema.actions.append(ActionSpec(
            name=name, description=f"Observed in {len(items)} transition(s) of the traces; not described by the schema inducer.",
            provenance=provenance))
        declared.add(name)
        notes.append(f"action {name!r} was used by {len(items)} evidence record(s) but not declared by the schema inducer; declared from the evidence")
    return notes


def declare_observed_arguments(schema: SchemaInductionResult, evidence: list[LocalTransitionEvidence]) -> list[str]:
    """Every argument name the evidence records for an action is declared on its ActionSpec.

    The rule and renderer stages may only reference declared arguments, so an argument the schema
    inducer left out would quarantine every rule that uses it. Missing names are added with the
    JSON type observed in the evidence (``any`` when it varies); each addition is reported.
    """
    aliases = action_aliases(schema.action_schema)
    specs = {spec.name: spec for spec in schema.action_schema.actions}
    observed: dict[str, dict[str, set[str]]] = {}
    for item in evidence:
        name = aliases.get(item.action.type, item.action.type)
        for argument, value in item.action.arguments.items():
            try:
                kind = value_type(value, None)
            except ValueError:  # literal text that happens to look like a template token
                kind = "any"
            observed.setdefault(name, {}).setdefault(argument, set()).add(kind)
    notes: list[str] = []
    for name, arguments in observed.items():
        spec = specs.get(name)
        if spec is None:
            continue
        for argument, types in arguments.items():
            if argument in spec.arguments:
                continue
            kind = next(iter(types)) if len(types) == 1 else "any"
            spec.arguments[argument] = ArgumentSpec(type=kind if kind in ARGUMENT_TYPES else "any",
                                                    description="Declared from the evidence records of this action")
            notes.append(f"Argument {argument!r} of action {name!r} was observed in evidence but not declared; added as {kind}")
    return notes


ARGUMENT_TYPES = {"string", "integer", "number", "boolean", "object", "array", "any"}
REFUSED_MARKER = "Evidence extraction refused by the model provider"
INVALID_MARKER = "Evidence extraction produced invalid structured output after repair attempts"


def repair_schema_aliases(schema: SchemaInductionResult) -> list[str]:
    """Drop induced aliases that cannot be resolved: an alias equal to another declared action name or state path, an
    alias claimed by two entries (dropped from both), or an entry's own name. The inducer produced all three on the
    android phone corpora (``surface.chrome.viewport`` as the alias of two fields, a field path listed as another
    field's alias); ``action_aliases`` / ``state_path_aliases`` would otherwise raise and fail the stage. Every drop
    is reported."""
    notes: list[str] = []
    for kind, entries, key in (("action", schema.action_schema.actions, "name"), ("state field", schema.state_schema.fields, "path")):
        names = {getattr(entry, key) for entry in entries}
        claims: dict[str, list[str]] = {}
        for entry in entries:
            for alias in entry.aliases:
                claims.setdefault(alias, []).append(getattr(entry, key))
        for entry in entries:
            kept = []
            for alias in entry.aliases:
                own = getattr(entry, key)
                if alias == own:
                    continue  # redundant: an entry is its own alias
                if alias in names:
                    notes.append(f"Dropped alias {alias!r} of {kind} {own!r}: it is a declared {kind} itself")
                elif len(set(claims[alias])) > 1:
                    notes.append(f"Dropped alias {alias!r} of {kind} {own!r}: also claimed by {sorted(set(claims[alias]) - {own})}")
                elif alias in kept:
                    continue
                else:
                    kept.append(alias)
            if kept != list(entry.aliases):
                entry.aliases = kept
    return notes


def declare_observed_paths(schema: SchemaInductionResult, evidence: list[LocalTransitionEvidence]) -> list[str]:
    """Every state path the (canonicalized) evidence mutates or asserts is declared in the state schema.

    The inducer may declare only a child (``world.mailman.lists``) while evidence also edits the
    parent (``world.mailman``), or skip a rarely used path; rules and tracked mutations on an
    undeclared path would be quarantined. Missing paths are declared with the type observed in the
    evidence (``any`` when it varies) and reported.
    """
    paths = state_path_aliases(schema.state_schema)
    observed: dict[str, set[str]] = {}
    for item in evidence:
        for mutation in item.mutations:
            path = canonical_state_path(mutation.path, paths)
            kind = "object" if mutation.op in {"merge", "remove"} else None
            if kind is None:
                try:
                    kind = value_type(mutation.value, None)
                except ValueError:
                    kind = "any"
            observed.setdefault(path, set()).add(kind)
        for fact in item.preconditions + item.observation_facts:
            path = canonical_state_path(fact.subject, paths)
            if fact.predicate in {"contains", "not_contains"}:
                observed.setdefault(path, set()).add("object" if isinstance(fact.value, dict) else "array")
            elif fact.predicate in {"equals", "eq", "="}:
                try:
                    observed.setdefault(path, set()).add(value_type(fact.value, None))
                except ValueError:
                    observed.setdefault(path, set()).add("any")
    notes: list[str] = []
    for path, kinds in sorted(observed.items()):
        try:
            state_field(path, schema.state_schema, None)
            continue
        except ValueError:
            pass
        namespace = path.split(".", 1)[0]
        if namespace not in {"world", "session", "surface", "epistemic"}:
            continue
        kind = next(iter(kinds)) if len(kinds) == 1 else "any"
        schema.state_schema.fields.append(StateField(
            path=path, type=kind if kind in ARGUMENT_TYPES else "any", visibility=namespace,  # type: ignore[arg-type]
            description="Declared from evidence: the schema stage did not declare this path"))
        notes.append(f"State path {path!r} was used by the evidence but not declared; added as {kind}")
    return notes


def refused_evidence(transition: TransitionSlice, action_events: list[RawEvent], reason: str, *,
                     marker: str = REFUSED_MARKER, rationale: str = "provider refusal") -> LocalTransitionEvidence:
    """A zero-confidence record for a slice that yielded no usable evidence (refusal or invalid output); no claims."""
    content = action_events[0].content if action_events else None
    if isinstance(content, dict) and isinstance(content.get("type"), str):
        action = NormalizedAction(type=content["type"], arguments=dict(content.get("arguments") or {}))
    else:
        action = NormalizedAction(type="unknown", arguments={}, raw=content)
    return LocalTransitionEvidence(
        id=f"evidence_{transition.id}", episode_id=transition.episode_id, transition_id=transition.id,
        action=action, outcome=Outcome.UNKNOWN, confidence=Confidence(value=0.0, rationale=rationale),
        ambiguities=[f"{marker}: {reason[:300]}"],
    )


def refused_count(evidence: list[LocalTransitionEvidence]) -> int:
    return sum(1 for item in evidence if any(note.startswith(REFUSED_MARKER) for note in item.ambiguities))


def invalid_count(evidence: list[LocalTransitionEvidence]) -> int:
    """Slices whose extraction never produced a valid record; they carry zero confidence and no claims."""
    return sum(1 for item in evidence if any(note.startswith(INVALID_MARKER) for note in item.ambiguities))


def environment_payload(config: ReconstructionConfig) -> dict:
    """Configuration as model context; a long domain description is clipped so it does not eat the byte budget."""
    payload = config.model_dump(mode="json")
    if len(payload.get("description") or "") > DESCRIPTION_CHARS:
        payload["description"] = payload["description"][:DESCRIPTION_CHARS] + "\n... [description truncated]"
    return payload


SCOPE_KEYS = {"environment_id", "tenant", "tenant_scope", "environment_version", "version"}
NAMESPACES = {"world", "session", "surface", "epistemic"}
UNSET_SCOPE_VALUES = {"", "none", "null", "unknown", "n/a"}
PLACEHOLDER = re.compile(r"<[A-Za-z][^<>\n]* [^<>\n]*>")  # `<shell prompt>`, `<target output>`: prose, not output
RENDERER_ID = re.compile(r"[A-Za-z0-9_.:-]{1,80}")


def repair_rule_selectors(rule: TransitionRule) -> list[str]:
    """Keep a rule's scope and template within what the runtime can evaluate.

    Scope keys are matched against the runtime context (tenant, version, environment id) or
    against state paths; any other key — an episode id, a command string, a free-text qualifier —
    would make the rule silently ineligible forever, so it is dropped and reported. Scope values
    that spell "None" for an unset selector are dropped for the same reason. A template that
    contains placeholder prose (``<shell prompt>``) is not literal output; it is removed so the
    renderer contract or the agent renders the observation instead.
    """
    notes: list[str] = []
    dropped_keys = [key for key in rule.scope
                    if key not in SCOPE_KEYS and key.split(".", 1)[0] not in NAMESPACES]
    dropped_unset = [key for key, value in rule.scope.items()
                     if key not in dropped_keys and str(value).strip().lower() in UNSET_SCOPE_VALUES]
    if dropped_keys or dropped_unset:
        rule.scope = {key: value for key, value in rule.scope.items() if key not in dropped_keys + dropped_unset}
        if dropped_keys:
            notes.append(f"Rule {rule.id}: scope keys {dropped_keys} are neither tenant/version/environment selectors "
                         "nor state paths and were dropped (they would never match a runtime context)")
        if dropped_unset:
            notes.append(f"Rule {rule.id}: scope entries {dropped_unset} carried an unset value and were dropped")
    if rule.observation_template and PLACEHOLDER.search(rule.observation_template):
        notes.append(f"Rule {rule.id}: observation_template contained placeholder prose "
                     f"({PLACEHOLDER.search(rule.observation_template).group(0)!r}) and was removed")
        rule.observation_template = None
    return notes


def reconcile_artifacts(
    schema: SchemaInductionResult,
    rules: list[TransitionRule],
    invariants: list[Invariant],
    renderers: list[RenderContract],
    notes: list[EnvironmentNote],
) -> tuple[list[TransitionRule], list[Invariant], list[RenderContract], list[EnvironmentNote], list[str], list[dict]]:
    """Close references across independently induced stages instead of failing the whole build.

    Renderer ids that rules cite but no contract defines get a contract synthesized from the rule;
    rules, invariants, contracts, and notes that still violate the schema are quarantined with a
    recorded reason. Nothing is silently dropped: every decision lands in ``unresolved``.
    """
    actions = {action.name: action for action in schema.action_schema.actions}
    state_schema = schema.state_schema
    unresolved: list[str] = []
    rejected: list[dict] = []
    contracts = {renderer.id: renderer for renderer in renderers}
    for rule in rules:
        unresolved.extend(repair_rule_selectors(rule))
        if rule.renderer not in contracts and rule.action_type in actions:
            contracts[rule.renderer] = RenderContract(
                id=rule.renderer, action_types=[rule.action_type],
                instructions=f"Contract synthesized at compile time for rule {rule.id}: {rule.description}",
                provenance=list(rule.provenance),
            )
            unresolved.append(f"Renderer {rule.renderer!r} was referenced by rule {rule.id} but not induced; a "
                              "contract without a template was synthesized")
    kept_rules: list[TransitionRule] = []
    for rule in rules:
        issues = rule_issues(rule, actions, state_schema, set(contracts))
        if issues:
            unresolved.append(f"Rule {rule.id} quarantined: " + "; ".join(issues))
            rejected.append({"kind": "rule", "id": rule.id, "issues": issues, "artifact": rule.model_dump(mode="json")})
        else:
            kept_rules.append(rule)
    kept_invariants: list[Invariant] = []
    for invariant in invariants:
        try:
            for operand in (invariant.condition.left, invariant.condition.right):
                if operand is not None and operand.path is not None:
                    state_field(operand.path, state_schema, None)
                if operand is not None and operand.action_arg is not None:
                    # Invariants hold on state alone; an action-argument operand cannot be checked without an action.
                    raise ValueError(f"invariants cannot reference action arguments (action_arg={operand.action_arg!r})")
            kept_invariants.append(invariant)
        except ValueError as exc:
            unresolved.append(f"Invariant {invariant.id} quarantined: {exc}")
            rejected.append({"kind": "invariant", "id": invariant.id, "issues": [str(exc)],
                             "artifact": invariant.model_dump(mode="json")})
    kept_renderers: list[RenderContract] = []
    for renderer in contracts.values():
        if renderer.template and PLACEHOLDER.search(renderer.template):
            unresolved.append(f"Renderer {renderer.id}: template contained placeholder prose "
                              f"({PLACEHOLDER.search(renderer.template).group(0)!r}) and was removed; its instructions remain")
            renderer = renderer.model_copy(update={"template": None})
            contracts[renderer.id] = renderer
        known = [action_type for action_type in renderer.action_types if action_type in actions]
        if len(known) != len(renderer.action_types):
            unresolved.append(f"Renderer {renderer.id} referenced unknown actions "
                              f"{sorted(set(renderer.action_types) - set(known))}; they were dropped")
            renderer = renderer.model_copy(update={"action_types": known})
        bad_fields, bad_template = invalid_render_references(renderer, actions, state_schema, kept_rules)
        if bad_fields:
            # Prose labels and invented paths are not checkable at runtime; the instructions still are.
            dropped = [path for path, _ in bad_fields]
            unresolved.append(f"Renderer {renderer.id}: required fields {dropped} are not renderable references "
                              f"({bad_fields[0][1]}) and were dropped; its instructions remain")
            renderer = renderer.model_copy(update={"required_fields": [p for p in renderer.required_fields if p not in dropped]})
            contracts[renderer.id] = renderer
        if bad_template:
            unresolved.append(f"Renderer {renderer.id}: template referenced {[path for path, _ in bad_template]} "
                              f"({bad_template[0][1]}); the template was removed and its instructions remain")
            renderer = renderer.model_copy(update={"template": None})
            contracts[renderer.id] = renderer
        issues = renderer_issues(renderer, actions, state_schema, kept_rules)
        if issues:
            unresolved.append(f"Renderer {renderer.id} quarantined: " + "; ".join(issues))
            rejected.append({"kind": "renderer", "id": renderer.id, "issues": issues, "artifact": renderer.model_dump(mode="json")})
            kept_rules = [rule if rule.renderer != renderer.id else rule.model_copy(update={"renderer": "default"})
                          for rule in kept_rules]
        else:
            kept_renderers.append(renderer)
    if any(rule.renderer == "default" for rule in kept_rules) and "default" not in {r.id for r in kept_renderers}:
        kept_renderers.append(RenderContract(id="default", instructions="Fallback contract: render from the resulting state."))
    kept_notes: list[EnvironmentNote] = []
    for note in notes:
        unknown = sorted(set(note.action_types) - set(actions))
        if unknown:
            unresolved.append(f"Note {note.id} referenced unknown actions {unknown}; it now applies to all actions")
            note = note.model_copy(update={"action_types": [a for a in note.action_types if a in actions]})
        kept_notes.append(note)
    return kept_rules, kept_invariants, kept_renderers, kept_notes, unresolved, rejected


def reconcile_provenance(
    schema: SchemaInductionResult,
    rules: list[TransitionRule],
    invariants: list[Invariant],
    renderers: list[RenderContract],
    notes: list[EnvironmentNote],
    episodes: list[Episode],
    evidence: list[LocalTransitionEvidence],
    transitions: list[TransitionSlice],
) -> list[str]:
    """Repair citations in every induced artifact in place; downgrade supported claims that lose their support."""
    unresolved: list[str] = []
    by_transition = {transition.id: transition for transition in transitions}
    for spec in schema.action_schema.actions:
        spec.provenance = repair_provenance(spec.provenance, episodes, unresolved, f"action {spec.name}")
    for field in schema.state_schema.fields:
        field.provenance = repair_provenance(field.provenance, episodes, unresolved, f"state field {field.path}")
    def anchored(refs: list[SourceRef], owner: str) -> list[SourceRef]:
        """Executable claims need event anchors; a citation of a whole episode or source is not one."""
        kept = [ref for ref in refs if ref.episode_id and ref.event_ids]
        if len(kept) != len(refs):
            unresolved.append(f"{owner}: dropped {len(refs) - len(kept)} citation(s) without event anchors")
        return kept

    for rule in rules:
        rule.provenance = repair_provenance(rule.provenance, episodes, unresolved, f"rule {rule.id}")
        rule.counterexamples = repair_provenance(rule.counterexamples, episodes, unresolved, f"rule {rule.id} counterexamples")
        if rule.status == "supported":
            rule.provenance = anchored(rule.provenance, f"rule {rule.id}")
        if rule.status == "supported" and not rule.provenance:
            rule.status = "tentative"
            unresolved.append(f"Rule {rule.id} downgraded to tentative: no event-anchored provenance")
        elif rule.status == "supported" and not rule_has_observed_support(rule, evidence, by_transition):
            rule.status = "tentative"
            unresolved.append(f"Rule {rule.id} downgraded to tentative: no unambiguous observed transition supports it")
    kept_invariants = []
    for invariant in invariants:
        invariant.provenance = anchored(repair_provenance(invariant.provenance, episodes, unresolved, f"invariant {invariant.id}"),
                                        f"invariant {invariant.id}")
        if invariant.provenance:
            kept_invariants.append(invariant)
        else:
            unresolved.append(f"Invariant {invariant.id} dropped: no event-anchored provenance")
    invariants[:] = kept_invariants
    for renderer in renderers:
        renderer.provenance = repair_provenance(renderer.provenance, episodes, unresolved, f"renderer {renderer.id}")
    for note in notes:
        note.provenance = repair_provenance(note.provenance, episodes, unresolved, f"note {note.id}")
        if note.status == "supported":
            note.provenance = anchored(note.provenance, f"note {note.id}")
        if note.status == "supported" and not note.provenance:
            note.status = "tentative"
            unresolved.append(f"Note {note.id} downgraded to tentative: no event-anchored provenance")
    return unresolved


class TraceReconstructor:
    """Stages LLM judgment so local evidence remains auditable and revisable."""

    def __init__(self, llm: StructuredLLM, config: ReconstructionConfig, work_dir: str | Path, workers: int = 1):
        self.llm = llm
        self.config = config
        self.work_dir = Path(work_dir)
        self.workers = max(1, workers)  # concurrent extract_transition calls; every other stage is sequential
        self.refused_episodes: set[str] = set()  # episodes with a provider-refused slice: suspects for induction refusals
        self.bounded = BoundedInduction(llm, config.max_prompt_bytes, config.induction_batch_size, suspect=self._suspect)
        self.source_snapshots: dict[str, str] = {}
        self.split_assignments = {}
        self.episode_groups = {}
        self.episode_scopes = {}

    def ingest(self, paths: Iterable[str | Path], split_manifest: SplitManifest | None = None) -> tuple[list[Episode], list[TransitionSlice]]:
        episodes = []
        self.source_snapshots = {}
        self.episode_groups = {}
        self.episode_scopes = {}
        self.split_assignments = split_manifest.assignments if split_manifest else {}
        seen = set()
        for path in paths:
            loaded = load_raw_traces(path)
            raw = Path(path).read_bytes()
            source_id = "src_" + hashlib.sha256(raw).hexdigest()
            if any(ep.source_id != source_id for ep in loaded):
                raise ValueError("Source changed while being ingested")
            assignment = self.split_assignments.get(source_id)
            if split_manifest and (assignment is None or assignment.split != "train"):
                raise ValueError("Reconstruction input is absent from the training split manifest")
            self.source_snapshots[source_id] = raw.decode("utf-8")
            for ep in loaded:
                if ep.metadata.get("split", "train") != "train":
                    raise ValueError("Held-out episodes cannot be used for reconstruction")
                if assignment:
                    ep.metadata["trajectory_group"] = assignment.trajectory_group
                    ep.metadata["split"] = assignment.split
                context = construction_scope(self.config)
                for key in ("tenant", "environment_version", "environment_id"):
                    if key in ep.metadata:
                        if key in context and context[key] != str(ep.metadata[key]):
                            raise ValueError(f"Input episode contradicts construction scope: {key}")
                        context[key] = str(ep.metadata[key])
                self.episode_scopes[ep.id] = context
                self.episode_groups[ep.id] = str(ep.metadata.get("trajectory_group", ep.id))
                signature = _json_payload([ep.metadata.get("original_episode_id"),
                    ep.metadata.get("trajectory_group"), [(e.actor.value, e.kind.value, e.content) for e in ep.events]])
                if signature in seen:
                    continue
                seen.add(signature)
                episodes.append(ep)
        write_json(self.work_dir / "raw" / "sources.json", self.source_snapshots)
        write_jsonl(self.work_dir / "raw" / "episodes.jsonl", episodes)
        normalized: list[Episode] = []
        for episode in episodes:
            kinds = {event.kind.value for event in episode.events}
            if "action" not in kinds or not ({"observation", "error", "state"} & kinds):
                annotations = self.bounded.call(TRACE_NORMALIZER,
                    {"raw_episode": episode.model_dump(mode="json")}, TraceAnnotations, "normalize_trace")
                normalized.append(annotate_episode(episode, annotations))
            else:
                normalized.append(episode)
        episodes = normalized
        transitions = [
            transition
            for episode in episodes
            for transition in segment_transitions(episode, self.config.max_history_events)
        ]
        self.work_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.work_dir / "normalized" / "episodes.jsonl", episodes)
        write_jsonl(self.work_dir / "normalized" / "transitions.jsonl", transitions)
        return episodes, transitions

    def extraction_payload(self, episode: Episode, transition: TransitionSlice) -> dict:
        """The exact user payload one ``extract_transition`` call receives (also used to size prompts).

        Events are shown once, as content plus metadata (no raw payload copy, no task-agent
        commentary), and long text is clipped head-and-tail with an explicit omission marker:
        history events to ``extraction_history_chars``, the slice's own events to
        ``extraction_event_chars``. The recorded ``observation_text`` is taken from the source
        events afterwards, so clipping never changes what the evidence stores.
        """
        selected = events_for_slice(episode, transition)
        limits = {"history": self.config.extraction_history_chars}
        return {
            "environment_context": {
                "environment_id": self.config.environment_id,
                "description": environment_payload(self.config)["description"],
                "domains": self.config.domains,
                "tenant_scope": self.config.tenant_scope,
            },
            "transition": transition.model_dump(mode="json"),
            "episode_scope": self.episode_scopes.get(episode.id, {}),
            "events": {
                key: [event_view(event, limits.get(key, self.config.extraction_event_chars)) for event in values]
                for key, values in selected.items()
            },
            "requirements": {
                "evidence_id": f"evidence_{transition.id}",
                "episode_id": episode.id,
                "transition_id": transition.id,
                "provenance_source_id": episode.source_id,
            },
        }

    def _extract_one(self, episode: Episode, transition: TransitionSlice) -> LocalTransitionEvidence:
        selected = events_for_slice(episode, transition)
        try:
            result = self.bounded.call(LOCAL_EVIDENCE_EXTRACTOR, self.extraction_payload(episode, transition),
                                       LocalTransitionEvidence, "extract_transition")
        except ProviderRefusal as exc:
            # The provider would not look at this slice (content policy). Keep the transition with an
            # explicit zero-confidence record so the gap is visible instead of failing the stage.
            result = refused_evidence(transition, selected["actions"], str(exc))
        except StructuredOutputError as exc:
            # The model never produced a record that satisfies the evidence contract. One malformed
            # answer must not fail a stage of hundreds of slices: record the gap, claim nothing.
            result = refused_evidence(transition, selected["actions"], str(exc), marker=INVALID_MARKER,
                                      rationale="invalid structured output")
        result.id = f"evidence_{transition.id}"
        result.episode_id = episode.id
        result.transition_id = transition.id
        result.observation_text = "\n".join(event.content if isinstance(event.content, str)
            else _json_payload(event.content) for event in selected["observations"])
        result.ambiguities = list(dict.fromkeys(result.ambiguities + transition.ambiguities))
        if transition.alignment == "ambiguous" or transition.concurrent_action_event_ids:
            result.mutations = []
            result.confidence.value = min(result.confidence.value, 0.5)
        # Citations must point at events of this slice. Anything else is dropped and recorded,
        # never trusted; a record left without provenance falls back to the slice's own events.
        allowed_ids = set(transition.history_event_ids + transition.action_event_ids + transition.observation_event_ids)
        dropped = 0

        def kept(refs: list[SourceRef]) -> list[SourceRef]:
            nonlocal dropped
            valid = []
            for ref in refs:
                try:
                    validate_source_ref(ref, {episode.id: episode}, require_event=True)
                except ValueError:
                    dropped += 1
                    continue
                if not set(ref.event_ids) <= allowed_ids:
                    dropped += 1
                    continue
                valid.append(ref)
            return valid

        result.provenance = kept(result.provenance)
        for fact in result.preconditions + result.observation_facts:
            fact.provenance = kept(fact.provenance)
        if dropped:
            result.ambiguities.append(f"Extractor cited {dropped} event reference(s) outside its transition slice; they were dropped")
        if not result.provenance:
            result.provenance = [
                SourceRef(
                    source_id=episode.source_id,
                    episode_id=episode.id,
                    event_ids=transition.action_event_ids + transition.observation_event_ids,
                )
            ]
        write_json(self.work_dir / "evidence" / f"{transition.id}.json", result)
        return result

    def extract_evidence(
        self, episodes: list[Episode], transitions: list[TransitionSlice]
    ) -> list[LocalTransitionEvidence]:
        episode_index = {episode.id: episode for episode in episodes}
        if self.workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                evidence = list(pool.map(lambda t: self._extract_one(episode_index[t.episode_id], t), transitions))
        else:
            evidence = [self._extract_one(episode_index[t.episode_id], t) for t in transitions]
        write_jsonl(self.work_dir / "evidence" / "all.jsonl", evidence)
        return evidence

    def _suspect(self, record: dict) -> bool:
        """A record from an episode the provider already refused once is dropped first when a chunk is refused."""
        if not isinstance(record, dict):
            return False
        if record.get("episode_id") in self.refused_episodes:
            return True
        return any(isinstance(ref, dict) and ref.get("episode_id") in self.refused_episodes
                   for ref in record.get("provenance") or [])

    def _note_refusals(self, evidence: list[LocalTransitionEvidence], result, role: str) -> None:
        """Remember refused episodes and report records an induction stage had to exclude."""
        self.refused_episodes |= {item.episode_id for item in evidence
                                  if any(note.startswith(REFUSED_MARKER) for note in item.ambiguities)}
        if self.bounded.refused:
            labels = sorted({str(record.get("id") or record.get("transition_id") or "?") for record in self.bounded.refused
                             if isinstance(record, dict)})
            result.unresolved.append(
                f"{len(self.bounded.refused)} record(s) were refused by the model provider during {role} and excluded "
                f"from this stage: {', '.join(labels[:12])}{' ...' if len(labels) > 12 else ''}")

    def induction_record(self, item: LocalTransitionEvidence) -> dict:
        """An evidence record as the induction stages see it: values and observation clipped, provenance kept."""
        record = item.model_dump(mode="json")
        limit = self.config.induction_value_chars
        for mutation in record["mutations"]:
            mutation["value"], _ = clip_leaves(mutation["value"], limit)
        for key in ("preconditions", "observation_facts"):
            for fact in record[key]:
                fact["value"], _ = clip_leaves(fact["value"], limit)
        record["action"]["arguments"], _ = clip_leaves(record["action"]["arguments"], limit)
        record["action"].pop("raw", None)
        record["observation_text"], _ = clip_text(record["observation_text"], self.config.induction_observation_chars)
        return record

    def induce_schema(self, evidence: list[LocalTransitionEvidence]) -> SchemaInductionResult:
        self._note_refusals(evidence, SchemaInductionResult(action_schema=ActionSchema(actions=[]),
                                                             state_schema=StateSchema(fields=[])), "induce_schema")
        result = self.bounded.induce(
            [{**self.induction_record(item), "observed_scope": self.episode_scopes.get(item.episode_id, {})}
             for item in sorted(evidence, key=lambda e: e.id)],
            SCHEMA_INDUCER, SchemaInductionResult, "induce_schema",
            {"environment": environment_payload(self.config)}, "local_evidence", "partial_schemas")
        self._note_refusals(evidence, result, "induce_schema")
        result.unresolved.extend(repair_schema_aliases(result))
        result.unresolved.extend(declare_observed_actions(result, evidence))
        result.unresolved.extend(declare_observed_arguments(result, evidence))
        result.unresolved.extend(declare_observed_paths(result, evidence))
        write_json(self.work_dir / "induction" / "schema.json", result)
        return result

    def induce_rules(
        self, evidence: list[LocalTransitionEvidence], schema: SchemaInductionResult
    ) -> RuleInductionResult:
        cohorts = {}
        for item in sorted(evidence, key=lambda e: (e.action.type, e.id)):
            cohorts.setdefault(item.action.type, []).append(item)
        contrasts = build_contrasts(evidence, self.episode_groups, self.episode_scopes)
        write_json(self.work_dir / "induction" / "contrast_sets.json", contrasts)
        contrast_index = {}
        for contrast in contrasts:
            for key in ("success_evidence_id", "failure_evidence_id"):
                contrast_index.setdefault(contrast[key], []).append(contrast)
        records = []
        for items in cohorts.values():
            for item in items:
                records.append({**self.induction_record(item),
                    "observed_scope": self.episode_scopes.get(item.episode_id, {}),
                    "matched_contrasts": contrast_index.get(item.id, [])[:2]})
        base = {"environment": environment_payload(self.config),
                "schema": schema.model_dump(mode="json")}
        self._note_refusals(evidence, RuleInductionResult(rules=[]), "induce_rules")
        proposal = self.bounded.induce(
            records,
            RULE_INDUCER, RuleInductionResult, "induce_rules", base,
            "local_evidence", "candidate_rule_sets")
        self._note_refusals(evidence, proposal, "induce_rules")
        write_json(self.work_dir / "induction" / "rules.proposed.json", proposal)
        payload = {**base, "proposed_rules": proposal.model_dump(mode="json"),
                   "local_evidence": [self.induction_record(item) for item in evidence],
                   "minimum_confidence": self.config.min_rule_confidence}
        if (len(evidence) <= self.config.induction_batch_size
                and self.bounded.size(RULE_FALSIFIER, payload, RuleInductionResult) <= self.config.max_prompt_bytes):
            reviewed = self.bounded.call(RULE_FALSIFIER, payload, RuleInductionResult, "falsify_rules")
        else:
            audits = []
            # Audit each action's rules against every evidence record for that action.
            # The hard budget also catches oversized schemas or indivisible rule sets.
            for action, items in cohorts.items():
                scoped = proposal.model_copy(update={"rules": [r for r in proposal.rules if r.action_type == action]})
                audit = self.bounded.induce(
                    [self.induction_record(item) for item in items],
                    RULE_FALSIFIER, RuleInductionResult, "falsify_rules_chunk",
                    {**base, "proposed_rules": scoped.model_dump(mode="json")},
                    "local_evidence", "batch_audits")
                audits.append(audit.model_dump(mode="json"))
            reviewed = self.bounded.reduce(audits, RULE_FALSIFIER, RuleInductionResult,
                                            "falsify_rules", base, "batch_audits")
        self._note_refusals(evidence, reviewed, "falsify_rules")
        reviewed.unresolved = list(dict.fromkeys(proposal.unresolved + reviewed.unresolved))
        for rule in reviewed.rules:
            if not RENDERER_ID.fullmatch(rule.renderer):
                # The model described the output format where an identifier belongs; keep the description.
                reviewed.unresolved.append(f"Rule {rule.id}: renderer was prose, renamed to {rule.action_type}.{rule.id}")
                rule.description = f"{rule.description.rstrip('.')}. Rendering: {rule.renderer}"
                rule.renderer = f"{rule.action_type}.{rule.id}"
        # Persist complete reviewed candidates, including excluded rules and their effects.
        for rule in reviewed.rules:
            if rule.confidence < self.config.min_rule_confidence and rule.status == "supported":
                rule.status = "tentative"
            if rule.status != "supported":
                reviewed.unresolved.append(f"Candidate {rule.id} is not executable: {rule.status}")
        write_json(self.work_dir / "induction" / "rules.reviewed.json", reviewed)
        return reviewed

    def induce_renderers(
        self, evidence: list[LocalTransitionEvidence], rules: RuleInductionResult
    ) -> RenderInductionResult:
        outputs = []
        for item in sorted(evidence, key=lambda e: e.id):
            record = self.induction_record(item)
            outputs.append({"action": record["action"], "outcome": record["outcome"],
                            "observation": record["observation_text"], "facts": record["observation_facts"],
                            "provenance": record["provenance"]})
        required = sorted({rule.renderer for rule in rules.rules})
        self._note_refusals(evidence, RenderInductionResult(renderers=[]), "induce_renderers")
        result = self.bounded.induce(outputs, RENDERER_INDUCER, RenderInductionResult,
            "induce_renderers", {"rules": rules.model_dump(mode="json"), "required_renderer_ids": required},
            "observed_outputs", "partial_contracts")
        self._note_refusals(evidence, result, "induce_renderers")
        write_json(self.work_dir / "induction" / "renderers.json", result)
        return result

    def induce_notes(
        self, evidence: list[LocalTransitionEvidence], rules: RuleInductionResult, renderers: RenderInductionResult
    ) -> NoteInductionResult:
        """Conventions, formats, constraints, and facts that rules and contracts do not express."""
        records = []
        for item in sorted(evidence, key=lambda e: e.id):
            record = self.induction_record(item)
            records.append({"action": record["action"], "outcome": record["outcome"],
                            "observation": record["observation_text"],
                            "facts": record["observation_facts"] + record["preconditions"],
                            "provenance": record["provenance"]})
        self._note_refusals(evidence, NoteInductionResult(), "induce_notes")
        result = self.bounded.induce(records, NOTE_INDUCER, NoteInductionResult, "induce_notes",
            {"rules": rules.model_dump(mode="json"), "renderers": renderers.model_dump(mode="json")},
            "observed_transitions", "partial_notes")
        self._note_refusals(evidence, result, "induce_notes")
        for note in result.notes:
            if note.confidence < self.config.min_rule_confidence and note.status == "supported":
                note.status = "tentative"
        write_json(self.work_dir / "induction" / "notes.json", result)
        return result

    @staticmethod
    def select_demonstrations(evidence: list[LocalTransitionEvidence], limit_per_action: int = 3,
                              rules=None) -> list[Demonstration]:
        selected: list[Demonstration] = []
        counts: dict[str, int] = {}
        remaining = sorted(evidence, key=lambda item: (-item.confidence.value, item.id))
        outcomes, episodes = set(), set()
        while remaining:
            item = max(remaining, key=lambda e: ((e.action.type, e.outcome) not in outcomes,
                       (e.action.type, e.episode_id) not in episodes, e.confidence.value))
            remaining.remove(item)
            action_type = item.action.type
            if counts.get(action_type, 0) >= limit_per_action:
                continue
            counts[action_type] = counts.get(action_type, 0) + 1
            outcomes.add((action_type, item.outcome))
            episodes.add((action_type, item.episode_id))
            before, after = {}, {}
            for facts, projection in [(item.preconditions, before), (item.observation_facts, after)]:
                for fact in facts:
                    if (fact.status.value == "observed" and fact.provenance and fact.predicate in {"eq", "equals", "="}
                            and fact.subject.split(".", 1)[0] in {"world", "session", "surface", "epistemic"}):
                        set_path(projection, fact.subject, fact.value)
            event_ids = {event for ref in item.provenance for event in ref.event_ids}
            linked = [rule.id for rule in (rules or []) if rule.action_type == action_type
                      and event_ids & {event for ref in rule.provenance for event in ref.event_ids}]
            selected.append(
                Demonstration(
                    id=f"demo_{item.id}",
                    action=item.action,
                    observation=item.observation_text,
                    state_before=before, state_after=after, rule_ids=linked, outcome=item.outcome,
                    provenance=item.provenance
                    or [SourceRef(source_id="unknown", episode_id=item.episode_id)],
                )
            )
        return selected

    def run(self, paths: Iterable[str | Path], output_dir: str | Path,
            split_manifest: SplitManifest | None = None) -> Path:
        if Path(output_dir).exists():
            raise FileExistsError("Choose a new package destination before reconstruction")
        episodes, transitions = self.ingest(paths, split_manifest)
        if not transitions:
            raise ValueError("No action-to-observation transitions were found in the supplied traces")
        evidence = self.extract_evidence(episodes, transitions)
        schema = self.induce_schema(evidence)
        evidence = canonical_evidence(evidence, schema)
        rules = self.induce_rules(evidence, schema)
        renderers = self.induce_renderers(evidence, rules)
        notes = self.induce_notes(evidence, rules, renderers)
        kept_rules, kept_invariants, kept_renderers, kept_notes, reconciled, rejected = reconcile_artifacts(
            schema, rules.rules, rules.invariants, renderers.renderers, notes.notes)
        if rejected:
            write_json(self.work_dir / "induction" / "rejected.json", rejected)
        reconciled += reconcile_provenance(schema, kept_rules, kept_invariants, kept_renderers, kept_notes,
                                           episodes, evidence, transitions)
        unresolved = list(dict.fromkeys(schema.unresolved + rules.unresolved + renderers.unresolved
                                        + notes.unresolved + reconciled))
        artifacts = ReconstructionArtifacts(
            episodes=episodes,
            transitions=transitions,
            evidence=evidence,
            action_schema=schema.action_schema,
            state_schema=schema.state_schema,
            rules=kept_rules,
            invariants=kept_invariants,
            renderers=kept_renderers,
            demonstrations=self.select_demonstrations(evidence, rules=kept_rules),
            notes=kept_notes,
            unresolved=unresolved,
            rejected=rejected,
            source_snapshots=self.source_snapshots,
            split_assignments=self.split_assignments,
        )
        write_json(self.work_dir / "artifacts.json", artifacts)
        return EnvironmentCompiler(self.config).compile(artifacts, output_dir)
