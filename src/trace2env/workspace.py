"""Workspace tools: how the world-model agent inspects knowledge, state, and episodic memory.

Every tool returns a bounded, JSON-serializable view and registers the artifact identifiers it
exposed, so a submitted transition can only cite what the agent actually retrieved. Rules are
offered as dry-run tools rather than executed behind the agent's back; nothing here mutates the
session, and the harness alone verifies and commits a transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from trace2env.engine import (
    MISSING,
    apply_mutations,
    check_invariants,
    deterministic_plan,
    evaluate_condition,
    get_path,
    render_template,
)
from trace2env.eligibility import rule_eligible
from trace2env.memory import EpisodicMemory
from trace2env.models import (
    SUBMIT_TOOL_NAME,
    EnvironmentState,
    MemoryEntry,
    NormalizedAction,
    StateMutation,
    TransitionRule,
    TransitionSubmission,
)
from trace2env.package import KNOWLEDGE_KINDS, EnvironmentPackage, PackageInspector
from trace2env.trace_corpus import TraceCorpus, action_query
from trace2env.validation import validate_mutations, validate_state_types

_ABSENT = object()
ABSENCE_MEANS_UNKNOWN = ("An absent key or path means 'not observed in this episode yet', not 'does not exist': "
                         "the tracker records only what observations established.")


def clip_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} characters]"


def clip_span(text: str, limit: int, *, turn: int | None = None) -> str:
    """Head-and-tail clip that tells the agent how to read the omitted middle exactly (``read_turn``)."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit * 3 // 4]
    tail = text[-(limit - len(head)):] if limit > len(head) else ""
    hint = f"read_turn(turn={turn}, offset={len(head)})" if turn is not None else "read_turn"
    return f"{head}\n[... {len(text) - len(head) - len(tail)} characters omitted; {hint} shows them ...]\n{tail}"


def compact_action(action: NormalizedAction | dict[str, Any]) -> dict[str, Any]:
    """An action view without redundant copies: ``keystrokes`` repeat ``commands``/``command``; ``raw`` repeats both."""
    data = action.model_dump(mode="json", exclude={"raw"}) if isinstance(action, NormalizedAction) else dict(action)
    data.pop("raw", None)
    arguments = dict(data.get("arguments") or {})
    if "keystrokes" in arguments and ("commands" in arguments or "command" in arguments):
        entries = arguments.pop("keystrokes")
        if isinstance(entries, list):
            total = sum(float(entry.get("duration", 0) or 0) for entry in entries if isinstance(entry, dict))
            arguments["duration_total"] = round(total, 3)
    data["arguments"] = arguments
    return data


def clip_value(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return clip_text(value, limit)
    if isinstance(value, dict):
        return {str(key): clip_value(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [clip_value(item, limit) for item in value]
    return value


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Dotted leaf paths; empty containers contribute no leaves, so filling a namespace is not itself a change."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value} if prefix else {}


def state_diff(before: EnvironmentState, after: EnvironmentState, *, limit: int = 50, text_limit: int = 300) -> dict[str, Any]:
    old = flatten(before.model_dump(mode="json"))
    new = flatten(after.model_dump(mode="json"))
    changes: dict[str, Any] = {}
    for path in sorted(set(old) | set(new)):
        if path.startswith("revision") or old.get(path, _ABSENT) == new.get(path, _ABSENT):
            continue
        changes[path] = {
            "before": clip_value(old[path], text_limit) if path in old else "<absent>",
            "after": clip_value(new[path], text_limit) if path in new else "<absent>",
        }
        if len(changes) >= limit:
            changes["..."] = "further changes omitted"
            break
    return changes


def state_summary(state: EnvironmentState, *, text_limit: int = 160) -> dict[str, Any]:
    summary: dict[str, Any] = {"step": state.step, "revision": state.revision, "pending_events": len(state.pending_events)}
    for namespace in ("world", "session", "surface", "epistemic"):
        summary[namespace] = clip_value(getattr(state, namespace), text_limit)
    return summary


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any] | None = None

    def spec(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class WorkspaceTools:
    package: EnvironmentPackage
    inspector: PackageInspector
    memory: EpisodicMemory | None
    state: EnvironmentState
    action: NormalizedAction
    text_limit: int = 1500
    list_limit: int = 8
    minimum_confidence: float = 0.0
    # Online-harness features (see RuntimeHarness): "history" adds read_turn and full-text recall
    # hits, "compact" slims duplicated views, "unknown" spells out absence semantics.
    features: set[str] = field(default_factory=set)
    brief_action_type: str | None = None  # the action whose retrieval the brief already shows
    turn_chars: int = 6000  # verbatim chars per memory observation before head/tail clipping
    recall_full: int = 3  # memory hits returned in full (up to full_chars) per recall/recent_turns call
    full_chars: int = 24000
    # Raw construction traces offered instead of reconstructed knowledge (the agentic_raw_traces control).
    trace_corpus: TraceCorpus | None = None
    # False for the trace2env_no_state control: no read_state / state_schema / apply_rule / dry_run.
    state_tools: bool = True
    # False for the harness_only control: no list_actions / inspect_action / search_knowledge / read_evidence.
    knowledge_tools: bool = True
    # Harness v5.2: applicability decisions and sanitized views for every package item shown (None = v5.1 behaviour).
    gate: Any = None
    withhold_templates: bool = False  # option B: rule templates withheld when the official input documents the format
    retrieved_ids: set[str] = field(default_factory=set)
    discovered_rules: dict[str, TransitionRule] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    # ─── Catalog ──────────────────────────────────────────────────────────────

    def tools(self) -> list[Tool]:
        string = {"type": "string"}
        integer = {"type": "integer", "minimum": 1}
        return [
            *([Tool("list_actions", "List the environment's known actions with their aliases and argument names.",
                    {"type": "object", "properties": {}}, self._list_actions),
               Tool("inspect_action", "Retrieve an action's contract, the rules eligible for it now, its observation "
                    "contracts, demonstrations, notes, and the global invariants.",
                    {"type": "object", "properties": {"action_type": string}, "required": ["action_type"]}, self._inspect_action),
               Tool("search_knowledge", "Full-text search over reconstructed knowledge: rules, notes, demonstrations, "
                    "evidence (real observed transitions), and actions. Use it for formats, error shapes, and examples.",
                    {"type": "object", "properties": {"query": string, "kinds": {"type": "array", "items": {"type": "string", "enum": KNOWLEDGE_KINDS}},
                     "limit": integer}, "required": ["query"]}, self._search_knowledge)] if self.knowledge_tools else []),
            *([Tool("read_state", "Read the current session state: a dotted path such as world.balance, or omit the "
                    "path for a summary of every namespace.",
                    {"type": "object", "properties": {"path": string}}, self._read_state),
               Tool("state_schema", "Describe declared state fields (types, mutability, descriptions), optionally under a path prefix.",
                    {"type": "object", "properties": {"prefix": string}}, self._state_schema)] if self.state_tools else []),
            Tool("recall", "Search this episode's memory of earlier actions and real observations.",
                 {"type": "object", "properties": {"query": string, "limit": integer}, "required": ["query"]}, self._recall),
            Tool("recent_turns", "Return the most recent turns of this episode with their observations.",
                 {"type": "object", "properties": {"limit": integer}}, self._recent_turns),
            *([Tool("read_turn", "Read an exact span of one earlier turn's real observation (turn numbers as shown in "
                    "recent_memory/recall; turn 0 is the initial screen). Page with offset/length to reproduce long "
                    "content verbatim.",
                    {"type": "object", "properties": {"turn": {"type": "integer", "minimum": 0}, "offset": {"type": "integer", "minimum": 0},
                     "length": integer}, "required": ["turn"]}, self._read_turn)] if "history" in self.features else []),
            *([Tool("read_evidence", "Read one raw turn of the episodes this package was built from: its action and an exact "
                    "span of its real observation (page with offset/length), plus the neighbouring turns' action types. "
                    "Ids come from similar_turns, search_knowledge (evidence/demonstration), or demonstrations.",
                    {"type": "object", "properties": {"id": string, "offset": {"type": "integer", "minimum": 0}, "length": integer},
                     "required": ["id"]}, self._read_evidence)] if "evidence" in self.features and self.knowledge_tools else []),
            *([Tool("search_traces", "Full-text search over the raw action -> observation turns of other recorded episodes "
                    "of this environment (different tasks). Returns turn ids, the action, and a snippet; use "
                    "read_trace_turn for the exact observation.",
                    {"type": "object", "properties": {"query": string, "limit": integer}, "required": ["query"]}, self._search_traces),
               Tool("read_trace_turn", "Read one raw trace turn: its action and an exact span of its real observation "
                    "(page with offset/length), plus the neighbouring turns' action types.",
                    {"type": "object", "properties": {"episode": string, "turn": {"type": "integer", "minimum": 0},
                     "offset": {"type": "integer", "minimum": 0}, "length": integer}, "required": ["episode", "turn"]},
                    self._read_trace_turn)] if self.trace_corpus is not None else []),
            *([Tool("apply_rule", "Dry-run a rule against the current state: whether it applies, its resolved effects, "
                    "the resulting state diff, the rendered observation when a template exists, and invariant checks.",
                    {"type": "object", "properties": {"rule_id": string}, "required": ["rule_id"]}, self._apply_rule),
               Tool("dry_run", "Apply candidate effects to a copy of the state and report the diff and invariant checks.",
                    {"type": "object", "properties": {"effects": {"type": "array", "items": StateMutation.model_json_schema()}},
                     "required": ["effects"]}, self._dry_run)] if self.state_tools else []),
            Tool(SUBMIT_TOOL_NAME, "Finish: submit the transition (effects, outcome, exact observation, rule_ids, "
                 "citations, uncertainty). This ends the simulation of this action.",
                 TransitionSubmission.model_json_schema(), None),
        ]

    def specs(self) -> list[dict[str, Any]]:
        return [tool.spec() for tool in self.tools()]

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = next((item for item in self.tools() if item.name == name), None)
        record: dict[str, Any] = {"tool": name, "arguments": clip_value(arguments, 200)}
        if tool is None or tool.handler is None:
            record["error"] = f"Unknown tool {name!r}"
            self.calls.append(record)
            return {"error": record["error"], "available_tools": [item.name for item in self.tools()]}
        try:
            result = tool.handler(arguments or {})
        except Exception as exc:  # noqa: BLE001 - tool failures are reported to the agent, not raised
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, dict) and "error" in result:
            record["error"] = result["error"]
        self.calls.append(record)
        return result

    # ─── Knowledge ────────────────────────────────────────────────────────────

    def _list_actions(self, arguments: dict[str, Any]) -> Any:
        self.retrieved_ids.add("schema:actions")
        return [
            {"name": spec.name, "description": spec.description, "aliases": spec.aliases, "arguments": sorted(spec.arguments)}
            for spec in self.package.action_schema.actions
        ]

    # ─── Harness v5.2 gate helpers (no-ops when self.gate is None) ────────────

    def _gate_record(self, reference: str):
        evidence_id = self.inspector.resolve_evidence_id(reference)
        return self.inspector._evidence_index()["by_id"].get(evidence_id) if evidence_id else None

    def _gate_summary(self, summary: dict[str, Any], *, kind: str = "evidence") -> dict[str, Any]:
        """An evidence summary (hit or search result) with its applicability and a sanitized snippet when needed."""
        if self.gate is None:
            return summary
        record = self._gate_record(summary.get("id", ""))
        if record is None:
            return summary
        decision = self.gate.decide_evidence(record, kind=kind)
        if decision.disposition == "rejected":
            return None  # withheld (format_only ablation): the item is recorded, not shown
        view = dict(summary)
        if decision.disposition == "sanitized":
            for key in ("snippet", "action_arguments", "observation_head"):
                if key in view:
                    view[key] = self.gate.sanitize_value(view[key], record, decision)
        view["applicability"] = decision.view()
        return view

    def _gate_notes(self, notes: list[Any]) -> list[Any]:
        """Notes with their applicability; a note contradicted by this episode's transcript is dropped."""
        if self.gate is None:
            return notes
        kept = []
        for note in notes:
            data = note if isinstance(note, dict) else note.model_dump(mode="json")
            decision = self.gate.decide_note(data)
            if decision.disposition == "rejected":
                continue
            kept.append({**data, "applicability": decision.view()})
        return kept

    def _inspect_action(self, arguments: dict[str, Any]) -> Any:
        result = self.inspector.inspect_action(str(arguments.get("action_type", "")), state=self.state)
        if self.gate is not None:
            result = dict(result)
            result["notes"] = self._gate_notes(result.get("notes") or [])
            if getattr(self.gate, "judge_llm", None) is not None:
                self.gate.prejudge([r for r in (self._gate_record(d.get("id", "")) for d in result.get("demonstrations") or []) if r is not None])
            demos = []
            for demo in result.get("demonstrations") or []:
                record = self._gate_record(demo.get("id", ""))
                if record is None:
                    demos.append(demo)
                    continue
                decision = self.gate.decide_evidence(record, kind="demonstration")
                if decision.disposition == "rejected":
                    continue
                view = dict(demo)
                if decision.disposition == "sanitized":
                    view["observation"] = self.gate.sanitize(str(demo.get("observation") or ""), record, decision)
                    view["action"] = self.gate.sanitize_value(demo.get("action"), record, decision)
                view["applicability"] = decision.view()
                demos.append(view)
            result["demonstrations"] = demos
        for item in result["rules"]:
            rule = TransitionRule.model_validate(item)
            self.discovered_rules.setdefault(rule.id, rule)
        if self.withhold_templates:  # option B: the returned view carries no template (the rule objects above are intact)
            result = dict(result)
            result["rules"] = [{**item, "observation_template": None,
                                "template_withheld": "the official input documents this action's observation format; follow its example"}
                               for item in result["rules"]]
        self.retrieved_ids.update(result["artifact_ids"])
        self.retrieved_ids.update(f"invariant:{item['id']}" for item in result["invariants"])
        if "compact" in self.features and result["canonical_action_type"] == self.brief_action_type:
            # The brief already carries this action's contract, notes, contracts, and demonstrations;
            # only the rule bodies (conditions, effects) add information.
            return {
                "note": "This action's contract, notes, observation contracts, and demonstrations are already in your "
                        "brief (retrieved); only the rule bodies are listed here. Use apply_rule to dry-run one.",
                "canonical_action_type": result["canonical_action_type"],
                "rules": [{key: item[key] for key in ("id", "description", "conditions", "effects", "outcome", "confidence", "status")}
                          for item in result["rules"]],
                "invariant_ids": [item["id"] for item in result["invariants"]],
            }
        return clip_value(result, self.text_limit)

    def _search_knowledge(self, arguments: dict[str, Any]) -> Any:
        limit = min(int(arguments.get("limit") or self.list_limit), 25)
        kinds = arguments.get("kinds") or None
        results = self.inspector.search_knowledge(str(arguments.get("query", "")), kinds=kinds, limit=limit)
        if self.gate is not None and getattr(self.gate, "judge_llm", None) is not None:
            self.gate.prejudge([r for r in (self._gate_record(item["ref_id"]) for item in results if item["kind"] in ("evidence", "demonstration")) if r is not None])
        shaped: list[Any] = []
        for item in results:
            self.retrieved_ids.add(item["id"])
            if item["kind"] == "rule":
                rule = next((rule for rule in self.package.rules if rule.id == item["ref_id"]), None)
                if rule is not None:
                    self.discovered_rules.setdefault(rule.id, rule)
            if "evidence" in self.features and item["kind"] in ("evidence", "demonstration"):
                # A raw turn is never clipped into the result; the hit says how long it is and how to read it exactly.
                evidence_id = self.inspector.resolve_evidence_id(item["ref_id"])
                summary = self.inspector.evidence_summary(evidence_id, snippet=item.get("snippet")) if evidence_id else None
                if summary is not None:
                    self.retrieved_ids.add(summary["id"])
                    gated = self._gate_summary(summary, kind=item["kind"])
                    if gated is not None:
                        shaped.append({**gated, "kind": item["kind"]})
                    continue
            if self.gate is not None:
                if item["kind"] == "note":
                    kept = self._gate_notes([{"id": item["ref_id"], "statement": item.get("text") or ""}])
                    if not kept:
                        continue  # contradicted by this episode's transcript
                    item = {**item, "applicability": kept[0]["applicability"]}
                elif item["kind"] in ("evidence", "demonstration"):
                    record = self._gate_record(item["ref_id"])
                    if record is not None:
                        decision = self.gate.decide_evidence(record, kind=item["kind"])
                        if decision.disposition == "rejected":
                            continue
                        item = dict(item)
                        if decision.disposition == "sanitized":
                            for key in ("text", "snippet"):
                                if key in item:
                                    item[key] = self.gate.sanitize(str(item[key]), record, decision)
                        item["applicability"] = decision.view()
            shaped.append(clip_value(item, self.text_limit))
        return shaped

    # ─── Evidence tier (v3 read path) ─────────────────────────────────────────

    def evidence_hits(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Raw turns whose action or observation matches, as summaries; registered as citable."""
        if not query.strip():
            return []
        hits = self.inspector.search_evidence(query, min(int(limit or self.list_limit), 25))
        self.retrieved_ids.update(hit["id"] for hit in hits)
        if self.gate is not None and getattr(self.gate, "judge_llm", None) is not None:
            self.gate.prejudge([r for r in (self._gate_record(hit["id"]) for hit in hits) if r is not None])  # one batched call
        return [gated for gated in (self._gate_summary(hit) for hit in hits) if gated is not None]

    def demo_view(self, demo: Any) -> dict[str, Any]:
        """A demonstration as a summary with a read pointer instead of a clipped observation."""
        data = demo if isinstance(demo, dict) else demo.model_dump(mode="json")
        evidence_id = self.inspector.resolve_evidence_id(data["id"])
        observation = data.get("observation") or ""
        view = {
            "id": f"demo:{data['id']}", "action": compact_action(data["action"]), "outcome": data.get("outcome"),
            "observation_chars": len(observation), "observation_head": observation[:400],
        }
        if evidence_id is not None:
            view["evidence"] = f"evidence:{evidence_id}"
            view["read"] = f"read_evidence(id='evidence:{evidence_id}')"
            self.retrieved_ids.add(f"evidence:{evidence_id}")
            if self.gate is not None:
                record = self._gate_record(f"evidence:{evidence_id}")
                if record is not None:
                    decision = self.gate.decide_evidence(record, kind="demonstration")
                    if decision.disposition == "rejected":
                        return {"id": view["id"], "withheld": True, "applicability": decision.view()}
                    if decision.disposition == "sanitized":
                        view["observation_head"] = self.gate.sanitize(view["observation_head"], record, decision)
                        view["action"] = self.gate.sanitize_value(view["action"], record, decision)
                    view["applicability"] = decision.view()
        return view

    def _read_evidence(self, arguments: dict[str, Any]) -> Any:
        evidence_id = self.inspector.resolve_evidence_id(str(arguments.get("id", "")))
        if evidence_id is None:
            return {"error": f"Unknown evidence id {arguments.get('id')!r}; use similar_turns, search_knowledge, or a demonstration's id"}
        offset = max(0, int(arguments.get("offset") or 0))
        length = max(1, min(int(arguments.get("length") or self.turn_chars), self.full_chars))
        view = self.inspector.evidence_view(evidence_id, offset=offset, length=length)
        if view is not None:
            self.retrieved_ids.add(view["id"])
            if self.gate is not None:
                record = self._gate_record(view["id"])
                if record is not None:
                    decision = self.gate.decide_evidence(record)
                    if decision.disposition == "rejected":
                        return {"id": view["id"], "withheld": "This turn is task-specific or uncertain evidence and is not available "
                                "in this configuration; predict from the episode's transcript and state.", "applicability": decision.view()}
                    view = dict(view)
                    if decision.disposition == "sanitized":
                        view["text"] = self.gate.sanitize(view["text"], record, decision)
                        view["action_arguments"] = self.gate.sanitize_value(view.get("action_arguments"), record, decision)
                    view["applicability"] = decision.view()
        return view

    # ─── Raw traces (agentic_raw_traces control) ──────────────────────────────

    def trace_hits(self, query: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Search the raw-trace corpus and register the hits as citable artifacts."""
        if self.trace_corpus is None:
            return []
        hits = self.trace_corpus.search(query, min(int(limit or self.list_limit), 25))
        self.retrieved_ids.update(hit["id"] for hit in hits)
        return clip_value(hits, self.text_limit)

    def _search_traces(self, arguments: dict[str, Any]) -> Any:
        return self.trace_hits(str(arguments.get("query", "")), arguments.get("limit"))

    def _read_trace_turn(self, arguments: dict[str, Any]) -> Any:
        if self.trace_corpus is None:
            return {"error": "No raw-trace corpus is attached to this workspace"}
        episode = str(arguments.get("episode", ""))
        turn = int(arguments.get("turn", -1))
        offset = max(0, int(arguments.get("offset") or 0))
        length = max(1, min(int(arguments.get("length") or self.turn_chars), self.full_chars))
        result = self.trace_corpus.read(episode, turn, offset=offset, length=length)
        if result is None:
            return {"error": f"No trace turn {episode}:{turn}; use search_traces to find turn ids"}
        self.retrieved_ids.add(result["id"])
        return result

    def _state_schema(self, arguments: dict[str, Any]) -> Any:
        prefix = arguments.get("prefix") or None
        fields = self.inspector.inspect_state(prefix)
        self.retrieved_ids.update(f"schema:{item['path']}" for item in fields)
        return fields[: self.list_limit * 4]

    # ─── State and memory ─────────────────────────────────────────────────────

    def _read_state(self, arguments: dict[str, Any]) -> Any:
        path = arguments.get("path")
        if not path:
            self.retrieved_ids.add("state:*")
            summary = state_summary(self.state, text_limit=self.text_limit // 4)
            if "unknown" in self.features:
                summary["semantics"] = ABSENCE_MEANS_UNKNOWN
            return summary
        value = get_path(self.state.model_dump(mode="json"), str(path), MISSING)
        self.retrieved_ids.add(f"state:{path}")
        if value is MISSING:
            missing: dict[str, Any] = {"path": path, "missing": True}
            if "unknown" in self.features:
                missing["meaning"] = ABSENCE_MEANS_UNKNOWN
            return missing
        if "unknown" in self.features and isinstance(value, str):
            # Exact values can be paged like turns so long contents are never reproduced from a clip.
            offset = max(0, int(arguments.get("offset") or 0))
            length = max(1, min(int(arguments.get("length") or self.full_chars), self.full_chars))
            return {"path": path, "total_chars": len(value), "offset": offset, "text": value[offset:offset + length],
                    "next_offset": offset + length if offset + length < len(value) else None}
        return {"path": path, "value": clip_value(value, self.text_limit)}

    def _memory_view(self, entry: MemoryEntry, *, full: bool = False) -> dict[str, Any]:
        identifier = f"memory:{entry.id}"
        self.retrieved_ids.add(identifier)
        if "history" in self.features:
            observation = entry.observation[: self.full_chars] if full else clip_span(entry.observation, self.turn_chars, turn=entry.turn)
            if full and len(entry.observation) > self.full_chars:
                observation += f"\n[... {len(entry.observation) - self.full_chars} more characters; read_turn(turn={entry.turn}, offset={self.full_chars}) ...]"
        else:
            observation = clip_text(entry.observation, self.text_limit)
        action = compact_action(entry.action) if "compact" in self.features else entry.action.model_dump(mode="json", exclude={"raw"})
        return {
            "id": identifier,
            "turn": entry.turn,
            "kind": entry.kind,
            "action": action,
            "observation": observation,
            "observation_chars": len(entry.observation),
            "effects": [effect.model_dump(mode="json", exclude_none=True) for effect in entry.effects],
        }

    def _views(self, entries: list[MemoryEntry], *, newest_full: bool) -> list[dict[str, Any]]:
        """Memory views where the most relevant ``recall_full`` entries are complete and the rest are clipped."""
        full_ids = {entry.id for entry in (entries[-self.recall_full:] if newest_full else entries[: self.recall_full])}
        return [self._memory_view(entry, full=entry.id in full_ids) for entry in entries]

    def _recall(self, arguments: dict[str, Any]) -> Any:
        if self.memory is None:
            return []
        limit = min(int(arguments.get("limit") or self.list_limit), 25)
        return self._views(self.memory.search(str(arguments.get("query", "")), limit), newest_full=False)

    def _recent_turns(self, arguments: dict[str, Any]) -> Any:
        if self.memory is None:
            return []
        limit = min(int(arguments.get("limit") or self.list_limit), 25)
        return self._views(self.memory.recent(limit), newest_full=True)

    def _read_turn(self, arguments: dict[str, Any]) -> Any:
        if self.memory is None:
            return {"error": "No episodic memory in this session"}
        turn = int(arguments.get("turn", -1))
        entries = self.memory.by_turn(turn)
        if not entries:
            return {"error": f"No recorded turn {turn}", "recorded_turns": sorted({e.turn for e in self.memory.recent(200)})}
        entry = entries[0]
        offset = max(0, int(arguments.get("offset") or 0))
        length = max(1, min(int(arguments.get("length") or self.turn_chars), self.full_chars))
        text = entry.observation
        self.retrieved_ids.add(f"memory:{entry.id}")
        return {
            "id": f"memory:{entry.id}", "turn": entry.turn, "kind": entry.kind,
            "action": compact_action(entry.action),
            "total_chars": len(text), "offset": offset, "length": min(length, max(0, len(text) - offset)),
            "text": text[offset:offset + length],
            "next_offset": offset + length if offset + length < len(text) else None,
        }

    # ─── Simulation dry-runs ──────────────────────────────────────────────────

    def _spec(self):
        return next((spec for spec in self.package.action_schema.actions if spec.name == self.action.type), None)

    def _render_context(self, after: EnvironmentState, outcome: str) -> dict[str, Any]:
        return {
            "action": self.action.model_dump(mode="python"),
            "state_before": self.state.model_dump(mode="python"),
            "state_after": after.model_dump(mode="python"),
            "outcome": outcome,
        }

    def _apply_rule(self, arguments: dict[str, Any]) -> Any:
        rule_id = str(arguments.get("rule_id", ""))
        rule = next((item for item in self.package.rules if item.id == rule_id), None) or self.discovered_rules.get(rule_id)
        if rule is None:
            return {"error": f"Unknown rule {rule_id!r}; use inspect_action or search_knowledge to find rule ids"}
        self.retrieved_ids.add(f"rule:{rule.id}")
        self.discovered_rules.setdefault(rule.id, rule)
        if rule.action_type != self.action.type:
            return {"applicable": False, "rule_id": rule.id, "reason": f"Rule is for action {rule.action_type!r}, not {self.action.type!r}"}
        if not rule_eligible(rule, self.package.scope, self.state, minimum_confidence=self.minimum_confidence):
            return {"applicable": False, "rule_id": rule.id, "reason": "Rule is not eligible here (status, confidence, or scope)"}
        failed = [condition.model_dump(mode="json", exclude_none=True) for condition in rule.conditions
                  if not evaluate_condition(condition, self.state, self.action)]
        if failed:
            return {"applicable": False, "rule_id": rule.id, "reason": "Conditions do not hold on the current state", "failed_conditions": failed}
        try:
            plan = deterministic_plan(rule, self.action)
            validate_mutations(plan.effects, self.package.state_schema, self._spec())
            candidate = apply_mutations(self.state, plan.effects, action=self.action)
            validate_state_types(candidate, self.package.state_schema)
        except (KeyError, ValueError) as exc:
            return {"applicable": False, "rule_id": rule.id, "reason": f"Effects cannot be applied: {exc}"}
        contract = next((item for item in self.package.renderers if item.id == plan.renderer_id), None)
        template = plan.observation_template or (contract.template if contract else None)
        observation: str | None = None
        withheld = bool(template) and self.withhold_templates
        if template and not withheld:
            try:
                observation = render_template(template, self._render_context(candidate, plan.outcome.value))
            except KeyError as exc:
                observation = None
        invariants = check_invariants(self.package.invariants, candidate, self.action)
        return {
            "applicable": True,
            "rule_id": rule.id,
            "description": rule.description,
            "outcome": plan.outcome.value,
            "effects": [effect.model_dump(mode="json", exclude_none=True) for effect in plan.effects],
            "state_diff": state_diff(self.state, candidate),
            "observation": clip_text(observation, self.text_limit) if observation is not None else None,
            **({"observation_withheld": "the official input documents this action's observation format; follow its example"} if withheld else {}),
            "invariants_ok": invariants.accepted,
            "invariant_issues": [issue.message for issue in invariants.issues],
            "how_to_submit": "To apply this rule, submit exactly these effects with rule_ids=[rule_id]; "
                             "otherwise leave rule_ids empty and list the rule in citations.",
        }

    def _dry_run(self, arguments: dict[str, Any]) -> Any:
        try:
            effects = [StateMutation.model_validate(item) for item in arguments.get("effects", [])]
            validate_mutations(effects, self.package.state_schema, self._spec())
            candidate = apply_mutations(self.state, effects, action=self.action)
            validate_state_types(candidate, self.package.state_schema)
        except (KeyError, ValueError) as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        invariants = check_invariants(self.package.invariants, candidate, self.action)
        return {
            "ok": True,
            "state_diff": state_diff(self.state, candidate),
            "invariants_ok": invariants.accepted,
            "invariant_issues": [issue.message for issue in invariants.issues],
        }
