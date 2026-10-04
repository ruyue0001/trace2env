"""Predict AgentWorldBench observations with the Trace2Env harness or with a prompting baseline.

``prompting`` reproduces the official inference: the record's system prompt plus the full
alternating history goes to a chat model. ``agentic`` is the Trace2Env path: each trajectory
gets a session whose structured state and episodic memory are rebuilt from the prefix by
``StateTracker`` (teacher-forced on the real observations); the evaluated turn is then simulated
by the world-model agent through ``RuntimeHarness.step`` on a fork of that session, so nothing
predicted leaks back into the tracked trajectory. Records of one trajectory share the session,
so a prefix is reconstructed once.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Literal

from trace2env.agentworld import (
    case_from_row,
    inference_messages,
    initial_observation,
    observed_turns,
    prompt_sections,
    subtask_of,
    wrap_prediction,
)
from trace2env.envpack import action_query, envpack_view
from trace2env.llm import AgentLLM, ChatLLM, StructuredLLM
from trace2env.memory import EpisodicMemory
from trace2env.models import (
    AgentWorldCase,
    AuditRecord,
    EnvironmentState,
    FastPathPolicy,
    NormalizedAction,
    ObservedTurn,
    RenderedObservation,
    StateTrackingStep,
    StepRequest,
    TrustPolicy,
)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.prompts import RUNTIME_RENDERER
from trace2env.runtime import (DEFAULT_FEATURES, ENVIRONMENT_PROMPT_CHARS, HARNESS_FEATURES, PREDICTION_MODES, PredictionMode, RuntimeHarness, _payload,
                               environment_context, initial_state_from_package, last_usage, request_context)
from trace2env.session import SessionStore
from trace2env.state_tracking import StateTracker
from trace2env.trace_corpus import TraceCorpus
from trace2env.trace_corpus import action_query as corpus_action_query
from trace2env.workspace import clip_text

Mode = Literal["prompting", "agentic", "prompting_rag", "envpack_prompting"]
# prompting_rag retrieval styles. "terminal" is the AgentWorldBench setting of every reported run (2026-09-20 terminal,
# 2026-09-23/24 web, android, swe): the typed command words of a keystrokes block, else the action type and its argument
# words (envpack.action_query), under the terminal wording of the retrieved-examples block. "generic" is the EnvScaler
# setting: keystrokes, else the trace corpus's own normalized action query (type, program, command, name or tool words
# only), under environment-neutral wording.
RagStyle = Literal["terminal", "generic"]
RAG_STYLES = ("terminal", "generic")
INITIAL_STATE_ACTION = "session.start"
TRACKING_ACTION = "trace2env.track"
_ACTION_BLOCK = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)
_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")


def current_keystrokes(case: AgentWorldCase) -> str:
    """The raw keystrokes of the turn to predict, straight from its prompt (no schema, no normalization)."""
    match = _ACTION_BLOCK.search(case.current_prompt)
    if not match:
        return ""
    try:
        return "".join(str(item.get("keystrokes", "")) for item in json.loads(match.group(1)) if isinstance(item, dict))
    except (json.JSONDecodeError, AttributeError):
        return ""


def rag_query(text: str, limit: int = 12) -> str:
    """Fixed lexical query for the retrieval baseline: the first ``limit`` distinct command words, no flags."""
    tokens: list[str] = []
    for token in _TOKEN.findall(text):
        token = token.strip("./")
        if token and not token.startswith("-") and token not in tokens:
            tokens.append(token)
    return " ".join(tokens[:limit])


def rag_case_query(case: AgentWorldCase, style: RagStyle = "terminal") -> str:
    """Fixed lexical retrieval query for the current action (see ``RagStyle``).

    AgentWorldBench terminal actions encode keystrokes as a JSON list, whose command words are the query under both
    styles. Rows without keystrokes (tool calls, GUI actions) use the action type and its argument words
    (``envpack.action_query``, the ``terminal`` style of the reported AgentWorldBench runs) or the trace corpus's own
    normalized action query (``generic``, the EnvScaler runs).
    """
    if style not in RAG_STYLES:
        raise ValueError(f"Unknown rag style {style!r}; known: {RAG_STYLES}")
    if style == "terminal":
        return rag_query(current_keystrokes(case)) if _ACTION_BLOCK.search(case.current_prompt) else action_query(case)
    keystrokes = current_keystrokes(case)
    if keystrokes:
        return rag_query(keystrokes)
    return corpus_action_query(case.action) if case.action is not None else ""


def rag_block(hits: list[dict[str, Any]], corpus: TraceCorpus, turn_chars: int, style: RagStyle = "terminal") -> str:
    """Retrieved raw turns as extra demonstrations appended to the official system prompt."""
    if style not in RAG_STYLES:
        raise ValueError(f"Unknown rag style {style!r}; known: {RAG_STYLES}")
    if style == "terminal":
        parts = [
            "\n\n---\n\n# Retrieved examples from other recorded sessions of this environment\n\n"
            "The turns below were recorded in other tasks on the same kind of terminal (retrieved because their command "
            "resembles the current action). They show real output formats and program behaviour; their files and values "
            "belong to those tasks, not to the current session.\n"
        ]
    else:
        parts = [
            "\n\n---\n\n# Retrieved examples from other recorded sessions of this environment\n\n"
            "The turns below were recorded in other tasks in the same kind of environment (retrieved because their action "
            "resembles the current action). They show real output formats and environment behaviour; their entities and values "
            "belong to those tasks, not to the current session.\n"
        ]
    action_label = "**Action (keystrokes):**" if style == "terminal" else "**Action:**"
    for number, hit in enumerate(hits, start=1):
        page = corpus.read(hit["episode"], hit["turn"], length=turn_chars)
        if page is None:
            continue
        keystrokes = page["action"].get("arguments", {}).get("keystrokes")
        typed = "".join(str(item.get("keystrokes", "")) for item in keystrokes if isinstance(item, dict)) if isinstance(keystrokes, list) else None
        if not typed:
            typed = json.dumps(page["action"], ensure_ascii=False)
        text = page["text"]
        if page["next_offset"] is not None:
            text += f"\n[... {page['total_chars'] - page['length']} more characters omitted]"
        parts.append(f"\n## Retrieved example {number} ({hit['id']})\n{action_label}\n```text\n{typed}\n```\n"
                     f"**Environment Observation:**\n{text}\n")
    return "".join(parts)


def tracking_summary(steps: list[StateTrackingStep]) -> dict[str, int]:
    summary = {"rule_turns": 0, "model_turns": 0, "skipped_turns": 0, "model_calls": 0, "mutations_applied": 0,
               "mutations_dropped": 0, "rule_mismatches": 0}
    for step in steps:
        summary[f"{step.route}_turns"] += len(step.turns)
        summary["model_calls"] += step.route == "model"
        summary["mutations_applied"] += step.applied
        summary["mutations_dropped"] += len(step.dropped)
        summary["rule_mismatches"] += sum("predicted a different observation" in issue for issue in step.issues)
    return summary


class AgentWorldRunner:
    def __init__(
        self,
        *,
        mode: Mode,
        package_dir: str | None = None,
        llm: StructuredLLM | None = None,
        agent_llm: AgentLLM | None = None,
        chat_llm: ChatLLM | None = None,
        history_window: int = 6,
        state_window: int = 8,
        trust_policy: TrustPolicy = "schema_checked",
        fast_path: FastPathPolicy = "confident",
        max_tool_calls: int = 8,
        observation_chars: int = 4000,
        allow_unvalidated: bool = False,
        verify_package: bool = True,
        session_root: str | Path | None = None,
        features: set[str] | str | None = "default",
        history_turn_chars: int = 6000,
        history_total_chars: int = 60000,
        history_full: bool = False,
        official_input: bool = False,
        knowledge_gate: bool = False,
        knowledge_gate_mode: str = "full",
        knowledge_judge: bool = False,
        knowledge_names: str = "paths",
        prediction_mode: PredictionMode = "agent",
        trace_corpus: TraceCorpus | None = None,
        single_shot_llm: StructuredLLM | None = None,
        state_tracking: bool = True,
        package_knowledge: bool = True,
        knowledge_tools: bool = True,
        rag_corpus: TraceCorpus | None = None,
        rag_top_k: int = 5,
        rag_turn_chars: int = 4000,
        rag_style: RagStyle = "terminal",
        effect_rejection: str = "fail",
        envpack_top_k: int = 6,
        envpack_evidence_chars: int = 3000,
        environment_prompt_chars: int | None = ENVIRONMENT_PROMPT_CHARS,
    ):
        self.mode = mode
        self.state_tracking = state_tracking
        self.package_knowledge = package_knowledge
        self.knowledge_tools = knowledge_tools
        # prompting_rag: the official prompting input plus a fixed top-k of raw turns from a trace corpus.
        self.rag_corpus = rag_corpus
        self.rag_top_k = max(0, rag_top_k)
        self.rag_turn_chars = max(200, rag_turn_chars)
        if rag_style not in RAG_STYLES:
            raise ValueError(f"Unknown rag style {rag_style!r}; known: {RAG_STYLES}")
        self.rag_style: RagStyle = rag_style
        # Bound on the environment prompt seen by the state tracker, the agent and the renderer (None = whole prompt).
        self.environment_prompt_chars = environment_prompt_chars
        # envpack_prompting: the official prompting input plus a fixed, non-agentic view of the package (envpack.py).
        self.envpack_top_k = max(0, envpack_top_k)
        self.envpack_evidence_chars = max(200, envpack_evidence_chars)
        self.inspector: PackageInspector | None = None
        self.effect_rejection = effect_rejection
        self.llm = llm
        self.single_shot_llm = single_shot_llm if single_shot_llm is not None else llm
        self.agent_llm = agent_llm
        self.chat_llm = chat_llm
        if prediction_mode not in PREDICTION_MODES:
            raise ValueError(f"Unknown prediction mode {prediction_mode!r}; known: {PREDICTION_MODES}")
        self.prediction_mode: PredictionMode = prediction_mode
        self.trace_corpus = trace_corpus
        self.features = (set(HARNESS_FEATURES) if features == "all" else set(DEFAULT_FEATURES) if features == "default"
                         else set(features or ()))
        self.history_turn_chars = history_turn_chars
        self.history_total_chars = history_total_chars
        self.history_full = history_full  # harness v4: the whole observed transcript verbatim in every brief
        self.official_input = official_input  # harness v5.1: the benchmark's own model input placed once, verbatim, in the brief
        self.knowledge_gate = knowledge_gate  # harness v5.2: applicability-gated package knowledge (brief and tools)
        self.knowledge_gate_mode = knowledge_gate_mode
        self.knowledge_judge = knowledge_judge  # v5.3: the structured provider also judges package applicability
        self.knowledge_names = knowledge_names  # v5.3.2: "pages" makes page identity the supporting signal (web)
        self.history_window = max(0, history_window)
        self.state_window = state_window
        self.trust_policy: TrustPolicy = trust_policy
        self.fast_path: FastPathPolicy = fast_path
        self.max_tool_calls = max_tool_calls
        self.observation_chars = observation_chars
        self.session_root = Path(session_root) if session_root else None
        self.package: EnvironmentPackage | None = None
        self.package_dir = package_dir
        if mode == "agentic":
            if package_dir is None:
                raise ValueError("Agentic prediction requires an environment package")
            self.package = EnvironmentPackage(package_dir, verify_hashes=verify_package)
            if self.package.manifest.metadata.get("validation_status") == "candidate" and not allow_unvalidated:
                raise ValueError("Package is an unvalidated candidate; pass allow_unvalidated to evaluate it")
        elif chat_llm is None:
            raise ValueError("Prompting prediction requires a chat model")
        if mode == "envpack_prompting":
            if package_dir is None:
                raise ValueError("envpack_prompting requires an environment package")
            self.package = EnvironmentPackage(package_dir, verify_hashes=verify_package)
            if self.package.manifest.metadata.get("validation_status") == "candidate" and not allow_unvalidated:
                raise ValueError("Package is an unvalidated candidate; pass allow_unvalidated to evaluate it")
            self.inspector = PackageInspector(self.package)
        if mode == "prompting_rag" and rag_corpus is None:
            raise ValueError("prompting_rag requires a raw-trace corpus (rag_corpus / --trace-corpus)")

    # ─── Driver ───────────────────────────────────────────────────────────────

    def run(
        self, rows: list[dict[str, Any]], *, limit: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows = rows[:limit] if limit else list(rows)
        if self.mode in ("prompting", "prompting_rag", "envpack_prompting"):
            outputs = []
            for index, row in enumerate(rows):
                outputs.append(self._prompting(row))
                if progress:
                    progress(index + 1, len(rows))
            return outputs
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[(subtask_of(str(row.get("task", "mcp"))), str(row["id"]))].append(index)
        outputs: dict[int, dict[str, Any]] = {}
        done = 0
        with tempfile.TemporaryDirectory(prefix="trace2env-awb-sessions-") as scratch:
            for (task, trajectory_id), indices in grouped.items():
                cases = sorted(((case_from_row(rows[i]), i) for i in indices), key=lambda item: item[0].turn_idx)
                session_dir = (self.session_root or Path(scratch)) / f"{task}_{trajectory_id}"
                if session_dir.exists():
                    # An evaluation always starts a trajectory from scratch: resuming a persisted session
                    # (e.g. after a restarted run) would track the same turns a second time on top of the
                    # already tracked state and duplicate its episodic memory.
                    shutil.rmtree(session_dir)
                outputs.update(self._run_trajectory(rows, cases, session_dir))
                done += len(indices)
                if progress:
                    progress(done, len(rows))
        return [outputs[index] for index in range(len(rows))]

    def _prompting(self, row: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        info: dict[str, Any] = {"mode": self.mode, "route": self.mode}
        try:
            case = case_from_row(row)
            messages = inference_messages(case)
            if self.mode == "prompting_rag":
                assert self.rag_corpus is not None
                # Fixed lexical query and block wording per self.rag_style (RagStyle): "terminal" is the setting of every
                # reported AgentWorldBench run, "generic" the EnvScaler setting. See rag_case_query / rag_block.
                query = rag_case_query(case, self.rag_style)
                hits = self.rag_corpus.search(query, self.rag_top_k) if query else []
                block = rag_block(hits, self.rag_corpus, self.rag_turn_chars, self.rag_style) if hits else ""
                if block:
                    if messages and messages[0]["role"] == "system":
                        messages[0] = {"role": "system", "content": messages[0]["content"] + block}
                    else:
                        messages.insert(0, {"role": "system", "content": block.lstrip("\n-")})
                info["rag"] = {"query": query, "style": self.rag_style, "top_k": self.rag_top_k, "turn_chars": self.rag_turn_chars,
                               "hits": [hit["id"] for hit in hits], "block_chars": len(block)}
            elif self.mode == "envpack_prompting":
                assert self.inspector is not None
                block, manifest = envpack_view(self.inspector, case, top_k=self.envpack_top_k, evidence_chars=self.envpack_evidence_chars)
                if messages and messages[0]["role"] == "system":
                    messages[0] = {"role": "system", "content": messages[0]["content"] + block}
                else:
                    messages.insert(0, {"role": "system", "content": block.lstrip("\n-")})
                info["envpack"] = manifest
            gen = self.chat_llm.chat(messages=messages, role="agentworld_infer")
            info["usage"] = last_usage(self.chat_llm)
            info["prompt_chars"] = sum(len(str(message.get("content", ""))) for message in messages)
        except Exception as exc:  # noqa: BLE001 - a failed prediction is a scored failure, not a crash
            gen = ""
            info["error"] = f"{type(exc).__name__}: {exc}"
        info["latency_seconds"] = round(time.perf_counter() - started, 3)
        return {**row, "gen": gen, "trace2env": info}

    # ─── One trajectory: shared session, teacher-forced tracking ─────────────

    def _run_trajectory(
        self, rows: list[dict[str, Any]], cases: list[tuple[AgentWorldCase, int]], session_dir: Path
    ) -> dict[int, dict[str, Any]]:
        assert self.package is not None
        first_case = cases[0][0]
        session = SessionStore(session_dir, initial_state_from_package(self.package))
        memory = EpisodicMemory(session.database)
        tracker = StateTracker(self.package, self.llm, window=self.state_window, memory=memory,
                               observation_chars=self.observation_chars, environment_prompt=first_case.system_prompt,
                               environment_prompt_chars=self.environment_prompt_chars,
                               features=self.features, disabled=not self.state_tracking)
        state = session.load()
        tracked = 0
        steps: list[StateTrackingStep] = []
        initial = initial_observation(first_case)
        if initial and memory.count() == 0:
            # The first prompt shows the environment before any action; track it as turn 0 — without a
            # model call when it is just an idle prompt line (compact feature).
            initial_action = NormalizedAction(type=INITIAL_STATE_ACTION)
            deterministic = tracker.track_initial_prompt(state, initial, initial_action)
            if deterministic is not None:
                state, initial_step = deterministic
                initial_steps = [initial_step]
            else:
                state, initial_steps = tracker.advance(state, [ObservedTurn(turn=0, action=initial_action, observation=initial)])
            steps += initial_steps
            state = self._commit_tracked(session, state, initial_steps)
        outputs: dict[int, dict[str, Any]] = {}
        for case, index in cases:
            turns = observed_turns(case)
            new_turns = [turn for turn in turns if turn.turn > tracked]
            if new_turns:
                state, new_steps = tracker.advance(state, new_turns)
                steps += new_steps
                tracked = case.turn_idx - 1
                state = self._commit_tracked(session, state, new_steps)
            outputs[index] = self.predict(rows[index], case, session, state, tracking_summary(steps))
        return outputs

    @staticmethod
    def _commit_tracked(session: SessionStore, state: EnvironmentState, steps: list[StateTrackingStep]) -> EnvironmentState:
        """Persist tracked state so the prediction fork and later records start from it."""
        current = session.load()
        audit = AuditRecord(
            prior_revision=current.revision,
            request=StepRequest(action=NormalizedAction(type=TRACKING_ACTION),
                                metadata={"tracking": [step.model_dump(mode="json") for step in steps]}),
            route="tracking",
        )
        return session.commit(current.revision, state, audit)

    # ─── One prediction ───────────────────────────────────────────────────────

    def request_metadata(self, case: AgentWorldCase) -> dict[str, Any]:
        sections = prompt_sections(case.current_prompt)
        metadata: dict[str, Any] = {
            "benchmark": "agentworldbench",
            "task": subtask_of(case.task),
            "turn": case.turn_idx,
            "environment_prompt": case.system_prompt,
            "current_input": {key: clip_text(str(value), self.observation_chars * 3)
                              for key, value in sections.items() if key not in {"action", "turn"}},
            "raw_action": sections.get("action", ""),
            "history_window": self.history_window,
        }
        if self.official_input:
            # Source of truth: the same input-construction code the prompting baseline uses (no target observation).
            metadata["official_input"] = inference_messages(case)
        return metadata

    def predict(
        self, row: dict[str, Any], case: AgentWorldCase, session: SessionStore, state: EnvironmentState,
        tracking: dict[str, int],
    ) -> dict[str, Any]:
        assert self.package is not None
        started = time.perf_counter()
        action = case.action or NormalizedAction(type=f"{subtask_of(case.task)}.action")
        metadata = self.request_metadata(case)
        info: dict[str, Any] = {"mode": "agentic", "state_tracking": tracking, "state_step": state.step,
                                "state_tracking_enabled": self.state_tracking, "package_knowledge": self.package_knowledge,
                                "knowledge_tools": self.knowledge_tools,
                                "features": sorted(self.features), "prediction_mode": self.prediction_mode, "history_full": self.history_full, "official_input": self.official_input, "knowledge_gate_enabled": self.knowledge_gate, "knowledge_gate_mode": self.knowledge_gate_mode, "knowledge_judge": self.knowledge_judge, "knowledge_names": self.knowledge_names,
                                "trace_corpus": self.trace_corpus.describe() if self.trace_corpus is not None else None,
                                "package_ablation": self.package.manifest.metadata.get("ablation")}
        observation = ""
        predicted = False
        with tempfile.TemporaryDirectory(prefix="trace2env-awb-") as directory:
            # Fork the trajectory session (state + episodic memory) so the prediction never leaks back.
            shutil.copy2(session.database, Path(directory) / "session.sqlite")
            try:
                single_shot = self.prediction_mode == "single_shot"
                harness = RuntimeHarness(
                    self.package_dir, directory, package=self.package,
                    agent_llm=None if single_shot else self.agent_llm, renderer_llm=self.llm,
                    allow_unvalidated=True, trust_policy=self.trust_policy, fast_path=self.fast_path,
                    max_tool_calls=self.max_tool_calls, features=self.features,
                    history_turn_chars=self.history_turn_chars, history_total_chars=self.history_total_chars, history_full=self.history_full,
                    official_input=self.official_input, knowledge_gate=self.knowledge_gate, knowledge_gate_mode=self.knowledge_gate_mode,
                    knowledge_judge_llm=self.llm if self.knowledge_judge else None,
                    knowledge_names=self.knowledge_names,
                    prediction_mode=self.prediction_mode, single_shot_llm=self.single_shot_llm if single_shot else None,
                    trace_corpus=self.trace_corpus, state_tracking=self.state_tracking, package_knowledge=self.package_knowledge,
                    knowledge_tools=self.knowledge_tools, effect_rejection=self.effect_rejection,  # type: ignore[arg-type]
                    environment_prompt_chars=self.environment_prompt_chars,
                )
                result = harness.step(StepRequest(action=action, metadata=metadata))
                observation = result.observation
                # An empty observation is a legitimate prediction (a short wait that revealed nothing);
                # it is wrapped like any other so the official judge scores it instead of failing the row.
                predicted = True
                audit = harness.session.audit_records()[-1]
                info.update({
                    "route": result.route,
                    "action_type": result.plan.action.type,
                    "rule_ids": result.plan.rule_ids,
                    "effects": len(result.plan.effects),
                    "outcome": result.outcome.value,
                    "verification": [issue.code for issue in result.verification.issues],
                    "citations": result.citations,
                    "tool_calls": [call.get("tool") for call in audit.tool_calls
                                   if call.get("tool") not in {"_usage", "_knowledge_gate", "submit_transition"}],
                    "usage": next((call["arguments"] for call in audit.tool_calls if call.get("tool") == "_usage"), {}),
                    "knowledge_gate": next((call["arguments"] for call in audit.tool_calls if call.get("tool") == "_knowledge_gate"), None),
                    "retrieved_artifacts": result.retrieved_artifacts,
                })
            except Exception as exc:  # noqa: BLE001 - keep the evaluation running; the row records the failure
                info.update({"action_type": action.type, "harness_error": f"{type(exc).__name__}: {exc}"[:600]})
                try:
                    # The failed audit still records what the agent did before verification rejected it.
                    audit = SessionStore(directory).audit_records()[-1]
                    info.update({
                        "failed_route": audit.route,
                        "tool_calls": [call.get("tool") for call in audit.tool_calls if call.get("tool") not in {"_usage", "_knowledge_gate", "submit_transition"}],
                        "citations": audit.citations,
                        "verification": [issue.code for issue in (audit.verification.issues if audit.verification else [])],
                    })
                except Exception:  # noqa: BLE001 - diagnostics only
                    pass
                observation = self._fallback(session, state, action, metadata, info)
        info["latency_seconds"] = round(time.perf_counter() - started, 3)
        keep_empty = predicted and "wait" in self.features
        return {**row, "gen": wrap_prediction(observation) if (observation or keep_empty) else "", "trace2env": info}

    def _fallback(
        self, session: SessionStore, state: EnvironmentState, action: NormalizedAction, metadata: dict[str, Any],
        info: dict[str, Any],
    ) -> str:
        """Degrade gracefully when the agent step failed: first a single-shot prediction over the same brief (history,
        state, evidence; route ``fallback_single_shot``), then a render from the tracked state (route ``fallback``).
        Nothing is committed to the trajectory session either way."""
        if self.llm is None:
            info["route"] = "error"
            return ""
        assert self.package is not None
        try:
            with tempfile.TemporaryDirectory(prefix="trace2env-awb-fallback-") as directory:
                shutil.copy2(session.database, Path(directory) / "session.sqlite")
                harness = RuntimeHarness(
                    self.package_dir, directory, package=self.package, agent_llm=None, renderer_llm=self.llm,
                    allow_unvalidated=True, trust_policy=self.trust_policy, fast_path=self.fast_path,
                    max_tool_calls=self.max_tool_calls, features=self.features,
                    history_turn_chars=self.history_turn_chars, history_total_chars=self.history_total_chars, history_full=self.history_full,
                    official_input=self.official_input, knowledge_gate=self.knowledge_gate, knowledge_gate_mode=self.knowledge_gate_mode,
                    knowledge_judge_llm=self.llm if self.knowledge_judge else None,
                    prediction_mode="single_shot", single_shot_llm=self.single_shot_llm or self.llm,
                    trace_corpus=self.trace_corpus, state_tracking=self.state_tracking, package_knowledge=self.package_knowledge,
                    knowledge_tools=self.knowledge_tools, effect_rejection="keep_observation",
                    environment_prompt_chars=self.environment_prompt_chars,
                )
                result = harness.step(StepRequest(action=action, metadata=metadata))
                audit = harness.session.audit_records()[-1]
                info.update({"route": "fallback_single_shot", "fallback_citations": result.citations,
                             "usage": next((call["arguments"] for call in audit.tool_calls if call.get("tool") == "_usage"), {})})
                return result.observation
        except Exception as exc:  # noqa: BLE001 - fall through to the state render
            info["fallback_single_shot_error"] = f"{type(exc).__name__}: {exc}"[:300]
        try:
            retrieval = PackageInspector(self.package).inspect_action(action.type, state=state)
            rendered = self.llm.complete(
                system=RUNTIME_RENDERER + environment_context(metadata),
                user=_payload({
                    "state_before": state.model_dump(mode="json"),
                    "state_after": state.model_dump(mode="json"),
                    "action": action.model_dump(mode="json"),
                    "note": "The harness could not verify a transition; predict the observation from the tracked "
                            "state and retrieved knowledge without assuming unverified state changes.",
                    "render_contracts": retrieval["renderers"],
                    "demonstrations": retrieval["demonstrations"],
                    "rules": retrieval["rules"],
                    "notes": retrieval.get("notes", []),
                    "context": request_context(metadata),
                }),
                response_model=RenderedObservation,
                role="runtime_render",
            )
            info["route"] = "fallback"
            return rendered.observation
        except Exception as exc:  # noqa: BLE001
            info["route"] = "error"
            info["fallback_error"] = f"{type(exc).__name__}: {exc}"
            return ""
