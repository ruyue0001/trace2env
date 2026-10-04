from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.examples import build_ledger_demo
from trace2env.engine import apply_mutations, process_due_events
from trace2env.llm import OpenAIResponsesLLM, ScriptedLLM
from trace2env.models import (
    ActionSchema,
    ActionSpec,
    Actor,
    AgentTurn,
    ArgumentSpec,
    Confidence,
    Episode,
    EnvironmentState,
    EventKind,
    LocalTransitionEvidence,
    NormalizedAction,
    NoteInductionResult,
    Outcome,
    ReconstructionConfig,
    RenderContract,
    RenderInductionResult,
    RawEvent,
    RuleInductionResult,
    SchemaInductionResult,
    StateField,
    StateMutation,
    StateSchema,
    StepRequest,
    TransitionRule,
    TransitionSubmission,
    EventAnnotation,
    TraceAnnotations,
    SourceRef,
    ReplayCase,
    ReplayStep,
)
from trace2env.package import EnvironmentPackage, PackageIntegrityError, PackageInspector
from trace2env.reconstruction import TraceReconstructor
from trace2env.runtime import RuntimeHarness
from trace2env.replay import evaluate_package, promote_package


class AdapterTests(unittest.TestCase):
    def test_plain_trace_is_segmented_without_losing_multiline_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(
                "User: show balance\nAction: balance\nObservation: Balance: 100\nready>\n",
                encoding="utf-8",
            )
            episodes = load_raw_traces(path)
            transitions = segment_transitions(episodes[0])
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(transitions), 1)
            observation = episodes[0].events[-1]
            self.assertIn("ready>", observation.content)
            self.assertEqual(transitions[0].observation_event_ids, [observation.id])

    def test_jsonl_event_log_groups_by_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            rows = [
                {"episode_id": "a", "actor": "agent", "kind": "action", "content": {"type": "ping"}},
                {"episode_id": "a", "actor": "tool", "content": "pong"},
                {"episode_id": "b", "actor": "agent", "kind": "action", "content": {"type": "ping"}},
                {"episode_id": "b", "actor": "tool", "content": "pong"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            episodes = load_raw_traces(path)
            self.assertEqual({episode.metadata["original_episode_id"] for episode in episodes}, {"a", "b"})
            self.assertEqual(len({episode.id for episode in episodes}), 2)
            self.assertEqual(sum(len(segment_transitions(episode)) for episode in episodes), 2)


class RuntimeTests(unittest.TestCase):
    def test_deterministic_runtime_persists_contrastive_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = build_ledger_demo(root / "package")
            harness = RuntimeHarness(package, root / "session")

            first = harness.step(
                StepRequest(action=NormalizedAction(type="withdraw", arguments={"amount": 25}))
            )
            self.assertEqual(first.outcome, Outcome.SUCCESS)
            self.assertEqual(first.state.world["balance"], 75.0)
            self.assertEqual(first.observation, "Withdrew 25. Balance: 75.0")

            second = harness.step(
                StepRequest(action=NormalizedAction(type="withdraw", arguments={"amount": 1000}))
            )
            self.assertEqual(second.outcome, Outcome.FAILURE)
            self.assertEqual(second.state.world["balance"], 75.0)
            self.assertEqual(second.state.revision, 2)
            self.assertEqual(len(harness.session.audit_records()), 2)

    def test_failed_agentic_plan_rolls_back_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = build_ledger_demo(root / "package")
            llm = ScriptedLLM(
                {
                    "runtime_agent_turn": [
                        AgentTurn(
                            final=TransitionSubmission(
                                effects=[StateMutation(op="decrement", path="world.balance", value=1000)],
                                outcome=Outcome.SUCCESS,
                                uncertainty=["No reconstructed rule covers this action."],
                            )
                        )
                    ]
                }
            )
            harness = RuntimeHarness(package, root / "session", planner_llm=llm)
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                harness.step(StepRequest(action=NormalizedAction(type="unknown")))
            state = harness.session.load()
            self.assertEqual(state.world["balance"], 100.0)
            self.assertEqual(state.revision, 0)
            audit = harness.session.audit_records()[0]
            self.assertFalse(audit.committed)
            self.assertIn("verification failed", audit.error or "")
            self.assertEqual(harness.memory.count(), 0)

    def test_agent_can_inspect_the_workspace_before_submitting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = build_ledger_demo(root / "package")
            llm = ScriptedLLM(
                {
                    "runtime_agent_turn": [
                        AgentTurn(tool="list_actions", reason="Check the recovered interface."),
                        AgentTurn(final=TransitionSubmission(
                            outcome=Outcome.FAILURE, observation="unknown: command not found",
                            uncertainty=["The action is outside the reconstructed interface."])),
                    ]
                }
            )
            harness = RuntimeHarness(package, root / "session", planner_llm=llm)
            result = harness.step(StepRequest(action=NormalizedAction(type="unknown")))
            self.assertEqual((result.outcome, result.observation), (Outcome.FAILURE, "unknown: command not found"))
            self.assertEqual((result.route, result.tool_calls), ("agent", 1))
            self.assertIn("schema:actions", result.retrieved_artifacts)
            self.assertEqual([call["role"] for call in llm.calls], ["runtime_agent_turn"] * 2)
            self.assertEqual(harness.memory.recent(1)[0].observation, "unknown: command not found")

    def test_package_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = build_ledger_demo(root / "package")
            (package / "overview.md").write_text("changed", encoding="utf-8")
            with self.assertRaises(PackageIntegrityError):
                EnvironmentPackage(package)

    def test_inspector_returns_only_relevant_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            package = build_ledger_demo(Path(directory) / "package")
            result = PackageInspector(EnvironmentPackage(package)).inspect_action("deposit")
            self.assertEqual([rule["id"] for rule in result["rules"]], ["deposit_success"])

    def test_delayed_effect_is_applied_only_when_due(self):
        state = EnvironmentState(world={"status": "pending"})
        scheduled = apply_mutations(
            state,
            [
                StateMutation(
                    op="schedule",
                    path="world.status",
                    value="settled",
                    delay_steps=2,
                )
            ],
        )
        early, early_ids = process_due_events(scheduled, 1)
        due, due_ids = process_due_events(early, 2)
        self.assertEqual(early.world["status"], "pending")
        self.assertFalse(early_ids)
        self.assertEqual(due.world["status"], "settled")
        self.assertEqual(len(due_ids), 1)


class ReconstructionTests(unittest.TestCase):
    def test_ambiguous_plain_log_uses_llm_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "opaque.log"
            trace.write_text("clicked submit and the dashboard appeared", encoding="utf-8")
            source_event = load_raw_traces(trace)[0].events[0]
            recovered = TraceAnnotations(annotations=[
                EventAnnotation(source_event_id=source_event.id, actor=Actor.AGENT,
                                kind=EventKind.ACTION, start=0, end=14),
                EventAnnotation(source_event_id=source_event.id, actor=Actor.ENVIRONMENT,
                                kind=EventKind.OBSERVATION, start=23, end=41),
            ])
            llm = ScriptedLLM({"normalize_trace": [recovered]})
            reconstructor = TraceReconstructor(
                llm,
                ReconstructionConfig(environment_id="opaque", name="Opaque"),
                root / "work",
            )
            episodes, transitions = reconstructor.ingest([trace])
            self.assertEqual(len(transitions), 1)
            self.assertEqual(episodes[0].events[0].kind, EventKind.ACTION)
            self.assertEqual("".join(e.content for e in episodes[0].events), source_event.content)
            self.assertEqual(llm.calls[0]["role"], "normalize_trace")

    def test_scripted_reconstruction_compiles_valid_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            trace.write_text("Action: deposit 5\nObservation: Deposited 5. Balance: 105\n", encoding="utf-8")
            evidence = LocalTransitionEvidence(
                id="e1",
                episode_id="placeholder",
                transition_id="placeholder",
                action=NormalizedAction(type="deposit", arguments={"amount": 5}),
                outcome=Outcome.SUCCESS,
                mutations=[StateMutation(op="increment", path="world.balance", value=5)],
                observation_text="Deposited 5. Balance: 105",
                confidence=Confidence(value=0.95, rationale="Directly observed"),
            )
            schema = SchemaInductionResult(
                action_schema=ActionSchema(
                    actions=[
                        ActionSpec(
                            name="deposit",
                            arguments={"amount": ArgumentSpec(type="number", required=True)},
                        )
                    ]
                ),
                state_schema=StateSchema(
                    fields=[StateField(path="world.balance", type="number", default=100)]
                ),
            )
            rule = TransitionRule(
                id="deposit",
                action_type="deposit",
                description="Deposit adds funds",
                effects=[StateMutation(op="increment", path="world.balance", value={"$action_arg": "amount"})],
                observation_template="Deposited {action.arguments.amount}. Balance: {state_after.world.balance}",
                renderer="ledger",
                provenance=[SourceRef(source_id=load_raw_traces(trace)[0].source_id,
                    episode_id=load_raw_traces(trace)[0].id,
                    event_ids=[event.id for event in load_raw_traces(trace)[0].events])],
            )
            rules = RuleInductionResult(rules=[rule])
            renderers = RenderInductionResult(
                renderers=[RenderContract(id="ledger", action_types=["deposit"])]
            )
            llm = ScriptedLLM(
                {
                    "extract_transition": [evidence],
                    "induce_schema": [schema],
                    "induce_rules": [rules],
                    "falsify_rules": [rules],
                    "induce_renderers": [renderers],
                    "induce_notes": [NoteInductionResult()],
                }
            )
            config = ReconstructionConfig(environment_id="test", name="Test")
            package = TraceReconstructor(llm, config, root / "work").run([trace], root / "package")
            loaded = EnvironmentPackage(package)
            self.assertEqual(loaded.manifest.transition_count, 1)
            self.assertEqual(loaded.rules[0].id, "deposit")
            self.assertEqual(loaded.manifest.metadata["validation_status"], "candidate")
            with self.assertRaisesRegex(ValueError, "unvalidated candidate"):
                RuntimeHarness(package, root / "blocked-session")
            self.assertFalse((root / "blocked-session").exists())
            case = ReplayCase(id="heldout", trajectory_group="heldout-family", source_ids=["src_" + "f" * 64],
                split="validation", initial_state=EnvironmentState(world={"balance": 100}),
                steps=[ReplayStep(action=NormalizedAction(type="deposit", arguments={"amount": 5}),
                                  expected_observation="Deposited 5. Balance: 105", expected_state={"world.balance": 105})])
            leaked = case.model_copy(update={"source_ids": [load_raw_traces(trace)[0].source_id]})
            self.assertFalse(evaluate_package(package, [leaked]).promotion_eligible)
            promoted, report = promote_package(package, [case], root / "promoted")
            self.assertTrue(report.promotion_eligible)
            with self.assertRaisesRegex(ValueError, "explicit initial state"):
                RuntimeHarness(promoted, root / "missing-initial-state")
            self.assertEqual(RuntimeHarness(promoted, root / "approved-session", initial_state=case.initial_state).step(
                StepRequest(action=case.steps[0].action)).state.world["balance"], 105)
            self.assertEqual(
                [call["role"] for call in llm.calls],
                ["extract_transition", "induce_schema", "induce_rules", "falsify_rules", "induce_renderers", "induce_notes"],
            )


class OpenAIAdapterTests(unittest.TestCase):
    def test_responses_parse_boundary_uses_structured_model(self):
        class FakeResponse:
            output_parsed = Confidence(value=0.8, rationale="structured")

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def parse(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        llm = OpenAIResponsesLLM(model="test-model", client=client)
        result = llm.complete(
            system="system", user="user", response_model=Confidence, role="test_role"
        )
        self.assertEqual(result.value, 0.8)
        self.assertIs(client.responses.kwargs["text_format"], Confidence)
        self.assertEqual(client.responses.kwargs["instructions"], "system")

    def test_chat_transports_send_only_configured_sampling_parameters(self):
        from trace2env.llm import OpenAIChatLLM, OpenAIChatStructuredLLM, OpenAIToolLLM

        class FakeMessage:
            content = '{"value": 0.5, "rationale": "r"}'
            tool_calls = None

        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"choices": [type("Choice", (), {"message": FakeMessage()})()], "usage": None})()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        client = FakeClient()
        OpenAIChatStructuredLLM("m", client=client).complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertNotIn("temperature", client.chat.completions.kwargs)
        self.assertEqual(client.chat.completions.kwargs["max_tokens"], 32768)
        client = FakeClient()
        OpenAIChatLLM("m", client=client, temperature=0.6, reasoning_effort="low").chat(messages=[], role="r")
        self.assertEqual((client.chat.completions.kwargs["temperature"], client.chat.completions.kwargs["reasoning_effort"]), (0.6, "low"))
        client = FakeClient()
        OpenAIToolLLM("m", client=client, max_tokens=None).respond(messages=[], tools=[], role="r")
        self.assertEqual(set(client.chat.completions.kwargs) & {"temperature", "max_tokens"}, set())
        # OpenRouter provider routing rides in extra_body next to the reasoning object and is part of the cache identity
        from trace2env.llm import _provider_identity
        pin = {"order": ["DeepSeek"], "allow_fallbacks": False}
        client = FakeClient()
        pinned = OpenAIChatStructuredLLM("m", client=client, reasoning={"effort": "low"}, provider_routing=pin)
        pinned.complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual(client.chat.completions.kwargs["extra_body"], {"reasoning": {"effort": "low"}, "provider": pin})
        self.assertEqual(_provider_identity(pinned)["provider_routing"], pin)
        self.assertNotIn("provider_routing", _provider_identity(OpenAIChatStructuredLLM("m", client=FakeClient())))

    def test_forced_tool_choice_rejected_by_the_provider_is_retried_with_auto(self):
        import httpx
        from openai import BadRequestError
        from trace2env.llm import OpenAIToolLLM
        from trace2env.models import SUBMIT_TOOL_NAME

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs.get("tool_choice"))
                if isinstance(kwargs.get("tool_choice"), dict):
                    raise BadRequestError("Thinking mode does not support this tool_choice",
                                          response=httpx.Response(400, request=httpx.Request("POST", "https://api.example")), body=None)
                call = type("Call", (), {"id": "c1", "function": type("F", (), {"name": SUBMIT_TOOL_NAME, "arguments": '{"outcome": "success", "observation": "ok"}'})()})()
                message = type("Message", (), {"content": None, "tool_calls": [call]})()
                return type("Response", (), {"choices": [type("Choice", (), {"message": message})()], "usage": None})()

        class FakeClient:
            def __init__(self):
                self.chat = type("Chat", (), {"completions": FakeCompletions()})()

        client = FakeClient()
        reply = OpenAIToolLLM("m", client=client, max_tokens=None).respond(messages=[], tools=[{"type": "function", "function": {"name": SUBMIT_TOOL_NAME}}], role="r", force_tool=SUBMIT_TOOL_NAME)
        self.assertEqual(client.chat.completions.calls, [{"type": "function", "function": {"name": SUBMIT_TOOL_NAME}}, "auto"])
        self.assertEqual([c.name for c in reply.tool_calls], [SUBMIT_TOOL_NAME])
        self.assertEqual(reply.usage.get("forced_tool_unsupported"), 1)

        class OtherCompletions(FakeCompletions):
            def create(self, **kwargs):
                raise BadRequestError("something else", response=httpx.Response(400, request=httpx.Request("POST", "https://api.example")), body=None)

        client = FakeClient()
        client.chat.completions = OtherCompletions()
        with self.assertRaises(BadRequestError):
            OpenAIToolLLM("m", client=client, max_tokens=None).respond(messages=[], tools=[], role="r", force_tool=SUBMIT_TOOL_NAME)

    def test_invalid_structured_output_is_repaired_with_its_validation_errors(self):
        from trace2env.llm import OpenAIChatStructuredLLM

        answers = ['{"value": 5, "rationale": "too big"}', '{"value": 0.5, "rationale": "fixed"}']

        class FakeCompletions:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                message = type("Message", (), {"content": answers[len(self.calls) - 1], "tool_calls": None})()
                return type("Response", (), {"choices": [type("Choice", (), {"message": message})()], "usage": None})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})()
        result = OpenAIChatStructuredLLM("m", client=client).complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual(result.value, 0.5)
        calls = client.chat.completions.calls
        self.assertEqual(len(calls), 2)
        self.assertEqual([m["role"] for m in calls[1]["messages"]], ["system", "user", "assistant", "user"])
        self.assertIn("value: Input should be less than or equal to 1", calls[1]["messages"][-1]["content"])
        answers[:] = ['{"value": 5, "rationale": "x"}'] * 3
        client = type("Client", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()})()
        with self.assertRaisesRegex(RuntimeError, "failed validation after repair attempts"):
            OpenAIChatStructuredLLM("m", client=client).complete(system="s", user="u", response_model=Confidence, role="r")

    def test_truncated_structured_output_retries_once_then_fails(self):
        from trace2env.llm import OpenAIChatStructuredLLM

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                message = type("Message", (), {"content": '{"value": 0.', "tool_calls": None})()
                choice = type("Choice", (), {"message": message, "finish_reason": "length"})()
                return type("Response", (), {"choices": [choice], "usage": None})()

        completions = FakeCompletions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
        with self.assertRaisesRegex(RuntimeError, "cut off at the output limit"):
            OpenAIChatStructuredLLM("m", client=client, max_tokens=64).complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual(completions.calls, 2)  # one retry with a doubled cap, then the failure is final
        completions.calls = 0
        with self.assertRaisesRegex(RuntimeError, "cut off at the output limit"):
            OpenAIChatStructuredLLM("m", client=client, max_tokens=64, truncation_retries=0).complete(
                system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual(completions.calls, 1)

    def test_freeform_environment_values_use_flexible_schema_then_local_validation(self):
        evidence = LocalTransitionEvidence(
            id="e",
            episode_id="ep",
            transition_id="tr",
            action=NormalizedAction(type="tool", arguments={"dynamic_name": 3}),
            observation_text="ok",
            confidence=Confidence(value=1),
        )

        class FakeResponse:
            output_text = evidence.model_dump_json()

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeResponse()

        class FakeClient:
            def __init__(self):
                self.responses = FakeResponses()

        client = FakeClient()
        result = OpenAIResponsesLLM(client=client).complete(
            system="system",
            user="user",
            response_model=LocalTransitionEvidence,
            role="extract_transition",
        )
        self.assertEqual(result.action.arguments["dynamic_name"], 3)
        self.assertFalse(client.responses.kwargs["text"]["format"]["strict"])
        self.assertFalse(client.responses.kwargs["store"])


if __name__ == "__main__":
    unittest.main()


class ReasoningObjectTests(unittest.TestCase):
    """The OpenRouter-style `reasoning` object replaces `reasoning_effort` in chat requests when given."""

    def test_sampling_sends_reasoning_object_instead_of_effort(self):
        from trace2env.llm import _sampling

        self.assertEqual(_sampling(None, 100, "medium"), {"max_tokens": 100, "reasoning_effort": "medium"})
        params = _sampling(None, 100, "medium", {"enabled": False})
        self.assertEqual(params, {"max_tokens": 100, "extra_body": {"reasoning": {"enabled": False}}})
        self.assertNotIn("reasoning_effort", params)

    def test_cli_parses_reasoning_json_and_rejects_non_objects(self):
        from trace2env.cli import _chat_reasoning_json, build_parser

        args = build_parser().parse_args(["awb-run", "rows.jsonl", "--mode", "prompting", "--output", "out.jsonl", "--chat-reasoning-json", '{"enabled": false}'])
        self.assertEqual(_chat_reasoning_json(args), {"enabled": False})
        args = build_parser().parse_args(["awb-run", "rows.jsonl", "--mode", "prompting", "--output", "out.jsonl"])
        self.assertIsNone(_chat_reasoning_json(args))
        args = build_parser().parse_args(["awb-run", "rows.jsonl", "--mode", "prompting", "--output", "out.jsonl", "--chat-reasoning-json", "[1]"])
        with self.assertRaises(SystemExit):
            _chat_reasoning_json(args)


class ToolTransportRobustnessTests(unittest.TestCase):
    """A forced final offers only the forced tool, and the tools transport can be traced to the call log."""

    def test_forced_tool_offers_only_that_tool(self):
        from trace2env.llm import OpenAIToolLLM

        seen = {}

        class _Message:
            content = None
            tool_calls = []

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]
            usage = None

        class _Completions:
            def create(self, **kwargs):
                seen.update(kwargs)
                return _Response()

        class _Client:
            chat = type("chat", (), {"completions": _Completions()})()

        llm = OpenAIToolLLM(model="m", client=_Client())
        tools = [{"type": "function", "function": {"name": "read_state", "parameters": {}}},
                 {"type": "function", "function": {"name": "submit_transition", "parameters": {}}}]
        llm.respond(messages=[{"role": "user", "content": "x"}], tools=tools, role="runtime_agent_turn", force_tool="submit_transition")
        self.assertEqual([t["function"]["name"] for t in seen["tools"]], ["submit_transition"])
        self.assertEqual(seen["tool_choice"], {"type": "function", "function": {"name": "submit_transition"}})
        llm.respond(messages=[{"role": "user", "content": "x"}], tools=tools, role="runtime_agent_turn")
        self.assertEqual(len(seen["tools"]), 2)
        self.assertEqual(seen["tool_choice"], "auto")

    def test_tracing_agent_llm_logs_each_reply(self):
        import tempfile
        from pathlib import Path

        from trace2env.llm import ScriptedToolLLM, TracingAgentLLM
        from trace2env.models import AgentReply, ToolCall

        inner = ScriptedToolLLM([AgentReply(tool_calls=[ToolCall(id="1", name="submit_transition", arguments={})],
                                            usage={"prompt_tokens": 3, "completion_tokens": 2})])
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "calls.jsonl"
            traced = TracingAgentLLM(inner, log)
            reply = traced.respond(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], tools=[], role="runtime_agent_turn")
            self.assertEqual(reply.tool_calls[0].name, "submit_transition")
            record = json.loads(log.read_text().splitlines()[0])
            self.assertEqual(record["role"], "runtime_agent_turn")
            self.assertEqual(record["usage"], {"prompt_tokens": 3, "completion_tokens": 2})
            self.assertEqual(record["response"]["tool_calls"][0]["name"], "submit_transition")
            self.assertEqual(record["sequence"], 1)


class CacheIdentityTests(unittest.TestCase):
    def test_reasoning_object_is_part_of_the_cache_identity(self):
        from trace2env.llm import OpenAIChatLLM, _provider_identity

        plain = OpenAIChatLLM(model="m", client=object())
        off = OpenAIChatLLM(model="m", client=object(), reasoning={"enabled": False})
        low = OpenAIChatLLM(model="m", client=object(), reasoning={"effort": "low"})
        self.assertNotIn("reasoning", _provider_identity(plain))
        self.assertNotEqual(_provider_identity(off), _provider_identity(low))
        self.assertNotEqual(_provider_identity(plain), _provider_identity(off))


class SchemaModeTests(unittest.TestCase):
    def test_json_object_mode_quotes_the_schema_and_sends_json_object(self):
        from trace2env.llm import OpenAIChatStructuredLLM, _provider_identity
        from trace2env.models import AgentTurn

        seen = []

        def make(content):
            msg = type("M", (), {"content": content})()
            choice = type("C", (), {"message": msg, "finish_reason": "stop"})()
            return type("R", (), {"choices": [choice], "usage": None})()

        class _Completions:
            def create(self, **kwargs):
                seen.append(kwargs)
                return make('{"tool": "read_state", "arguments": {"path": "world.cwd"}}')

        client = type("Client", (), {"chat": type("chat", (), {"completions": _Completions()})()})()
        llm = OpenAIChatStructuredLLM(model="m", client=client, schema_mode="json_object")
        turn = llm.complete(system="s", user="u", response_model=AgentTurn, role="runtime_agent_turn")
        self.assertEqual((turn.tool, turn.arguments), ("read_state", {"path": "world.cwd"}))
        self.assertEqual(seen[0]["response_format"], {"type": "json_object"})
        self.assertIn('"properties"', seen[0]["messages"][0]["content"])  # the schema travels in the system text
        self.assertEqual(_provider_identity(llm)["schema_mode"], "json_object")
        default = OpenAIChatStructuredLLM(model="m", client=client)
        default.complete(system="s", user="u", response_model=AgentTurn, role="runtime_agent_turn")
        self.assertEqual(seen[1]["response_format"]["type"], "json_schema")
        self.assertNotIn("schema_mode", _provider_identity(default))  # the default keeps earlier cache keys valid
        with self.assertRaises(ValueError):
            OpenAIChatStructuredLLM(model="m", client=client, schema_mode="yaml")


class TruncationRetryTests(unittest.TestCase):
    def test_structured_chat_retries_once_with_a_doubled_cap(self):
        from trace2env.llm import OpenAIChatStructuredLLM, OutputTruncated
        from trace2env.models import RenderedObservation

        seen = []

        def make(finish, content):
            msg = type("M", (), {"content": content})()
            choice = type("C", (), {"message": msg, "finish_reason": finish})()
            return type("R", (), {"choices": [choice], "usage": None})()

        class _Completions:
            def __init__(self, replies):
                self.replies = list(replies)

            def create(self, **kwargs):
                seen.append(kwargs.get("max_tokens"))
                return self.replies.pop(0)

        def client(replies):
            return type("Client", (), {"chat": type("chat", (), {"completions": _Completions(replies)})()})()

        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", ""), make("stop", '{"observation": "ok"}')]), max_tokens=100)
        result = llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_render")
        self.assertEqual((result.observation, seen), ("ok", [100, 200]))
        self.assertIn("retrying with max_tokens=200", llm.last_attempts[0]["error"])
        seen.clear()
        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", ""), make("length", "")]), max_tokens=100)
        with self.assertRaises(OutputTruncated):
            llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_render")
        self.assertEqual(seen, [100, 200])

    def test_degenerate_repetition_at_the_cap_is_resampled_once_at_the_same_cap(self):
        from trace2env.llm import TRUNCATION_CEILING, OpenAIChatStructuredLLM, OutputTruncated, looks_degenerate
        from trace2env.models import RenderedObservation

        seen = []

        def make(finish, content):
            msg = type("M", (), {"content": content})()
            choice = type("C", (), {"message": msg, "finish_reason": finish})()
            return type("R", (), {"choices": [choice], "usage": None})()

        class _Completions:
            def __init__(self, replies):
                self.replies = list(replies)

            def create(self, **kwargs):
                seen.append(kwargs.get("max_tokens"))
                return self.replies.pop(0)

        def client(replies):
            return type("Client", (), {"chat": type("chat", (), {"completions": _Completions(replies)})()})()

        loop = '{"tool": "inspect_action' + ".json?action_type=shell" * 400  # the live pattern: an unclosed string repeated
        self.assertTrue(looks_degenerate(loop))
        self.assertFalse(looks_degenerate('{"observation": "' + "".join(chr(97 + i % 26) * (i % 7 + 1) for i in range(2000)) + '"}'))
        self.assertFalse(looks_degenerate("ab" * 100))  # too short to judge
        # At the ceiling a doubled cap is impossible; the resample at the same cap recovers the answer.
        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", loop), make("stop", '{"observation": "ok"}')]), max_tokens=TRUNCATION_CEILING)
        result = llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_agent_turn")
        self.assertEqual((result.observation, seen), ("ok", [TRUNCATION_CEILING, TRUNCATION_CEILING]))
        self.assertIn("degenerate repetition", llm.last_attempts[0]["error"])
        # A second loop is not resampled again: the call fails as a truncation.
        seen.clear()
        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", loop), make("length", loop)]), max_tokens=TRUNCATION_CEILING)
        with self.assertRaises(OutputTruncated):
            llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_agent_turn")
        self.assertEqual(seen, [TRUNCATION_CEILING, TRUNCATION_CEILING])
        # Below the ceiling a loop is resampled at the same cap and a genuine overflow still doubles the cap.
        seen.clear()
        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", loop), make("length", ""), make("stop", '{"observation": "ok"}')]), max_tokens=100)
        llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_agent_turn")
        self.assertEqual(seen, [100, 100, 200])
        # loop_retries=0 restores the previous behaviour.
        seen.clear()
        llm = OpenAIChatStructuredLLM(model="m", client=client([make("length", loop), make("length", loop)]), max_tokens=TRUNCATION_CEILING, loop_retries=0)
        with self.assertRaises(OutputTruncated):
            llm.complete(system="s", user="u", response_model=RenderedObservation, role="runtime_agent_turn")
        self.assertEqual(seen, [TRUNCATION_CEILING])
