"""Call tracing, usage capture, and the extraction safeguards used by the stage-by-stage pilot driver."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trace2env.adapters import load_raw_traces
from trace2env.llm import CachedLLM, OpenAIChatStructuredLLM, ScriptedLLM, TracingLLM, usage_dict
from trace2env.models import (Confidence, Fact, LocalTransitionEvidence, NormalizedAction, Outcome,
                              ReconstructionConfig, SourceRef)
from trace2env.reconstruction import TraceReconstructor


def _evidence(refs: list[SourceRef]) -> LocalTransitionEvidence:
    return LocalTransitionEvidence(
        id="x", episode_id="x", transition_id="x", action=NormalizedAction(type="a"), outcome=Outcome.SUCCESS,
        observation_text="", confidence=Confidence(value=0.9), provenance=list(refs),
        observation_facts=[Fact(subject="world.x", predicate="eq", value=1, provenance=list(refs))])


class TracingTests(unittest.TestCase):
    def test_stage_usage_summary_sums_only_numeric_usage(self):
        from scripts.pilot_stages import usage_summary

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "calls.jsonl"
            log.write_text(json.dumps({
                "cache_hit": False,
                "error": None,
                "elapsed_s": 1.5,
                "usage": {"prompt_tokens": 12, "cost_usd": 0.25, "provider": "OpenAI"},
            }) + "\n", encoding="utf-8")
            self.assertEqual(usage_summary(log), {
                "calls": 1,
                "cache_hits": 0,
                "errors": 0,
                "elapsed_s": 1.5,
                "prompt_tokens": 12,
                "cost_usd": 0.25,
            })

    def test_chat_structured_reports_usage_and_tracing_logs_every_call(self):
        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                message = SimpleNamespace(content='{"value": 0.5, "rationale": "r"}', tool_calls=None)
                usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150,
                                        prompt_tokens_details={"cached_tokens": 100}, cost=0.0012)
                return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        provider = OpenAIChatStructuredLLM("m", client=client)
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "calls.jsonl"
            llm = TracingLLM(CachedLLM(provider, Path(directory) / "cache"), log)
            first = llm.complete(system="s", user="u", response_model=Confidence, role="r")
            second = llm.complete(system="s", user="u", response_model=Confidence, role="r")
            self.assertEqual((first.value, second.value), (0.5, 0.5))
            self.assertEqual(client.chat.completions.calls, 1)
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([record["cache_hit"] for record in records], [False, True])
            self.assertEqual(records[0]["usage"], {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                                                   "cached_tokens": 100, "cost_usd": 0.0012, "attempts": 1})
            self.assertIsNone(records[1]["usage"])
            self.assertEqual(records[0]["response"], {"value": 0.5, "rationale": "r"})
            self.assertEqual((records[0]["role"], records[0]["system"], records[0]["user"]), ("r", "s", "u"))
            self.assertEqual([record["sequence"] for record in records], [1, 2])

    def test_transient_gateway_failures_are_retried(self):
        from trace2env.llm import TransientProviderError

        answers = [
            SimpleNamespace(choices=None, usage=None, error={"code": 502, "message": "upstream unavailable"}),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None), finish_reason="error")], usage=None),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"value": 0.25, "rationale": "ok"}', tool_calls=None),
                                                     finish_reason="stop")], usage=None),
        ]

        class FakeCompletions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return answers[self.calls - 1]

        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        llm = OpenAIChatStructuredLLM("m", client=client, retry_backoff_s=0)
        result = llm.complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual((result.value, client.chat.completions.calls), (0.25, 3))
        self.assertEqual(llm.last_usage, None)

        class AlwaysBroken:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[], usage=None, error="down")

        broken = OpenAIChatStructuredLLM("m", client=SimpleNamespace(chat=SimpleNamespace(completions=AlwaysBroken())), retry_backoff_s=0)
        with self.assertRaises(TransientProviderError):
            broken.complete(system="s", user="u", response_model=Confidence, role="r")

    def test_content_policy_refusals_are_not_retried_and_become_zero_confidence_evidence(self):
        from trace2env.llm import ProviderRefusal

        class Refusing:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return SimpleNamespace(choices=None, usage=None,
                                       error={"message": "This content was flagged for possible cybersecurity risk."})

        completions = Refusing()
        llm = OpenAIChatStructuredLLM("m", client=SimpleNamespace(chat=SimpleNamespace(completions=completions)), retry_backoff_s=0)
        with self.assertRaises(ProviderRefusal):
            llm.complete(system="s", user="u", response_model=Confidence, role="r")
        self.assertEqual(completions.calls, 1)

        class RefusingStructured:
            calls: list = []

            def complete(self, *, system, user, response_model, role):
                raise ProviderRefusal("Provider refused role extract_transition: flagged")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            trace.write_text("Action: a\nObservation: 1\n", encoding="utf-8")
            recon = TraceReconstructor(RefusingStructured(), ReconstructionConfig(environment_id="x", name="X"), root / "work")
            episodes, transitions = recon.ingest([trace])
            [item] = recon.extract_evidence(episodes, transitions)
            self.assertEqual((item.confidence.value, item.outcome.value, item.mutations), (0.0, "unknown", []))
            self.assertTrue(item.ambiguities[0].startswith("Evidence extraction refused"))
            self.assertEqual(item.observation_text, "1")
            self.assertEqual(item.provenance[0].event_ids, transitions[0].action_event_ids + transitions[0].observation_event_ids)

    def test_unrepairable_structured_output_becomes_zero_confidence_evidence(self):
        from trace2env.llm import StructuredOutputError
        from trace2env.reconstruction import INVALID_MARKER, invalid_count, refused_count

        class Malformed:
            def complete(self, *, system, user, response_model, role):
                raise StructuredOutputError("Structured output for role extract_transition failed validation after repair attempts: x")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.txt"
            trace.write_text("Action: a\nObservation: 1\n", encoding="utf-8")
            recon = TraceReconstructor(Malformed(), ReconstructionConfig(environment_id="x", name="X"), root / "work")
            episodes, transitions = recon.ingest([trace])
            [item] = recon.extract_evidence(episodes, transitions)
            self.assertEqual((item.confidence.value, item.outcome.value, item.mutations), (0.0, "unknown", []))
            self.assertTrue(item.ambiguities[0].startswith(INVALID_MARKER))
            self.assertEqual((invalid_count([item]), refused_count([item])), (1, 0))
            recon._note_refusals([item], SimpleNamespace(unresolved=[]), "induce_schema")
            self.assertEqual(recon.refused_episodes, set())  # not a content-policy suspect

    def test_usage_dict_accepts_dicts_and_missing_usage(self):
        self.assertIsNone(usage_dict(None))
        self.assertEqual(usage_dict({"input_tokens": 5, "output_tokens": 2, "unrelated": "x"}), {"input_tokens": 5, "output_tokens": 2})

    def test_upstream_provider_is_recorded_and_survives_retry_accumulation(self):
        from trace2env.llm import _add_usage, _response_provider
        self.assertEqual(usage_dict({"prompt_tokens": 3}, provider="DeepSeek"), {"prompt_tokens": 3, "provider": "DeepSeek"})
        self.assertEqual(usage_dict({"prompt_tokens": 3}, provider=""), {"prompt_tokens": 3})

        class Response:  # the OpenAI SDK keeps OpenRouter's top-level `provider` as an extra field
            model_extra = {"provider": "Novita"}

        self.assertEqual(_response_provider(Response()), "Novita")
        self.assertIsNone(_response_provider(object()))
        merged = _add_usage({"prompt_tokens": 3, "provider": "DeepSeek", "attempts": 1}, {"prompt_tokens": 4, "provider": "Novita"})
        self.assertEqual(merged, {"prompt_tokens": 7, "provider": "Novita", "attempts": 2})


class ExtractionSafeguardTests(unittest.TestCase):
    def _episode(self, root: Path):
        trace = root / "trace.txt"
        trace.write_text("Action: a\nObservation: 1\nAction: b\nObservation: 2\n", encoding="utf-8")
        return trace, load_raw_traces(trace)[0]

    def test_citations_outside_the_slice_are_dropped_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, episode = self._episode(root)
            bogus = SourceRef(source_id=episode.source_id, episode_id=episode.id, event_ids=["evt_missing"])
            cross_slice = SourceRef(source_id=episode.source_id, episode_id=episode.id, event_ids=[episode.events[1].id])
            llm = ScriptedLLM({"extract_transition": [_evidence([bogus]), _evidence([cross_slice])]})
            recon = TraceReconstructor(llm, ReconstructionConfig(environment_id="x", name="X", max_history_events=0),
                                       root / "work")
            episodes, transitions = recon.ingest([trace])
            results = recon.extract_evidence(episodes, transitions)
            self.assertEqual(len(results), 2)
            for item, transition in zip(results, transitions):
                self.assertEqual(item.provenance[0].event_ids, transition.action_event_ids + transition.observation_event_ids)
                self.assertEqual(item.observation_facts[0].provenance, [])
                self.assertTrue(any("outside its transition slice" in note for note in item.ambiguities), item.ambiguities)

    def test_parallel_extraction_keeps_transition_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace, _ = self._episode(root)
            llm = ScriptedLLM({"extract_transition": [_evidence([]), _evidence([])]})
            recon = TraceReconstructor(llm, ReconstructionConfig(environment_id="x", name="X"), root / "work", workers=2)
            episodes, transitions = recon.ingest([trace])
            results = recon.extract_evidence(episodes, transitions)
            self.assertEqual([item.transition_id for item in results], [transition.id for transition in transitions])
            self.assertEqual(len(llm.calls), 2)


class RuleSelectorRepairTests(unittest.TestCase):
    def test_unevaluable_scope_keys_and_placeholder_templates_are_repaired(self):
        from trace2env.models import TransitionRule
        from trace2env.reconstruction import repair_rule_selectors

        rule = TransitionRule(
            id="r", action_type="cat", description="d",
            scope={"environment_id": "tb2", "episode_id": "ep_1", "command": "cat x", "tenant_scope": "None",
                   "session.cwd": "/app", "environment_version": "1.0"},
            observation_template="{action.arguments.command}\n<terminal rendering of operand bytes>\n<shell prompt>")
        notes = repair_rule_selectors(rule)
        self.assertEqual(rule.scope, {"environment_id": "tb2", "session.cwd": "/app", "environment_version": "1.0"})
        self.assertIsNone(rule.observation_template)
        self.assertEqual(len(notes), 3)
        clean = TransitionRule(id="ok", action_type="cat", description="d", scope={"tenant": "a"},
                               observation_template="Balance: {state_after.world.balance}\n<stdin>")
        self.assertEqual(repair_rule_selectors(clean), [])
        self.assertEqual(clean.observation_template, "Balance: {state_after.world.balance}\n<stdin>")


    def test_renderer_placeholder_templates_are_removed_at_reconciliation(self):
        from trace2env.models import ActionSchema, ActionSpec, RenderContract, SchemaInductionResult, StateSchema
        from trace2env.reconstruction import reconcile_artifacts

        schema = SchemaInductionResult(action_schema=ActionSchema(actions=[ActionSpec(name="cat")]),
                                       state_schema=StateSchema(fields=[]))
        contract = RenderContract(id="terminal", action_types=["cat"], template="<file bytes>\n<shell prompt>",
                                  instructions="Echo the command, then the file, then the prompt.")
        _, _, renderers, _, unresolved, rejected = reconcile_artifacts(schema, [], [], [contract], [])
        self.assertIsNone(renderers[0].template)
        self.assertEqual(renderers[0].instructions, contract.instructions)
        self.assertTrue(any("placeholder prose" in note for note in unresolved))
        self.assertEqual(rejected, [])


class InductionRefusalTests(unittest.TestCase):
    """A provider refusal of one chunk excludes the offending records instead of failing the stage."""

    def _llm(self, poison: str):
        from trace2env.llm import ProviderRefusal

        class Fake:
            def __init__(self):
                self.calls = []

            def complete(self, *, system, user, response_model, role):
                self.calls.append((role, user))
                if poison in user:
                    raise ProviderRefusal("flagged")
                ids = [record["id"] for record in json.loads(user).get("records", [])]
                merged = [record for partial in json.loads(user).get("merged", []) for record in partial["rationale"].split(",")]
                return Confidence(value=0.5, rationale=",".join(ids or merged))

        return Fake()

    def test_suspect_records_are_dropped_first_then_chunks_are_bisected(self):
        from trace2env.induction import BoundedInduction

        records = [{"id": f"r{i}", "episode_id": "ep-bad" if i == 2 else "ep-ok", "text": "poison" if i == 2 else "fine"}
                   for i in range(4)]
        llm = self._llm("poison")
        bounded = BoundedInduction(llm, max_bytes=100000, batch_size=10, suspect=lambda r: r["episode_id"] == "ep-bad")
        result = bounded.induce(records, "sys", Confidence, "induce_x", {}, "records", "merged")
        self.assertEqual(result.rationale, "r0,r1,r3")
        self.assertEqual([record["id"] for record in bounded.refused], ["r2"])
        self.assertEqual([role for role, _ in llm.calls], ["induce_x", "induce_x_chunk"])

        llm = self._llm("poison")
        bounded = BoundedInduction(llm, max_bytes=100000, batch_size=10)  # no suspects known: bisect
        result = bounded.induce(records, "sys", Confidence, "induce_x", {}, "records", "merged")
        self.assertEqual(set(result.rationale.split(",")), {"r0", "r1", "r3"})
        self.assertEqual([record["id"] for record in bounded.refused], ["r2"])
        self.assertTrue(all(role.endswith(("_chunk", "_merge")) or role == "induce_x" for role, _ in llm.calls))

    def test_everything_refused_raises(self):
        from trace2env.induction import BoundedInduction
        from trace2env.llm import ProviderRefusal

        bounded = BoundedInduction(self._llm("r0"), max_bytes=100000, batch_size=10)
        with self.assertRaises(ProviderRefusal):
            bounded.induce([{"id": "r0"}], "sys", Confidence, "induce_x", {}, "records", "merged")


class TrackerOverflowTests(unittest.TestCase):
    """A tracking window whose structured answer overflows the output limit is bisected, then skipped per turn."""

    def test_overflowing_window_is_bisected_then_skipped(self):
        from trace2env.examples import build_ledger_demo
        from trace2env.models import EnvironmentState, ObservedTurn, StateTrackingResult, StateMutation
        from trace2env.package import EnvironmentPackage
        from trace2env.state_tracking import StateTracker

        class Overflowing:
            def __init__(self):
                self.calls = []

            def complete(self, *, system, user, response_model, role):
                turns = json.loads(user)["observed_turns"]
                self.calls.append(len(turns))
                if len(turns) > 1 or turns[0]["turn"] == 3:
                    raise RuntimeError("Structured output for role track_state was cut off at the output limit (max_tokens=1)")
                return StateTrackingResult(mutations=[StateMutation(op="set", path="world.balance", value=10 * turns[0]["turn"])])

        with tempfile.TemporaryDirectory() as directory:
            package = EnvironmentPackage(build_ledger_demo(Path(directory) / "pkg"))
            llm = Overflowing()
            tracker = StateTracker(package, llm, window=4)
            turns = [ObservedTurn(turn=i, action=NormalizedAction(type="noop", arguments={}), observation=f"obs {i}") for i in range(1, 5)]
            state, steps = tracker.advance(EnvironmentState(world={"balance": 0}), turns)
            self.assertEqual(state.world["balance"], 40)  # turns 1, 2, 4 tracked individually; turn 3 skipped
            self.assertEqual(llm.calls, [4, 2, 1, 1, 2, 1, 1])
            self.assertEqual([step.route for step in steps], ["model"])
            self.assertTrue(any("turn 3" in issue and "skipped" not in issue for issue in steps[0].issues) or
                            any("state left unchanged" in issue for issue in steps[0].issues))


class RunnerSessionResetTests(unittest.TestCase):
    """A persisted per-trajectory session from an earlier run is reset, never resumed, by ``awb-run``."""

    def test_second_run_does_not_track_the_same_turns_twice(self):
        from trace2env.agentworld import load_rows
        from trace2env.agentworld_runner import AgentWorldRunner
        from trace2env.examples import build_ledger_demo
        from trace2env.memory import EpisodicMemory
        from trace2env.models import EnvironmentState
        from trace2env.session import SessionStore

        rows = load_rows([Path("examples/agentworldbench/synthetic_terminal.jsonl")])
        with tempfile.TemporaryDirectory() as directory:
            package = build_ledger_demo(Path(directory) / "pkg")
            root = Path(directory) / "sessions"
            for _ in range(2):
                AgentWorldRunner(mode="agentic", package_dir=str(package), session_root=root).run(rows)
            counts = {}
            for session_dir in root.iterdir():
                memory = EpisodicMemory(SessionStore(session_dir, EnvironmentState()).database)
                counts[session_dir.name] = memory.count()
            expected = {}
            for row in rows:
                expected[row["id"]] = max(expected.get(row["id"], 0), int(row["turn_idx"]))
            self.assertTrue(counts)
            for name, count in counts.items():
                trajectory = next(tid for tid in expected if name.endswith(f"_{tid}"))
                # observed turns before the last evaluated turn, plus the initial state entry when present
                self.assertLessEqual(count, expected[trajectory] + 1, (name, count))


class SchemaCanonicalizationTests(unittest.TestCase):
    def test_evidence_is_rewritten_through_state_and_action_aliases(self):
        from trace2env.models import ActionSchema, ActionSpec, SchemaInductionResult, StateField, StateMutation, StateSchema
        from trace2env.reconstruction import canonical_evidence
        from trace2env.validation import canonical_state_path, state_path_aliases

        schema = SchemaInductionResult(
            action_schema=ActionSchema(actions=[ActionSpec(name="git", aliases=["git-cmd"])]),
            state_schema=StateSchema(fields=[
                StateField(path="world.repositories", type="object", aliases=["world.git", "world.git_repositories"]),
                StateField(path="world.files", type="object")]))
        aliases = state_path_aliases(schema.state_schema)
        self.assertEqual(canonical_state_path("world.git", aliases), "world.repositories")
        self.assertEqual(canonical_state_path("world.git.repos.x", aliases), "world.repositories.repos.x")
        self.assertEqual(canonical_state_path("world.files.a", aliases), "world.files.a")
        self.assertEqual(canonical_state_path("session.cwd", aliases), "session.cwd")
        item = LocalTransitionEvidence(
            id="e", episode_id="p", transition_id="t", action=NormalizedAction(type="git-cmd"), outcome=Outcome.SUCCESS,
            confidence=Confidence(value=0.9), mutations=[StateMutation(op="merge", path="world.git", value={"/r": {}})],
            observation_facts=[Fact(subject="world.git_repositories", predicate="contains", value={"/r": {}})])
        rewritten = canonical_evidence([item], schema)[0]
        self.assertEqual((rewritten.action.type, rewritten.mutations[0].path, rewritten.observation_facts[0].subject),
                         ("git", "world.repositories", "world.repositories"))
        self.assertEqual(item.mutations[0].path, "world.git")  # the extracted record is untouched
        with self.assertRaisesRegex(ValueError, "Ambiguous state path alias"):
            state_path_aliases(StateSchema(fields=[StateField(path="world.a", aliases=["world.x"]),
                                                   StateField(path="world.b", aliases=["world.x"])]))

    def test_terminal_actions_carry_program_names(self):
        from trace2env.agentworld import command_programs, terminal_keystrokes_action

        single = terminal_keystrokes_action([{"keystrokes": "FOO=1 /usr/bin/cat /app/x.txt\n", "duration": 0.5}])
        self.assertEqual((single.type, single.arguments["program"], single.arguments["argv"]), ("cat", "cat", ["/app/x.txt"]))
        batch = terminal_keystrokes_action([{"keystrokes": "cat > f.py << 'EOF'\nimport os\nEOF\n# run it\npython3 f.py\n", "duration": 1.0}])
        self.assertEqual(batch.type, "shell")
        self.assertEqual(batch.arguments["programs"], ["cat", "python3"])
        self.assertEqual(command_programs(["cd /app && ls", "git status"]), ["cd", "git"])

    def test_observed_paths_are_declared_on_the_schema(self):
        from trace2env.models import ActionSchema, SchemaInductionResult, StateField, StateMutation, StateSchema
        from trace2env.reconstruction import declare_observed_paths

        schema = SchemaInductionResult(
            action_schema=ActionSchema(actions=[]),
            state_schema=StateSchema(fields=[StateField(path="world.mailman.lists", type="object"),
                                             StateField(path="world.files", type="object")]))
        evidence = [LocalTransitionEvidence(
            id="e", episode_id="p", transition_id="t", outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
            action=NormalizedAction(type="shell"),
            mutations=[StateMutation(op="merge", path="world.mailman", value={"version": "3.3"}),
                       StateMutation(op="merge", path="world.files", value={"/x": {}}),
                       StateMutation(op="set", path="world.last_exit_status", value=1)],
            observation_facts=[Fact(subject="surface.mode", predicate="equals", value="prompt")])]
        notes = declare_observed_paths(schema, evidence)
        declared = {field.path: field.type for field in schema.state_schema.fields}
        self.assertEqual(declared["world.mailman"], "object")
        self.assertEqual(declared["world.last_exit_status"], "integer")
        self.assertEqual(declared["surface.mode"], "string")
        self.assertEqual(len(notes), 3)
        self.assertEqual(next(f for f in schema.state_schema.fields if f.path == "surface.mode").visibility, "surface")

    def test_observed_arguments_are_declared_on_the_schema(self):
        from trace2env.models import ActionSchema, ActionSpec, ArgumentSpec, SchemaInductionResult, StateSchema
        from trace2env.reconstruction import declare_observed_arguments

        schema = SchemaInductionResult(
            action_schema=ActionSchema(actions=[ActionSpec(name="cat", arguments={"argv": ArgumentSpec(type="array")})]),
            state_schema=StateSchema(fields=[]))
        evidence = [LocalTransitionEvidence(
            id="e", episode_id="p", transition_id="t", outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
            action=NormalizedAction(type="cat", arguments={"argv": ["x"], "command": "cat x", "keystrokes": [{"keystrokes": "cat x\n"}]}))]
        notes = declare_observed_arguments(schema, evidence)
        declared = schema.action_schema.actions[0].arguments
        self.assertEqual((declared["command"].type, declared["keystrokes"].type, declared["argv"].type), ("string", "array", "array"))
        self.assertEqual(len(notes), 2)


class MapMutationTests(unittest.TestCase):
    """``merge`` and ``remove`` edit entries of object-valued state such as a file map keyed by paths."""

    def test_merge_and_remove_edit_map_entries_without_replacing_the_map(self):
        from trace2env.engine import apply_mutations
        from trace2env.models import EnvironmentState, StateMutation

        state = EnvironmentState(world={"files": {"/app/a.txt": {"type": "file"}}})
        merged = apply_mutations(state, [StateMutation(op="merge", path="world.files",
                                                        value={"/app/b.csv": {"type": "file", "size": 3}})])
        self.assertEqual(set(merged.world["files"]), {"/app/a.txt", "/app/b.csv"})
        removed = apply_mutations(merged, [StateMutation(op="remove", path="world.files", value="/app/a.txt")])
        self.assertEqual(list(removed.world["files"]), ["/app/b.csv"])
        created = apply_mutations(EnvironmentState(), [StateMutation(op="merge", path="world.env", value={"HOME": "/root"})])
        self.assertEqual(created.world["env"], {"HOME": "/root"})
        self.assertEqual(apply_mutations(EnvironmentState(), [StateMutation(op="remove", path="world.env", value=["X"])]).world, {})
        with self.assertRaisesRegex(ValueError, "non-object"):
            apply_mutations(EnvironmentState(world={"count": 1}), [StateMutation(op="merge", path="world.count", value={"a": 1})])

    def test_file_names_with_dots_address_map_entries(self):
        """``world.files./app/report.txt`` is the entry ``/app/report.txt``, not a nested ``txt`` key (DeepSeek emitted
        such effects live: a ``delete`` then left an empty ``/app/report`` behind and the file seemed to survive ``mv``)."""
        from trace2env.engine import apply_mutations, get_path
        from trace2env.models import EnvironmentState, StateMutation

        created = apply_mutations(EnvironmentState(), [
            StateMutation(op="create", path="world.files./app/report.txt", value={"type": "file", "content": "x\n"})])
        self.assertEqual(created.world["files"], {"/app/report.txt": {"type": "file", "content": "x\n"}})
        edited = apply_mutations(created, [StateMutation(op="set", path="world.files./app/report.txt.content", value="y\n"),
                                           StateMutation(op="set", path="world.files./app/data.tar.gz.size", value=3)])
        self.assertEqual(edited.world["files"]["/app/report.txt"]["content"], "y\n")
        self.assertEqual(edited.world["files"]["/app/data.tar.gz"], {"size": 3}, "a scalar write names an attribute")
        self.assertEqual(get_path(edited.model_dump(mode="python"), "world.files./app/report.txt.content"), "y\n")
        moved = apply_mutations(edited, [StateMutation(op="create", path="world.files./app/summary.txt", value={"type": "file"}),
                                         StateMutation(op="delete", path="world.files./app/report.txt")])
        self.assertEqual(set(moved.world["files"]), {"/app/summary.txt", "/app/data.tar.gz"}, "mv leaves nothing behind")
        backup = apply_mutations(moved, [StateMutation(op="create", path="world.files./app/summary.txt.bak", value={"type": "file"})])
        self.assertIn("/app/summary.txt.bak", backup.world["files"], "a key extending an existing one is its own entry")
        merged = apply_mutations(backup, [StateMutation(op="merge", path="world.files./app/summary.txt", value={"mode": "0644"})])
        self.assertEqual(merged.world["files"]["/app/summary.txt"], {"type": "file", "mode": "0644"})
        pages = apply_mutations(EnvironmentState(), [StateMutation(op="set", path="world.pages.https://shop.example.com/cart.items", value=2)])
        self.assertEqual(pages.world["pages"], {"https://shop.example.com/cart": {"items": 2}}, "URLs are names too")
        nested = apply_mutations(EnvironmentState(), [StateMutation(op="set", path="world.git.branches.main", value="abc")])
        self.assertEqual(nested.world["git"], {"branches": {"main": "abc"}}, "ordinary segments still nest")
        # bracket notation, the other spelling models emit, is the same path (it was rejected as undeclared live)
        from trace2env.models import StateField, StateSchema, normalize_state_path
        from trace2env.validation import validate_mutations
        self.assertEqual(normalize_state_path('world.files["/app/report.txt"].content'), "world.files./app/report.txt.content")
        self.assertEqual(normalize_state_path("world.files['/app/report.txt']['mode']"), "world.files./app/report.txt.mode")
        self.assertEqual(normalize_state_path("world.files[/app/x]"), "world.files./app/x")
        bracket = StateMutation(op="create", path='world.files["/app/report.txt"]', value={"type": "file"})
        self.assertEqual(bracket.path, "world.files./app/report.txt")
        validate_mutations([bracket], StateSchema(fields=[StateField(path="world.files", type="object")]), None)
        self.assertEqual(apply_mutations(EnvironmentState(), [bracket]).world["files"], {"/app/report.txt": {"type": "file"}})

    def test_action_tokens_resolve_inside_map_keys(self):
        from trace2env.engine import apply_mutations
        from trace2env.models import ActionSpec, ArgumentSpec, EnvironmentState, StateField, StateMutation, StateSchema
        from trace2env.validation import validate_mutations

        action = NormalizedAction(type="mkdir", arguments={"argv": ["/app/out"], "command": "mkdir /app/out"})
        effect = StateMutation(op="merge", path="world.files", value={"{action.arguments.argv.0}": {"type": "dir"}})
        after = apply_mutations(EnvironmentState(), [effect], action=action)
        self.assertEqual(after.world["files"], {"/app/out": {"type": "dir"}})
        spec = ActionSpec(name="mkdir", arguments={"argv": ArgumentSpec(type="array"), "command": ArgumentSpec(type="string")})
        schema = StateSchema(fields=[StateField(path="world.files", type="object")])
        validate_mutations([effect], schema, spec)
        with self.assertRaisesRegex(ValueError, "Undeclared action argument"):
            validate_mutations([StateMutation(op="merge", path="world.files", value={"{action.arguments.target}": {}})], schema, spec)

    def test_state_template_tokens_are_rejected_inside_effect_values(self):
        from trace2env.models import StateField, StateMutation, StateSchema
        from trace2env.validation import validate_mutations

        schema = StateSchema(fields=[StateField(path="surface.mode", type="string", visibility="surface")])
        with self.assertRaisesRegex(ValueError, "State-value interpolation"):
            validate_mutations([StateMutation(op="set", path="surface.mode", value="{state_before.surface.mode}")], schema, None)
        validate_mutations([StateMutation(op="set", path="surface.mode", value="prompt")], schema, None)

    def test_map_mutations_are_validated_against_the_schema(self):
        from trace2env.models import StateField, StateMutation, StateSchema
        from trace2env.validation import validate_mutations

        schema = StateSchema(fields=[StateField(path="world.files", type="object"),
                                     StateField(path="world.count", type="integer")])
        validate_mutations([StateMutation(op="merge", path="world.files", value={"/x": {}}),
                            StateMutation(op="remove", path="world.files", value=["/x", "/y"])], schema, None)
        with self.assertRaisesRegex(ValueError, "requires object state"):
            validate_mutations([StateMutation(op="merge", path="world.count", value={"a": 1})], schema, None)
        with self.assertRaisesRegex(ValueError, "object value"):
            validate_mutations([StateMutation(op="merge", path="world.files", value="text")], schema, None)
        with self.assertRaisesRegex(ValueError, "list of keys"):
            validate_mutations([StateMutation(op="remove", path="world.files", value={"/x": 1})], schema, None)


if __name__ == "__main__":
    unittest.main()


class DeclareObservedActionsTests(unittest.TestCase):
    def test_undeclared_evidence_actions_are_added_deterministically(self):
        from trace2env.models import (ActionSchema, ActionSpec, Confidence, LocalTransitionEvidence, NormalizedAction, Outcome,
                                      SchemaInductionResult, SourceRef, StateSchema)
        from trace2env.reconstruction import declare_observed_actions

        schema = SchemaInductionResult(action_schema=ActionSchema(actions=[ActionSpec(name="shell", aliases=["sh"])]),
                                       state_schema=StateSchema(fields=[]))
        ref = SourceRef(source_id="src_x", episode_id="ep_1", event_ids=["ev_1"])
        evidence = [
            LocalTransitionEvidence(id="e1", episode_id="ep_1", transition_id="t1", action=NormalizedAction(type="sh"),
                                    outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9), provenance=[ref]),
            LocalTransitionEvidence(id="e2", episode_id="ep_1", transition_id="t2", action=NormalizedAction(type="python", arguments={"argv": ["x.py"]}),
                                    outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9), provenance=[ref]),
        ]
        notes = declare_observed_actions(schema, evidence)
        self.assertEqual([spec.name for spec in schema.action_schema.actions], ["shell", "python"])  # alias resolved, python added
        self.assertEqual(schema.action_schema.actions[1].provenance, [ref])
        self.assertEqual(len(notes), 1)
        self.assertIn("python", notes[0])
        self.assertEqual(declare_observed_actions(schema, evidence), [])  # idempotent


class StageUsageSummaryTests(unittest.TestCase):
    def test_usage_summary_skips_non_numeric_labels(self):
        import importlib.util
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location("pilot_stages", root / "scripts" / "pilot_stages.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            calls = Path(tmp) / "calls.jsonl"
            calls.write_text("\n".join(json.dumps(r) for r in [
                {"elapsed_s": 1.5, "usage": {"prompt_tokens": 10, "completion_tokens": 2, "provider": "OpenAI", "cost": 0.5}},
                {"elapsed_s": 0.5, "cache_hit": True, "usage": {"prompt_tokens": 5, "completion_tokens": 1, "provider": "Azure"}},
            ]) + "\n", encoding="utf-8")
            totals = module.usage_summary(calls)
        self.assertEqual((totals["calls"], totals["cache_hits"], totals["prompt_tokens"], totals["completion_tokens"]), (2, 1, 15, 3))
        self.assertAlmostEqual(totals["cost"], 0.5)
        self.assertNotIn("provider", totals)
