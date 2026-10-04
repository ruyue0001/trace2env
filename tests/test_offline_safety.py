from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trace2env.adapters import annotate_episode, events_for_slice, load_raw_traces, segment_transitions
from trace2env.cli import main
from trace2env.compiler import EnvironmentCompiler
from trace2env.engine import (AmbiguousRuleError, MISSING, apply_mutations, deterministic_plan,
                             evaluate_condition, get_path, select_rule, verify_rule_support)
from trace2env.examples import build_ledger_demo
from trace2env.induction import BoundedInduction, PromptBudgetExceeded
from trace2env.llm import ScriptedLLM
from trace2env.models import (ActionSpec, AgentTurn, ArgumentSpec, ArtifactEdit, Confidence, Condition, EnvironmentState,
    EventAnnotation, Fact, LocalTransitionEvidence, NormalizedAction, Outcome, PackagePatch,
    ReconstructionArtifacts, ReconstructionConfig, ReplayCase, ReplayStep, SourceRef, StateField, StateMutation,
    StepRequest, TraceAnnotations, TransitionSubmission, SplitManifest)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.patching import apply_package_patch
from trace2env.reconstruction import TraceReconstructor
from trace2env.replay import evaluate_package, promote_package
from trace2env.runtime import NoApplicableRule, RuntimeHarness
from trace2env.validation import artifact_digest, validate_artifacts, validate_mutations
from trace2env.contrasts import build_contrasts


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class AdapterSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def calls(self, replies, ids=("a", "b")):
        path = self.root / "calls.json"
        save(path, {"messages": [{"role": "assistant", "tool_calls": [
            {"id": value, "type": "function", "function": {"name": value, "arguments": "{}"}}
            for value in ids]}, *replies]})
        return load_raw_traces(path)[0]

    def test_concurrent_replies_match_ids_even_out_of_order(self):
        episode = self.calls([{"role": "tool", "tool_call_id": "b", "content": "B"},
                              {"role": "tool", "tool_call_id": "a", "content": "A"}])
        slices = segment_transitions(episode)
        self.assertEqual(len(slices), 2)
        pairs = [events_for_slice(episode, item) for item in slices]
        self.assertEqual([(x["actions"][0].metadata["call_id"], x["observations"][0].content) for x in pairs],
                         [("a", "A"), ("b", "B")])
        self.assertTrue(all(item.concurrent_action_event_ids for item in slices))
        self.assertTrue(slices[0].delayed_observation_event_ids)

    def test_missing_and_foreign_replies_are_not_guessed(self):
        episode = self.calls([{"role": "tool", "tool_call_id": "foreign", "content": "X"},
                              {"role": "tool", "tool_call_id": "b", "content": "B"}])
        slices = segment_transitions(episode)
        self.assertEqual(slices[0].alignment, "ambiguous")
        self.assertEqual(slices[0].observation_event_ids, [])
        self.assertEqual(len(slices[1].observation_event_ids), 1)

    def test_duplicate_call_ids_are_ambiguous(self):
        episode = self.calls([{"role": "tool", "tool_call_id": "a", "content": "A"}], ids=("a", "a"))
        self.assertTrue(all(t.alignment == "ambiguous" and not t.observation_event_ids for t in segment_transitions(episode)))

    def test_overlapping_calls_across_messages_are_both_marked(self):
        path = self.root / "separate.json"
        save(path, {"messages": [
            {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "a"}}]},
            {"role": "assistant", "tool_calls": [{"id": "b", "function": {"name": "b"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
            {"role": "tool", "tool_call_id": "b", "content": "B"}]})
        self.assertTrue(all(t.concurrent_action_event_ids for t in segment_transitions(load_raw_traces(path)[0])))

    def test_content_identity_is_location_independent_and_changes_with_bytes(self):
        a, b = self.root / "a.txt", self.root / "b.txt"
        a.write_bytes(b"Action: read\r\nObservation: 1\r\n")
        b.write_bytes(a.read_bytes())
        first, copied = load_raw_traces(a)[0], load_raw_traces(b)[0]
        self.assertEqual(first.source_id, copied.source_id)
        a.write_bytes(b"Action: read\r\nObservation: 2\r\n")
        self.assertNotEqual(first.source_id, load_raw_traces(a)[0].source_id)

    def test_episode_and_event_ids_are_qualified_across_sources(self):
        episodes = []
        for index in range(2):
            path = self.root / f"{index}.json"
            save(path, {"episode_id": "shared", "events": [
                {"id": "action", "kind": "action", "content": str(index)},
                {"id": "reply", "kind": "observation", "content": "ok"}]})
            episodes.extend(load_raw_traces(path))
        self.assertEqual(len({e.id for e in episodes}), 2)
        self.assertEqual(len({event.id for e in episodes for event in e.events}), 4)
        self.assertEqual([e.metadata["original_episode_id"] for e in episodes], ["shared", "shared"])

    def test_annotations_preserve_unclassified_text_and_reject_invalid_spans(self):
        path = self.root / "opaque.log"
        path.write_text("prefix ACTION suffix", encoding="utf-8")
        episode = load_raw_traces(path)[0]
        event = episode.events[0]
        annotated = annotate_episode(episode, TraceAnnotations(annotations=[EventAnnotation(
            source_event_id=event.id, actor="agent", kind="action", start=7, end=13)]))
        self.assertEqual("".join(e.content for e in annotated.events), event.content)
        self.assertEqual(annotated.events[1].content, "ACTION")
        for start, end in [(3, 99), (5, 5)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                annotate_episode(episode, TraceAnnotations(annotations=[EventAnnotation(
                    source_event_id=event.id, actor="agent", kind="action", start=start, end=end)]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            annotate_episode(episode, TraceAnnotations(annotations=[
                EventAnnotation(source_event_id=event.id, actor="agent", kind="action", start=0, end=9),
                EventAnnotation(source_event_id=event.id, actor="tool", kind="observation", start=8, end=15)]))

    def test_ingest_preserves_exact_raw_bytes_and_deduplicates_copies(self):
        a, b = self.root / "a.txt", self.root / "b.txt"
        raw = b"Action: read\r\nObservation: 1\r\n"
        a.write_bytes(raw)
        b.write_bytes(raw)
        recon = TraceReconstructor(ScriptedLLM({}), ReconstructionConfig(environment_id="x", name="X"), self.root / "work")
        episodes, transitions = recon.ingest([a, b])
        self.assertEqual((len(episodes), len(transitions)), (1, 1))
        self.assertEqual(next(iter(recon.source_snapshots.values())).encode("utf-8"), raw)

    def test_ingest_rejects_nontraining_manifest_sources_before_model_calls(self):
        path = self.root / "heldout.txt"
        path.write_text("Action: read\nObservation: 1", encoding="utf-8")
        source = load_raw_traces(path)[0].source_id
        manifest = SplitManifest(assignments={source: {"split": "validation", "trajectory_group": "heldout"}})
        llm = ScriptedLLM({})
        recon = TraceReconstructor(llm, ReconstructionConfig(environment_id="x", name="X"), self.root / "work")
        with self.assertRaisesRegex(ValueError, "training split"):
            recon.ingest([path], manifest)
        self.assertEqual(llm.calls, [])

    def test_split_manifest_rejects_cross_split_trajectory_family(self):
        with self.assertRaisesRegex(ValueError, "cross data splits"):
            SplitManifest(assignments={"a": {"split": "train", "trajectory_group": "same"},
                                       "b": {"split": "test", "trajectory_group": "same"}})

    def test_reconciliation_closes_renderers_and_quarantines_invalid_rules(self):
        from trace2env.models import (ActionSchema, EnvironmentNote, Invariant, Operand, RenderContract, SchemaInductionResult,
                                      StateSchema, TransitionRule)
        from trace2env.reconstruction import reconcile_artifacts
        schema = SchemaInductionResult(
            action_schema=ActionSchema(actions=[ActionSpec(name="ls", arguments={"argv": ArgumentSpec(type="array")})]),
            state_schema=StateSchema(fields=[StateField(path="session.cwd", type="string", visibility="session")]))
        good = TransitionRule(id="ls_ok", action_type="ls", description="lists", renderer="terminal.listing",
                              observation_template="{state_after.session.cwd}")
        bad_path = TransitionRule(id="ls_bad", action_type="ls", description="bad", renderer="terminal.listing",
                                  effects=[StateMutation(op="set", path="world.unknown", value=1)])
        bad_action = TransitionRule(id="cd_bad", action_type="cd", description="no such action", renderer="terminal.listing")
        invariants = [Invariant(id="ok", description="cwd exists", condition=Condition(left=Operand(path="session.cwd"), op="exists")),
                      Invariant(id="broken", description="x", condition=Condition(left=Operand(path="world.x"), op="exists")),
                      Invariant(id="argy", description="needs an action", severity="warning",
                                condition=Condition(left=Operand(action_arg="sql"), op="contains", right=Operand(literal="SELECT")))]
        renderers = [RenderContract(id="terminal.other", action_types=["ls", "ghost"]),
                     RenderContract(id="terminal.prose", action_types=["ls"], instructions="Render the listing.",
                                    required_fields=["total row count", "outcome.rows", "state_after.session.cwd"],
                                    template="{outcome.rows} rows in {state_after.session.cwd}")]
        notes = [EnvironmentNote(id="n", statement="s", action_types=["ls", "rm"])]
        rules, invs, contracts, kept_notes, unresolved, rejected = reconcile_artifacts(schema, [good, bad_path, bad_action], invariants, renderers, notes)
        self.assertEqual([rule.id for rule in rules], ["ls_ok"])
        self.assertEqual({item["id"] for item in rejected}, {"ls_bad", "cd_bad", "broken", "argy"})
        self.assertEqual({contract.id for contract in contracts}, {"terminal.other", "terminal.listing", "terminal.prose"})
        self.assertEqual(next(c for c in contracts if c.id == "terminal.other").action_types, ["ls"])
        prose = next(c for c in contracts if c.id == "terminal.prose")  # kept: prose fields and bad template dropped
        self.assertEqual((prose.required_fields, prose.template, prose.instructions),
                         (["state_after.session.cwd"], None, "Render the listing."))
        self.assertTrue(any("terminal.prose" in item and "were dropped" in item for item in unresolved))
        self.assertEqual([inv.id for inv in invs], ["ok"])
        self.assertEqual(kept_notes[0].action_types, ["ls"])
        self.assertTrue(any("synthesized" in item for item in unresolved))
        ReconstructionArtifacts(episodes=[], transitions=[], evidence=[], action_schema=schema.action_schema,
                                state_schema=schema.state_schema, rules=rules, invariants=invs, renderers=contracts, notes=kept_notes)

    def test_provenance_is_re_anchored_or_dropped_and_support_downgraded(self):
        from trace2env.models import ActionSchema, SchemaInductionResult, StateSchema, TransitionRule, TransitionSlice
        from trace2env.reconstruction import reconcile_provenance
        from trace2env.validation import repair_provenance
        a, b = self.root / "a.txt", self.root / "b.txt"
        a.write_text("Action: ls\nObservation: one", encoding="utf-8"); b.write_text("Action: ls\nObservation: two", encoding="utf-8")
        episodes = load_raw_traces(a) + load_raw_traces(b)
        ea, eb = episodes
        unresolved = []
        refs = [SourceRef(source_id=ea.source_id, episode_id=ea.id, event_ids=[ea.events[0].id, eb.events[1].id, "ghost"]),
                SourceRef(source_id="nowhere", episode_id="nowhere", event_ids=["missing"])]
        repaired = repair_provenance(refs, episodes, unresolved, "rule r")
        self.assertEqual({(ref.episode_id, tuple(ref.event_ids)) for ref in repaired},
                         {(ea.id, (ea.events[0].id,)), (eb.id, (eb.events[1].id,))})
        self.assertEqual(repaired[1].source_id, eb.source_id)
        self.assertEqual(len(unresolved), 4)  # dropped ghost, re-anchored, unknown ids + unresolvable citation
        schema = SchemaInductionResult(action_schema=ActionSchema(actions=[ActionSpec(name="ls")]), state_schema=StateSchema(fields=[]))
        transitions = [t for ep in episodes for t in segment_transitions(ep)]
        evidence = [LocalTransitionEvidence(id=f"e{i}", episode_id=t.episode_id, transition_id=t.id, action=NormalizedAction(type="ls"),
                                            observation_text="o", confidence=Confidence(value=1)) for i, t in enumerate(transitions)]
        rule = TransitionRule(id="r", action_type="ls", description="d", provenance=[SourceRef(source_id="x", event_ids=["ghost"])])
        from trace2env.models import Condition, EnvironmentNote, Invariant, Operand
        episode_only = SourceRef(source_id=ea.source_id, episode_id=ea.id, event_ids=[])  # cites a whole episode, no event
        note = EnvironmentNote(id="n", statement="s", status="supported", provenance=[episode_only])
        invariant = Invariant(id="i", description="d", condition=Condition(left=Operand(path="session.cwd"), op="exists"),
                              provenance=[episode_only])
        invariants = [invariant]
        notes = reconcile_provenance(schema, [rule], invariants, [], [note], episodes, evidence, transitions)
        self.assertEqual((rule.status, rule.provenance), ("tentative", []))
        self.assertTrue(any("downgraded" in item for item in notes))
        self.assertEqual((note.status, note.provenance), ("tentative", []))  # supported claims need event anchors
        self.assertEqual(invariants, [])  # an invariant without anchors is dropped in place
        self.assertTrue(any("Invariant i dropped" in item for item in notes))

    def test_rule_scope_values_are_coerced_to_text(self):
        from trace2env.models import TransitionRule
        rule = TransitionRule(id="r", action_type="serve", description="d", scope={"port": 8080, "tls": False, "tenant": "a"})
        self.assertEqual(rule.scope, {"port": "8080", "tls": "False", "tenant": "a"})

    def test_unparsable_timestamps_are_kept_as_metadata(self):
        path = self.root / "stamped.jsonl"
        rows = [{"episode_id": "a", "actor": "agent", "kind": "action", "content": "ls", "timestamp": "Tuesday noon"},
                {"episode_id": "a", "actor": "tool", "content": "ok", "timestamp": "2026-01-01T00:00:00Z"}]
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        events = load_raw_traces(path)[0].events
        self.assertIsNone(events[0].timestamp)
        self.assertEqual(events[0].metadata["raw_timestamp"], "Tuesday noon")
        self.assertEqual(events[1].timestamp.year, 2026)


class PackageSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = build_ledger_demo(self.root / "base")
        self.artifacts = ReconstructionArtifacts.model_validate_json((self.base / "construction/artifacts.json").read_text(encoding="utf-8"))
        self.config = ReconstructionConfig.model_validate_json((self.base / "construction/config.json").read_text(encoding="utf-8"))

    def compile(self, artifacts=None, name="candidate", config=None):
        return EnvironmentCompiler(config or self.config).compile(artifacts or self.artifacts, self.root / name)

    def full_case(self, split="validation", group="heldout"):
        return ReplayCase(id="ledger", trajectory_group=group, source_ids=["src_" + "9" * 64], split=split,
            initial_state=EnvironmentState(world={"balance": 100.0}), steps=[
                ReplayStep(action=NormalizedAction(type="balance"), expected_observation="Balance: 100.0"),
                ReplayStep(action=NormalizedAction(type="deposit", arguments={"amount": 5}),
                           expected_observation="Deposited 5. Balance: 105.0", expected_state={"world.balance": 105.0}),
                ReplayStep(action=NormalizedAction(type="withdraw", arguments={"amount": 25}),
                           expected_observation="Withdrew 25. Balance: 80.0", expected_outcome=Outcome.SUCCESS),
                ReplayStep(action=NormalizedAction(type="withdraw", arguments={"amount": 1000}),
                           expected_observation="Insufficient funds. Balance: 80.0", expected_outcome=Outcome.FAILURE,
                           expected_state={"world.balance": 80.0}),
            ])

    def test_missing_path_is_distinct_from_null_and_supports_new_fields(self):
        self.assertIs(get_path({}, "x", MISSING), MISSING)
        self.assertIsNone(get_path({"x": None}, "x", MISSING))
        with self.assertRaises(KeyError):
            get_path({}, "x")
        condition = Condition(left={"path": "world.x"}, op="not_exists")
        self.assertTrue(evaluate_condition(condition, EnvironmentState(), NormalizedAction(type="x")))
        self.assertEqual(apply_mutations(EnvironmentState(), [StateMutation(op="set", path="world.x", value=1)]).world, {"x": 1})

    def test_immutable_state_cannot_be_changed_through_parent_or_child(self):
        from trace2env.models import StateSchema
        schema = StateSchema(fields=[StateField(path="world.account", type="object", mutable=False),
                                     StateField(path="world.account.balance", type="number")])
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_mutations([StateMutation(op="set", path="world.account.balance", value=0)], schema, None)
        schema.fields[0].mutable, schema.fields[1].mutable = True, False
        with self.assertRaisesRegex(ValueError, "immutable"):
            validate_mutations([StateMutation(op="set", path="world.account", value={})], schema, None)

    def test_replay_does_not_assume_zero_for_unobserved_arithmetic(self):
        self.artifacts.rules[0].conditions = []
        candidate = self.compile()
        case = self.full_case()
        case.initial_state = EnvironmentState()
        case.steps = [ReplayStep(action=NormalizedAction(type="deposit", arguments={"amount": 5}),
                                expected_observation="Deposited 5. Balance: 5")]
        report = evaluate_package(candidate, [case])
        self.assertFalse(report.cases[0].passed)
        self.assertIn("updates unobserved state", report.cases[0].errors[0])

    def test_disputed_rules_are_retained_but_not_executable(self):
        for index, status in enumerate(["tentative", "conflicted", "deprecated"]):
            with self.subTest(status=status):
                data = self.artifacts.model_copy(deep=True)
                data.rules[0].status = status
                data.rules[0].confidence = 1
                package_dir = self.compile(data, f"status-{index}")
                package = EnvironmentPackage(package_dir)
                self.assertNotIn(data.rules[0].id, [r.id for r in package.rules])
                self.assertIn(data.rules[0].id, (package_dir / "candidates/rules.jsonl").read_text())
                self.assertEqual(PackageInspector(package).inspect_action("deposit")["rules"], [])
                with self.assertRaises(NoApplicableRule):
                    RuntimeHarness(package_dir, self.root / f"session-{index}").step(
                        StepRequest(action=NormalizedAction(type="deposit", arguments={"amount": 5})))

    def test_foreign_or_unknown_scope_is_excluded(self):
        self.artifacts.rules[0].scope = {"tenant": "B"}
        scoped = self.config.model_copy(update={"tenant_scope": "A"})
        package = EnvironmentPackage(self.compile(config=scoped))
        self.assertEqual(PackageInspector(package).inspect_action("deposit")["rules"], [])
        self.assertIsNone(select_rule([self.artifacts.rules[0]], EnvironmentState(world={"balance": 100}),
            NormalizedAction(type="deposit", arguments={"amount": 5})))

    def test_matching_scope_and_state_scope_execute(self):
        self.artifacts.rules[0].scope = {"tenant_scope": "A", "world.balance": "100.0"}
        package = self.compile(config=self.config.model_copy(update={"tenant_scope": "A"}))
        harness = RuntimeHarness(package, self.root / "session")
        action = StepRequest(action=NormalizedAction(type="deposit", arguments={"amount": 5}))
        self.assertEqual(harness.step(action).state.world["balance"], 105)
        with self.assertRaises(NoApplicableRule):
            harness.step(action)
        self.assertEqual(harness.session.load().revision, 1)

    def test_conflicting_equal_priority_rules_roll_back(self):
        conflict = self.artifacts.rules[0].model_copy(deep=True)
        conflict.id = "opposite"
        conflict.effects[0].op = "decrement"
        self.artifacts.rules.append(conflict)
        harness = RuntimeHarness(self.compile(), self.root / "session")
        with self.assertRaises(AmbiguousRuleError):
            harness.step(StepRequest(action=NormalizedAction(type="deposit", arguments={"amount": 5})))
        self.assertEqual(harness.session.load().world["balance"], 100)
        self.assertFalse(harness.session.audit_records()[0].committed)

    def test_conflicts_beyond_retrieval_limit_cannot_be_hidden(self):
        for index in range(20):
            rule = self.artifacts.rules[0].model_copy(deep=True)
            rule.id = f"duplicate-{index}"
            if index == 0:
                rule.confidence = 0.56
                rule.effects[0].op = "decrement"
            self.artifacts.rules.append(rule)
        harness = RuntimeHarness(self.compile(), self.root / "session")
        with self.assertRaises(AmbiguousRuleError):
            harness.step(StepRequest(action=NormalizedAction(type="deposit", arguments={"amount": 5})))

    def test_invalid_semantic_references_and_types_fail_before_writes(self):
        def change_path(a): a.rules[0].effects[0].path = "world.unknown"
        def change_arg(a): a.rules[0].effects[0].value = {"$action_arg": "unknown"}
        def ghost(a): a.rules[0].provenance = [SourceRef(source_id="ghost", event_ids=["missing"])]
        def immutable(a): a.state_schema.fields[0].mutable = False
        def wrong_type(a): a.rules[0].effects[0].value = "five"
        def renderer(a): a.rules[0].observation_template = "{computed.fake}"
        def duplicate(a): a.action_schema.actions.append(a.action_schema.actions[0].model_copy())
        for index, mutate in enumerate([change_path, change_arg, ghost, immutable, wrong_type, renderer, duplicate]):
            data = self.artifacts.model_copy(deep=True)
            mutate(data)
            with self.subTest(mutation=mutate.__name__), self.assertRaises(ValueError):
                self.compile(data, f"invalid-{index}")
            self.assertFalse((self.root / f"invalid-{index}").exists())

    def test_dynamic_object_paths_are_permitted_and_argument_references_checked(self):
        self.artifacts.action_schema.actions[0].arguments["name"] = ArgumentSpec(type="string", required=True)
        self.artifacts.state_schema.fields.append(StateField(path="world.records", type="object", default={}))
        self.artifacts.rules[0].effects = [StateMutation(op="set", path="world.records.{action.arguments.name}", value=1)]
        package = self.compile()
        result = RuntimeHarness(package, self.root / "session").step(StepRequest(
            action=NormalizedAction(type="deposit", arguments={"amount": 0, "name": "x"})))
        self.assertEqual(result.state.world["records"], {"x": 1})

    def test_compiler_never_overwrites_existing_package(self):
        before = (self.base / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            EnvironmentCompiler(self.config).compile(self.artifacts, self.base)
        self.assertEqual((self.base / "manifest.json").read_bytes(), before)

    def test_failed_build_never_publishes_destination(self):
        with patch.object(EnvironmentCompiler, "_build_index", side_effect=RuntimeError("interrupted")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.compile()
        self.assertFalse((self.root / "candidate").exists())
        EnvironmentPackage(self.base)

    def test_reconstruction_requires_raw_evidence(self):
        with self.assertRaisesRegex(ValueError, "trace evidence"):
            self.compile(config=self.config.model_copy(update={"construction_kind": "reconstructed"}))

    def test_plan_cannot_duplicate_omit_or_reorder_effects(self):
        rule = self.artifacts.rules[0]
        action = NormalizedAction(type="deposit", arguments={"amount": 5})
        state = EnvironmentState(world={"balance": 100})
        for effects in [[], [StateMutation(op="increment", path="world.balance", value=5)] * 2]:
            plan = deterministic_plan(rule, action).model_copy(update={"effects": effects})
            self.assertTrue(any(i.severity == "error" for i in verify_rule_support(plan, [rule], state)))

    def test_delayed_preconditions_use_state_after_due_events(self):
        self.artifacts.rules[0].conditions = [Condition(left={"path": "world.balance"}, op="eq", right={"literal": 120})]
        state = apply_mutations(EnvironmentState(world={"balance": 100}), [StateMutation(
            op="schedule", path="world.balance", value=120, delay_steps=1)])
        result = RuntimeHarness(self.compile(), self.root / "session", initial_state=state).step(
            StepRequest(action=NormalizedAction(type="deposit", arguments={"amount": 5})))
        self.assertEqual(result.state.world["balance"], 125)

    def test_replay_covers_success_failure_and_persistent_state(self):
        report = evaluate_package(self.base, [self.full_case()])
        self.assertTrue(report.promotion_eligible, report.reasons)
        self.assertEqual(report.attempted_steps, 4)
        self.assertEqual(report.uncovered_rule_ids, [])
        self.assertFalse((self.base / "session.sqlite").exists())

    def test_promotion_requires_coverage_and_independent_validation(self):
        cases = [self.full_case(split="train")]
        report = evaluate_package(self.base, cases)
        self.assertTrue(report.cases[0].passed)
        self.assertFalse(report.promotion_eligible)
        case = self.full_case().model_copy(update={"steps": self.full_case().steps[:1]})
        report = evaluate_package(self.base, [case])
        self.assertFalse(report.promotion_eligible)
        self.assertEqual(len(report.uncovered_rule_ids), 3)

    def test_test_data_never_promotes_and_split_overlap_rejected(self):
        report = evaluate_package(self.base, [self.full_case(split="test")])
        self.assertFalse(report.promotion_eligible)
        a, b = self.full_case(), self.full_case(split="train").model_copy(update={"id": "second"})
        with self.assertRaisesRegex(ValueError, "crosses replay splits"):
            evaluate_package(self.base, [a, b])

    def test_wrong_candidate_reports_regression_and_keeps_baseline(self):
        self.artifacts.rules[1].effects[0].op = "increment"
        candidate = self.compile()
        report = evaluate_package(candidate, [self.full_case()], self.base)
        self.assertFalse(report.promotion_eligible)
        self.assertEqual(report.baseline_passed_cases, 1)
        self.assertEqual(report.regressions, ["ledger"])
        self.assertTrue(evaluate_package(self.base, [self.full_case()]).cases[0].passed)

    def test_missing_state_is_unscorable_and_not_seeded_from_defaults(self):
        case = self.full_case().model_copy(update={"initial_state": EnvironmentState()})
        report = evaluate_package(self.base, [case])
        self.assertFalse(report.promotion_eligible)
        self.assertEqual(report.attempted_steps, 1)
        self.assertEqual(report.total_steps, 4)

    def test_unknown_absence_does_not_activate_not_exists_rule(self):
        self.artifacts.rules[0].conditions = [Condition(left={"path": "world.balance"}, op="not_exists")]
        self.artifacts.rules[0].effects = [StateMutation(op="set", path="world.balance", value=5)]
        package = self.compile()
        case = ReplayCase(id="absence", trajectory_group="x", source_ids=["src_" + "8" * 64], split="validation",
            initial_state=EnvironmentState(), steps=[ReplayStep(action=NormalizedAction(type="deposit", arguments={"amount": 5}),
                                                             expected_observation="Deposited 5. Balance: 5")])
        report = evaluate_package(package, [case])
        self.assertIn("unobserved state", report.cases[0].errors[0])
        known = case.model_copy(update={"known_absent_paths": ["world.balance"]})
        self.assertTrue(evaluate_package(package, [known]).cases[0].passed)

    def test_promotion_rejects_modified_artifacts_or_scope(self):
        report = evaluate_package(self.base, [self.full_case()])
        changed = self.artifacts.model_copy(deep=True)
        changed.rules[0].description += " changed"
        with self.assertRaisesRegex(ValueError, "exact artifacts"):
            EnvironmentCompiler(self.config).compile(changed, self.root / "wrong", report)
        with self.assertRaisesRegex(ValueError, "exact artifacts"):
            EnvironmentCompiler(self.config.model_copy(update={"tenant_scope": "B"})).compile(
                self.artifacts, self.root / "wrong-scope", report)

    def test_successful_promotion_creates_new_verified_package(self):
        destination, report = promote_package(self.base, [self.full_case()], self.root / "promoted")
        loaded = EnvironmentPackage(destination)
        self.assertEqual(loaded.manifest.metadata["validation_status"], "validated")
        self.assertEqual(loaded.manifest.metadata["artifact_digest"], report.artifact_digest)
        self.assertEqual(EnvironmentPackage(self.base).manifest.metadata["validation_status"], "authored")

    def test_cli_replay_writes_report_and_promotes_without_model(self):
        cases = self.root / "cases.jsonl"
        cases.write_text(self.full_case().model_dump_json() + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            status = main(["replay", str(self.base), "--cases", str(cases), "--output", str(self.root / "report.json"),
                           "--promote-to", str(self.root / "promoted")])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads((self.root / "report.json").read_text())["promotion_eligible"])

    def test_typed_patches_reject_stale_bases_and_preserve_original(self):
        value = self.artifacts.rules[0].model_dump(mode="json")
        value["description"] = "Updated description"
        change = PackagePatch(base_artifact_digest=artifact_digest(self.artifacts), rationale="clarify",
            failed_case_ids=["ledger"], edits=[ArtifactEdit(collection="rules", key=value["id"], operation="replace", value=value)])
        updated = apply_package_patch(self.artifacts, change)
        self.assertNotEqual(updated.rules[0].description, self.artifacts.rules[0].description)
        with self.assertRaisesRegex(ValueError, "stale"):
            apply_package_patch(updated, change)
        conflicting = change.model_copy(update={"edits": change.edits * 2})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            apply_package_patch(self.artifacts, conflicting)

    def test_schema_checked_policy_accepts_declared_effects_only(self):
        def planner(path):
            submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="audited",
                                              effects=[StateMutation(op="set", path=path, value=42.0)])
            return ScriptedLLM({"runtime_agent_turn": [AgentTurn(final=submission)]})
        request = StepRequest(action=NormalizedAction(type="audit"))
        # Declared, mutable, correctly typed: accepted as a flagged model proposal under schema_checked.
        harness = RuntimeHarness(self.base, self.root / "graded", planner_llm=planner("world.balance"),
                                 trust_policy="schema_checked")
        result = harness.step(request)
        self.assertEqual(result.state.world["balance"], 42.0)
        self.assertIn("model_proposed_effects", [issue.code for issue in result.verification.issues])
        # The same plan is rejected under the default policy, and undeclared paths are rejected under both.
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            RuntimeHarness(self.base, self.root / "strict", planner_llm=planner("world.balance")).step(request)
        with self.assertRaisesRegex(ValueError, "Undeclared state path"):
            RuntimeHarness(self.base, self.root / "undeclared", planner_llm=planner("world.unknown"),
                           trust_policy="schema_checked").step(request)
        self.assertEqual(RuntimeHarness(self.base, self.root / "strict").session.load().world["balance"], 100.0)

    def test_rules_needing_absent_arguments_do_not_apply_or_crash(self):
        rule = self.artifacts.rules[0]  # deposit: increments by $action_arg amount
        state = EnvironmentState(world={"balance": 100})
        self.assertIsNone(select_rule([rule], state, NormalizedAction(type="deposit", arguments={})))
        harness = RuntimeHarness(self.base, self.root / "session")
        with self.assertRaisesRegex(ValueError, "Missing required argument"):
            harness.step(StepRequest(action=NormalizedAction(type="deposit", arguments={})))
        self.assertEqual(harness.session.load().revision, 0)


class ConditionArgumentTests(unittest.TestCase):
    """A rule condition that names an argument the action does not carry never holds; it must not crash rule selection."""

    def test_unresolvable_argument_token_makes_the_condition_false(self):
        from trace2env.engine import evaluate_condition, select_rule
        from trace2env.models import Condition, Operand, TransitionRule, StateMutation
        state = EnvironmentState(surface={"visible_elements": {"3": {"editable": True}}})
        indexed = Condition(left=Operand(path="surface.visible_elements.{action.arguments.index}.editable"), op="eq", right=Operand(literal=True))
        self.assertTrue(evaluate_condition(indexed, state, NormalizedAction(type="click", arguments={"index": 3})))
        self.assertFalse(evaluate_condition(indexed, state, NormalizedAction(type="click", arguments={"coordinate": [1, 2]})))
        absent = Condition(left=Operand(path="surface.visible_elements.{action.arguments.index}"), op="not_exists")
        self.assertFalse(evaluate_condition(absent, state, NormalizedAction(type="click", arguments={})))  # never holds either
        rule = TransitionRule(id="click_editable", action_type="click", description="tap an editable element", conditions=[indexed],
                              effects=[StateMutation(op="set", path="surface.keyboard", value=True)], outcome="success", status="supported",
                              confidence=0.9, provenance=[])
        self.assertIsNone(select_rule([rule], state, NormalizedAction(type="click", arguments={"coordinate": [1, 2]})))
        self.assertIs(select_rule([rule], state, NormalizedAction(type="click", arguments={"index": 3})), rule)


class SchemaAliasRepairTests(unittest.TestCase):
    """An induced schema whose aliases collide is repaired (and reported) instead of failing the stage."""

    def test_colliding_aliases_are_dropped_and_reported(self):
        from trace2env.reconstruction import repair_schema_aliases
        from trace2env.validation import action_aliases, state_path_aliases
        from trace2env.models import ActionSchema, ActionSpec, SchemaInductionResult, StateField, StateSchema
        result = SchemaInductionResult(
            action_schema=ActionSchema(actions=[
                ActionSpec(name="click", description="tap", aliases=["tap", "press", "click"]),
                ActionSpec(name="long_press", description="hold", aliases=["press", "hold"]),
                ActionSpec(name="hold", description="hold too", aliases=[]),
            ]),
            state_schema=StateSchema(fields=[
                StateField(path="surface.scroll_state", type="object", default={}, visibility="surface", aliases=["surface.chrome.viewport"]),
                StateField(path="surface.visible_content", type="string", default="", visibility="surface", aliases=["surface.chrome.viewport", "surface.content"]),
                StateField(path="surface.ui_elements", type="array", default=[], visibility="surface", aliases=["surface.ui_elements", "surface.files.action_mode"]),
                StateField(path="surface.files.action_mode", type="string", default="", visibility="surface", aliases=[]),
            ]),
        )
        notes = repair_schema_aliases(result)
        self.assertEqual([a.aliases for a in result.action_schema.actions], [["tap"], [], []])  # 'press' claimed twice, 'hold' is an action
        self.assertEqual([f.aliases for f in result.state_schema.fields], [[], ["surface.content"], [], []])
        self.assertEqual(len(notes), 6)
        self.assertTrue(all(note.startswith("Dropped alias") for note in notes))
        self.assertEqual(action_aliases(result.action_schema)["tap"], "click")  # resolvable again
        self.assertEqual(state_path_aliases(result.state_schema)["surface.content"], "surface.visible_content")


class InductionSafetyTests(unittest.TestCase):
    def test_contrasts_match_arguments_scope_and_independent_episodes(self):
        a = LocalTransitionEvidence(id="a", episode_id="a", transition_id="a", action=NormalizedAction(type="withdraw", arguments={"amount": 25}),
            outcome="success", observation_text="ok", confidence=Confidence(value=1))
        b = a.model_copy(update={"id": "b", "episode_id": "b", "outcome": Outcome.FAILURE})
        self.assertEqual(len(build_contrasts([a, b], {}, {})), 1)
        self.assertEqual(build_contrasts([a, b], {"a": "family", "b": "family"}, {}), [])
        self.assertEqual(build_contrasts([a, b], {}, {"a": {"tenant": "A"}, "b": {"tenant": "B"}}), [])
        b = b.model_copy(update={"action": NormalizedAction(type="withdraw", arguments={"amount": 50})})
        self.assertEqual(build_contrasts([a, b], {}, {}), [])

    def test_recursive_reduction_bounds_every_call(self):
        class Provider:
            def __init__(self): self.calls = []
            def complete(self, **kwargs):
                self.calls.append(kwargs)
                return Confidence(value=1)
        provider = Provider()
        bounded = BoundedInduction(provider, 2500, 2)
        values = [{"id": i, "text": "x" * 200} for i in range(17)]
        bounded.induce(values, "induce", Confidence, "induce", {}, "evidence", "partials")
        self.assertTrue(any(call["role"] == "induce_merge" for call in provider.calls))
        for call in provider.calls:
            self.assertLessEqual(bounded.size(call["system"], json.loads(call["user"]), Confidence), 2500)
        seen = [item["id"] for call in provider.calls if call["role"] == "induce_chunk" for item in json.loads(call["user"])["evidence"]]
        self.assertEqual(seen, list(range(17)))

    def test_oversized_evidence_fails_before_llm_call(self):
        llm = ScriptedLLM({})
        with self.assertRaises(PromptBudgetExceeded):
            BoundedInduction(llm, 1000, 2).induce(["x" * 5000], "s", Confidence, "r", {}, "e", "p")
        self.assertEqual(llm.calls, [])

    def test_demonstrations_retain_failure_branch_and_observed_state(self):
        evidence = []
        for i in range(4):
            ref = SourceRef(source_id="s", episode_id=str(i), event_ids=[str(i)])
            evidence.append(LocalTransitionEvidence(id=str(i), episode_id="success" if i < 3 else "failure", transition_id=str(i),
                action=NormalizedAction(type="withdraw"), outcome="success" if i < 3 else "failure",
                observation_text="OK" if i < 3 else "REJECTED", confidence=Confidence(value=0.99-i*0.05),
                preconditions=[Fact(subject="world.balance", predicate="eq", value=10, provenance=[ref])], provenance=[ref]))
        demos = TraceReconstructor.select_demonstrations(evidence)
        self.assertEqual(len(demos), 3)
        self.assertIn(Outcome.FAILURE, [d.outcome for d in demos])
        self.assertEqual(demos[0].state_before, {"world": {"balance": 10}})


if __name__ == "__main__":
    unittest.main()
