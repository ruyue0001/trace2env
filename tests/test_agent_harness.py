"""World-model agent harness: episodic memory, workspace tools, the tool loop, routing policies, and notes."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from trace2env.agent import AgentLoop, AgentProtocolError
from trace2env.compiler import EnvironmentCompiler
from trace2env.examples import build_ledger_demo
from trace2env.llm import ScriptedLLM, ScriptedToolLLM, StructuredAgentLLM
from trace2env.memory import EpisodicMemory
from trace2env.models import (
    AgentReply,
    AgentTurn,
    ArtifactEdit,
    EnvironmentNote,
    EnvironmentState,
    MemoryEntry,
    NormalizedAction,
    Outcome,
    PackagePatch,
    ReconstructionArtifacts,
    ReconstructionConfig,
    StateMutation,
    StepRequest,
    ToolCall,
    TransitionSubmission,
)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.patching import apply_package_patch
from trace2env.runtime import RuntimeHarness
from trace2env.storage import read_json
from trace2env.validation import artifact_digest
from trace2env.workspace import WorkspaceTools, state_diff


def reply(name: str, **arguments) -> AgentReply:
    return AgentReply(tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)])


def submit(**kwargs) -> AgentReply:
    return reply("submit_transition", **TransitionSubmission(**kwargs).model_dump(mode="json"))


WITHDRAW = NormalizedAction(type="withdraw", arguments={"amount": 25})
DECREMENT = [StateMutation(op="decrement", path="world.balance", value=25)]


class MemoryTests(unittest.TestCase):
    def test_record_recent_and_search(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EpisodicMemory(Path(directory) / "session.sqlite")
            first = memory.record(MemoryEntry(turn=1, kind="observed", action=NormalizedAction(type="ls", arguments={"argv": ["-la"]}),
                                              observation="total 16\nmaze_game.sh\ntests"))
            memory.record(MemoryEntry(turn=2, kind="predicted", action=NormalizedAction(type="cat", arguments={"argv": ["README.md"]}),
                                      observation="# Maze"))
            self.assertEqual(first.id, 1)
            self.assertEqual([entry.turn for entry in memory.recent(5)], [1, 2])
            self.assertEqual([entry.id for entry in memory.search("maze_game.sh")], [1])
            self.assertEqual([entry.action.type for entry in memory.search("cat README.md")], ["cat"])
            self.assertEqual(memory.search("", 5), [])
            self.assertEqual(memory.count(), 2)
            self.assertEqual(EpisodicMemory(Path(directory) / "session.sqlite").count(), 2)  # reopening keeps entries


class WorkspaceToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = EnvironmentPackage(build_ledger_demo(self.root / "package"))
        self.memory = EpisodicMemory(self.root / "session.sqlite")
        self.tools = WorkspaceTools(self.package, PackageInspector(self.package), self.memory,
                                    EnvironmentState(world={"balance": 100.0}), WITHDRAW)

    def test_templates_withheld_when_the_official_input_documents_the_format(self):
        # Option B: rule templates are not shown or rendered when the official input already documents the format.
        withheld = WorkspaceTools(self.package, PackageInspector(self.package), self.memory,
                                  EnvironmentState(world={"balance": 100.0}), WITHDRAW, withhold_templates=True)
        rules = withheld.call("inspect_action", {"action_type": "withdraw"})["rules"]
        self.assertTrue(rules)
        self.assertTrue(all(item["observation_template"] is None and item.get("template_withheld") for item in rules))
        applied = withheld.call("apply_rule", {"rule_id": rules[0]["id"]})
        self.assertTrue(applied["applicable"])
        self.assertIsNone(applied["observation"])
        self.assertIn("observation_withheld", applied)
        shown = self.tools.call("apply_rule", {"rule_id": rules[0]["id"]})
        self.assertIsNotNone(shown["observation"])

    def test_knowledge_tools_register_what_they_expose(self):
        names = [tool.name for tool in self.tools.tools()]
        self.assertEqual(names[-1], "submit_transition")
        self.assertIn("apply_rule", names)
        self.assertIn("withdraw", [item["name"] for item in self.tools.call("list_actions", {})])
        self.assertIn("schema:actions", self.tools.retrieved_ids)
        inspected = self.tools.call("inspect_action", {"action_type": "withdraw"})
        self.assertEqual({rule["id"] for rule in inspected["rules"]}, {"withdraw_success", "withdraw_insufficient"})
        self.assertIn("balance_format", [note["id"] for note in inspected["notes"]])
        self.assertLessEqual({"rule:withdraw_success", "note:balance_format", "invariant:nonnegative_balance"}, self.tools.retrieved_ids)
        found = self.tools.call("search_knowledge", {"query": "fractional digit"})
        self.assertIn("note:balance_format", [item["id"] for item in found])
        rules = self.tools.call("search_knowledge", {"query": "withdrawal", "kinds": ["rule"]})
        self.assertEqual({item["kind"] for item in rules}, {"rule"})
        self.assertIn("rule:withdraw_success", [item["id"] for item in rules])
        self.assertIn("withdraw_success", self.tools.discovered_rules)
        unknown = self.tools.call("nonexistent", {})
        self.assertIn("error", unknown)
        self.assertEqual(self.tools.calls[-1]["tool"], "nonexistent")

    def test_state_and_dry_run_tools_never_commit(self):
        self.assertEqual(self.tools.call("read_state", {"path": "world.balance"})["value"], 100.0)
        self.assertIn("state:world.balance", self.tools.retrieved_ids)
        self.assertEqual(self.tools.call("read_state", {})["world"]["balance"], 100.0)
        self.assertTrue(self.tools.call("read_state", {"path": "world.nothing"})["missing"])
        applied = self.tools.call("apply_rule", {"rule_id": "withdraw_success"})
        self.assertTrue(applied["applicable"])
        self.assertEqual(applied["effects"], [{"op": "decrement", "path": "world.balance", "value": 25}])
        self.assertEqual(applied["observation"], "Withdrew 25. Balance: 75.0")
        self.assertEqual(applied["state_diff"], {"world.balance": {"before": 100.0, "after": 75.0}})
        self.assertFalse(self.tools.call("apply_rule", {"rule_id": "withdraw_insufficient"})["applicable"])
        self.assertFalse(self.tools.call("apply_rule", {"rule_id": "deposit_success"})["applicable"])
        self.assertIn("error", self.tools.call("apply_rule", {"rule_id": "ghost"}))
        dry = self.tools.call("dry_run", {"effects": [{"op": "set", "path": "world.balance", "value": -5}]})
        self.assertEqual((dry["ok"], dry["invariants_ok"]), (True, False))
        self.assertFalse(self.tools.call("dry_run", {"effects": [{"op": "set", "path": "world.nope", "value": 1}]})["ok"])
        self.assertEqual([field["path"] for field in self.tools.call("state_schema", {"prefix": "world"})], ["world.balance"])
        self.assertEqual(self.tools.state.world["balance"], 100.0)
        self.assertEqual(state_diff(EnvironmentState(), EnvironmentState(world={"x": 1})), {"world.x": {"before": "<absent>", "after": 1}})

    def test_memory_tools(self):
        self.memory.record(MemoryEntry(turn=1, kind="observed", action=NormalizedAction(type="balance"), observation="Balance: 100.0"))
        recalled = self.tools.call("recall", {"query": "Balance"})
        self.assertEqual((recalled[0]["id"], recalled[0]["turn"]), ("memory:1", 1))
        self.assertIn("memory:1", self.tools.retrieved_ids)
        self.assertEqual(len(self.tools.call("recent_turns", {"limit": 3})), 1)
        self.assertEqual(WorkspaceTools(self.package, PackageInspector(self.package), None, EnvironmentState(), WITHDRAW)
                         .call("recall", {"query": "x"}), [])


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = EnvironmentPackage(build_ledger_demo(self.root / "package"))

    def tools(self) -> WorkspaceTools:
        return WorkspaceTools(self.package, PackageInspector(self.package), None, EnvironmentState(world={"balance": 100.0}), WITHDRAW)

    def test_tool_results_flow_back_and_submission_ends_the_loop(self):
        llm = ScriptedToolLLM([
            reply("apply_rule", rule_id="withdraw_success"),
            submit(outcome="success", observation="Withdrew 25. Balance: 75.0", rule_ids=["withdraw_success"],
                   effects=[effect.model_dump(mode="json") for effect in DECREMENT], citations=["rule:withdraw_success"]),
        ])
        submission, trace, usage = AgentLoop(llm, self.tools(), system="sys", max_tool_calls=4).run("brief")
        self.assertEqual(submission.rule_ids, ["withdraw_success"])
        self.assertEqual([item["tool"] for item in trace], ["apply_rule", "submit_transition"])
        messages = llm.calls[1]["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "tool"])
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "apply_rule")
        self.assertTrue(json.loads(messages[3]["content"])["applicable"])
        self.assertEqual(usage["model_calls"], 2)

    def test_repeated_invalid_submissions_are_an_error_and_recovery_is_allowed(self):
        # Two invalid submissions may be repaired; a third is a protocol error (the runner then falls back), never a spin.
        recovers = ScriptedToolLLM([reply("final"), reply("submit_transition"), submit(outcome="success", observation="ok")])
        submission, trace, usage = AgentLoop(recovers, self.tools(), system="s").run("brief")
        self.assertEqual(submission.observation, "ok")
        self.assertEqual([t.get("error") for t in trace], ["invalid submission", "invalid submission", None])
        self.assertEqual(usage["model_calls"], 3)
        stubborn = ScriptedToolLLM([reply("final"), reply("final"), reply("final"), submit(outcome="success", observation="late")])
        with self.assertRaisesRegex(AgentProtocolError, "invalid transition 3 times"):
            AgentLoop(stubborn, self.tools(), system="s").run("brief")

    def test_round_cap_bounds_every_path(self):
        llm = ScriptedToolLLM([reply("final"), reply("list_actions"), reply("final"), reply("list_actions"), submit(outcome="success", observation="x")])
        with self.assertRaisesRegex(AgentProtocolError, "calls without a valid submission"):
            AgentLoop(llm, self.tools(), system="s", max_tool_calls=8, max_rounds=3).run("brief")
        loop = AgentLoop(ScriptedToolLLM([submit(outcome="success", observation="x")]), self.tools(), system="s", max_tool_calls=8)
        self.assertEqual(loop.max_rounds, 20)

    def test_budget_forces_submission_and_refusal_is_an_error(self):
        llm = ScriptedToolLLM([reply("list_actions"), reply("list_actions"), submit(outcome="failure", observation="x")])
        AgentLoop(llm, self.tools(), system="s", max_tool_calls=2).run("brief")
        self.assertEqual([call["force_tool"] for call in llm.calls], [None, None, "submit_transition"])
        # A refusal of the forced final is nudged once (models that ignore tool_choice usually comply when told), then fails.
        stubborn = ScriptedToolLLM([reply("list_actions"), reply("list_actions"), reply("list_actions")])
        with self.assertRaisesRegex(AgentProtocolError, "budget"):
            AgentLoop(stubborn, self.tools(), system="s", max_tool_calls=1).run("brief")
        self.assertEqual([call["force_tool"] for call in stubborn.calls], [None, "submit_transition", "submit_transition"])
        self.assertIn("budget is exhausted", stubborn.calls[2]["messages"][-1]["content"])
        complies = ScriptedToolLLM([reply("list_actions"), reply("list_actions"), submit(outcome="success", observation="late")])
        submission, trace, _ = AgentLoop(complies, self.tools(), system="s", max_tool_calls=1).run("brief")
        self.assertEqual(submission.observation, "late")
        self.assertEqual(trace[1], {"tool": "list_actions", "error": "refused: tool budget exhausted"})

    def test_final_is_accepted_as_an_alias_of_submit_transition(self):
        llm = ScriptedToolLLM([reply("final", **TransitionSubmission(outcome="success", observation="ok").model_dump(mode="json"))])
        submission, trace, _ = AgentLoop(llm, self.tools(), system="s", max_tool_calls=2).run("brief")
        self.assertEqual((submission.observation, trace[0]["tool"]), ("ok", "submit_transition"))

    def test_plain_text_is_nudged_once(self):
        llm = ScriptedToolLLM([AgentReply(content="thinking out loud"), submit(outcome="success", observation="ok")])
        submission, _, _ = AgentLoop(llm, self.tools(), system="s", max_tool_calls=2).run("brief")
        self.assertEqual(submission.observation, "ok")
        self.assertIn("submit_transition", llm.calls[1]["messages"][-1]["content"])
        with self.assertRaises(AgentProtocolError):
            AgentLoop(ScriptedToolLLM([AgentReply(content="a"), AgentReply(content="b")]), self.tools(), system="s").run("brief")

    def test_invalid_submission_is_reported_and_retried(self):
        llm = ScriptedToolLLM([reply("submit_transition", outcome="nonsense"), submit(outcome="success", observation="ok")])
        submission, trace, _ = AgentLoop(llm, self.tools(), system="s", max_tool_calls=2).run("brief")
        self.assertEqual((submission.observation, trace[0]["error"]), ("ok", "invalid submission"))
        last = llm.calls[1]["messages"][-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("Invalid submission", last["content"])

    def test_structured_adapter_maps_typed_turns_to_tool_calls(self):
        inner = ScriptedLLM({"runtime_agent_turn": [
            AgentTurn(tool="read_state", arguments={"path": "world.balance"}, reason="check"),
            TransitionSubmission(outcome=Outcome.SUCCESS, observation="ok"),  # the forced final asks for the submission schema
        ]})
        adapter = StructuredAgentLLM(inner)
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        first = adapter.respond(messages=messages, tools=self.tools().specs(), role="x")
        self.assertEqual((first.tool_calls[0].name, first.tool_calls[0].arguments, first.content),
                         ("read_state", {"path": "world.balance"}, "check"))
        payload = json.loads(inner.calls[0]["user"])
        self.assertNotIn("submit_transition", [tool["name"] for tool in payload["tools"]])
        self.assertEqual(payload["conversation"], [{"role": "user", "content": "U"}])
        self.assertTrue(inner.calls[0]["system"].startswith("S"))
        second = adapter.respond(messages=messages, tools=self.tools().specs(), role="x", force_tool="submit_transition")
        self.assertEqual((second.tool_calls[0].name, second.tool_calls[0].arguments["observation"]), ("submit_transition", "ok"))
        self.assertIn("transition submission itself", json.loads(inner.calls[1]["user"])["instruction"])
        self.assertIn("TransitionSubmission", inner.calls[1]["system"])


class HarnessRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = build_ledger_demo(self.root / "package")
        self.request = StepRequest(action=WITHDRAW)

    def test_confident_fast_path_skips_the_agent_for_templated_rules(self):
        llm = ScriptedToolLLM([])
        result = RuntimeHarness(self.package_dir, self.root / "fast", agent_llm=llm).step(self.request)
        self.assertEqual((result.route, result.tool_calls, result.observation), ("rule", 0, "Withdrew 25. Balance: 75.0"))
        self.assertEqual(llm.calls, [])

    def test_agent_route_applies_rules_as_tools_with_citations(self):
        llm = ScriptedToolLLM([
            reply("apply_rule", rule_id="withdraw_success"),
            submit(outcome="success", observation="the template wins over this text", rule_ids=["withdraw_success"],
                   effects=[effect.model_dump(mode="json") for effect in DECREMENT],
                   citations=["rule:withdraw_success", "note:balance_format", "evidence:ghost"]),
        ])
        harness = RuntimeHarness(self.package_dir, self.root / "agent", agent_llm=llm, fast_path="never")
        result = harness.step(self.request)
        self.assertEqual((result.route, result.tool_calls, result.observation), ("agent", 1, "Withdrew 25. Balance: 75.0"))
        self.assertEqual(result.state.world["balance"], 75.0)
        codes = [issue.code for issue in result.verification.issues]
        self.assertIn("uncited_reference", codes)  # evidence:ghost was never retrieved
        self.assertTrue(result.verification.accepted)
        self.assertIn("rule:withdraw_success", result.citations)
        audit = harness.session.audit_records()[-1]
        self.assertEqual((audit.route, [call["tool"] for call in audit.tool_calls]), ("agent", ["apply_rule", "submit_transition", "_usage"]))
        remembered = harness.memory.recent(1)[0]
        self.assertEqual((remembered.kind, remembered.observation, remembered.citations[0]),
                         ("predicted", "Withdrew 25. Balance: 75.0", "rule:withdraw_success"))
        first_message = llm.calls[0]["messages"][0]["content"]
        self.assertIn("You are the environment", first_message)
        self.assertIn("rules_only", first_message)

    def test_untemplated_rules_go_to_the_agent_under_the_confident_policy(self):
        artifacts = ReconstructionArtifacts.model_validate(read_json(self.package_dir / "construction" / "artifacts.json"))
        config = ReconstructionConfig.model_validate(read_json(self.package_dir / "construction" / "config.json"))
        for rule in artifacts.rules:
            rule.observation_template = None
        package = EnvironmentCompiler(config).compile(artifacts, self.root / "untemplated")
        llm = ScriptedToolLLM([submit(outcome="success", observation="Took 25; 75.0 left.", rule_ids=["withdraw_success"],
                                      effects=[effect.model_dump(mode="json") for effect in DECREMENT])])
        result = RuntimeHarness(package, self.root / "u", agent_llm=llm).step(self.request)
        self.assertEqual((result.route, result.observation, result.state.world["balance"]), ("agent", "Took 25; 75.0 left.", 75.0))
        self.assertEqual(RuntimeHarness(package, self.root / "always", agent_llm=ScriptedToolLLM([]), fast_path="always")
                         .step(self.request).route, "rule")

    def test_agent_effects_require_rule_support_or_schema_checked_trust(self):
        balance = StepRequest(action=NormalizedAction(type="balance"))
        llm = ScriptedToolLLM([submit(outcome="success", observation="Balance: 1.0",
                                      effects=[{"op": "set", "path": "world.balance", "value": 1.0}])])
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            RuntimeHarness(self.package_dir, self.root / "strict", agent_llm=llm, fast_path="never").step(balance)
        llm = ScriptedToolLLM([submit(outcome="success", observation="Balance: 1.0",
                                      effects=[{"op": "set", "path": "world.balance", "value": 1.0}])])
        result = RuntimeHarness(self.package_dir, self.root / "graded", agent_llm=llm, fast_path="never",
                                trust_policy="schema_checked").step(balance)
        self.assertIn("model_proposed_effects", [issue.code for issue in result.verification.issues])
        self.assertEqual((result.observation, result.state.world["balance"]), ("Balance: 1.0", 1.0))

    def test_preexisting_invariant_violations_do_not_block_a_transition(self):
        # The ledger invariant (balance >= 0) already fails in this state; the agent's step does not touch balance.
        broken = EnvironmentState(world={"balance": -5.0})
        llm = ScriptedToolLLM([submit(outcome="failure", observation="Insufficient funds. Balance: -5.0")])
        harness = RuntimeHarness(self.package_dir, self.root / "pre", agent_llm=llm, fast_path="never", initial_state=broken)
        result = harness.step(StepRequest(action=NormalizedAction(type="balance")))
        codes = [issue.code for issue in result.verification.issues]
        self.assertIn("invariant_preexisting:nonnegative_balance", codes)
        self.assertTrue(result.verification.accepted)
        # A step that introduces the violation is still rejected.
        llm = ScriptedToolLLM([submit(outcome="success", observation="x", effects=[{"op": "set", "path": "world.balance", "value": -1.0}])])
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            RuntimeHarness(self.package_dir, self.root / "intro", agent_llm=llm, fast_path="never",
                           trust_policy="schema_checked").step(StepRequest(action=NormalizedAction(type="balance")))

    def test_brief_carries_state_memory_retrieval_and_environment_prompt(self):
        llm = ScriptedToolLLM([submit(outcome="success", observation="Balance: 100.0", citations=["memory:1"])])
        harness = RuntimeHarness(self.package_dir, self.root / "brief", agent_llm=llm, fast_path="never")
        harness.memory.record(MemoryEntry(turn=0, kind="observed", action=NormalizedAction(type="balance"), observation="Balance: 100.0"))
        result = harness.step(StepRequest(action=NormalizedAction(type="balance"),
                                          metadata={"environment_prompt": "You are a ledger.", "history_window": 3}))
        self.assertNotIn("uncited_reference", [issue.code for issue in result.verification.issues])
        system, user = (message["content"] for message in llm.calls[0]["messages"][:2])
        self.assertIn("You are a ledger.", system)
        brief = json.loads(user)
        self.assertEqual(brief["recent_memory"][0]["id"], "memory:1")
        self.assertEqual(brief["state_summary"]["world"]["balance"], 100.0)
        self.assertEqual(brief["retrieved"]["applicable_rule"], "balance_read")
        self.assertEqual(brief["budget"], {"tool_calls": 8})
        self.assertIn("note:balance_format", result.retrieved_artifacts)


class NotesTests(unittest.TestCase):
    def test_notes_are_compiled_indexed_patched_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = build_ledger_demo(root / "package")
            package = EnvironmentPackage(package_dir)
            self.assertEqual({note.id for note in package.notes}, {"balance_format", "single_account"})
            inspector = PackageInspector(package)
            self.assertEqual({note["id"] for note in inspector.inspect_action("deposit")["notes"]}, {"balance_format", "single_account"})
            self.assertIn("note:single_account", [item["id"] for item in inspector.search_knowledge("exactly one account")])
            self.assertEqual(inspector.search_knowledge("   "), [])
            self.assertEqual(inspector.summary()["note_count"], 2)
            artifacts = ReconstructionArtifacts.model_validate(read_json(package_dir / "construction" / "artifacts.json"))
            config = ReconstructionConfig.model_validate(read_json(package_dir / "construction" / "config.json"))
            patch = PackagePatch(base_artifact_digest=artifact_digest(artifacts), rationale="document fees", failed_case_ids=["x"],
                                 edits=[ArtifactEdit(collection="notes", key="fees", operation="add",
                                                     value={"id": "fees", "kind": "fact", "statement": "No fees apply."})])
            self.assertIn("fees", [note.id for note in apply_package_patch(artifacts, patch).notes])
            broken = artifacts.model_copy(deep=True)
            broken.notes.append(EnvironmentNote(id="x", statement="s", action_types=["transfer"]))
            with self.assertRaisesRegex(ValueError, "unknown action"):
                EnvironmentCompiler(config).compile(broken, root / "broken")


if __name__ == "__main__":
    unittest.main()


class InvalidSubmissionMessageTests(AgentLoopTests):
    def test_invalid_submission_error_does_not_echo_the_input(self):
        """A malformed submission (e.g. a truncated JSON payload from a native tool call) must not be echoed back into
        the conversation: with the raw input in the error message a 16k-token payload overflowed a 262k context."""
        huge = "x" * 200_000
        llm = ScriptedToolLLM([
            reply("submit_transition", _raw=huge),
            submit(outcome="success", observation="ok", rule_ids=[], effects=[], citations=[]),
        ])
        submission, trace, usage = AgentLoop(llm, self.tools(), system="sys", max_tool_calls=4).run("brief")
        self.assertEqual(submission.observation, "ok")
        error_message = llm.calls[1]["messages"][-1]["content"]
        self.assertIn("Invalid submission", error_message)
        self.assertLess(len(error_message), 2000)


class EffectRejectionPolicyTests(HarnessRoutingTests):
    def test_keep_observation_drops_rejected_effects_and_keeps_the_prediction(self):
        balance = StepRequest(action=NormalizedAction(type="balance"))
        llm = ScriptedToolLLM([submit(outcome="success", observation="Balance: 1.0",
                                      effects=[{"op": "set", "path": "world.balance", "value": 1.0}])])
        harness = RuntimeHarness(self.package_dir, self.root / "keep", agent_llm=llm, fast_path="never",
                                 effect_rejection="keep_observation")
        result = harness.step(balance)  # rules_only would have rejected the unsupported effect
        self.assertEqual((result.route, result.observation, result.plan.effects), ("agent", "Balance: 1.0", []))
        self.assertEqual(result.state.world["balance"], 100.0)  # the state is untouched
        codes = [issue.code for issue in result.verification.issues]
        self.assertIn("effects_rejected", codes)
        self.assertTrue(result.verification.accepted)
        audit = harness.session.audit_records()[-1]
        self.assertEqual((audit.route, audit.observation, audit.error), ("agent", "Balance: 1.0", None))
        # a malformed effect (remove on a non-object path) is handled the same way
        llm = ScriptedToolLLM([submit(outcome="success", observation="ok", effects=[{"op": "remove", "path": "world.balance", "value": "x"}])])
        result = RuntimeHarness(self.package_dir, self.root / "keep2", agent_llm=llm, fast_path="never",
                                effect_rejection="keep_observation").step(balance)
        self.assertEqual((result.observation, result.plan.effects), ("ok", []))
