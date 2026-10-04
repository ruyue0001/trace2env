"""trace2env_no_state control: the agent loop, knowledge, and history stay; persistent state and state tools go."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import terminal_package, terminal_rows
from trace2env.agentworld import RESPONSE_TAG, parse_model_output
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.llm import ScriptedLLM, StructuredAgentLLM
from trace2env.models import AgentTurn, Outcome, StateMutation, StateTrackingResult, TransitionSubmission


class NoStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package_dir = terminal_package(Path(self.temporary.name))
        self.rows = terminal_rows()

    def test_no_state_keeps_history_and_drops_state_and_effects(self):
        submission = TransitionSubmission(
            outcome=Outcome.SUCCESS, observation="root@host:/app/tests# pwd\n/app/tests\nroot@host:/app/tests#",
            effects=[StateMutation(op="set", path="session.cwd", value="/app/tests")], citations=["memory:2"], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3,
                           "runtime_agent_turn": [AgentTurn(tool="recent_turns", arguments={"limit": 3}), AgentTurn(final=submission)] * 3})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  state_tracking=False)
        outputs = runner.run(self.rows)
        by_turn = {row["turn_idx"]: row for row in outputs}
        self.assertNotIn("track_state", [call["role"] for call in llm.calls])  # nothing tracks state
        self.assertEqual([by_turn[t]["trace2env"]["route"] for t in (1, 2, 3)], ["agent", "agent", "agent"])  # no rule route either
        info = by_turn[3]["trace2env"]
        self.assertFalse(info["state_tracking_enabled"])
        self.assertEqual(info["state_tracking"]["skipped_turns"], 3)  # turn 0, 1, 2 recorded in memory only
        self.assertEqual((info["effects"], info["rule_ids"]), (0, []))
        self.assertIn("state_tracking_disabled", info["verification"])
        self.assertNotIn("model_proposed_effects", info["verification"])
        self.assertEqual(parse_model_output(by_turn[3]["gen"], RESPONSE_TAG), submission.observation)
        call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        self.assertIn("No structured state is tracked", call["system"])
        payload = json.loads(call["user"])
        tools = [tool["name"] for tool in payload["tools"]]
        self.assertTrue({"read_state", "state_schema", "apply_rule", "dry_run"}.isdisjoint(tools))
        self.assertTrue({"recall", "recent_turns", "read_turn", "search_knowledge", "inspect_action"} <= set(tools))
        brief = json.loads(payload["conversation"][0]["content"])
        self.assertNotIn("state_summary", brief)
        self.assertNotIn("state_fields", brief)
        self.assertNotIn("known_files", brief)
        self.assertIn("not tracked", brief["state"])
        self.assertEqual([entry["turn"] for entry in brief["recent_memory"]], [0])  # the initial screen is remembered
        last = [call for call in llm.calls if call["role"] == "runtime_agent_turn"][-2]
        last_brief = json.loads(json.loads(last["user"])["conversation"][0]["content"])
        self.assertEqual([entry["turn"] for entry in last_brief["recent_memory"]], [0, 1, 2])  # observed turns stay in history

    def test_default_runner_still_tracks(self):
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_agent_turn": [AgentTurn(final=TransitionSubmission(
            outcome=Outcome.SUCCESS, observation="x", effects=[], citations=[], rationale=""))] * 3})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm))
        outputs = runner.run(self.rows)  # turn 3 needs the cd turn tracked by the model (no rule explains it)
        self.assertTrue(outputs[0]["trace2env"]["state_tracking_enabled"])
        self.assertIn("track_state", [call["role"] for call in llm.calls])


if __name__ == "__main__":
    unittest.main()


class HarnessOnlyTests(unittest.TestCase):
    def test_agent_sees_nothing_from_the_package(self):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="root@host:/app# pwd\n/app\nroot@host:/app#",
                                          effects=[], citations=["memory:1"], rationale="")
        llm = ScriptedLLM({"runtime_agent_turn": [AgentTurn(tool="recent_turns", arguments={"limit": 2}), AgentTurn(final=submission)] * 3})
        with tempfile.TemporaryDirectory() as directory:
            package_dir = terminal_package(Path(directory))
            runner = AgentWorldRunner(mode="agentic", package_dir=str(package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                      state_tracking=False, package_knowledge=False)
            outputs = runner.run(terminal_rows())
            info = outputs[0]["trace2env"]
            self.assertEqual((info["route"], info["package_knowledge"], info["state_tracking_enabled"]), ("agent", False, False))
            self.assertEqual([row["trace2env"]["route"] for row in outputs], ["agent"] * 3)  # the pwd rule never routes
            call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
            payload = json.loads(call["user"])
            self.assertEqual(sorted(tool["name"] for tool in payload["tools"]), ["read_turn", "recall", "recent_turns"])
            brief = json.loads(payload["conversation"][0]["content"])
            self.assertNotIn("retrieved", brief)
            self.assertNotIn("state_summary", brief)
            self.assertNotIn("similar_turns", brief)
            self.assertIn("none attached", brief["package"])
            self.assertIn("No environment knowledge package", call["system"])
            self.assertEqual(parse_model_output(outputs[2]["gen"], RESPONSE_TAG), submission.observation)


class SchemaOnlyHv3Tests(unittest.TestCase):
    def test_no_knowledge_tools_keeps_schemas_state_and_memory(self):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="root@host:/app/tests# pwd\n/app/tests\nroot@host:/app/tests#",
                                          effects=[], citations=["memory:2"], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3,
                           "runtime_agent_turn": [AgentTurn(tool="read_state", arguments={}), AgentTurn(final=submission)] * 3})
        with tempfile.TemporaryDirectory() as directory:
            package_dir = terminal_package(Path(directory))
            runner = AgentWorldRunner(mode="agentic", package_dir=str(package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                      knowledge_tools=False)
            outputs = runner.run(terminal_rows())
            by_turn = {row["turn_idx"]: row for row in outputs}
            self.assertEqual(by_turn[1]["trace2env"]["route"], "rule")  # the confident pwd rule still routes (schemas + rules kept here)
            info = by_turn[2]["trace2env"]
            self.assertEqual((info["route"], info["knowledge_tools"], info["state_tracking_enabled"]), ("agent", False, True))
            self.assertIn("track_state", [call["role"] for call in llm.calls])  # tracking still runs
            call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
            payload = json.loads(call["user"])
            tools = sorted(tool["name"] for tool in payload["tools"])
            self.assertTrue({"list_actions", "inspect_action", "search_knowledge", "read_evidence"}.isdisjoint(tools))
            self.assertTrue({"read_state", "state_schema", "recall", "recent_turns", "read_turn", "apply_rule", "dry_run"} <= set(tools))
            brief = json.loads(payload["conversation"][0]["content"])
            self.assertIn("state_summary", brief)
            self.assertIn("state_fields", brief)
            self.assertIn("action_specs", brief["retrieved"])
            self.assertNotIn("similar_turns", brief)
            self.assertIn("No knowledge tools", call["system"])
