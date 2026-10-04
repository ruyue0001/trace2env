"""Closed-loop rollouts and evaluation-only EnvScaler scoring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from trace2env.cli import build_parser
from trace2env.examples import build_ledger_demo
from trace2env.llm import ScriptedLLM
from trace2env.long_horizon import LongHorizonRunner, aggregate_pass_at_k, evaluate_envscaler_attempt
from trace2env.models import EnvironmentState
from trace2env.runtime import RuntimeHarness
from trace2env.storage import read_json, read_jsonl


METADATA = {
    "env_id": "tiny",
    "env_class_name": "CounterEnvironment",
    "env_class_code": (
        "class CounterEnvironment:\n"
        "    def __init__(self, config):\n"
        "        self.counter = config.get('counter', 0)\n"
        "    def increment(self, amount):\n"
        "        self.counter += amount\n"
        "        return f'counter={self.counter}'\n"
    ),
    "tools": [{"function": {"name": "increment", "description": "Increment the counter.",
                             "parameters": {"type": "object", "properties": {"amount": {"type": "integer"}},
                                            "required": ["amount"]}}}],
}
SCENARIO = {"env_id": "tiny", "task_id": "tiny-1", "task": "Make counter equal 2.",
            "init_config": {"counter": 0},
            "checklist_with_func": [{"check_item": "counter is 2",
                                     "check_func": "def check_func(final_state):\n    return final_state['counter'] == 2\n"}]}


class FakeHarness:
    instances = []

    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.steps = []
        self.instances.append(self)

    def step(self, request):
        self.steps.append(request)
        return SimpleNamespace(observation=f"simulated {request.action.arguments['amount']}", route="agent")


class LongHorizonTests(unittest.TestCase):
    def test_oracle_replay_checks_final_state_and_catches_errors(self):
        passed = evaluate_envscaler_attempt(SCENARIO, [{"name": "increment", "arguments": {"amount": 2}}], METADATA)
        self.assertTrue(passed["success"])
        self.assertEqual(passed["checklist_passed"], 1)
        failed = evaluate_envscaler_attempt(SCENARIO, [{"name": "increment", "arguments": {"amount": 1}}], METADATA)
        self.assertFalse(failed["success"])
        invalid = evaluate_envscaler_attempt(SCENARIO, [{"name": "no_such_tool", "arguments": {}}], METADATA)
        self.assertFalse(invalid["success"])
        self.assertIn("AttributeError", invalid["oracle_errors"][0]["error"])
        recovered = evaluate_envscaler_attempt(SCENARIO, [
            {"name": "no_such_tool", "arguments": {}},
            {"name": "increment", "arguments": {"amount": 2}},
        ], METADATA)
        self.assertTrue(recovered["success"])

    def test_fresh_attempts_task_agent_only_sees_simulated_observations(self):
        FakeHarness.instances = []
        task_llm = ScriptedLLM({"task_agent": [
            {"kind": "action", "name": "increment", "arguments": {"amount": 1}},
            {"kind": "finish"},
            {"kind": "action", "name": "increment", "arguments": {"amount": 2}},
            {"kind": "finish"},
        ]})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            runner = LongHorizonRunner(package_dir="unused", metadata=METADATA, task_llm=task_llm,
                world_agent_llm=None, output_dir=output, attempts_per_task=2,
                harness_factory=FakeHarness)
            summary = runner.run([SCENARIO])
            self.assertEqual(summary["pass_at_k"], {"1": 0.0, "2": 1.0})
            self.assertEqual(summary["attempt_success_rate"], 0.5)
            self.assertEqual(len(FakeHarness.instances), 2)
            self.assertNotEqual(FakeHarness.instances[0].session_dir, FakeHarness.instances[1].session_dir)
            self.assertIn("simulated 1", task_llm.calls[1]["user"])
            self.assertNotIn("check_func", task_llm.calls[1]["user"])
            self.assertNotIn("counter=1", task_llm.calls[1]["user"])
            self.assertEqual(len(list((output / "attempts").glob("*.json"))), 2)
            with self.assertRaises(FileExistsError):
                runner.run([SCENARIO])

    def test_invalid_tool_call_can_be_corrected(self):
        FakeHarness.instances = []
        task_llm = ScriptedLLM({"task_agent": [
            {"kind": "action", "name": "no_such_tool", "arguments": {}},
            {"kind": "action", "name": "increment", "arguments": {"amount": 2}},
            {"kind": "finish"},
        ]})
        with tempfile.TemporaryDirectory() as temporary:
            runner = LongHorizonRunner(package_dir="unused", metadata=METADATA, task_llm=task_llm,
                world_agent_llm=None, output_dir=Path(temporary) / "run", harness_factory=FakeHarness)
            summary = runner.run([SCENARIO])
            self.assertEqual(summary["pass_at_k"]["1"], 1.0)
            self.assertEqual(len(FakeHarness.instances[0].steps), 1)
            self.assertIn("Unknown tool", task_llm.calls[1]["user"])

    def test_two_actions_share_one_world_model_session(self):
        FakeHarness.instances = []
        task_llm = ScriptedLLM({"task_agent": [
            {"kind": "action", "name": "increment", "arguments": {"amount": 1}},
            {"kind": "action", "name": "increment", "arguments": {"amount": 1}},
            {"kind": "finish"},
        ]})
        with tempfile.TemporaryDirectory() as temporary:
            runner = LongHorizonRunner(package_dir="unused", metadata=METADATA, task_llm=task_llm,
                world_agent_llm=None, output_dir=Path(temporary) / "run", harness_factory=FakeHarness)
            summary = runner.run([SCENARIO])
            self.assertEqual(summary["pass_at_k"]["1"], 1.0)
            self.assertEqual(len(FakeHarness.instances), 1)
            self.assertEqual(len(FakeHarness.instances[0].steps), 2)
            self.assertIn("simulated 1", task_llm.calls[2]["user"])

    def test_real_harness_commits_state_and_memory_across_actions(self):
        task_llm = ScriptedLLM({"task_agent": [
            {"kind": "action", "name": "deposit", "arguments": {"amount": 1}},
            {"kind": "action", "name": "deposit", "arguments": {"amount": 2}},
            {"kind": "finish"},
        ]})
        metadata = {"env_id": "ledger", "tools": [{"function": {"name": "deposit",
                    "description": "Add funds.", "parameters": {"type": "object", "properties": {
                        "amount": {"type": "number"}}, "required": ["amount"]}}}]}
        scenario = {"task_id": "ledger-1", "task": "Deposit 1, then 2."}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = build_ledger_demo(root / "package")
            runner = LongHorizonRunner(package_dir=package, metadata=metadata, task_llm=task_llm,
                world_agent_llm=None, output_dir=root / "run",
                harness_factory=lambda path: RuntimeHarness(package, path,
                    initial_state=EnvironmentState(world={"balance": 100})),
                evaluator=lambda _scenario, actions, _metadata: {"success": len(actions) == 2})
            summary = runner.run([scenario])
            self.assertEqual(summary["pass_at_k"]["1"], 1.0)
            attempt = read_json(next((root / "run" / "attempts").glob("*.json")))
            session = Path(attempt["session_dir"])
            self.assertEqual(read_json(session / "state.json")["world"]["balance"], 103)
            self.assertEqual(len(read_jsonl(session / "audit.jsonl")), 2)
            self.assertIn("Balance: 101", task_llm.calls[1]["user"])

    def test_pass_at_k_denominator_and_cli(self):
        records = [{"task_id": "a", "success": False}, {"task_id": "a", "success": True},
                   {"task_id": "b", "success": False}, {"task_id": "b", "success": False}]
        result = aggregate_pass_at_k(records, ["a", "b"], 2)
        self.assertEqual(result["pass_at_k"], {"1": 0.0, "2": 0.5})
        args = build_parser().parse_args(["envscaler-long-horizon", "package", "--env-defs", "defs",
                                          "--env-id", "env_160_rl", "--task-id", "env_160_rl-task_1",
                                          "--output", "run", "--attempts", "3"])
        self.assertEqual(args.attempts, 3)


if __name__ == "__main__":
    unittest.main()
