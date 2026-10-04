"""Focused checks for full-history prompting in the ALFWorld WM baseline."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "run_alfworld_prompting_wm.py"
spec = importlib.util.spec_from_file_location("run_alfworld_prompting_wm", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ScriptedCompletions:
    def __init__(self):
        self.calls = []

    def create(self, *, model, messages):
        self.calls.append((model, [dict(row) for row in messages]))
        outputs = [
            "Thought: inspect first.\nAction: look",
            "You see a closed fridge.",
            "Thought: inspect again.\nAction: open fridge 1",
            "The fridge opens. [SUCCESS]",
        ]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outputs[len(self.calls) - 1]))])


class PromptingWMTests(unittest.TestCase):
    def test_world_model_prompt_uses_exact_alfworld_command_forms(self):
        prompt = runner.WM_INSTRUCTION.format(
            command_reference=runner.COMMAND_REFERENCE,
            environment_description="On the bed, a pillow.",
            initial_observation="Put a pillow on armchair.")
        self.assertIn("`move <object> to <receptacle>`", prompt)
        self.assertIn("`use <object>`", prompt)
        self.assertIn("`Nothing happens.`", prompt)
        self.assertIn("On the bed, a pillow.", prompt)

    def test_command_reference_rejects_mismatched_grammar(self):
        with TemporaryDirectory() as directory:
            grammar = Path(directory) / "alfred.twl2"
            actual = (Path(__file__).resolve().parents[1] / "baseline" / "Word2World" /
                      "scripts" / "download_data" / "alfred.twl2")
            grammar.write_text(actual.read_text(encoding="utf-8"), encoding="utf-8")
            runner.verify_command_reference(grammar)
            grammar.write_text('template :: "put {o} on {r}";\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_command_reference(grammar)

    def test_every_wm_request_contains_prior_action_and_observation(self):
        calls = ScriptedCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=calls))
        initial = [{"role": "user", "content": "intro"},
                   {"role": "assistant", "content": "ack"},
                   {"role": "user", "content": "initial observation"}]
        context = {"task": {"item_id": 1, "task_id": "task", "task_type": "type"},
                   "agent_initial": initial,
                   "wm_initial": [{"role": "system", "content": "world context"}]}
        record = runner.run_task(context, client, agent_model="agent", wm_model="wm", max_steps=5)

        self.assertEqual(record["status"], "wm_success")
        self.assertEqual(record["steps"], 2)
        self.assertEqual(calls.calls[1][1], [
            {"role": "system", "content": "world context"},
            {"role": "user", "content": "look"},
        ])
        self.assertEqual(calls.calls[3][1], [
            {"role": "system", "content": "world context"},
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": "You see a closed fridge."},
            {"role": "user", "content": "open fridge 1"},
        ])
        self.assertEqual(record["wm_history"][-1]["content"], "The fridge opens.")
        runner.audit_history(record)


if __name__ == "__main__":
    unittest.main()
