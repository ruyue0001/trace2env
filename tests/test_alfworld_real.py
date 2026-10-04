"""API-free checks of the Word2World-compatible real-ALFWorld runner."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_alfworld_real import load_tasks, original_react_prompt, parse_react_action, run_attempt, summarize


class FakeWrapper:
    def __init__(self):
        self._lock = threading.Lock()
        self.env_init = {}
        self.env = {}
        self.info = {}
        self.ls = []

    def create(self):
        identifier = len(self.ls)
        self.ls.append(identifier)
        self.env_init[identifier] = SimpleNamespace(close=lambda: None)
        return {"id": identifier}

    def reset(self, identifier, game, world_type):
        assert game == 2420 and world_type == "Text"
        return {"observation": "A kitchen. Your task is to take the cup.", "available_actions": ["take cup"]}

    def step(self, identifier, action):
        assert action == "take cup"
        return {"observation": "You win.", "available_actions": [], "reward": 1.0, "done": True}


class FakeCompletions:
    def create(self, *, model, messages):
        assert model == "gpt-5.6-sol"
        assert messages[-1]["content"].endswith("AVAILABLE ACTIONS: take cup")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Thought:\nTake the cup.\n\nAction:\ntake cup"))])


class AlfworldRealTests(unittest.TestCase):
    def test_selection_matches_mapping_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test.json").write_text(json.dumps([{"item_id": "alfworld_2420"},
                                                         {"item_id": "alfworld_2421"}]))
            (root / "mapping.json").write_text(json.dumps([
                {"item_id": 2420, "task_type": "a", "task_id": "x"},
                {"item_id": 2421, "task_type": "b", "task_id": "y"},
            ]))
            rows = load_tasks(root / "test.json", root / "mapping.json", limit=2)
            self.assertEqual([row["item_id"] for row in rows], [2420, 2421])
            second = load_tasks(root / "test.json", root / "mapping.json", limit=1,
                                start_index=1)
            self.assertEqual([row["item_id"] for row in second], [2421])

    def test_original_prompt_and_react_parser(self):
        intro, ack = original_react_prompt()
        self.assertIn("Thought:", intro)
        self.assertIn("available actions", intro)
        self.assertTrue(ack.startswith("OK."))
        self.assertEqual(parse_react_action("Thought:\nGo.\n\nAction:\ngoto desk"), "goto desk")
        self.assertEqual(parse_react_action("no action"), "")

    def test_real_reward_and_pass_at_k(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        task = {"item_id": 2420, "task_type": "test", "task_id": "trial"}
        intro, ack = original_react_prompt()
        record = run_attempt(FakeWrapper(), client, task=task, attempt=1, intro=intro, ack=ack,
                             model="gpt-5.6-sol", max_rounds=50)
        self.assertTrue(record["success"])
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["actions"], ["take cup"])
        summary = summarize([record], [2420], 1)
        self.assertEqual(summary["pass_at_k"], {"1": 1.0})
        self.assertEqual(summary["real_task_success_rate"], 1.0)
        records = [dict(record, item_id=2420, attempt=1, success=False),
                   dict(record, item_id=2420, attempt=2, success=True),
                   dict(record, item_id=2421, attempt=1, success=False),
                   dict(record, item_id=2421, attempt=2, success=False)]
        self.assertEqual(summarize(records, [2420, 2421], 2)["pass_at_k"], {"1": 0.0, "2": 0.5})


if __name__ == "__main__":
    unittest.main()
