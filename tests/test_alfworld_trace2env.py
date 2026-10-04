"""Construction split and turn-interface checks for the ALFWorld package run."""

import importlib.util
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("collect_alfworld_trace2env", ROOT / "scripts" / "collect_alfworld_trace2env.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
runner_spec = importlib.util.spec_from_file_location("run_alfworld_trace2env", ROOT / "scripts" / "run_alfworld_trace2env.py")
runner = importlib.util.module_from_spec(runner_spec)
runner_spec.loader.exec_module(runner)


class AlfworldConstructionTests(TestCase):
    def test_current_turn_memory_survives_reload_and_rejects_mismatch(self):
        from trace2env.memory import EpisodicMemory
        from trace2env.models import EnvironmentState, MemoryEntry, NormalizedAction

        with TemporaryDirectory() as directory:
            database = Path(directory) / "session.sqlite"
            memory = EpisodicMemory(database)
            memory.record(MemoryEntry(turn=0, kind="observed", action=NormalizedAction(type="session.start"),
                                      observation="Initial room and goal"))
            memory.record(MemoryEntry(turn=1, kind="predicted", action=NormalizedAction(type="look"),
                                      observation="You see a desk."))
            reloaded = EpisodicMemory(database)
            harness = SimpleNamespace(session=SimpleNamespace(load=lambda: EnvironmentState(step=1)),
                                      memory=reloaded)
            runner.assert_memory_contiguous(harness, expected_observation="You see a desk.")
            with self.assertRaisesRegex(RuntimeError, "differs from the committed observation"):
                runner.assert_memory_contiguous(harness, expected_observation="A different room")

    def test_selection_excludes_evaluation_signatures_and_duplicates(self):
        evaluation = [{"item_id": item_id, "task_type": f"pick_and_place_simple-Obj{item_id}-None-Desk-1",
                       "task_id": f"eval_{item_id}"} for item_id in range(2420, 2520)]
        construction = [{"item_id": item_id, "task_type": f"pick_and_place_simple-Obj{item_id}-None-Desk-1",
                         "task_id": f"train_{item_id}"} for item_id in range(2520, 2620)]
        construction[0]["task_type"] = evaluation[0]["task_type"]
        construction[2]["task_type"] = construction[1]["task_type"]
        with patch.object(collector, "load_tasks", side_effect=[evaluation, construction]), \
                patch.object(collector, "file_sha256", side_effect=lambda path: str(path)):
            selection = collector.choose_candidates(Path("test.json"), Path("data"))
        ids = {row["item_id"] for row in selection["candidate_order"]}
        self.assertEqual(len(ids), 98)
        self.assertNotIn(2520, ids)
        self.assertNotIn(2522, ids)
        self.assertEqual(len({tuple(row["signature"]) for row in selection["candidate_order"]}), 98)

    def test_success_trace_export_preserves_initial_and_per_turn_observations(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "attempts").mkdir()
            collector.write_json(output / "selection.json", {"frozen": True})
            collector.write_json(output / "collection_summary.json", {
                "complete": True, "successes": 1, "success_ids": [2520]})
            record = {"item_id": 2520, "status": "done", "success": True,
                      "actions": ["go to bed 1", "put pillow 1 on armchair 1",
                                  "move pillow 1 to armchair 1"],
                      "conversation": [
                          {"role": "user", "content": "intro"},
                          {"role": "assistant", "content": "ack"},
                          {"role": "user", "content": "initial task and room"},
                          {"role": "assistant", "content": "Action: go to bed 1"},
                          {"role": "user", "content": "You arrive at bed 1."},
                          {"role": "assistant", "content": "Action: put pillow 1 on armchair 1"},
                          {"role": "user", "content": "Nothing happens."},
                          {"role": "assistant", "content": "Action: move pillow 1 to armchair 1"},
                          {"role": "user", "content": "You move the pillow 1 to armchair 1."},
                      ]}
            collector.write_json(output / "attempts" / "alfworld_2520.json", record)
            selection = {"candidate_order": [{"item_id": 2520, "signature": ["place", "Pillow", "ArmChair"]}],
                         "evaluation_ids": list(range(2420, 2520))}
            result = collector.export_traces(selection, output, 1)
            trace = json.loads((output / result["traces"][0]["trace"]).read_text())
            self.assertEqual(len(trace["events"]), 7)
            self.assertEqual(trace["events"][0]["content"], "initial task and room")
            self.assertEqual(trace["events"][1]["content"]["type"], "go_to")
            self.assertEqual(trace["events"][3]["content"]["type"], "invalid_command")
            self.assertEqual(trace["events"][4]["content"], "Nothing happens.")
            self.assertEqual(trace["events"][5]["content"]["type"], "move")


if __name__ == "__main__":
    import unittest
    unittest.main()
