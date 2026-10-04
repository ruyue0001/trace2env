"""EnvScaler oracle, probe rows, and exact scoring. The environment and its rollout are synthetic."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from trace2env.agentworld import case_from_row, wrap_prediction
from trace2env.cli import main
from trace2env.envscaler import episode_rows, load_environment, rollout_episode
from trace2env.envscaler_oracle import (EnvironmentOracle, load_metadata, perturbations, probe_rows, replay_check, reproduces,
                                        state_diff, system_prompt_for)
from trace2env.envscaler_scoring import UNPARSED, aggregate, parse_observation, score_row, score_rows, values_equal
from trace2env.storage import read_json, write_json
from tests.test_envscaler import ENV, ITEMS, write_collection

SOURCE = '''
import uuid
from datetime import datetime


class ItemSystem:
    def __init__(self, init_config: dict):
        self.items = {}
        self.init_config = init_config

    def get_item(self, item_id: str) -> dict:
        item = self.items.get(item_id)
        if not item:
            return {"success": False, "error": "Item not found."}
        return {"success": True, "data": item}

    def update_item_status(self, item_id: str, new_status: str) -> dict:
        if item_id not in self.items:
            return {"success": False, "error": "Item not found."}
        if new_status not in ("new", "done", "never-observed"):
            return {"success": False, "error": f"Invalid status: {new_status}"}
        if self.items[item_id]["status"] == new_status:
            return {"success": False, "error": f"Item {item_id} is already {new_status}."}
        self.items[item_id]["status"] = new_status
        return {"success": True, "message": "Status updated."}

    def add_item(self, status: str) -> dict:
        item_id = str(uuid.uuid4())
        self.items[item_id] = {"item_id": item_id, "status": status, "created": datetime.utcnow().isoformat()}
        return {"success": True, "message": "Item added.", "item_id": item_id}
'''


def executable_collection(root: Path) -> Path:
    """The converter's fixture with a real environment class behind it (the recorded steps follow this code)."""
    rollout = write_collection(root, directory="benchmark")
    path = root / "env_defs" / f"{ENV}_metadata.json"
    metadata = read_json(path)
    metadata.update(env_class_name="ItemSystem", env_class_code=SOURCE)
    metadata["tools"].append({"type": "function", "function": {"name": "add_item", "description": "Add an item.",
                                                                "parameters": {"type": "object", "properties": {"status": {"type": "string"}}}}})
    write_json(path, metadata)
    return rollout


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rollout_path = executable_collection(self.root)
        self.rollout = read_json(self.rollout_path)
        self.oracle = EnvironmentOracle(load_metadata(self.root / "env_defs", ENV))

    def test_step_runs_the_environment_source_on_a_copy_of_the_state(self):
        state = {"items": json.loads(json.dumps(ITEMS)), "init_config": {}}
        result = self.oracle.step(state, {"name": "update_item_status", "arguments": {"item_id": "I1", "new_status": "done"}})
        self.assertEqual((result["observation"], result["raised"]), ("{'success': True, 'message': 'Status updated.'}", None))
        self.assertEqual(result["state"]["items"]["I1"]["status"], "done")
        self.assertEqual(state["items"]["I1"]["status"], "new")  # the caller's state is untouched
        self.assertEqual(state_diff(state, result["state"]), {"items": {"I1": {"status": {"changed": {"old": "new", "new": "done"}}}}})
        # A call the environment cannot execute claims no observation: what the agent saw then is the collector's business.
        for action in ({"name": "no_such_tool", "arguments": {}}, {"name": "get_item", "arguments": {"unexpected": 1}}):
            failed = self.oracle.step(state, action)
            self.assertIsNone(failed["observation"])
            self.assertTrue(failed["raised"])

    def test_generated_values_are_seeded_and_the_clock_is_fixed(self):
        state = {"items": {}, "init_config": {}}
        first = EnvironmentOracle(load_metadata(self.root / "env_defs", ENV), seed=7).step(state, {"name": "add_item", "arguments": {"status": "new"}})
        again = EnvironmentOracle(load_metadata(self.root / "env_defs", ENV), seed=7).step(state, {"name": "add_item", "arguments": {"status": "new"}})
        other = EnvironmentOracle(load_metadata(self.root / "env_defs", ENV), seed=8).step(state, {"name": "add_item", "arguments": {"status": "new"}})
        self.assertEqual(first, again)
        self.assertNotEqual(first["observation"], other["observation"])
        record = next(iter(first["state"]["items"].values()))
        self.assertEqual(record["created"], "2026-01-01T12:00:00")

    def test_replay_check_gates_on_reproducing_the_recording(self):
        check = replay_check([self.rollout], self.oracle)
        self.assertEqual(check, {"steps": 3, "observation_identical": 3, "state_identical_up_to_generated_values": 3})  # agent-side and unknown tools skipped
        self.assertTrue(reproduces(check))
        tampered = json.loads(json.dumps(self.rollout))
        tampered["trajectory"][1]["observation"] = "{'success': True, 'message': 'Updated.'}"
        self.assertFalse(reproduces(replay_check([tampered], self.oracle)))
        self.assertFalse(reproduces({}))  # nothing replayed is not a pass

    def test_perturbations_change_one_argument_with_values_from_the_database(self):
        state = self.rollout["trajectory"][1]["state_before"]
        variants = perturbations(state, {"name": "update_item_status", "arguments": {"item_id": "I1", "new_status": "done"}})
        found = {(operator, json.dumps(action["arguments"], sort_keys=True)) for operator, action in variants}
        self.assertIn(("other_record", '{"item_id": "I2", "new_status": "done"}'), found)
        self.assertIn(("missing_record", '{"item_id": "I901", "new_status": "done"}'), found)
        self.assertIn(("empty", '{"item_id": "", "new_status": "done"}'), found)
        # "done" names no record and is not yet a status in this database, so it is not treated as an enum value ...
        self.assertFalse(any(operator == "other_value" for operator, _ in variants))
        # ... but "new" is: it gets the field's other value and an unknown one.
        enum = perturbations(state, {"name": "update_item_status", "arguments": {"item_id": "I1", "new_status": "new"}})
        self.assertIn(("other_value", "never-observed"), {(operator, action["arguments"]["new_status"]) for operator, action in enum})
        self.assertIn(("invalid_value", "invalid_value"), {(operator, action["arguments"]["new_status"]) for operator, action in enum})

    def test_probe_rows_are_counterfactual_turns_with_recorded_history_and_oracle_truth(self):
        environment = load_environment(self.root / "env_defs", ENV)
        episode = rollout_episode(self.rollout_path, split="test", environment=environment)
        coverage: Counter = Counter()
        rows, truth = probe_rows(self.rollout, episode, self.oracle, system_prompt=system_prompt_for(self.oracle),
                                 initial_state=self.rollout["init_state"], per_trajectory=6, coverage=coverage)
        self.assertEqual((len(rows), len(truth)), (6, 6))
        recorded = episode_rows(episode, tool_definitions=environment["tools"], initial_state=self.rollout["init_state"])
        for row, item in zip(rows, truth):
            base = recorded[row["turn_idx"] - 1]
            self.assertEqual((row["id"], row["prompt"][:-1], row["response"][:-1]), (base["id"], base["prompt"][:-1], base["response"][:-1]))
            self.assertNotEqual(row["current_prompt"], base["current_prompt"])  # a different action at the same point
            self.assertEqual(row["response"][-1], "**Environment Observation:**\n" + item["observation"])
            self.assertEqual((row["probe"], row["split"], row["env_id"]), (item["probe"], "test", ENV))
            case = case_from_row(row)
            self.assertEqual((case.action.type, case.action.arguments), (item["action"]["name"], item["action"]["arguments"]))
            rerun = self.oracle.step(self.rollout["trajectory"][item["turn"] - 1]["state_before"], item["action"])
            self.assertEqual(rerun["observation"], item["observation"])
        labels = Counter(item["outcome_label"] for item in truth)
        self.assertGreaterEqual(labels["rejected"], 3)
        wordings = {item["signature"]["wording"] for item in truth if item["signature"]["kind"] == "error"}
        self.assertEqual(wordings, {"Item not found.", "Invalid status:", "Item {} is already {}."})  # two of them the rollout never reached
        # The state-dependent rejection: the recorded update, replayed after it took effect.
        replayed = next(item for item in truth if item["operator"] == "replayed_call" and item["signature"]["wording"] == "Item {} is already {}.")
        self.assertEqual((replayed["turn"], replayed["observation"]), (3, "{'success': False, 'error': 'Item I1 is already done.'}"))
        self.assertTrue(all(item["state_diff"] == {} for item in truth if item["outcome_label"] != "success-effect"))
        self.assertEqual(sum(coverage.values()), 6)
        # Deterministic: the same inputs give the same rows.
        again, _ = probe_rows(self.rollout, episode, EnvironmentOracle(load_metadata(self.root / "env_defs", ENV)),
                              system_prompt=system_prompt_for(self.oracle), initial_state=self.rollout["init_state"], per_trajectory=6)
        self.assertEqual(rows, again)

    def test_cli_probe_refuses_a_source_that_does_not_reproduce_the_rollouts(self):
        arguments = ["envscaler-probe", str(self.root / "benchmark"), "--env-defs", str(self.root / "env_defs"),
                     "--output", str(self.root / "probes.jsonl"), "--ground-truth", str(self.root / "truth"), "--per-trajectory", "4"]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(arguments), 0)
        report = json.loads(out.getvalue())
        self.assertEqual((report["rows"], report["environments"][ENV]["probes"]), (4, 4))
        self.assertTrue(reproduces(report["environments"][ENV]["replay_check"]))
        self.assertEqual(len(read_json(self.root / "truth" / "probes" / f"{ENV}-task_1.json")), 4)
        metadata = read_json(self.root / "env_defs" / f"{ENV}_metadata.json")
        metadata["env_class_code"] = SOURCE.replace("Status updated.", "Status changed.")
        write_json(self.root / "env_defs" / f"{ENV}_metadata.json", metadata)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "does not reproduce"):
            main(arguments[:5] + [str(self.root / "refused.jsonl")])
        self.assertFalse((self.root / "refused.jsonl").exists())


class ExactScoringTests(unittest.TestCase):
    @staticmethod
    def row(truth: str, prediction: str | None, *, history: str = "", **extra) -> dict:
        row = {"id": "t", "turn_idx": 2, "env_id": ENV, "prompt": [f"### Turn 1\n{history}", "### Turn 2\n**Action:**\n```json\n{}\n```"],
               "response": ["**Environment Observation:**\n{'success': True}", f"**Environment Observation:**\n{truth}"], **extra}
        if prediction is not None:
            row["gen"] = wrap_prediction(prediction)
        return row

    def test_values_are_compared_not_text(self):
        truth = "{'success': True, 'data': [{'item_id': 'I1', 'status': 'new'}, {'item_id': 'I2', 'status': 'done'}]}"
        as_json = '{"success": true, "data": [{"item_id": "I1", "status": "new"}, {"item_id": "I2", "status": "done"}]}'
        score = score_row(self.row(truth, as_json))
        self.assertEqual((score["kind"], score["parsed"], score["outcome_match"], score["match"], score["match_strict"], score["text_match"]),
                         ("data", True, True, True, True, False))
        self.assertTrue(score_row(self.row(truth, f"```python\n{truth}\n```"))["text_match"])  # a code fence is not part of the answer
        reordered = "{'success': True, 'data': [{'item_id': 'I2', 'status': 'done'}, {'item_id': 'I1', 'status': 'new'}]}"
        self.assertFalse(score_row(self.row(truth, reordered))["match"])  # records follow the database's order
        wrong = score_row(self.row(truth, "{'success': True, 'data': [{'item_id': 'I1', 'status': 'done'}]}"))
        self.assertEqual((wrong["outcome_match"], wrong["match"]), (True, False))
        self.assertIs(parse_observation("Item not found"), UNPARSED)
        prose = score_row(self.row(truth, "The item is returned."))
        self.assertEqual((prose["predicted"], prose["parsed"], prose["match"]), (True, False, False))
        self.assertFalse(values_equal(True, 1))
        self.assertFalse(values_equal({"success": True}, {"success": True, "extra": 1}))

    def test_generated_values_match_by_shape_only_when_the_context_does_not_hold_them(self):
        truth = "{'success': True, 'message': 'Item added.', 'item_id': '4ac08a88-3e4e-4d77-8e84-996b5e071907'}"
        guess = "{'success': True, 'message': 'Item added.', 'item_id': '11111111-2222-4333-8444-555555555555'}"
        fresh = score_row(self.row(truth, guess))
        self.assertEqual((fresh["generated"], fresh["match"], fresh["text_match"]), (True, True, True))
        self.assertFalse(score_row(self.row(truth, guess.replace("11111111-2222-4333-8444-555555555555", "not-an-id")))["match"])
        self.assertFalse(score_row(self.row(truth, guess.replace("Item added.", "Added.")))["match"])  # everything around it stays exact
        known = score_row(self.row(truth, guess, history="item 4ac08a88-3e4e-4d77-8e84-996b5e071907 exists"))
        self.assertEqual((known["generated"], known["match"]), (False, False))  # an id the model was shown must be reproduced
        clock = "{'success': True, 'data': {'_id': 'U1:G1:1788453640', 'join_date': '2026-09-04T00:40:40.098262'}}"
        self.assertTrue(score_row(self.row(clock, clock.replace("1788453640", "1767268800").replace("2026-09-04T00:40:40.098262", "2026-01-01T12:00:00")))["match"])
        self.assertFalse(score_row(self.row(clock, clock.replace("U1:G1", "U2:G1")))["match"])

    def test_set_built_lists_are_unordered_in_match_and_ordered_in_match_strict(self):
        truth = "{'success': True, 'data': {'allowed': ['ING3', 'ING8', 'ING1']}}"
        score = score_row(self.row(truth, "{'success': True, 'data': {'allowed': ['ING1', 'ING3', 'ING8']}}"))
        self.assertEqual((score["match"], score["match_strict"]), (True, False))
        self.assertFalse(score_row(self.row(truth, "{'success': True, 'data': {'allowed': ['ING1', 'ING3']}}"))["match"])
        message = "{'success': False, 'error': \"Ingredient(s) ['ING3', 'ING1'] present in both lists.\"}"
        listed = score_row(self.row(message, message.replace("['ING3', 'ING1']", "['ING1', 'ING3']")))
        self.assertEqual((listed["kind"], listed["match"], listed["match_strict"]), ("error", True, False))
        # A set printed inside a message has no order either (its repr follows the interpreter's hash seed).
        allowed = "{'success': False, 'error': \"Status 'locked' is not valid. Allowed: {'pending', 'active', 'removed'}\"}"
        printed = score_row(self.row(allowed, allowed.replace("{'pending', 'active', 'removed'}", "{'removed', 'active', 'pending'}")))
        self.assertEqual((printed["match"], printed["match_strict"], printed["text_match"]), (True, False, False))
        self.assertFalse(score_row(self.row(allowed, allowed.replace("{'pending', 'active', 'removed'}", "{'pending', 'active'}")))["match"])
        self.assertFalse(score_row(self.row(allowed, allowed.replace("'locked'", "'closed'")))["match"])  # the rest of the message stays exact

    def test_generated_ids_are_bound_consistently_distinctly_and_must_be_new(self):
        new, other = "4ac08a88-3e4e-4d77-8e84-996b5e071907", "50f76b8b-7610-48d9-814d-bed6871381ac"
        guess, second = "11111111-2222-4333-8444-555555555555", "66666666-7777-4888-8999-aaaaaaaaaaaa"
        existing = "99999999-aaaa-4bbb-8ccc-dddddddddddd"
        truth = f"{{'success': True, 'message': 'Enrollment {new} created.', 'enrollment_id': '{new}'}}"
        self.assertTrue(score_row(self.row(truth, truth.replace(new, guess)))["match"])
        # One generated id keeps one predicted value throughout the response.
        split = f"{{'success': True, 'message': 'Enrollment {guess} created.', 'enrollment_id': '{second}'}}"
        self.assertFalse(score_row(self.row(truth, split))["match"])
        # Two generated ids need two different predicted ids.
        pair = f"{{'success': True, 'data': [{{'id': '{new}'}}, {{'id': '{other}'}}]}}"
        self.assertTrue(score_row(self.row(pair, pair.replace(new, guess).replace(other, second)))["match"])
        self.assertFalse(score_row(self.row(pair, pair.replace(new, guess).replace(other, guess)))["match"])
        # A predicted id that the context already holds names another record, so it is not a new id.
        reused = score_row(self.row(truth, truth.replace(new, existing), history=f"record {existing} exists"))
        self.assertEqual((reused["generated"], reused["match"], reused["text_match"]), (True, False, False))
        self.assertTrue(score_row(self.row(truth, truth.replace(new, guess), history=f"record {existing} exists"))["match"])
        # Records filed under generated keys: any new key whose record agrees, whatever the order of the keys.
        keyed = f"{{'success': True, 'data': {{'{new}': {{'name': 'a'}}, '{other}': {{'name': 'b'}}}}}}"
        swapped = f"{{'success': True, 'data': {{'{guess}': {{'name': 'b'}}, '{second}': {{'name': 'a'}}}}}}"
        self.assertTrue(score_row(self.row(keyed, swapped))["match"])
        self.assertFalse(score_row(self.row(keyed, swapped.replace("'name': 'a'", "'name': 'c'")))["match"])
        # JSON can only write text keys.
        self.assertTrue(values_equal({"success": True, "data": {1: "a"}}, {"success": True, "data": {"1": "a"}}))

    def test_only_what_the_environment_can_generate_matches_by_shape(self):
        # A deadline computed from a session's start: the model never saw the text, but the environment did not generate it.
        deadline = "{'success': True, 'data': {'registration_allowed': True, 'registration_deadline': '2024-07-04 16:00'}}"
        wrong = deadline.replace("16:00", "18:00")
        self.assertTrue(score_row(self.row(deadline, wrong))["match"])  # no shape list in the row: every shape, as before
        no_clock = score_row(self.row(deadline, wrong, generated_shapes=[]))
        self.assertEqual((no_clock["match"], no_clock["text_match"], no_clock["outcome_match"], no_clock["generated"]), (False, False, True, False))
        self.assertTrue(score_row(self.row(deadline, deadline, generated_shapes=[]))["match"])
        # An environment that draws ids but never reads the clock: a new id is accepted, another date is not.
        created = "{'success': True, 'item_id': '5f0c0f6e-1d0a-4c55-9a53-0f2c8f0e7a11', 'due': '2024-07-04'}"
        other_id = created.replace("5f0c0f6e-1d0a-4c55-9a53-0f2c8f0e7a11", "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d")
        self.assertTrue(score_row(self.row(created, other_id, generated_shapes=["uuid"]))["match"])
        self.assertFalse(score_row(self.row(created, other_id.replace("2024-07-04", "2024-07-05"), generated_shapes=["uuid"]))["match"])
        self.assertTrue(score_row(self.row(created, other_id.replace("2024-07-04", "2024-07-05"), generated_shapes=["uuid", "date"]))["match"])

    def test_outcome_and_missing_predictions(self):
        rejected = "{'success': False, 'error': 'Item not found.'}"
        close = score_row(self.row(rejected, "{'success': False, 'error': 'No such item.'}"))
        self.assertEqual((close["outcome_match"], close["match"]), (True, False))
        self.assertFalse(score_row(self.row(rejected, "{'success': True, 'message': 'Status updated.'}"))["outcome_match"])
        missing = score_row(self.row(rejected, None))
        self.assertEqual((missing["predicted"], missing["outcome_match"], missing["match"]), (False, False, False))
        none = score_row(self.row("None", "None"))
        self.assertEqual((none["kind"], none["outcome_match"], none["match"]), ("other", True, True))
        self.assertFalse(score_row(self.row("None", rejected))["outcome_match"])

    def test_aggregate_separates_recorded_turns_from_probes_and_cli_reports(self):
        rejected = "{'success': False, 'error': 'Item not found.'}"
        rows = [self.row("{'success': True, 'message': 'Status updated.'}", "{'success': True, 'message': 'Status updated.'}"),
                self.row(rejected, "{'success': True, 'message': 'Status updated.'}", probe="t:t02:p1"),
                self.row(rejected, rejected, probe="t:t02:p2")]
        summary = aggregate(score_rows(rows))
        self.assertEqual((summary["all"]["rows"], summary["all"]["match"]), (3, 66.67))
        self.assertEqual((summary["set:recorded"]["match"], summary["set:probe"]["match"], summary["set:probe"]["outcome_match"]), (100.0, 50.0, 50.0))
        self.assertEqual(summary[f"env:{ENV}|set:probe"]["rows"], 2)
        self.assertEqual(summary["set:probe|kind:error"]["rows"], 2)
        self.assertNotIn("exact", rows[0])  # scoring does not modify its input
        self.assertNotIn("reads:excluded|kind:data", summary)
        self.assertEqual(summary["reads:excluded"], summary["all"])  # no read among these rows
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "pred.jsonl"
            predictions.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(["envscaler-score", "--predictions", str(predictions), "--summary", str(Path(directory) / "s.json"),
                                       "--scored", str(Path(directory) / "scored.jsonl")]), 0)
            self.assertIn("set:probe", out.getvalue())
            self.assertEqual(read_json(Path(directory) / "s.json")["all"]["rows"], 3)
            self.assertIn('"exact"', (Path(directory) / "scored.jsonl").read_text(encoding="utf-8"))

    def test_reported_accuracy_leaves_out_successful_reads_and_keeps_rejected_lookups(self):
        record = "{'success': True, 'data': {'item_id': 'I1', 'status': 'open'}}"
        not_found = "{'success': False, 'error': 'Item not found.'}"
        updated = "{'success': True, 'message': 'Status updated.'}"
        rows = [self.row(record, record), self.row(record, record),   # reads, both right: they would lift the score
                self.row(not_found, not_found),                       # a rejected lookup is not a read row
                self.row(updated, not_found),
                self.row(not_found, updated, probe="t:t02:p1")]
        summary = aggregate(score_rows(rows))
        self.assertEqual((summary["all"]["rows"], summary["all"]["match"]), (5, 60.0))
        self.assertEqual((summary["reads:excluded"]["rows"], summary["reads:excluded"]["match"]), (3, 33.33))
        self.assertEqual(summary["reads:excluded"]["outcome_match"], 33.33)
        self.assertEqual(summary[f"env:{ENV}|reads:excluded"], summary["reads:excluded"])
        self.assertEqual((summary["set:recorded|reads:excluded"]["rows"], summary["set:probe|reads:excluded"]["rows"]), (2, 1))
        self.assertEqual(summary["kind:data"]["rows"], 2)  # still reported on their own


if __name__ == "__main__":
    unittest.main()
