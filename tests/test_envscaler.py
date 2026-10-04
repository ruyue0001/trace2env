"""EnvScaler rollouts → Trace2Env episodes, ground truth, and benchmark rows. Fixtures are synthetic."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import (build_judge_messages, case_from_row, inference_messages, initial_observation, load_rows,
                                  prompt_sections, split_of)
from trace2env.atif import BENCHMARK_SUFFIX
from trace2env.cli import main
from trace2env.envscaler import (
    database,
    domain_description,
    episode_rows,
    evaluation_tool_description,
    export_rollouts,
    few_shot_examples,
    limit_few_shot_examples,
    render_few_shot_examples,
    ground_truth,
    iter_rollouts,
    load_environment,
    rollout_episode,
    rollout_turns,
    sanitize_evaluation_system_prompt,
    generated_shapes,
    state_layout,
    unique_turns,
)
from trace2env.llm import ScriptedLLM
from trace2env.models import ReconstructionConfig
from trace2env.reconstruction import TraceReconstructor
from trace2env.storage import read_json, write_json

ENV = "env_9_rl"
TERMINAL_ROWS = Path(__file__).resolve().parents[1] / "examples" / "agentworldbench" / "synthetic_terminal.jsonl"
ITEMS = {"I1": {"item_id": "I1", "status": "new"}, "I2": {"item_id": "I2", "status": "never-observed"}}
DONE = {"I1": {"item_id": "I1", "status": "done"}, "I2": ITEMS["I2"]}


def step(number: int, name: str, arguments: dict, observation: str, raw, *, before: dict, after: dict, diff: dict | None = None,
         label: str = "success-noop", error: str | None = None) -> dict:
    state_before, state_after = {"items": before, "init_config": {"items": ITEMS}}, {"items": after, "init_config": {"items": ITEMS}}
    action = {"name": name, "arguments": arguments}
    return {"step": number, "action": action, "observation": observation, "observation_raw": raw, "error": error,
            "state_before": state_before, "state_after": state_after, "state_diff": diff or {}, "n_delta": len(diff or {}),
            "obs_success": label != "rejected", "sigma_t": label, "outcome": label,
            "S_t": state_before, "A_t": action, "O_tplus1": raw, "S_tplus1": state_after, "Delta_t": diff or {}}


def write_collection(root: Path, *, task: str = f"{ENV}-task_1", directory: str = "reserve") -> Path:
    """One rollout plus the environment definitions, laid out like the collection (rollouts beside ``env_defs/``)."""
    changed = {"items": {"I1": {"status": {"changed": {"old": "new", "new": "done"}}}}}
    trajectory = [
        # The recorder kept a live reference: the parsed observation already shows the status step 2 writes.
        step(1, "get_item", {"item_id": "I1"}, "{'success': True, 'data': {'item_id': 'I1', 'status': 'new'}}",
             {"success": True, "data": {"item_id": "I1", "status": "done"}}, before=ITEMS, after=ITEMS),
        step(2, "update_item_status", {"item_id": "I1", "new_status": "done"}, "{'success': True, 'message': 'Status updated.'}",
             {"success": True, "message": "Status updated."}, before=ITEMS, after=DONE, diff=changed, label="success-effect"),
        step(3, "update_item_status", {"item_id": "I9", "new_status": "done"}, "{'success': False, 'error': 'Item not found.'}",
             {"success": False, "error": "Item not found."}, before=DONE, after=DONE, label="rejected"),
        step(4, "get_items", {}, "None", None, before=DONE, after=DONE, label="rejected",
             error='Traceback (most recent call last):\n  File "/collector/env_loader.py", line 457, in apply_action\n'
                   "AttributeError: 'ItemSystem' object has no attribute 'get_items'\n"),
        step(5, "chat_with_user", {"content": "Task Completed"}, "Task Completed", "Task Completed", before=DONE, after=DONE),
    ]
    path = root / directory / ENV / f"{ENV}-{task}.json"
    write_json(path, {"env_id": ENV, "task_id": task, "steps": 4, "reward": 1.0, "checklist_pass": 2, "checklist_total": 2,
                      "checklist_failed": [], "terminated": True, "truncated": False,
                      "init_state": {"items": ITEMS, "init_config": {"items": ITEMS}},
                      "final_state": {"items": DONE, "init_config": {"items": ITEMS}}, "trajectory": trajectory,
                      "backend": "meta", "model": "gpt-5.4", "elapsed_s": 3.5, "attempts_used": 1, "label": "success"})
    write_json(root / "env_defs" / f"{ENV}_metadata.json", {
        "env_id": ENV, "environment_summary": "Item tracking system", "environment_introduction": "Tracks items\nand their status.",
        "constraints_rules": ["HIDDEN-CONSTRAINT: only new items may be updated."],
        "env_class_code": "class ItemSystem: ...",
        "tools": [{"type": "function", "function": {"name": "get_item", "description": """Fetch one item. HIDDEN-DOCSTRING

Args:
    item_id (str): Item identifier.

Returns:
    dict: The exact item payload or 'Item not found.'.

Constraints:
    - item_id must exist.

Notes:
    Reads do not mutate the item.""",
                                                    "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}}, "required": ["item_id"]}}},
                  {"type": "function", "function": {"name": "update_item_status", "description": "Set an item's status.",
                                                    "parameters": {"type": "object", "properties": {"item_id": {"type": "string"}, "new_status": {"type": "string"}},
                                                                   "required": ["item_id", "new_status"]}}}]})
    write_json(root / "env_defs" / f"{ENV}_scenarios.json", [{"env_id": ENV, "task_id": task, "init_config": {"items": ITEMS},
                                                              "task": "Mark item I1 as done.", "checklist_with_func": []}])
    write_json(root / "selection_manifest.json", {"experiment": "fixture", "envs": {}})
    return path


class EnvScalerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rollout = write_collection(self.root)
        self.environment = load_environment(self.root / "env_defs", ENV)

    def test_evaluation_tool_descriptions_remove_returns_and_constraints(self):
        description = self.environment["tools"][0]["description"]
        self.assertIn("Fetch one item. HIDDEN-DOCSTRING", description)
        self.assertIn("Args:\n    item_id (str): Item identifier.", description)
        self.assertIn("Notes:\n    Reads do not mutate the item.", description)
        self.assertNotIn("Returns:", description)
        self.assertNotIn("Constraints:", description)
        rendered = ("#### **Description**:\nSummary.\n\nArgs:\n    item_id (str)\n\nReturns:\n    exact output\n\n"
                    "Constraints:\n    must exist\n\nNotes:\n    retained\n\n#### **Parameters**:\n"
                    "- **item_id** `string` (REQUIRED): ")
        cleaned = sanitize_evaluation_system_prompt(rendered)
        self.assertIn("Args:\n    item_id (str)", cleaned)
        self.assertIn("Notes:\n    retained", cleaned)
        self.assertNotIn("Returns:", cleaned)
        self.assertNotIn("Constraints:", cleaned)
        self.assertEqual(evaluation_tool_description("Summary.\n\nReturns: exact output"), "Summary.")

    def test_turns_keep_the_text_the_agent_saw_and_drop_agent_side_tools(self):
        turns = rollout_turns(read_json(self.rollout))
        self.assertEqual([turn["name"] for turn in turns], ["get_item", "update_item_status", "update_item_status", "get_items"])
        # Step 1's parsed observation was overwritten by step 2; the recorded text is the evidence.
        self.assertEqual(turns[0]["output"], "{'success': True, 'data': {'item_id': 'I1', 'status': 'new'}}")
        self.assertEqual([turn["stale_raw"] for turn in turns], [True, False, False, False])
        self.assertEqual(turns[3]["output"], "None")
        self.assertEqual(turns[3]["harness_error"], "AttributeError: 'ItemSystem' object has no attribute 'get_items'")
        self.assertEqual(turns[1]["action"].model_dump(mode="json"),
                         {"type": "update_item_status", "arguments": {"item_id": "I1", "new_status": "done"},
                          "raw": {"name": "update_item_status", "arguments": {"item_id": "I1", "new_status": "done"}}})

    def test_collector_completion_mislabeled_as_a_tool_result_is_dropped(self):
        rollout = read_json(self.rollout)
        rollout["trajectory"][-1]["action"] = {"name": "get_item", "arguments": {"item_id": "I1"}}
        turns = rollout_turns(rollout)
        self.assertEqual([turn["name"] for turn in turns], ["get_item", "update_item_status", "update_item_status", "get_items"])
        path = self.root / "mislabeled-completion.json"
        write_json(path, rollout)
        episode = rollout_episode(path, environment=self.environment)
        self.assertEqual((episode["turns"], episode["task_complete"]), (4, True))
        self.assertNotIn("Task Completed", json.dumps(episode["events"]))

    def test_episode_holds_what_crossed_the_tool_interface_and_no_ground_truth(self):
        episode = rollout_episode(self.rollout, environment=self.environment)
        self.assertEqual([event["kind"] for event in episode["events"]], ["message"] + ["action", "observation"] * 4)
        self.assertEqual(episode["events"][0]["content"], "Mark item I1 as done.")
        self.assertEqual((episode["turns"], episode["task_complete"], episode["task"], episode["source"]), (4, True, "mcp", "envscaler"))
        self.assertEqual((episode["trajectory_group"], episode["trajectory_id"]), (f"envscaler:{ENV}-task_1", f"envscaler-{ENV}-task_1"))
        self.assertEqual(episode["split"], split_of(f"envscaler:{ENV}-task_1"))
        self.assertEqual(episode["outcome"]["reward"], 1.0)
        self.assertEqual(episode["events"][-1]["metadata"]["harness_error"], "AttributeError: 'ItemSystem' object has no attribute 'get_items'")
        self.assertNotIn("environment_id", episode)  # that key would be read as a construction-scope selector
        text = json.dumps(episode)
        for hidden in ("never-observed", "state_diff", "state_before", "init_state", "sigma_t", "success-effect", "observation_raw",
                       "/collector/", "HIDDEN"):
            self.assertNotIn(hidden, text)
        self.assertEqual(rollout_episode(self.rollout)["instruction"], "")  # no definitions: no instruction event
        self.assertEqual(rollout_episode(self.rollout)["events"][0]["kind"], "action")

    def test_episode_ingests_one_sequential_slice_per_tool_call_and_only_for_construction(self):
        paths, manifest, _ = export_rollouts([self.rollout], self.root / "train", env_defs=self.root / "env_defs", split="train")
        loaded = load_raw_traces(paths[0])[0]
        slices = segment_transitions(loaded)
        self.assertEqual((len(slices), {s.alignment for s in slices}), (4, {"sequential"}))
        self.assertFalse(any(s.ambiguities or s.concurrent_action_event_ids for s in slices))
        self.assertEqual(loaded.metadata["trajectory_group"], f"envscaler:{ENV}-task_1")
        recon = TraceReconstructor(ScriptedLLM({}), ReconstructionConfig(environment_id=ENV, name="T"), self.root / "work")
        self.assertEqual(tuple(len(item) for item in recon.ingest(paths, manifest)), (1, 4))
        held_out, held_out_manifest, _ = export_rollouts([self.rollout], self.root / "eval", split="test")
        with self.assertRaises(ValueError):
            recon.ingest(held_out, held_out_manifest)

    def test_export_writes_episodes_tools_manifest_and_separate_ground_truth(self):
        found = [path.name for path, _ in iter_rollouts([self.root])]  # definitions and the manifest are not rollouts
        self.assertEqual(found, [self.rollout.name])
        with self.assertRaisesRegex(ValueError, "not an EnvScaler rollout"):
            list(iter_rollouts([self.root / "selection_manifest.json"]))
        output = self.root / "episodes"
        paths, manifest, summaries = export_rollouts([self.root], output, env_defs=self.root / "env_defs", split="validation",
                                                     ground_truth_dir=self.root / "truth")
        self.assertEqual([path.name for path in paths], [f"{ENV}-task_1.json"])
        self.assertEqual(sorted(path.name for path in output.glob("*.json")), [f"{ENV}-task_1.json"])  # nothing else to ingest
        episode = read_json(paths[0])
        self.assertEqual(episode["tool_definitions"], f"tools/{ENV}.json")
        self.assertEqual([tool["name"] for tool in read_json(output / episode["tool_definitions"])], ["get_item", "update_item_status"])
        assignment = next(iter(manifest.assignments.values()))
        self.assertEqual((assignment.split, assignment.trajectory_group), ("validation", f"envscaler:{ENV}-task_1"))
        self.assertEqual({key: summaries[0][key] for key in ("turns", "effect_turns", "rejected_turns", "harness_errors",
                                                             "stale_raw_observations", "instruction")},
                         {"turns": 4, "effect_turns": 1, "rejected_turns": 2, "harness_errors": 1, "stale_raw_observations": 1,
                          "instruction": True})
        truth = read_json(self.root / "truth" / f"{ENV}-task_1.json")
        self.assertEqual(truth, json.loads(json.dumps(ground_truth(self.rollout, episode))))
        self.assertEqual([(turn["turn"], turn["tool"], turn["outcome"], turn["outcome_label"]) for turn in truth["turns"]],
                         [(1, "get_item", "success", "success-noop"), (2, "update_item_status", "success", "success-effect"),
                          (3, "update_item_status", "failure", "rejected"), (4, "get_items", "failure", "rejected")])
        self.assertEqual(truth["turns"][1]["state_diff"]["items"]["I1"]["status"]["changed"], {"old": "new", "new": "done"})
        self.assertEqual(truth["initial_state"]["items"]["I2"]["status"], "never-observed")  # verbatim, config entry included
        self.assertIn("init_config", truth["initial_state"])
        with self.assertRaisesRegex(ValueError, "ingested as traces"):
            export_rollouts([self.root], self.root / "mixed", ground_truth_dir=self.root / "mixed")
        write_collection(self.root, directory="benchmark")  # the same task a second time
        with self.assertRaisesRegex(ValueError, "supplied twice"):
            export_rollouts([self.root], self.root / "twice")

    def test_rows_show_the_initial_database_once_in_the_official_layout(self):
        episode = rollout_episode(self.rollout, environment=self.environment)
        initial = read_json(self.rollout)["init_state"]
        rows = episode_rows(episode, tool_definitions=self.environment["tools"], initial_state=initial)
        self.assertEqual([row["turn_idx"] for row in rows], [1, 2, 3, 4])
        last = rows[-1]
        # The terminal rows' layout: the initial state as text right under the header, then the action block.
        self.assertTrue(last["prompt"][0].startswith('### Turn 1\n**Current State:**\n{"items": {"I1": {"item_id": "I1", "status": "new"}'))
        self.assertIn('}}\n\n**Action:**\n```json\n{\n  "name": "get_item"', last["prompt"][0])
        self.assertNotIn("init_config", last["prompt"][0])  # the environment's bookkeeping copy is not database state
        self.assertEqual(["Current State" in prompt for prompt in last["prompt"]], [True, False, False, False])
        self.assertTrue(last["prompt"][-1].endswith(BENCHMARK_SUFFIX))
        self.assertEqual(last["current_prompt"] + BENCHMARK_SUFFIX, last["prompt"][-1])
        self.assertEqual(last["response"], ["**Environment Observation:**\n{'success': True, 'data': {'item_id': 'I1', 'status': 'new'}}",
                                            "**Environment Observation:**\n{'success': True, 'message': 'Status updated.'}",
                                            "**Environment Observation:**\n{'success': False, 'error': 'Item not found.'}",
                                            "**Environment Observation:**\nNone"])
        self.assertIn("### 2. update_item_status\n#### **Description**:\nSet an item's status.", last["system_str"])
        self.assertNotIn("\nReturns:", last["system_str"])
        self.assertNotIn("\nConstraints:", last["system_str"])
        self.assertFalse(any(token in last["system_str"] for token in ("{tool_definitions}", "{demonstrations}")))
        self.assertEqual((last["task"], last["id"], last["env_id"], last["source"]), ("mcp", f"envscaler-{ENV}-task_1", ENV, "envscaler"))
        case = case_from_row(rows[1])
        self.assertEqual((case.action.type, case.action.arguments), ("update_item_status", {"item_id": "I1", "new_status": "done"}))
        self.assertEqual(json.loads(initial_observation(case)), database(initial))  # what the runner tracks as turn 0
        latent = episode_rows(episode, tool_definitions=self.environment["tools"], turns_per_trajectory=2)
        self.assertEqual([row["turn_idx"] for row in latent], [2, 4])
        self.assertIsNone(initial_observation(case_from_row(latent[0])))

    def test_rows_have_the_structure_of_agentworldbench_terminal_records(self):
        """Same record keys and evaluator inputs; this explicit ablation also checks state-before-turn-1 parsing."""
        terminal = load_rows([TERMINAL_ROWS])[-1]  # the bundled terminal record evaluating turn 3 of 3
        episode = rollout_episode(self.rollout, environment=self.environment)
        row = episode_rows(episode, tool_definitions=self.environment["tools"], initial_state=read_json(self.rollout)["init_state"])[2]
        self.assertEqual(set(terminal) - set(row), set())  # every official key; ours adds bookkeeping keys only
        # Same value types; the trajectory id is free-form (the evaluation code reads it through str()).
        self.assertEqual({key: type(row[key]) for key in terminal if key != "id"},
                         {key: type(value) for key, value in terminal.items() if key != "id"})

        def structure(record: dict) -> tuple:
            case = case_from_row(record)
            messages = inference_messages(case)
            return ([message["role"] for message in messages],                       # system, alternating history, current turn
                    [sorted(prompt_sections(prompt)) for prompt in case.prompts],     # Current State before turn 1 only
                    [response.split("\n", 1)[0] for response in case.responses],      # the observation marker
                    messages[-1]["content"].endswith("</predicted_observation> tags."),
                    messages[-1]["content"].startswith(case.current_prompt),
                    initial_observation(case) is not None,
                    [message["role"] for message in build_judge_messages(record, "x")])

        self.assertEqual(structure(row), structure(terminal))
        self.assertEqual(structure(row)[1], [["action", "current_state", "turn"], ["action", "turn"], ["action", "turn"]])
        # The prediction target is the evaluated turn's observation (the tool's response), not the database.
        self.assertEqual(case_from_row(row).ground_truth, "**Environment Observation:**\n{'success': False, 'error': 'Item not found.'}")

    def test_few_shot_examples_follow_the_official_mcp_layout_and_come_from_construction_episodes_only(self):
        construction = rollout_episode(self.rollout, split="train", environment=self.environment)
        examples = few_shot_examples([construction], self.environment["tools"])
        # One example per tool (its first call), in the order of the tool definitions; undefined tools last.
        self.assertEqual([(item["name"], item["arguments"]) for item in examples],
                         [("get_item", {"item_id": "I1"}), ("update_item_status", {"item_id": "I1", "new_status": "done"}), ("get_items", {})])
        section = render_few_shot_examples(examples)
        self.assertTrue(section.startswith("\n---\n\n# Few-shot Examples\n\nBelow are examples of interactions for the tools available in this "
                                           "environment:\n\n## Example: get_item\n**Action:**\n```json\n{\n  \"name\": \"get_item\",\n"))
        self.assertIn("```\n**Environment Observation:**\n{'success': True, 'data': {'item_id': 'I1', 'status': 'new'}}\n\n## Example: update_item_status\n",
                      section)
        self.assertTrue(section.endswith("**Environment Observation:**\nNone\n\n"))
        self.assertEqual(render_few_shot_examples([]), "")
        self.assertEqual([item["name"] for item in limit_few_shot_examples(examples, 2)], ["get_item", "get_items"])
        self.assertEqual(limit_few_shot_examples(examples, 0), [])
        self.assertEqual(limit_few_shot_examples(examples, None), examples)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            limit_few_shot_examples(examples, -1)
        held_out = rollout_episode(self.rollout, split="test", environment=self.environment)
        with self.assertRaisesRegex(ValueError, "construction episodes only"):
            few_shot_examples([held_out])
        rows = episode_rows(held_out, tool_definitions=self.environment["tools"], examples=examples)
        system = rows[0]["system_str"]
        self.assertIn("consistent with the session context.\n\n---\n\n# Few-shot Examples\n", system)  # the official junction
        self.assertTrue(system.endswith("\n\n"))
        self.assertFalse(any(token in system for token in ("{tool_definitions}", "{demonstrations}")))
        self.assertLess(system.index("### 2. update_item_status"), system.index("# Few-shot Examples"))
        self.assertNotIn("# Few-shot Examples", episode_rows(held_out, tool_definitions=self.environment["tools"])[0]["system_str"])

    @staticmethod
    def held_out_episode(task: str, calls: list[tuple[str, dict, str]], *, artifact_turns: tuple[int, ...] = ()) -> dict:
        events = []
        for turn, (tool, arguments, answer) in enumerate(calls, start=1):
            events.append({"actor": "agent", "kind": "action", "content": {"type": tool, "arguments": arguments}, "metadata": {"turn": turn}})
            metadata = {"turn": turn, **({"harness_error": "AttributeError: no such tool"} if turn in artifact_turns else {})}
            events.append({"actor": "environment", "kind": "observation", "content": answer, "metadata": metadata})
        return {"episode_id": f"envscaler:{task}", "trajectory_id": f"envscaler-{task}", "benchmark_task": task, "env_id": ENV,
                "split": "test", "turns": len(calls), "tool_names": sorted({call[0] for call in calls}), "events": events}

    def test_unique_turns_keep_the_last_of_a_repeated_call_and_one_of_each_duplicate(self):
        listed, relisted, done = "{'success': True, 'data': ['new']}", "{'success': True, 'data': ['done']}", "{'success': True, 'message': 'Status updated.'}"
        missing = "{'success': False, 'error': 'Item not found.'}"
        first = self.held_out_episode("t1", [("list_items", {}, listed),                                    # 1 repeated at 3 with a NEW answer
                                             ("update_item_status", {"item_id": "I1", "new_status": "done"}, done),
                                             ("list_items", {}, relisted),
                                             ("get_item", {"item_id": "I9"}, missing),                   # 4 repeated at 6 with the SAME answer
                                             ("get_items", {}, "None"),                                  # 5 a tool the environment lacks
                                             ("get_item", {"item_id": "I9"}, missing)], artifact_turns=(5,))
        second = self.held_out_episode("t2", [("get_item", {"item_id": "I9"}, missing),                   # the same call and answer as t1 turn 6
                                              ("update_item_status", {"item_id": "I2", "new_status": "done"}, done)])  # same tool and answer text as t1 turn 2
        kept, report = unique_turns([first, second], "episode")
        self.assertEqual(kept, {"envscaler-t1": [2, 3, 6], "envscaler-t2": [1, 2]})
        self.assertEqual(report["dropped_by_reason"], {"loader_artifact": 1, "repeated_action": 2})
        self.assertEqual([(item["turn"], item["reason"], item.get("kept")) for item in report["dropped"]],
                         [(1, "repeated_action", {"benchmark_task": "t1", "turn": 3}), (4, "repeated_action", {"benchmark_task": "t1", "turn": 6}),
                          (5, "loader_artifact", None)])
        # The kept last occurrence at turn 6 repeats the answer of turn 4, which is in its own history: flagged, not dropped.
        self.assertEqual(report["kept_with_answer_in_own_history"], [{"benchmark_task": "t1", "turn": 6, "tool": "get_item"}])
        kept, report = unique_turns([first, second], "environment")
        self.assertEqual(kept, {"envscaler-t1": [2, 3, 6], "envscaler-t2": [2]})  # of the two identical calls the longer history stays
        self.assertEqual(report["dropped_by_reason"]["same_call_and_answer_elsewhere"], 1)
        self.assertEqual((report["turns"], report["kept"], report["level"], report["env_id"]), (8, 4, "environment", ENV))
        kept, report = unique_turns([first, second], "answer")
        self.assertEqual(kept, {"envscaler-t1": [3, 6], "envscaler-t2": [2]})      # one 'Status updated.' per tool: equal turns, the later task wins
        self.assertEqual(report["dropped_by_reason"]["same_tool_and_answer_elsewhere"], 1)
        self.assertEqual(unique_turns([first, second], "answer"), (kept, report))   # deterministic
        # same-answer: a repeated call loses an occurrence only to a later one with the SAME answer, so the listing before
        # the update (turn 1, a different answer than turn 3) stays, and the repeated identical failure (turn 4) still goes.
        kept, report = unique_turns([first, second], "episode", repeated="same-answer")
        self.assertEqual(kept, {"envscaler-t1": [1, 2, 3, 6], "envscaler-t2": [1, 2]})
        self.assertEqual((report["repeated_calls"], report["dropped_by_reason"]), ("same-answer", {"loader_artifact": 1, "repeated_action": 1}))
        self.assertEqual(report["kept_with_answer_in_own_history"], [{"benchmark_task": "t1", "turn": 6, "tool": "get_item"}])
        with self.assertRaisesRegex(ValueError, "unknown level"):
            unique_turns([first], "everything")
        with self.assertRaisesRegex(ValueError, "repeated calls"):
            unique_turns([first], "episode", repeated="first")
        with self.assertRaisesRegex(ValueError, "one environment"):
            unique_turns([first, {**second, "env_id": "env_other"}])
        # Only the evaluated turn is filtered: a kept row still has every recorded turn in its history.
        rows = episode_rows(first, turns=[3, 6], system_prompt="SYS")
        self.assertEqual([row["turn_idx"] for row in rows], [3, 6])
        self.assertEqual((len(rows[1]["prompt"]), len(rows[1]["response"]), rows[1]["total_turns"]), (6, 6, 6))
        self.assertIn('"name": "get_items"', rows[1]["prompt"][4])  # the artifact turn is history, just never the evaluated turn
        self.assertEqual([row["turn_idx"] for row in episode_rows(first, turns=[2, 3, 6], turns_per_trajectory=2, system_prompt="SYS")], [3, 6])

    def test_generated_shapes_are_read_off_the_environment_source(self):
        self.assertEqual(generated_shapes("import uuid\nclass E:\n    def add(self):\n        return str(uuid.uuid4())\n"), ["uuid"])
        self.assertEqual(generated_shapes("from datetime import datetime\nclass E:\n    def add(self):\n        return datetime.now().isoformat()\n"),
                         ["date", "epoch", "stamp"])
        self.assertEqual(generated_shapes("from time import time\nclass E:\n    def add(self):\n        return str(time())\n"), ["date", "epoch", "stamp"])
        self.assertEqual(generated_shapes("import time\nclass E:\n    def add(self):\n        return time.time()\n"), ["date", "epoch", "stamp"])
        # Arithmetic on stored dates reads no clock: what it returns is computed, and the scorer compares it exactly.
        computed = ("from datetime import datetime, timedelta\nclass E:\n    def deadline(self, start):\n"
                    "        return (datetime.strptime(start, '%Y-%m-%d %H:%M') - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')\n")
        self.assertEqual(generated_shapes(computed), [])
        self.assertIsNone(generated_shapes("class E: ..."[:-1] + "("))  # unparsable: nothing is known
        self.assertEqual(generated_shapes("class ItemSystem: ..."), [])

    def test_description_draft_uses_the_database_layout_and_none_of_the_answer_key(self):
        layout = state_layout([read_json(self.rollout)["init_state"], None])
        self.assertEqual(layout, {"items": ["item_id", "status"]})
        text = domain_description(self.environment, layout, ["update_item_status", "get_item", "get_item"])
        self.assertIn("Environment: Item tracking system (EnvScaler `env_9_rl`). Tracks items and their status.", text)
        self.assertIn("- world.items (object): record id -> {item_id, status}", text)
        self.assertIn("Tools seen in these traces: `get_item`, `update_item_status`.", text)
        for withheld in ("HIDDEN", "init_config", "never-observed", "class ItemSystem"):
            self.assertNotIn(withheld, text)

    def test_cli_export(self):
        arguments = ["envscaler-export", str(self.root / "reserve"), "--output", str(self.root / "episodes"),
                     "--env-defs", str(self.root / "env_defs"), "--ground-truth", str(self.root / "truth"),
                     "--descriptions", str(self.root / "descriptions"), "--rows", str(self.root / "rows.jsonl"),
                     "--summary", str(self.root / "summary.json")]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(arguments + ["--assign-split", "train"]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual({key: report[key] for key in ("episodes", "skipped", "turns", "splits", "environments", "rows", "rows_by_split",
                                                       "without_instruction", "stale_raw_observations")},
                         {"episodes": 1, "skipped": 0, "turns": 4, "splits": {"train": 1}, "environments": {ENV: 1}, "rows": 4,
                          "rows_by_split": {"train": 4}, "without_instruction": 0, "stale_raw_observations": 1})
        self.assertTrue((self.root / "episodes" / "split_manifest.json").exists())
        self.assertTrue((self.root / "truth" / f"{ENV}-task_1.json").exists())
        self.assertIn("world.items", (self.root / "descriptions" / f"{ENV}.md").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (self.root / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertNotIn("**Current State:**", rows[0]["prompt"][0])
        # With the definitions at hand a row says what the environment can generate (this source: nothing); the
        # episode, a construction input, does not carry it.
        self.assertEqual({tuple(row["generated_shapes"]) for row in rows}, {()})
        self.assertNotIn("generated_shapes", (self.root / "episodes" / f"{ENV}-task_1.json").read_text(encoding="utf-8"))
        # A description is edited by hand before a build: exporting again keeps it.
        (self.root / "descriptions" / f"{ENV}.md").write_text("edited by hand\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["envscaler-export", str(self.root / "reserve"), "--output", str(self.root / "again"),
                                   "--env-defs", str(self.root / "env_defs"), "--assign-split", "train",
                                   "--descriptions", str(self.root / "descriptions")]), 0)
        again = json.loads(out.getvalue())
        self.assertEqual((again["descriptions"], len(again["descriptions_kept"])), ([], 1))
        self.assertEqual((self.root / "descriptions" / f"{ENV}.md").read_text(encoding="utf-8"), "edited by hand\n")
        # A held-out export drafts no description (it would let evaluation data shape a construction input), and
        # keeps state latent by default, matching the official MCP prompt layout.
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["envscaler-export", str(self.rollout), "--output", str(self.root / "eval"), "--assign-split", "test",
                                   "--descriptions", str(self.root / "eval-descriptions"), "--rows", str(self.root / "eval-rows.jsonl"),
                                   "--rows-split", "test", "--manifest", str(self.root / "eval-split.json")]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual((report["descriptions"], report["without_instruction"], report["rows_by_split"]), ([], 1, {"test": 4}))
        self.assertEqual(report["few_shot_examples"], {ENV: 0})  # none asked for
        self.assertEqual(report["unique_turns"], {})             # nor a filter
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["envscaler-export", str(self.rollout), "--output", str(self.root / "eval-unique"), "--assign-split", "test",
                                   "--rows", str(self.root / "unique-rows.jsonl"), "--unique-turns", "environment",
                                   "--filter-report", str(self.root / "unique.json")]), 0)
        filtered = json.loads(out.getvalue())
        self.assertEqual(filtered["unique_turns"], {ENV: {"level": "environment", "repeated_calls": "last", "turns": 4, "kept": 3,
                                                          "dropped_by_reason": {"loader_artifact": 1}}})
        self.assertEqual((filtered["rows"], [item["turn"] for item in read_json(self.root / "unique.json")[ENV]["dropped"]]), (3, [4]))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["envscaler-export", str(self.rollout), "--output", str(self.root / "eval-examples"), "--assign-split", "test",
                                   "--env-defs", str(self.root / "env_defs"), "--rows", str(self.root / "eval-examples-rows.jsonl"),
                                   "--examples-from", str(self.root / "episodes")]), 0)  # the construction export made above
        self.assertEqual(json.loads(out.getvalue())["few_shot_examples"], {ENV: 3})
        self.assertIn("# Few-shot Examples", json.loads((self.root / "eval-examples-rows.jsonl").read_text(encoding="utf-8").splitlines()[0])["system_str"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["envscaler-export", str(self.rollout), "--output", str(self.root / "eval-limited-examples"),
                                   "--assign-split", "test", "--env-defs", str(self.root / "env_defs"), "--rows",
                                   str(self.root / "eval-limited-examples-rows.jsonl"), "--examples-from",
                                   str(self.root / "episodes"), "--few-shot-limit", "1"]), 0)
        self.assertEqual(json.loads(out.getvalue())["few_shot_examples"], {ENV: 1})
        limited_system = json.loads((self.root / "eval-limited-examples-rows.jsonl").read_text(encoding="utf-8").splitlines()[0])["system_str"]
        self.assertEqual(limited_system.count("## Example:"), 1)
        self.assertIn("## Example: get_item", limited_system)
        self.assertNotIn("## Example: update_item_status", limited_system)
        with self.assertRaisesRegex(ValueError, "construction episodes only"):  # a held-out directory is refused as a source of examples
            main(["envscaler-export", str(self.rollout), "--output", str(self.root / "eval-bad"), "--assign-split", "test",
                  "--rows", str(self.root / "bad-rows.jsonl"), "--examples-from", str(self.root / "eval")])
        self.assertFalse((self.root / "eval-descriptions").exists())
        self.assertTrue((self.root / "eval-split.json").exists())
        self.assertNotIn("Current State", (self.root / "eval-rows.jsonl").read_text(encoding="utf-8"))

        # The prior state-visible protocol remains available only as an explicit ablation.
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["envscaler-export", str(self.rollout), "--output", str(self.root / "state-visible"),
                                   "--assign-split", "test", "--rows", str(self.root / "state-visible.jsonl"),
                                   "--initial-state"]), 0)
        self.assertIn("**Current State:**", (self.root / "state-visible.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
