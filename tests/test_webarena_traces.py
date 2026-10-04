import json
import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.webarena_traces import (
    AGENT_TOOLS, SITES, agent_system_prompt, build_episode, call_string, final_answer, fold_snapshots, load_tasks,
    prune_tool_schema, resolve_url, split_manifest_for,
)

SNAPSHOT = "### Page\n- Page URL: http://gitlab.example.com/users/sign_in\n- Page Title: Sign in · GitLab\n### Snapshot\n```yaml\n- generic [ref=e1]\n```"
CODE_ONLY = "### Ran Playwright code\n```js\nawait page.getByRole('textbox', { name: 'Username' }).fill('byteblaze');\n```"


def turns():
    return [
        {"call_id": "c1", "name": "browser_navigate", "arguments": {"url": "http://gitlab.example.com/users/sign_in"},
         "output": "### Ran Playwright code\n```js\nawait page.goto('http://gitlab.example.com/users/sign_in');\n```\n" + SNAPSHOT},
        {"call_id": "c2", "name": "browser_type", "arguments": {"element": "Username", "ref": "e5", "text": "byteblaze"}, "output": CODE_ONLY},
        {"call_id": "c3", "name": "browser_snapshot", "arguments": {}, "output": SNAPSHOT},
        {"call_id": "c4", "name": "browser_click", "arguments": {"element": "Sign in button", "ref": "e9"}, "output": SNAPSHOT},
    ]


class WebArenaTraceTests(unittest.TestCase):
    def test_placeholders_resolve_to_benchmark_hosts(self):
        self.assertEqual(resolve_url("__GITLAB__/byteblaze/dotfiles"), "http://gitlab.example.com/byteblaze/dotfiles")
        self.assertEqual(resolve_url("__SHOPPING_ADMIN__"), "http://magento-admin.example.com/admin")
        self.assertEqual(resolve_url("__REDDIT__/f/books"), "http://forum.example.com/f/books")

    def test_load_tasks_keeps_requested_order_and_rejects_unknown_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.raw.json"
            path.write_text(json.dumps([{"task_id": 1, "sites": ["gitlab"], "start_url": "__GITLAB__", "intent": "a", "intent_template_id": 10},
                                        {"task_id": 2, "sites": ["reddit"], "start_url": "__REDDIT__", "intent": "b", "intent_template_id": 11}]))
            tasks = load_tasks(path, [2, 1])
            self.assertEqual([t["task_id"] for t in tasks], [2, 1])
            self.assertEqual(tasks[0]["start_url"], "http://forum.example.com")
            with self.assertRaises(ValueError):
                load_tasks(path, [3])

    def test_brief_names_only_the_task_site_and_its_account(self):
        brief = agent_system_prompt({"task_id": 5, "sites": ["shopping_admin"], "start_url": "http://magento-admin.example.com/admin", "intent": "x"})
        self.assertIn(SITES["shopping_admin"]["password"], brief)
        self.assertNotIn(SITES["gitlab"]["password"], brief)
        self.assertIn("FINAL ANSWER:", brief)

    def test_fold_snapshots_replaces_the_previous_observation(self):
        folded = fold_snapshots(turns())
        self.assertEqual([t["name"] for t in folded], ["browser_navigate", "browser_type", "browser_click"])
        self.assertEqual(folded[1]["output"], SNAPSHOT)
        self.assertEqual(folded[1]["folded_call_ids"], ["c3"])
        self.assertEqual([t["turn"] for t in folded], [1, 2, 3])
        self.assertEqual(fold_snapshots([{"call_id": "c0", "name": "browser_snapshot", "arguments": {}, "output": SNAPSHOT}]), [])
        self.assertIn("call browser_snapshot to observe", agent_system_prompt({"task_id": 1, "sites": ["gitlab"], "start_url": "x", "intent": "y"}))

    def test_tool_schemas_are_pruned_to_the_benchmark_arguments(self):
        schema = {"type": "object", "properties": {"element": {"type": "string"}, "ref": {"type": "string"},
                                                   "doubleClick": {"type": "boolean"}, "button": {"type": "string"}},
                  "required": ["element", "ref"]}
        pruned = prune_tool_schema("browser_click", schema)
        self.assertEqual(sorted(pruned["properties"]), ["element", "ref"])
        self.assertEqual(pruned["required"], ["element", "ref"])
        self.assertNotIn("browser_hover", AGENT_TOOLS)
        self.assertEqual(prune_tool_schema("browser_unknown", schema), schema)

    def test_call_string_matches_the_benchmark_action_text(self):
        self.assertEqual(call_string("browser_click", {"element": "Sign In link", "ref": "e12"}),
                         'browser_click(element="Sign In link", ref="e12")')

    def test_final_answer_protocol(self):
        self.assertEqual(final_answer("Some reasoning.\nFINAL ANSWER: 42"), ("answer", "42"))
        self.assertEqual(final_answer("DONE"), ("done", ""))
        self.assertEqual(final_answer("I will now click the button."), (None, None))

    def test_episode_ingests_and_pairs_actions_with_observations_by_call_id(self):
        task = {"task_id": 105, "sites": ["gitlab"], "start_url": "http://gitlab.example.com", "intent": "List the issues", "intent_template_id": 349}
        episode = build_episode(task, turns(), run="r1", agent="scripted", outcome={"status": "done", "answer": None, "error": None},
                                metrics={"prompt_tokens": 10})
        self.assertEqual(episode["turns"], 3)
        self.assertEqual(episode["raw_turns"], 4)
        self.assertEqual(episode["trajectory_group"], "webarena:105")
        self.assertEqual(episode["tool_names"], ["browser_click", "browser_navigate", "browser_type"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "webarena__105__r1.json"
            path.write_text(json.dumps(episode), encoding="utf-8")
            loaded = load_raw_traces(path)
            self.assertEqual(len(loaded), 1)
            kinds = [e.kind.value for e in loaded[0].events]
            self.assertEqual(kinds, ["message", "action", "observation", "action", "observation", "action", "observation"])
            slices = segment_transitions(loaded[0])
            self.assertEqual(len(slices), 3)
            self.assertTrue(all(s.alignment == "correlated" for s in slices))
            actions = [e for e in loaded[0].events if e.kind.value == "action"]
            self.assertEqual(actions[1].content["type"], "browser_type")
            self.assertEqual(actions[1].content["raw"], 'browser_type(element="Username", ref="e5", text="byteblaze")')
            manifest = split_manifest_for([path])
            assignment = next(iter(manifest.assignments.values()))
            self.assertEqual((assignment.split, assignment.trajectory_group), ("train", "webarena:105"))


if __name__ == "__main__":
    unittest.main()
