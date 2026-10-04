"""``envpack_prompting``: one deterministic free-text call per row over the official input plus a fixed package view."""
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import train_trajectory_id
from tests.test_knowledge_gate import evidence_package, rows_with_history
from trace2env.agentworld import case_from_row
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.envpack import action_query, envpack_view
from trace2env.llm import ScriptedChatLLM
from trace2env.package import EnvironmentPackage, PackageInspector


class EnvpackPromptingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.package_dir = evidence_package(Path(self.tmp.name))
        self.rows = rows_with_history(train_trajectory_id())

    def tearDown(self):
        self.tmp.cleanup()

    def test_query_uses_the_typed_command_words(self):
        case = case_from_row(self.rows[1])
        self.assertEqual(action_query(case), "ls warriors")

    def test_view_is_deterministic_and_lists_what_it_placed(self):
        inspector = PackageInspector(EnvironmentPackage(self.package_dir))
        case = case_from_row(self.rows[1])
        block, manifest = envpack_view(inspector, case, top_k=3, evidence_chars=500)
        block2, manifest2 = envpack_view(inspector, case, top_k=3, evidence_chars=500)
        self.assertEqual((block, manifest), (block2, manifest2))
        self.assertEqual(manifest["canonical_action_type"], "ls")
        self.assertEqual(manifest["query"], "ls warriors")
        self.assertIn("# Reconstructed environment package", block)
        self.assertIn("## Current action `ls` (schema)", block)
        self.assertIn("## State schema", block)
        self.assertIn("## Notes", block)
        self.assertTrue(manifest["evidence"], "lexical retrieval over the package catalog found no evidence")
        for evidence_id in manifest["evidence"]:
            self.assertIn(evidence_id, block)
        self.assertEqual(manifest["block_chars"], len(block))
        self.assertEqual(manifest["top_k"], 3)
        self.assertEqual(sorted(manifest), sorted(["canonical_action_type", "query", "top_k", "evidence_chars", "rule_limit", "demo_limit", "demo_chars",
                                                   "state_fields_shown", "action_specs", "rules", "renderers", "invariants", "notes", "demonstrations",
                                                   "evidence", "block_chars", "package"]))

    def test_runner_makes_exactly_one_chat_call_per_row_with_the_block_in_the_system_prompt(self):
        chat = ScriptedChatLLM({"agentworld_infer": ["<predicted_observation>one</predicted_observation>", "<predicted_observation>two</predicted_observation>"]})
        runner = AgentWorldRunner(mode="envpack_prompting", chat_llm=chat, package_dir=str(self.package_dir), allow_unvalidated=True,
                                  envpack_top_k=2, envpack_evidence_chars=400)
        outputs = runner.run(self.rows)
        self.assertEqual(len(chat.calls), 2)
        for row, call, expected in zip(outputs, chat.calls, ("one", "two")):
            info = row["trace2env"]
            self.assertEqual((info["mode"], info["route"]), ("envpack_prompting", "envpack_prompting"))
            self.assertIn(expected, row["gen"])
            system = call["messages"][0]
            self.assertEqual(system["role"], "system")
            self.assertTrue(system["content"].startswith(row["system_str"]))  # the official system prompt first, verbatim
            self.assertIn("# Reconstructed environment package", system["content"])
            self.assertEqual(call["messages"][-1]["content"], row["prompt"][-1])  # the official current message, untouched
            self.assertEqual(len(call["messages"]), 1 + 2 * (row["turn_idx"] - 1) + 1)
            manifest = info["envpack"]
            for evidence_id in manifest["evidence"]:
                self.assertIn(evidence_id, system["content"])
            self.assertEqual(manifest["top_k"], 2)
            self.assertNotIn("tool_calls", info)
            self.assertNotIn("state_tracking", info)
        self.assertFalse(list(Path(self.tmp.name).glob("**/session.sqlite")))  # no runtime session, no memory

    def test_runner_requires_a_package(self):
        with self.assertRaises(ValueError):
            AgentWorldRunner(mode="envpack_prompting", chat_llm=ScriptedChatLLM({}))


if __name__ == "__main__":
    unittest.main()


class ActionQueryOnToolRowsTests(unittest.TestCase):
    def test_tool_rows_fall_back_to_the_action_type_and_argument_words(self):
        from trace2env.agentworld import case_from_row
        prompt = "### Turn 3\n**Current State:**\n### Page\n- Page URL: http://forum.example.com/\n\n**Action:**\n```text\nbrowser_click(element=\"Log in link\", ref=\"e23\")\n```"
        row = {"task": "web", "id": 4242, "prompt": [prompt], "response": ["x"], "current_prompt": prompt, "system_str": "s", "turn_idx": 1, "total_turns": 1}
        query = action_query(case_from_row(row))
        self.assertTrue(query.startswith("browser_click element Log in link ref e23"), query)
