"""prompting+rag baseline: the official prompting input plus a fixed top-k of raw turns from a trace corpus."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import terminal_rows
from trace2env.agentworld import RESPONSE_TAG, case_from_row, inference_messages, parse_model_output
from trace2env.agentworld_runner import AgentWorldRunner, current_keystrokes, rag_block, rag_case_query, rag_query
from trace2env.cli import main
from trace2env.llm import ScriptedChatLLM
from trace2env.trace_corpus import TraceCorpus

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROWS = ROOT / "examples" / "agentworldbench" / "synthetic_terminal.jsonl"


class PromptingRagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["awb-export", str(SYNTHETIC_ROWS), "--output", str(self.root / "episodes"), "--split", "all"]), 0)
        self.corpus = TraceCorpus.build(self.root / "episodes", self.root / "corpus.sqlite")
        self.rows = terminal_rows()

    def test_query_and_block_are_fixed_and_lexical(self):
        self.assertEqual(rag_query("cat > /app/solve.py << 'EOF'\npython3 solve.py --verbose -n 3\n"), "cat app/solve.py EOF python3 solve.py 3")
        self.assertEqual(current_keystrokes(case_from_row(self.rows[1])), "cd tests\n")
        hits = self.corpus.search("pwd", 2)
        block = rag_block(hits, self.corpus, 50)
        self.assertIn("# Retrieved examples from other recorded sessions", block)
        self.assertIn(hits[0]["id"], block)
        # Default style "terminal": the wording of every reported AgentWorldBench run.
        self.assertIn("same kind of terminal", block)
        self.assertIn("**Action (keystrokes):**", block)
        self.assertIn("**Environment Observation:**", block)
        # Style "generic" (the EnvScaler runs): environment-neutral wording, same examples.
        generic = rag_block(hits, self.corpus, 50, "generic")
        self.assertIn("same kind of environment", generic)
        self.assertIn("**Action:**", generic)
        self.assertNotIn("same kind of terminal", generic)
        self.assertNotIn("Action (keystrokes)", generic)
        self.assertEqual(generic.count("## Retrieved example"), block.count("## Retrieved example"))
        with self.assertRaises(ValueError):
            rag_block(hits, self.corpus, 50, "other")

    def test_object_action_query_depends_on_the_rag_style(self):
        row = {
            "task": "mcp", "id": "mcp-1", "turn_idx": 1, "total_turns": 1,
            "system_str": "system",
            "current_prompt": "### Turn 1\n**Action:**\n```json\n{\"name\": \"get_user_profile_by_id\", "
                              "\"arguments\": {\"user_id\": \"USR1\"}}\n```",
            "prompt": [], "response": ["**Environment Observation:**\n{}"],
        }
        row["prompt"] = [row["current_prompt"]]
        case = case_from_row(row)
        self.assertEqual(current_keystrokes(case), "")
        # "terminal" (default, the reported AgentWorldBench runs): action type and argument words (envpack.action_query).
        self.assertEqual(rag_case_query(case), "get_user_profile_by_id user_id USR1")
        # "generic" (the EnvScaler runs): the trace corpus's normalized action query.
        self.assertEqual(rag_case_query(case, "generic"), "get_user_profile_by_id")
        # Terminal keystrokes give the same command-word query under both styles.
        terminal_case = case_from_row(self.rows[1])
        self.assertEqual(rag_case_query(terminal_case), rag_case_query(terminal_case, "generic"))
        self.assertEqual(rag_case_query(terminal_case), rag_query(current_keystrokes(terminal_case)))
        with self.assertRaises(ValueError):
            rag_case_query(case, "other")

    def test_runner_appends_retrieved_turns_to_the_official_input(self):
        llm = ScriptedChatLLM({"agentworld_infer": ["<predicted_observation>\nout\n</predicted_observation>"] * 3})
        runner = AgentWorldRunner(mode="prompting_rag", chat_llm=llm, rag_corpus=self.corpus, rag_top_k=2, rag_turn_chars=300)
        outputs = runner.run(self.rows)
        self.assertEqual(parse_model_output(outputs[0]["gen"], RESPONSE_TAG), "out")
        info = outputs[0]["trace2env"]
        self.assertEqual((info["mode"], info["route"]), ("prompting_rag", "prompting_rag"))
        self.assertLessEqual(len(info["rag"]["hits"]), 2)
        self.assertTrue(info["rag"]["hits"])
        self.assertEqual(info["rag"]["query"], "pwd")
        official = inference_messages(case_from_row(self.rows[0]))
        sent = llm.calls[0]["messages"]
        self.assertEqual(sent[1:], official[1:])  # history and current prompt untouched
        self.assertTrue(sent[0]["content"].startswith(official[0]["content"]))  # official system prompt first ...
        self.assertIn("# Retrieved examples", sent[0]["content"])  # ... then the retrieved turns
        self.assertGreater(info["prompt_chars"], sum(len(m["content"]) for m in official))
        self.assertIn("usage", info)

    def test_mode_requires_a_corpus(self):
        with self.assertRaises(ValueError):
            AgentWorldRunner(mode="prompting_rag", chat_llm=ScriptedChatLLM({}))


if __name__ == "__main__":
    unittest.main()
