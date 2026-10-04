"""MCPMark / Toolathlon trajectories → Trace2Env episodes and benchmark rows. Fixtures are synthetic."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import case_from_row
from trace2env.atif import BENCHMARK_SUFFIX
from trace2env.cli import main
from trace2env.mcp_traces import (
    episode_rows,
    export_manifest,
    mcpmark_episode,
    mcpmark_turns,
    render_tool_definitions,
    toolathlon_episode,
    toolathlon_tool_name,
    toolathlon_turns,
)
from trace2env.storage import read_json, write_json

ENVELOPE = '{"type":"text","text":"%s","annotations":null}'


def mcpmark_run(root: Path) -> Path:
    task = root / "claude-opus-4-1__postgres" / "run-1" / "chinook__customer_data_migration"
    task.mkdir(parents=True)
    write_json(task / "meta.json", {"task_name": "chinook__customer_data_migration", "model_name": "claude-opus-4-1", "mcp": "postgres",
                                    "execution_result": {"success": False, "error_message": None, "verification_error": "x"},
                                    "token_usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "turn_count": 3})
    write_json(task / "messages.json", [
        {"role": "user", "content": "Migrate the customer table."},
        {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": "Listing schemas first."}]},
        {"type": "function_call", "call_id": "c1", "name": "list_schemas", "arguments": ""},
        {"type": "function_call_output", "call_id": "c1", "output": ENVELOPE % "public"},
        {"role": "assistant", "type": "message", "content": [{"type": "output_text", "text": ""}]},
        {"type": "function_call", "call_id": "c2", "name": "execute_sql", "arguments": "{\"sql\": \"SELECT 1\"}"},
        {"type": "function_call", "call_id": "c3", "name": "execute_sql", "arguments": "{\"sql\": \"SELECT 2\"}"},
        {"type": "function_call_output", "call_id": "c2", "output": ENVELOPE % "1"},
        {"type": "function_call_output", "call_id": "c3", "output": ENVELOPE % "2"},
        {"type": "function_call", "call_id": "c4", "name": "execute_sql", "arguments": "not json"},  # never answered
    ])
    return task


def toolathlon_run(root: Path, run: str = "1", *, passed: bool = True) -> Path:
    task = root / f"aws_anthropic_bedrock-claude-opus-4-8_{run}" / "finalpool" / "ab-testing"
    task.mkdir(parents=True)
    write_json(task / "traj_log.json", {
        "config": {"task_dir": "finalpool/ab-testing", "needed_mcp_servers": ["google-cloud", "filesystem"],
                   "needed_local_tools": ["claim_done", "python_execute", "manage_context"]},
        "status": "success",
        "key_stats": {"interaction_turns": 1, "tool_calls": 3, "total_tokens": 100},
        "tool_calls": {"tools": [
            {"type": "function", "function": {"name": "filesystem_read_file", "description": "Read a file.",
                                              "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Path"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "google_cloud_bigquery_list_datasets", "description": "List datasets.", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "local_claim_done", "description": "Done.", "parameters": {"type": "object", "properties": {}}}},
        ]},
        "messages": [
            {"role": "user", "content": "Analyze the A/B test."},
            {"role": "assistant", "content": "Exploring.", "thinking": "private", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "filesystem_read_file", "arguments": "{\"path\": \"/w/record.csv\"}"}},
                {"id": "t2", "type": "function", "function": {"name": "google_cloud_bigquery_list_datasets", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": ENVELOPE % "a,b"},
            {"role": "tool", "tool_call_id": "t2", "content": ENVELOPE % "Found 1 datasets"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t3", "type": "function", "function": {"name": "local_check_context_status", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t3", "content": "context ok"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t4", "type": "function", "function": {"name": "local_claim_done", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t4", "content": [{"type": "text", "text": "Task claimed as done."}]},
        ],
    })
    write_json(task / "eval_res.json", {"pass": passed, "details": "All evaluation checks passed"})
    return task


class McpTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.mcpmark = mcpmark_run(self.root / "mcpmark")
        self.toolathlon = toolathlon_run(self.root / "toolathlon")

    def test_mcpmark_turns_pair_calls_with_outputs_by_call_id(self):
        turns = mcpmark_turns(read_json(self.mcpmark / "messages.json"))
        self.assertEqual([turn["name"] for turn in turns], ["list_schemas", "execute_sql", "execute_sql"])  # c4 unanswered
        self.assertEqual(turns[0]["message"], "Listing schemas first.")
        self.assertEqual([turn["batch_size"] for turn in turns], [1, 2, 2])
        episode = mcpmark_episode(self.mcpmark)
        self.assertEqual((episode["turns"], episode["service"], episode["trajectory_group"]), (3, "postgres", "mcpmark:postgres:chinook"))
        self.assertEqual(episode["events"][1]["content"], {"type": "list_schemas", "arguments": {}, "raw": {"name": "list_schemas", "arguments": ""}})
        self.assertEqual(episode["events"][3]["content"]["arguments"], {"sql": "SELECT 1"})
        self.assertEqual(episode["events"][2]["content"], ENVELOPE % "public")  # verbatim envelope
        self.assertEqual(episode["outcome"], {"solved": False, "harness_error": None, "verification_error": "x"})
        self.assertIsNone(episode["tool_definitions"])

    def test_toolathlon_names_follow_the_benchmark_and_context_tools_are_dropped(self):
        servers = ["google-cloud", "filesystem"]
        self.assertEqual(toolathlon_tool_name("google_cloud_bigquery_list_datasets", servers), "google-cloud-bigquery_list_datasets")
        self.assertEqual(toolathlon_tool_name("filesystem_read_file", servers), "filesystem-read_file")
        self.assertEqual(toolathlon_tool_name("local_claim_done", servers), "local-claim_done")
        turns = toolathlon_turns(read_json(self.toolathlon / "traj_log.json"))
        self.assertEqual([turn["type"] for turn in turns], ["filesystem-read_file", "google-cloud-bigquery_list_datasets", "local-claim_done"])
        self.assertEqual([turn["batch_size"] for turn in turns], [2, 2, 1])
        self.assertEqual(turns[2]["output"], "Task claimed as done.")  # content-part list joined
        episode = toolathlon_episode(self.toolathlon)
        self.assertEqual((episode["turns"], episode["trajectory_id"], episode["split"]), (3, "toolathlon:ab-testing:1", episode["split"]))
        self.assertEqual(episode["outcome"]["solved"], True)
        self.assertEqual(episode["tool_definitions"][0]["name"], "filesystem-read_file")
        self.assertNotIn("thinking", json.dumps(episode["events"]))
        self.assertEqual(episode["events"][1]["metadata"]["agent_message"], "Exploring.")

    def test_episodes_ingest_and_segment_one_slice_per_tool_call(self):
        for episode in (mcpmark_episode(self.mcpmark), toolathlon_episode(self.toolathlon)):
            path = self.root / f"{episode['source']}.json"
            write_json(path, episode)
            loaded = load_raw_traces(path)[0]
            slices = segment_transitions(loaded)
            self.assertEqual(len(slices), episode["turns"])
            self.assertFalse(any(s.alignment == "ambiguous" or s.concurrent_action_event_ids for s in slices))
            self.assertEqual(loaded.metadata["trajectory_group"], episode["trajectory_group"])

    def test_export_manifest_writes_episodes_tools_and_split_manifest(self):
        manifest = self.root / "sample.json"
        write_json(manifest, {"entries": [
            {"source": "mcpmark", "dir": str(self.mcpmark), "split": "train"},
            {"source": "toolathlon", "dir": str(self.toolathlon), "split": "validation"},
        ]})
        paths, split_manifest, summaries = export_manifest(manifest, self.root / "episodes")
        self.assertEqual(len(paths), 2)
        self.assertEqual(sorted(a.split for a in split_manifest.assignments.values()), ["train", "validation"])
        toolathlon = read_json(paths[1])
        self.assertEqual(toolathlon["tool_definitions"], "tools/toolathlon__ab-testing.json")
        self.assertEqual(read_json(self.root / "episodes" / toolathlon["tool_definitions"])[1]["harness_name"], "google_cloud_bigquery_list_datasets")
        self.assertEqual([s["split"] for s in summaries], ["train", "validation"])

    def test_rows_use_normalized_names_and_official_layout(self):
        episode = toolathlon_episode(self.toolathlon)
        rows = episode_rows(episode, tool_definitions=episode["tool_definitions"])
        self.assertEqual([row["turn_idx"] for row in rows], [1, 2, 3])
        first = rows[0]
        self.assertTrue(first["prompt"][0].startswith('### Turn 1\n**Action:**\n```json\n{\n  "name": "filesystem-read_file"'))
        self.assertTrue(first["prompt"][-1].endswith(BENCHMARK_SUFFIX))
        self.assertTrue(first["response"][0].startswith("**Environment Observation:**\n{\"type\":\"text\""))
        self.assertIn("### 1. filesystem-read_file\n#### **Description**:\nRead a file.\n\n#### **Parameters**:\n- **path** `string` (REQUIRED): Path",
                      first["system_str"])
        self.assertNotIn("{tool_definitions}", first["system_str"])
        case = case_from_row(rows[1])
        self.assertEqual((case.action.type, case.action.arguments), ("google-cloud-bigquery_list_datasets", {}))
        names_only = render_tool_definitions(None, ["execute_sql"])
        self.assertIn("### 1. execute_sql", names_only)

    def test_cli_export(self):
        manifest = self.root / "sample.json"
        write_json(manifest, {"entries": [
            {"source": "mcpmark", "dir": str(self.mcpmark), "split": "validation"},
            {"source": "toolathlon", "dir": str(self.toolathlon), "split": "train"},
        ]})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["mcp-export", str(manifest), "--output", str(self.root / "episodes"), "--rows", str(self.root / "rows.jsonl"),
                                   "--turns-per-trajectory", "2", "--summary", str(self.root / "summary.json")]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual((report["episodes"], report["turns"], report["rows"], report["splits"]), (2, 6, 2, {"validation": 1, "train": 1}))
        self.assertTrue((self.root / "episodes" / "split_manifest.json").exists())
        rows = [json.loads(line) for line in (self.root / "rows.jsonl").read_text().splitlines()]
        self.assertEqual({row["source"] for row in rows}, {"mcpmark"})  # rows default to the validation split


if __name__ == "__main__":
    unittest.main()
