"""ATIF (Harbor / Terminus-2) trajectories → Trace2Env episodes and benchmark rows. Fixtures are synthetic."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import case_from_row, inference_messages, split_of
from trace2env.atif import (
    BENCHMARK_SUFFIX,
    episode_rows,
    export_trajectories,
    find_trajectories,
    initial_context,
    load_trajectory,
    terminus_turns,
    trajectory_episode,
)
from trace2env.cli import main
from trace2env.reconstruction import TraceReconstructor
from trace2env.llm import ScriptedLLM
from trace2env.models import ReconstructionConfig
from trace2env.storage import read_json

PROMPT = ("You are an AI assistant tasked with solving command-line tasks.\n\nTask Description:\nFix the broken git "
          "repository in /app.\n\nCurrent terminal state:\nroot@abc:/app# ")


def agent_step(step_id: int, commands: list[dict], observation: str, *, complete: bool = False, copied: bool = False) -> dict:
    calls = [{"tool_call_id": f"call_{step_id}_{i + 1}", "function_name": "bash_command", "arguments": command}
             for i, command in enumerate(commands)]
    if complete:
        calls.append({"tool_call_id": f"call_{step_id}_task_complete", "function_name": "mark_task_complete", "arguments": {}})
    step = {"step_id": step_id, "timestamp": "2026-09-16T10:00:00+00:00", "source": "agent", "model_name": "openrouter/anthropic/claude-opus-4.6",
            "message": f"Analysis: step {step_id}\nPlan: continue", "tool_calls": calls or None,
            "observation": {"results": [{"source_call_id": calls[0]["tool_call_id"] if len(commands) == 1 else None, "content": observation}]},
            "metrics": {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 800, "cost_usd": 0.01}}
    if copied:
        step["is_copied_context"] = True
    return step


def trajectory(root: Path) -> Path:
    steps = [
        {"step_id": 1, "source": "user", "message": PROMPT},
        agent_step(2, [{"keystrokes": "git status\n", "duration": 0.1}], "root@abc:/app# git status\nfatal: not a git repository\nroot@abc:/app# "),
        {"step_id": 3, "source": "agent", "model_name": "m", "message": "not json", "observation": {"results": [{"content": "Previous response had parsing errors:\nERROR: bad"}]},
         "metrics": {"prompt_tokens": 10, "completion_tokens": 5}},
        agent_step(4, [{"keystrokes": "ls -la\n", "duration": 0.1}, {"keystrokes": "cat .git/HEAD\n", "duration": 0.1}],
                   "root@abc:/app# ls -la\ntotal 0\nroot@abc:/app# cat .git/HEAD\nref: refs/heads/main\nroot@abc:/app# "),
    ]
    first = {"schema_version": "ATIF-v1.8", "session_id": "s1", "agent": {"name": "terminus-2", "version": "0.23.0", "model_name": "openrouter/anthropic/claude-opus-4.6"},
             "steps": steps, "continued_trajectory_ref": "trajectory.1.json"}
    continuation = {"schema_version": "ATIF-v1.8", "session_id": "s1", "agent": first["agent"], "steps": [
        {"step_id": 1, "source": "user", "message": "handoff summary"},
        agent_step(2, [{"keystrokes": "ls -la\n", "duration": 0.1}], "copied", copied=True),
        agent_step(3, [{"keystrokes": "git log --oneline\n", "duration": 0.5}], "root@abc:/app# git log --oneline\nabc123 init\nroot@abc:/app# ", complete=True),
    ]}
    trial = root / "fix-git__abc12"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(json.dumps(first), encoding="utf-8")
    (trial / "agent" / "trajectory.1.json").write_text(json.dumps(continuation), encoding="utf-8")
    (trial / "result.json").write_text(json.dumps({"task_name": "fix-git", "trial_name": "fix-git__abc12",
                                                   "verifier_result": {"rewards": {"reward": 1.0}}}), encoding="utf-8")
    return trial / "agent" / "trajectory.json"


class AtifTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = trajectory(self.root / "jobs" / "sanity")

    def test_continuations_are_spliced_and_only_environment_turns_are_kept(self):
        loaded = load_trajectory(self.path)
        self.assertEqual((len(loaded["steps"]), loaded["continuation_files"]), (7, 1))
        self.assertEqual(initial_context(loaded), ("Fix the broken git repository in /app.", "root@abc:/app#"))
        turns = terminus_turns(loaded)
        self.assertEqual([len(turn["commands"]) for turn in turns], [1, 2, 1])  # parse error and copied steps skipped
        self.assertTrue(turns[-1]["task_complete"])
        self.assertEqual(turns[1]["commands"][1]["keystrokes"], "cat .git/HEAD\n")

    def test_episode_matches_the_export_shape_and_ingests(self):
        episode = trajectory_episode(self.path, task="fix-git", trial="fix-git__abc12")
        kinds = [event["kind"] for event in episode["events"]]
        self.assertEqual(kinds, ["message", "state", "action", "observation", "action", "observation", "action", "observation"])
        self.assertEqual(episode["events"][2]["content"]["type"], "git")
        self.assertEqual(episode["events"][4]["content"]["type"], "shell")  # two commands in one batch
        self.assertEqual((episode["turns"], episode["task_complete"], episode["metrics"]["cost_usd"]), (3, True, 0.03))
        self.assertEqual(episode["split"], split_of("tb2:fix-git"))
        paths, manifest, summaries = export_trajectories([self.root / "jobs"], self.root / "episodes")
        self.assertEqual(len(paths), 1)
        self.assertEqual(summaries[0]["reward"], 1.0)
        self.assertEqual(find_trajectories([self.root / "jobs"])[0][1:], ("fix-git", "fix-git__abc12"))
        loaded = load_raw_traces(paths[0])[0]
        self.assertEqual(len(segment_transitions(loaded)), 3)
        self.assertEqual(loaded.metadata["trajectory_group"], "tb2:fix-git")
        recon = TraceReconstructor(ScriptedLLM({}), ReconstructionConfig(environment_id="tb2", name="T"), self.root / "work")
        expected_split = read_json(paths[0])["split"]
        if expected_split == "train":
            episodes, slices = recon.ingest(paths, manifest)
            self.assertEqual((len(episodes), len(slices)), (1, 3))
        else:
            with self.assertRaises(ValueError):
                recon.ingest(paths, manifest)

    def test_rows_follow_the_benchmark_record_format(self):
        episode = trajectory_episode(self.path, task="fix-git", trial="fix-git__abc12")
        rows = episode_rows(episode, system_prompt="SYS")
        self.assertEqual([row["turn_idx"] for row in rows], [1, 2, 3])
        first = rows[0]
        self.assertTrue(first["prompt"][0].startswith("### Turn 1\n**Current State:**\nroot@abc:/app#\n\n**Action:**\n```json"))
        self.assertTrue(first["prompt"][-1].endswith(BENCHMARK_SUFFIX))
        self.assertEqual(first["current_prompt"] + BENCHMARK_SUFFIX, first["prompt"][-1])
        self.assertTrue(first["response"][0].startswith("**Environment Observation:**\nroot@abc:/app# git status"))
        case = case_from_row(rows[2])
        self.assertEqual((case.action.type, case.turn_idx, len(inference_messages(case))), ("git", 3, 6))
        sampled = episode_rows(episode, turns_per_trajectory=2, system_prompt="SYS")
        self.assertEqual([row["turn_idx"] for row in sampled], [2, 3])

    def test_cli_export(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["atif-export", str(self.root / "jobs"), "--output", str(self.root / "episodes"),
                                   "--rows", str(self.root / "rows.jsonl"), "--turns-per-trajectory", "2",
                                   "--summary", str(self.root / "summary.json")]), 0)
        report = json.loads(out.getvalue())
        self.assertEqual((report["trials"], report["turns"], report["rows"]), (1, 3, 2))
        self.assertTrue((self.root / "episodes" / "split_manifest.json").exists())
        self.assertEqual(len(read_json(self.root / "summary.json")), 1)


if __name__ == "__main__":
    unittest.main()
