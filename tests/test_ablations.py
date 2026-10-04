"""Baselines and ablations of the reconstructed workspace: schema-only packages, raw-trace retrieval, single-shot prediction."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import terminal_package, terminal_rows
from trace2env.agentworld import RESPONSE_TAG, parse_model_output
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.cli import main
from trace2env.llm import ScriptedLLM, StructuredAgentLLM
from trace2env.models import AgentTurn, Outcome, SingleShotPrediction, StateTrackingResult, TransitionSubmission
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.trace_corpus import TraceCorpus, action_query

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROWS = ROOT / "examples" / "agentworldbench" / "synthetic_terminal.jsonl"


def load_ablate():
    spec = importlib.util.spec_from_file_location("ablate_package", ROOT / "scripts" / "ablate_package.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def submission(observation: str, citations: list[str] | None = None) -> TransitionSubmission:
    return TransitionSubmission(outcome=Outcome.SUCCESS, observation=observation, effects=[], citations=citations or [], rationale="")


class AblatedPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = terminal_package(self.root)
        self.ablated = load_ablate().ablate(self.source, self.root / "schema-only")
        self.rows = terminal_rows()

    def test_schema_only_package_keeps_schemas_and_records_provenance(self):
        package = EnvironmentPackage(self.ablated)
        self.assertEqual([action.name for action in package.action_schema.actions], ["pwd", "cd"])
        self.assertEqual([field.path for field in package.state_schema.fields], ["session.cwd", "surface.prompt"])
        self.assertEqual((package.rules, package.renderers, package.demonstrations, package.notes, package.invariants), ([], [], [], [], []))
        metadata = package.manifest.metadata
        self.assertEqual((metadata["construction_kind"], metadata["validation_status"]), ("ablated", "candidate"))
        self.assertEqual(metadata["ablation"]["kept"], ["action_schema", "state_schema"])
        self.assertIn("rules", metadata["ablation"]["removed"])
        self.assertEqual(metadata["ablation"]["source_counts"]["rules"], 1)
        self.assertEqual(len(metadata["ablation"]["source_manifest_sha256"]), 64)
        hits = PackageInspector(package).search_knowledge("pwd")
        self.assertEqual({hit["kind"] for hit in hits}, {"action"})  # nothing reconstructed is left to find
        with self.assertRaises(FileExistsError):
            load_ablate().ablate(self.source, self.ablated)

    def test_tier_modes_keep_one_tier_each(self):
        structure = EnvironmentPackage(load_ablate().ablate(self.source, self.root / "structure-only", "structure"))
        self.assertEqual(([rule.id for rule in structure.rules], len(structure.renderers), structure.demonstrations), (["pwd_read"], 1, []))
        self.assertEqual(structure.manifest.metadata["ablation"]["mode"], "structure")
        self.assertTrue(all(not rule.provenance for rule in structure.rules))  # trace material is gone, so is its provenance
        examples = EnvironmentPackage(load_ablate().ablate(self.source, self.root / "examples-only", "examples"))
        self.assertEqual((examples.rules, examples.renderers, examples.notes), ([], [], []))
        self.assertEqual(examples.manifest.metadata["ablation"]["kept"][:4], ["action_schema", "state_schema", "evidence", "demonstrations"])

    def test_schema_only_runs_every_turn_through_the_agent(self):
        llm = ScriptedLLM({
            "track_state": [StateTrackingResult()] * 3,
            "runtime_agent_turn": [AgentTurn(final=submission("root@host:/app# pwd\n/app\nroot@host:/app#"))] * 3,
        })
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.ablated), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  allow_unvalidated=True)
        outputs = runner.run(self.rows)
        self.assertEqual([row["trace2env"]["route"] for row in outputs], ["agent", "agent", "agent"])  # no pwd rule any more
        self.assertEqual(outputs[0]["trace2env"]["package_ablation"]["mode"], "schemas")
        self.assertEqual(outputs[0]["trace2env"]["prediction_mode"], "agent")
        agent_call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        brief = json.loads(json.loads(agent_call["user"])["conversation"][0]["content"])
        self.assertEqual((brief["retrieved"]["rules"], brief["retrieved"]["demonstrations"], brief["retrieved"]["notes"]), ([], [], []))
        self.assertEqual([field["path"] for field in brief["state_fields"]], ["session.cwd", "surface.prompt"])
        self.assertNotIn("trace_hits", brief)
        self.assertNotIn("search_traces", [tool["name"] for tool in json.loads(agent_call["user"])["tools"]])


class TraceCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["awb-export", str(SYNTHETIC_ROWS), "--output", str(self.root / "episodes"), "--split", "all"]), 0)
        self.corpus = TraceCorpus.build(self.root / "episodes", self.root / "corpus.sqlite")
        self.ablated = load_ablate().ablate(terminal_package(self.root), self.root / "schema-only")
        self.rows = terminal_rows()

    def test_corpus_indexes_raw_turns_and_pages_them(self):
        self.assertGreater(self.corpus.turns, 0)
        self.assertGreater(self.corpus.episodes, 0)
        self.assertEqual(action_query({"type": "cd", "arguments": {"argv": ["tests"], "command": "cd tests"}}), "cd tests")
        hits = self.corpus.search("pwd")
        self.assertTrue(hits)
        self.assertTrue(hits[0]["id"].startswith("trace:"))
        self.assertIn("snippet", hits[0])
        page = self.corpus.read(hits[0]["episode"], hits[0]["turn"], offset=0, length=5)
        self.assertEqual((page["offset"], len(page["text"])), (0, min(5, page["total_chars"])))
        self.assertIsNone(self.corpus.read("no-such-episode", 1))
        self.assertEqual(TraceCorpus(self.root / "corpus.sqlite").turns, self.corpus.turns)  # reopenable

    def test_agent_can_search_and_read_raw_traces_instead_of_knowledge(self):
        hit = self.corpus.search("pwd")[0]
        llm = ScriptedLLM({
            "track_state": [StateTrackingResult()] * 3,
            "runtime_agent_turn": [
                AgentTurn(tool="search_traces", arguments={"query": "pwd"}),
                AgentTurn(tool="read_trace_turn", arguments={"episode": hit["episode"], "turn": hit["turn"]}),
                AgentTurn(final=submission("root@host:/app# pwd\n/app\nroot@host:/app#", citations=[hit["id"]])),
            ],
        })
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.ablated), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  allow_unvalidated=True, trace_corpus=self.corpus)
        [output] = runner.run([self.rows[0]])
        info = output["trace2env"]
        self.assertEqual((info["route"], info["tool_calls"]), ("agent", ["search_traces", "read_trace_turn"]))
        self.assertNotIn("uncited_reference", info["verification"])  # the cited trace turn was retrieved
        self.assertEqual(info["trace_corpus"]["turns"], self.corpus.turns)
        self.assertEqual(parse_model_output(output["gen"], RESPONSE_TAG), "root@host:/app# pwd\n/app\nroot@host:/app#")
        first = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        self.assertIn("Raw traces", first["system"])
        payload = json.loads(first["user"])
        self.assertIn("search_traces", [tool["name"] for tool in payload["tools"]])
        brief = json.loads(payload["conversation"][0]["content"])
        self.assertTrue(brief["trace_hits"])
        self.assertEqual(brief["trace_corpus"]["turns"], self.corpus.turns)
        read_result = json.loads(llm.calls[-1]["user"])["conversation"][-1]
        self.assertEqual(read_result["role"], "tool")
        self.assertIn("neighbours", read_result["content"])


class SingleShotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = terminal_package(self.root)
        self.rows = terminal_rows()

    def test_single_shot_makes_one_call_over_a_fixed_brief(self):
        prediction = SingleShotPrediction(observation="root@host:/app# cd tests\nroot@host:/app/tests#", citations=["memory:1"])
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_single_shot": [prediction]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, prediction_mode="single_shot")
        outputs = runner.run(self.rows)
        by_turn = {row["turn_idx"]: row for row in outputs}
        self.assertEqual(by_turn[1]["trace2env"]["route"], "rule")  # the confident pwd template still short-circuits
        info = by_turn[2]["trace2env"]
        self.assertEqual((info["route"], info["prediction_mode"], info["tool_calls"]), ("single_shot", "single_shot", []))
        self.assertEqual(parse_model_output(by_turn[2]["gen"], RESPONSE_TAG), prediction.observation)
        self.assertNotIn("uncited_reference", info["verification"])  # memory:1 was in the brief
        self.assertEqual(info["usage"]["model_calls"], 1)
        roles = [call["role"] for call in llm.calls]
        self.assertNotIn("runtime_agent_turn", roles)
        self.assertEqual(roles.count("runtime_single_shot"), 1)
        call = next(call for call in llm.calls if call["role"] == "runtime_single_shot")
        self.assertIn("You have no tools", call["system"])
        brief = json.loads(call["user"])
        self.assertNotIn("budget", brief)
        self.assertEqual(brief["fixed_retrieval"]["query"], "cd tests")
        self.assertEqual({hit["kind"] for hit in brief["fixed_retrieval"]["knowledge_hits"]} <= {"action", "rule", "note", "demonstration", "evidence"}, True)
        self.assertIsNone(brief["fixed_retrieval"]["applicable_rule_dry_run"])
        self.assertEqual([entry["turn"] for entry in brief["recent_memory"]], [0, 1])

    def test_unknown_prediction_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), prediction_mode="oracle")  # type: ignore[arg-type]


class CliModeTests(unittest.TestCase):
    def test_awb_run_accepts_the_ablation_flags_without_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ablated = load_ablate().ablate(terminal_package(root), root / "schema-only")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["awb-export", str(SYNTHETIC_ROWS), "--output", str(root / "episodes"), "--split", "all"]), 0)
                status = main(["awb-run", str(SYNTHETIC_ROWS), "--mode", "agentic", "--package", str(ablated), "--no-model",
                               "--allow-unvalidated", "--prediction-mode", "single_shot", "--trace-corpus", str(root / "episodes"),
                               "--session-root", str(root / "sessions"), "--output", str(root / "pred.jsonl")])
            self.assertEqual(status, 0)
            rows = [json.loads(line) for line in (root / "pred.jsonl").read_text().splitlines() if line.strip()]
            self.assertTrue(rows)
            self.assertEqual(rows[0]["trace2env"]["prediction_mode"], "single_shot")
            self.assertGreater(rows[0]["trace2env"]["trace_corpus"]["turns"], 0)
            self.assertTrue((root / "sessions" / "trace_corpus.sqlite").exists())


if __name__ == "__main__":
    unittest.main()


class GracefulFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = terminal_package(self.root)
        self.rows = terminal_rows()

    def test_failed_agent_step_degrades_to_a_single_shot_prediction(self):
        prediction = SingleShotPrediction(observation="root@host:/app# cd tests\nroot@host:/app/tests#", citations=["memory:1"])
        # The agent transport has no scripted turns, so every agent step raises; the fallback must answer from the brief.
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_single_shot": [prediction]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm))
        outputs = runner.run(self.rows)
        by_turn = {row["turn_idx"]: row for row in outputs}
        info = by_turn[2]["trace2env"]
        self.assertEqual(info["route"], "fallback_single_shot")
        self.assertIn("harness_error", info)
        self.assertEqual(parse_model_output(by_turn[2]["gen"], RESPONSE_TAG), prediction.observation)
        self.assertEqual(info["usage"]["model_calls"], 1)


class TraceCorpusLabelTests(unittest.TestCase):
    def test_files_sharing_a_name_prefix_get_distinct_episode_labels(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "episodes").mkdir()
            for task in ("105", "106"):
                (root / "episodes" / f"webarena__{task}__r1.json").write_text(json.dumps({
                    "episode_id": f"webarena:{task}:r1",
                    "events": [{"actor": "agent", "kind": "action", "content": {"type": "browser_navigate", "arguments": {"url": f"http://x/{task}"}, "raw": "n"}, "call_id": "c1"},
                               {"actor": "environment", "kind": "observation", "content": f"### Page\n- Page URL: http://x/{task}", "call_id": "c1"}]}), encoding="utf-8")
            (root / "episodes" / "build-pov-ray__run1.json").write_text(json.dumps({
                "episode_id": "tb2:build-pov-ray:run1",
                "events": [{"actor": "agent", "kind": "action", "content": {"type": "ls", "arguments": {}, "raw": "ls"}, "call_id": "c1"},
                           {"actor": "environment", "kind": "observation", "content": "a b", "call_id": "c1"}]}), encoding="utf-8")
            TraceCorpus.build(root / "episodes", root / "corpus.sqlite")
            labels = sorted(r[0] for r in sqlite3.connect(str(root / "corpus.sqlite")).execute("SELECT DISTINCT episode FROM turns"))
        self.assertEqual(labels, ["build-pov-ray", "webarena__105__r1", "webarena__106__r1"])
