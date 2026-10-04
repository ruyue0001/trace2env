"""v3 read path: the package's raw evidence tier is searchable, pageable, and cited like any other artifact."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import terminal_rows
from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import RESPONSE_TAG, parse_model_output
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.cli import _features, main
from trace2env.compiler import EnvironmentCompiler
from trace2env.llm import ScriptedLLM, StructuredAgentLLM
from trace2env.models import (ActionSchema, ActionSpec, AgentTurn, ArgumentSpec, Confidence, Demonstration, LocalTransitionEvidence,
                              NormalizedAction, Outcome, ReconstructionArtifacts, ReconstructionConfig, SingleShotPrediction, StateField,
                              StateSchema, StateTrackingResult, TransitionSubmission)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.runtime import DEFAULT_FEATURES, HARNESS_FEATURES, normalize_citations
from trace2env.workspace import WorkspaceTools

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROWS = ROOT / "examples" / "agentworldbench" / "synthetic_terminal.jsonl"


def _argument_type(value):
    return "array" if isinstance(value, list) else "object" if isinstance(value, dict) else "number" if isinstance(value, (int, float)) else "string"


def evidence_package(root: Path) -> Path:
    """An authored package whose evidence tier holds the raw turns of the synthetic terminal trajectories."""
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["awb-export", str(SYNTHETIC_ROWS), "--output", str(root / "episodes"), "--split", "all"]) == 0
    episodes, transitions, evidence, specs = [], [], [], {}
    for path in sorted((root / "episodes").glob("*.json")):
        for episode in load_raw_traces(path):
            episodes.append(episode)
            by_id = {event.id: event for event in episode.events}
            for slice_ in segment_transitions(episode):
                transitions.append(slice_)
                action_event = by_id[slice_.action_event_ids[0]]
                content = action_event.content if isinstance(action_event.content, dict) else {"type": "shell", "arguments": {"text": str(action_event.content)}}
                action = NormalizedAction(type=content["type"], arguments=dict(content.get("arguments") or {}))
                spec = specs.setdefault(action.type, {})
                for key, value in action.arguments.items():
                    spec.setdefault(key, ArgumentSpec(type=_argument_type(value)))
                observation = "\n".join(str(by_id[i].content) for i in slice_.observation_event_ids)
                evidence.append(LocalTransitionEvidence(
                    id=f"evidence_{slice_.id}", episode_id=episode.id, transition_id=slice_.id, action=action,
                    outcome=Outcome.SUCCESS, observation_text=observation, confidence=Confidence(value=0.9)))
    demos = [Demonstration(id=f"demo_{item.id}", action=item.action, observation=item.observation_text, outcome=Outcome.SUCCESS)
             for item in evidence[:2]]
    artifacts = ReconstructionArtifacts(
        episodes=episodes, transitions=transitions, evidence=evidence,
        action_schema=ActionSchema(actions=[ActionSpec(name=name, description=f"{name} command", arguments=arguments)
                                            for name, arguments in specs.items()]),
        state_schema=StateSchema(fields=[StateField(path="session.cwd", type="string", default="/app", visibility="session"),
                                         StateField(path="surface.prompt", type="string", default="root@host:/app#", visibility="surface")]),
        rules=[], renderers=[], demonstrations=demos,
    )
    config = ReconstructionConfig(environment_id="test.evidence", name="Synthetic terminal with evidence", construction_kind="authored")
    return EnvironmentCompiler(config).compile(artifacts, root / "evidence-package")


class EvidenceReadPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = evidence_package(self.root)
        self.package = EnvironmentPackage(self.package_dir)
        self.inspector = PackageInspector(self.package)

    def test_search_returns_summaries_and_read_pages_with_neighbours(self):
        hits = self.inspector.search_evidence("pwd", 5)
        self.assertTrue(hits)
        first = hits[0]
        self.assertTrue(first["id"].startswith("evidence:"))
        self.assertIn("snippet", first)
        self.assertNotIn("text", first)
        self.assertEqual(first["read"], f"read_evidence(id='{first['id']}')")
        evidence_id = first["id"].split(":", 1)[1]
        page = self.inspector.evidence_view(evidence_id, offset=0, length=4)
        self.assertEqual((page["offset"], len(page["text"])), (0, min(4, page["total_chars"])))
        self.assertGreaterEqual(page["turn"], 1)
        neighbours = {item["turn"] for item in page["neighbours"]}
        self.assertTrue(neighbours <= {page["turn"] - 1, page["turn"] + 1})
        demo = self.package.demonstrations[0]
        self.assertEqual(self.inspector.resolve_evidence_id(f"demo:{demo.id}"), demo.id[len("demo_"):])
        self.assertIsNone(self.inspector.resolve_evidence_id("evidence:nope"))

    def test_workspace_shapes_hits_and_registers_reads(self):
        state = self.package and __import__("trace2env.runtime", fromlist=["initial_state_from_package"]).initial_state_from_package(self.package)
        tools = WorkspaceTools(self.package, self.inspector, None, state, NormalizedAction(type="pwd"), features={"evidence", "history"})
        names = [tool.name for tool in tools.tools()]
        self.assertIn("read_evidence", names)
        hits = tools.call("search_knowledge", {"query": "pwd", "kinds": ["evidence"]})
        self.assertTrue(hits)
        self.assertNotIn("text", hits[0])
        self.assertIn(hits[0]["id"], tools.retrieved_ids)
        page = tools.call("read_evidence", {"id": hits[0]["id"], "length": 10})
        self.assertEqual(len(page["text"]), min(10, page["total_chars"]))
        self.assertIn("error", tools.call("read_evidence", {"id": "evidence:missing"}))
        view = tools.demo_view(self.package.demonstrations[0])
        self.assertTrue(view["read"].startswith("read_evidence("))
        legacy = WorkspaceTools(self.package, self.inspector, None, state, NormalizedAction(type="pwd"), features=set())
        self.assertNotIn("read_evidence", [tool.name for tool in legacy.tools()])
        self.assertIn("text", legacy.call("search_knowledge", {"query": "pwd", "kinds": ["evidence"]})[0])

    def test_agent_brief_lists_similar_turns_and_citations_normalize(self):
        hit = self.inspector.search_evidence("pwd", 1)[0]
        bare = hit["id"].split(":", 1)[1]  # the model cites the raw id without its kind prefix
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="root@host:/app# pwd\n/app\nroot@host:/app#",
                                          effects=[], citations=[bare, "memory:1"], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3,
                           "runtime_agent_turn": [AgentTurn(tool="read_evidence", arguments={"id": hit["id"]}), AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  features={"history", "wait", "compact", "unknown", "evidence"})
        [output] = runner.run([terminal_rows()[1]])
        info = output["trace2env"]
        self.assertEqual((info["route"], info["tool_calls"]), ("agent", ["read_evidence"]))
        self.assertEqual(info["citations"], [hit["id"], "memory:1"])
        self.assertNotIn("uncited_reference", info["verification"])
        call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        self.assertIn("read_evidence", call["system"])
        payload = json.loads(call["user"])
        brief = json.loads(payload["conversation"][0]["content"])
        self.assertTrue(brief["similar_turns"])
        self.assertIn("read", brief["similar_turns"][0])
        self.assertTrue(all("read" in demo or "observation_head" in demo for demo in brief["retrieved"]["demonstrations"]))
        self.assertIn("read_evidence", [tool["name"] for tool in payload["tools"]])

    def test_single_shot_gets_evidence_pages(self):
        prediction = SingleShotPrediction(observation="root@host:/app# cd tests\nroot@host:/app/tests#")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_single_shot": [prediction]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, prediction_mode="single_shot",
                                  features={"history", "wait", "compact", "unknown", "evidence"})
        [output] = runner.run([terminal_rows()[1]])
        self.assertEqual(output["trace2env"]["route"], "single_shot")
        call = next(call for call in llm.calls if call["role"] == "runtime_single_shot")
        self.assertIn("similar_turns holds", call["system"])
        fixed = json.loads(call["user"])["fixed_retrieval"]
        self.assertTrue(fixed["similar_turns"])
        self.assertIn("text", fixed["similar_turns"][0])
        self.assertTrue(all(hit["kind"] != "evidence" for hit in fixed["knowledge_hits"]))

    def test_normalize_citations_and_feature_lists(self):
        known = {"note:bash-continuation-prompt", "evidence:evidence_tr_1", "rule:r1", "memory:3"}
        self.assertEqual(normalize_citations(["bash-continuation-prompt", "evidence_tr_1", "memory:3", "unknown-thing"], known),
                         ["note:bash-continuation-prompt", "evidence:evidence_tr_1", "memory:3", "unknown-thing"])
        self.assertEqual(normalize_citations(["r1", "r1"], known | {"note:r1"}), ["r1"])  # ambiguous stays as written

        class Args:
            features = "default,evidence"
        self.assertEqual(_features(Args()), set(DEFAULT_FEATURES) | {"evidence"})
        Args.features = "all"
        self.assertEqual(_features(Args()), "all")
        self.assertIn("evidence", HARNESS_FEATURES)
        self.assertIn("evidence", DEFAULT_FEATURES)  # v3 default since the 2026-09-20 full-set result


if __name__ == "__main__":
    unittest.main()
