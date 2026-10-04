"""Harness v5.2: compatibility-gated package knowledge (labels, sanitized views, abstention, v5.1 reproduction)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import SUFFIX, SYSTEM, keystrokes_prompt, train_trajectory_id
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.compiler import EnvironmentCompiler
from trace2env.knowledge_gate import (
    same_screen, screen_signature,
    EpisodeContext,
    GateDecision,
    KnowledgeGate,
    build_episode_context,
    specific_tokens,
    normalize_page,
)
from trace2env.llm import ScriptedLLM, StructuredAgentLLM
from trace2env.models import (
    ActionSchema,
    ActionSpec,
    AgentTurn,
    ArgumentSpec,
    Confidence,
    EnvironmentNote,
    Episode,
    Fact,
    LocalTransitionEvidence,
    NormalizedAction,
    Outcome,
    RawEvent,
    ReconstructionArtifacts,
    ReconstructionConfig,
    StateField,
    StateSchema,
    StateTrackingResult,
    TransitionSlice,
    TransitionSubmission,
)
from trace2env.package import EnvironmentPackage, PackageInspector

FOREIGN_LISTING = ("root@06e31eef63ee:/app# ls -la /app/\ntotal 24\ndrwxr-xr-x 1 root root  4096 Oct 31  2025 .\n"
                   "-rwxr-xr-x 1 root root 14520 Oct 31  2025 vulnerable\nroot@06e31eef63ee:/app#")
SAME_LISTING = ("root@e88ac1109c97:/app# ls warriors/\ng2-clear.red  paper.red  stone.red\nroot@e88ac1109c97:/app# cat warriors/stone.red\n"
                ";redcode-94\n;name Stone\nroot@e88ac1109c97:/app#")


def evidence_package(root: Path) -> Path:
    """A package with two evidence turns: one from a run of this task family, one from another container."""
    config = ReconstructionConfig(environment_id="test.terminal", name="Synthetic terminal", construction_kind="authored")
    same = LocalTransitionEvidence(
        id="evidence_tr_same", episode_id="ep_same", transition_id="tr_same",
        action=NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/", "cat warriors/stone.red"]}),
        observation_text=SAME_LISTING, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
        preconditions=[Fact(subject="session.cwd", predicate="equals", value="/app")],
        observation_facts=[Fact(subject="world.files", predicate="contains", value={"/app/warriors/stone.red": {"type": "file"}})],
    )
    foreign = LocalTransitionEvidence(
        id="evidence_tr_foreign", episode_id="ep_foreign", transition_id="tr_foreign",
        action=NormalizedAction(type="ls", arguments={"argv": ["-la", "/app/"], "commands": ["ls -la /app/"]}),
        observation_text=FOREIGN_LISTING, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
        preconditions=[Fact(subject="session.cwd", predicate="equals", value="/app")],
        observation_facts=[Fact(subject="world.files", predicate="contains", value={"/app/vulnerable": {"type": "file"}})],
    )
    artifacts = ReconstructionArtifacts(
        episodes=[Episode(id="ep_same", source_id="src_same", metadata={"episode_id": "tb2:corewars-like:run1"},
                          events=[RawEvent(id="a1", content="ls warriors/"), RawEvent(id="o1", content=SAME_LISTING)]),
                  Episode(id="ep_foreign", source_id="src_foreign", metadata={"episode_id": "tb2:vulnerable-secret:run2"},
                          events=[RawEvent(id="a2", content="ls -la /app/"), RawEvent(id="o2", content=FOREIGN_LISTING)])],
        transitions=[TransitionSlice(id="tr_same", episode_id="ep_same", action_event_ids=["a1"], observation_event_ids=["o1"]),
                     TransitionSlice(id="tr_foreign", episode_id="ep_foreign", action_event_ids=["a2"], observation_event_ids=["o2"])],
        evidence=[same, foreign], rules=[], renderers=[],
        action_schema=ActionSchema(actions=[
            ActionSpec(name="ls", description="List files.", arguments={"argv": ArgumentSpec(type="array")}),
            ActionSpec(name="cat", description="Print a file.", arguments={"argv": ArgumentSpec(type="array")}),
        ]),
        state_schema=StateSchema(fields=[
            StateField(path="session.cwd", type="string", default="/app", visibility="session"),
            StateField(path="surface.prompt", type="string", default="root@host:/app#", visibility="surface"),
        ]),
        notes=[EnvironmentNote(id="bash-continuation-prompt", kind="convention",
                               statement="Heredoc input is displayed with the continuation prompt `> ` before continued lines.",
                               action_types=["ls", "cat"]),
               EnvironmentNote(id="ls-long-layout", kind="format", statement="`ls -la` begins with `total N` then one row per entry.",
                               action_types=["ls"])],
    )
    return EnvironmentCompiler(config).compile(artifacts, root / "evidence-package")


def rows_with_history(trajectory_id: str) -> list[dict]:
    """Turn 1 shows a file of this task family; turn 2 lists the directory (the evaluated row)."""
    prompts = [keystrokes_prompt(1, "cat warriors/stone.red", "root@host:/app#"), keystrokes_prompt(2, "ls warriors/")]
    responses = [
        "**Environment Observation:**\nroot@host:/app# cat warriors/stone.red\n;redcode-94\n;name Stone\nroot@host:/app#",
        "**Environment Observation:**\nroot@host:/app# ls warriors/\ng2-clear.red  paper.red  stone.red\nroot@host:/app#",
    ]
    rows = []
    for turn in (1, 2):
        history = prompts[:turn]
        rows.append({"task": "terminal", "id": int(trajectory_id), "prompt": history[:-1] + [history[-1] + SUFFIX],
                     "response": responses[:turn], "current_prompt": prompts[turn - 1], "system_str": SYSTEM,
                     "turn_idx": turn, "total_turns": 2})
    return rows


class TokenAndContextTests(unittest.TestCase):
    def test_specific_tokens_ignore_shell_vocabulary_and_generic_names(self):
        self.assertEqual(specific_tokens("ls -la /app/"), set())
        self.assertEqual(specific_tokens("python3 --version && pip install numpy"), set())
        self.assertEqual(specific_tokens("ls warriors/"), {"warriors"})
        self.assertEqual(specific_tokens("cat warriors/stone.red"), {"warriors/stone.red", "stone.red", "warriors"})
        self.assertNotIn("/usr/bin/python3", specific_tokens("/usr/bin/python3 -m venv .venv"))
        self.assertIn("gp_rstan.R", specific_tokens("cat /app/gp_rstan.R"))
        # v5.3: code attributes, standard headers and all-generic paths are not file names
        self.assertEqual(specific_tokens("python3 -c 'import numpy as np; print(np.abs(json.load(f)))' && time.sleep"), set())
        self.assertEqual(specific_tokens("gcc -o /app/sim /app/sim.c -include stdio.h Python.h"), {"app/sim", "app/sim.c", "sim.c"})
        self.assertEqual(specific_tokens("cat app/result.txt /app/solution.txt"), set())
        # v5.3.1: URLs, host:port and IP path fragments, protocol versions, media types and every container's
        # home dot-files are not names
        self.assertEqual(specific_tokens("curl -s -H 'Content-Type: application/json' http://localhost:2375/containers/json"), set())
        self.assertEqual(specific_tokens("HTTP/1.1 200 OK\ncurl http://169.254.169.254/latest/meta-data/ localhost:9000/minio/health/live"), set())
        self.assertEqual(specific_tokens("ls -la .bashrc .cache .config default cache application"), set())
        self.assertEqual(specific_tokens("cat /etc/ssh/sshd_config"), {"etc/ssh/sshd_config", "sshd_config"})

    def test_tool_call_arguments_are_operands_and_identifiers_are_names(self):
        # Tool environments (MCP): every string argument is an operand except payloads; UUIDs and long hex ids are names.
        from trace2env.knowledge_gate import action_text, transcript_names
        call = {"type": "API-patch-page", "arguments": {"page_id": "1f0a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8", "properties": {"title": "x"},
                                                       "content": "a very long body " * 50, "path": "/app/.mcpmark_backups/backup_desktop_1/notes.md"}}
        text = action_text(call)
        self.assertIn("1f0a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8", text)
        self.assertNotIn("a very long body", text)
        tokens = specific_tokens(text)
        self.assertIn("1f0a2b3c-4d5e-6f70-8192-a3b4c5d6e7f8", tokens)
        self.assertIn("app/.mcpmark_backups/backup_desktop_1/notes.md", tokens)
        self.assertIn("backup_desktop_1", tokens)
        self.assertEqual(specific_tokens("ls -la"), set())
        history = ('### Turn 3\n**Action:**\n```json\n{"name": "execute_sql", "arguments": {"sql": "SELECT * FROM rental_2024 LIMIT 5", '
                   '"database_id": "9f8e7d6c5b4a39281706f5e4d3c2b1a0"}}\n```\n**Environment Observation:**\n{"type": "text"}')
        names = transcript_names(history)
        self.assertIn("9f8e7d6c5b4a39281706f5e4d3c2b1a0", names)

    def test_one_signal_per_path(self):
        self.assertEqual(KnowledgeGate.distinct_signals({"etc/ssh/sshd_config", "sshd_config", "warriors/stone.red", "stone.red", "warriors"}),
                         ["etc/ssh/sshd_config", "warriors/stone.red"])
        self.assertEqual(KnowledgeGate.distinct_signals({"plus_comm.v", "warriors"}), ["plus_comm.v", "warriors"])

    def test_episode_context_reads_the_transcript_before_the_state(self):
        transcript = ["root@host:/app# cat warriors/stone.red\n;redcode\nroot@host:/app/sqlite# cat > x.txt << 'EOF'\n> line\n>\n> EOF\nroot@host:/app/sqlite#"]
        context = build_episode_context(transcript, None)
        self.assertEqual(context.cwd, "/app/sqlite")
        self.assertEqual(context.continuation_style, "bare")
        self.assertIn("warriors/stone.red", context.shown_names)
        self.assertIsNone(build_episode_context(["root@host:/app# ls\n"], None).continuation_style)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package = EnvironmentPackage(evidence_package(Path(self.temporary.name)))
        self.inspector = PackageInspector(self.package)
        self.records = self.inspector._evidence_index()["by_id"]

    def gate(self, action: NormalizedAction, shown: set[str] | None = None, cwd: str | None = "/app") -> KnowledgeGate:
        return KnowledgeGate(self.inspector, EpisodeContext(shown_names=shown or set(), cwd=cwd), action)

    def test_shared_operands_and_files_make_evidence_supporting_but_only_relevance(self):
        gate = self.gate(NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/"]}),
                         shown={"warriors/stone.red", "stone.red", "warriors"})
        decision = gate.decide_evidence(self.records["evidence_tr_same"])
        self.assertEqual((decision.label, decision.disposition), ("supporting", "shown"))
        self.assertEqual(decision.provenance["task"], "corewars-like")
        self.assertIn("warriors", decision.reason)
        # The same evidence with nothing shared in this episode is only a single weak signal.
        alone = self.gate(NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/"]}))
        self.assertEqual(alone.decide_evidence(self.records["evidence_tr_same"]).label, "uncertain")
        # A generic listing from another container shares nothing: format only, sanitized.
        foreign = gate.decide_evidence(self.records["evidence_tr_foreign"])
        self.assertEqual((foreign.label, foreign.disposition), ("format_only", "sanitized"))
        self.assertEqual(foreign.provenance["task"], "vulnerable-secret")
        # A differing working directory keeps the evidence from being supporting.
        elsewhere = self.gate(NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/"]}),
                              shown={"warriors/stone.red", "stone.red", "warriors"}, cwd="/app/other")
        self.assertEqual(elsewhere.decide_evidence(self.records["evidence_tr_same"]).label, "uncertain")

    def test_sanitized_view_masks_foreign_values_and_keeps_known_ones(self):
        gate = self.gate(NormalizedAction(type="ls", arguments={"argv": ["-la"], "commands": ["ls -la"]}), shown={"warriors"})
        decision = gate.decide_evidence(self.records["evidence_tr_foreign"])
        text = gate.sanitize(FOREIGN_LISTING, self.records["evidence_tr_foreign"], decision)
        self.assertNotIn("06e31eef63ee", text)
        self.assertNotIn("vulnerable", text)
        self.assertNotIn("14520", text)
        self.assertIn("root@<host>:/app#", text)
        self.assertIn("<name>", text)
        self.assertIn("<date>", text)
        self.assertIn("total <n>", text)
        self.assertGreater(decision.masked, 0)
        supporting = self.gate(NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/"]}),
                               shown={"warriors/stone.red", "stone.red", "warriors"}).decide_evidence(self.records["evidence_tr_same"])
        self.assertEqual(KnowledgeGate(self.inspector, EpisodeContext(), None).sanitize(SAME_LISTING, self.records["evidence_tr_same"], supporting), SAME_LISTING)

    def test_notes_are_checked_against_the_transcript(self):
        note = {"id": "bash-continuation-prompt", "statement": "Heredoc input shows the continuation prompt `> ` before continued lines."}
        bare = KnowledgeGate(self.inspector, EpisodeContext(continuation_style="bare"), None).decide_note(note)
        self.assertEqual((bare.label, bare.disposition), ("contradicted", "rejected"))
        space = KnowledgeGate(self.inspector, EpisodeContext(continuation_style="space"), None).decide_note(note)
        self.assertEqual((space.label, space.disposition), ("consistent", "shown"))
        untested = KnowledgeGate(self.inspector, EpisodeContext(), None).decide_note(note)
        self.assertEqual((untested.label, untested.disposition), ("untested", "shown"))
        summary = KnowledgeGate(self.inspector, EpisodeContext(), None).summary()
        self.assertTrue(summary["abstained"])


class SpecificityTests(unittest.TestCase):
    """v5.3.1: a shared name is a signal only when it is rare in the package, and a path counts once."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inspector = PackageInspector(EnvironmentPackage(evidence_package(Path(self.tmp.name))))

    def tearDown(self):
        self.tmp.cleanup()

    def _gate(self, shown, action="ls -la /app/"):
        context = EpisodeContext(shown_names=set(shown), cwd="/app", transcript="\n".join(sorted(shown)))
        return KnowledgeGate(self.inspector, context, NormalizedAction(type="ls", arguments={"argv": ["-la", "/app/"], "commands": [action]}))

    def test_names_owned_by_several_package_tasks_are_not_signals(self):
        gate = self._gate({"warriors/stone.red"})
        self.assertTrue(gate.specific_in_package("warriors/stone.red"))
        # make both package tasks own `default` and `solve.py`: neither identifies a task any more
        self.inspector._gate_inventories = {"ep_same": {"warriors/stone.red", "default", "solve.py"}, "ep_foreign": {"vulnerable", "default", "solve.py"}}
        self.inspector._gate_name_tasks = None
        self.assertFalse(gate.specific_in_package("default"))
        self.assertFalse(gate.specific_in_package("solve.py"))
        self.assertTrue(gate.specific_in_package("warriors/stone.red"))
        same = self.inspector._evidence_index()["by_id"]["evidence_tr_same"]
        label, _, reason, _, _ = self._gate({"default", "solve.py"})._rule_decision(same)
        self.assertEqual(label, "format_only", reason)
        label, _, reason, _, _ = self._gate({"default", "solve.py", "warriors/stone.red"})._rule_decision(same)
        self.assertEqual(label, "uncertain", reason)

    def test_a_path_and_its_components_count_once(self):
        self.inspector._gate_inventories = {"ep_same": {"etc/ssh/sshd_config", "sshd_config", "warriors/stone.red"}, "ep_foreign": {"vulnerable"}}
        self.inspector._gate_name_tasks = None
        same = self.inspector._evidence_index()["by_id"]["evidence_tr_same"]
        label, _, reason, _, _ = self._gate({"etc/ssh/sshd_config", "sshd_config"})._rule_decision(same)
        self.assertEqual(label, "uncertain", reason)  # one shared path, not two shared names
        label, _, reason, _, _ = self._gate({"etc/ssh/sshd_config", "sshd_config", "warriors/stone.red"})._rule_decision(same)
        self.assertEqual(label, "supporting", reason)

    def test_judge_anchors_must_be_specific_package_rare_names(self):
        gate = self._gate({"warriors/stone.red"})
        haystack = "root@host:/app# cat /app/result.txt warriors/stone.red default\nTheorem plus_comm"
        self.assertTrue(gate._verified_anchor("warriors/stone.red", haystack))
        self.assertFalse(gate._verified_anchor("/app/result.txt", haystack))  # an all-generic path
        self.assertFalse(gate._verified_anchor("default", haystack))  # a generic word
        self.assertFalse(gate._verified_anchor("Theorem plus_comm", haystack))  # a phrase, not a name
        self.assertFalse(gate._verified_anchor("warriors/paper.red", haystack))  # not in this episode
        self.inspector._gate_inventories = {"ep_same": {"warriors/stone.red"}, "ep_foreign": {"warriors/stone.red"}}
        self.inspector._gate_name_tasks = None
        self.assertFalse(gate._verified_anchor("warriors/stone.red", haystack))  # owned by two package tasks


class DocumentedFormatTests(unittest.TestCase):
    """Option B: when the official input documents the action's observation format, format-level items are withheld."""

    MCP_SYSTEM = ('# Role\n...\n## Example: list_allowed_directories\n**Action:**\n```json\n{\n  "name": "list_allowed_directories",\n'
                  '  "arguments": {}\n}\n```\n**Environment Observation:**\n{"type": "text", "text": "{\\"meta\\": null}"}\n')
    TERMINAL_SYSTEM = ('# Example\n**Action:**\n```json\n{"keystrokes": [{"keystrokes": "ls\\n"}]}\n```\n'
                       '**Environment Observation:**\nroot@host:/app# ls\n')

    def test_documented_format_detects_tool_examples_only(self):
        from trace2env.knowledge_gate import documented_format
        self.assertTrue(documented_format(self.MCP_SYSTEM, "list_allowed_directories"))
        self.assertFalse(documented_format(self.MCP_SYSTEM, "read_text_file"))
        self.assertFalse(documented_format(self.TERMINAL_SYSTEM, "ls"))
        self.assertFalse(documented_format("", "ls"))

    def test_gate_withholds_format_items_but_keeps_supporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            inspector = PackageInspector(EnvironmentPackage(evidence_package(Path(tmp))))
            index = inspector._evidence_index()["by_id"]
            context = EpisodeContext(shown_names={"warriors/stone.red", "warriors"}, cwd="/app", transcript="cat warriors/stone.red")
            action = NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/"]})
            gate = KnowledgeGate(inspector, context, action, format_documented=True)
            same, foreign = gate.decide_evidence(index["evidence_tr_same"]), gate.decide_evidence(index["evidence_tr_foreign"])
            self.assertEqual((same.label, same.disposition), ("supporting", "shown"))
            self.assertEqual((foreign.label, foreign.disposition), ("format_only", "rejected"))
            self.assertIn("documents this action's observation format", foreign.reason)
            summary = gate.summary()
            self.assertTrue(summary["format_documented"])
            self.assertEqual(summary["withheld_documented_format"], 1)
            self.assertTrue(gate.brief_notice()["format_documented"])
            plain = KnowledgeGate(inspector, context, action)
            self.assertEqual(plain.decide_evidence(index["evidence_tr_foreign"]).disposition, "sanitized")


class RunnerGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package_dir = evidence_package(Path(self.temporary.name))
        self.row = rows_with_history(train_trajectory_id())[1]

    def run_row(self, *, knowledge_gate: bool):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="root@host:/app# ls warriors/\ng2-clear.red  paper.red  stone.red\nroot@host:/app#",
                                          effects=[], citations=["evidence:evidence_tr_same"], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3,
                           "runtime_agent_turn": [AgentTurn(tool="read_evidence", arguments={"id": "evidence:evidence_tr_foreign"}),
                                                  AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  official_input=True, knowledge_gate=knowledge_gate)
        [output] = runner.run([self.row])
        calls = [c for c in llm.calls if c["role"] == "runtime_agent_turn"]
        return output, calls

    def test_gate_labels_brief_and_tool_results_and_records_decisions(self):
        output, calls = self.run_row(knowledge_gate=True)
        self.assertEqual(output["trace2env"]["route"], "agent")
        self.assertTrue(output["trace2env"]["knowledge_gate_enabled"])
        first = json.loads(calls[0]["user"])
        brief = json.loads(first["conversation"][0]["content"])
        self.assertIn("package_applicability", brief)
        self.assertFalse(brief["package_applicability"]["abstained"])
        self.assertIn("evidence:evidence_tr_same", brief["package_applicability"]["supporting_items"])
        labels = {hit["id"]: hit["applicability"]["label"] for hit in brief["similar_turns"]}
        self.assertEqual(labels.get("evidence:evidence_tr_same"), "supporting")
        self.assertEqual(labels.get("evidence:evidence_tr_foreign"), "format_only")
        self.assertIn("Package applicability (harness v5.2)", calls[0]["system"])
        self.assertEqual([m["role"] for m in first["conversation"]], ["user"])
        # Round 2 carries the read_evidence result: a sanitized view of the foreign turn, with its decision attached.
        second = json.loads(calls[1]["user"])
        tool_result = json.loads(next(m["content"] for m in second["conversation"] if m.get("role") == "tool"))
        self.assertEqual(tool_result["applicability"]["label"], "format_only")
        self.assertEqual(tool_result["applicability"]["provenance"]["task"], "vulnerable-secret")
        self.assertIn("root@<host>:/app#", tool_result["text"])
        self.assertNotIn("vulnerable", tool_result["text"])
        self.assertNotIn("14520", tool_result["text"])
        # The raw package file is unchanged: the same record read without the gate is verbatim.
        raw = PackageInspector(EnvironmentPackage(self.package_dir)).evidence_view("evidence_tr_foreign")
        self.assertEqual(raw["text"], FOREIGN_LISTING)
        # The row records every decision: label, provenance, reason, disposition.
        record = output["trace2env"]["knowledge_gate"]
        self.assertFalse(record["abstained"])
        self.assertGreater(record["masked_tokens"], 0)
        by_id = {d["id"]: d for d in record["decisions"]}
        self.assertEqual(by_id["evidence:evidence_tr_foreign"]["disposition"], "sanitized")
        self.assertEqual(by_id["evidence:evidence_tr_same"]["disposition"], "shown")
        self.assertIn("reason", by_id["evidence:evidence_tr_same"])
        # No heredoc in this transcript: the continuation note is untested and shown with that label.
        notes = {n["id"]: n for n in brief["retrieved"]["notes"]}
        self.assertEqual(notes["bash-continuation-prompt"]["applicability"]["label"], "untested")
        self.assertTrue(all("applicability" in n for n in notes.values()))
        # Memory citation of the supporting evidence is accepted without an uncited-reference warning.
        issues = [i.get("code") if isinstance(i, dict) else i for i in output["trace2env"].get("verification") or []]
        self.assertNotIn("uncited_reference", issues)

    def test_format_only_mode_withholds_supporting_and_uncertain_evidence(self):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="ok", effects=[], citations=[], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3,
                           "runtime_agent_turn": [AgentTurn(tool="read_evidence", arguments={"id": "evidence:evidence_tr_same"}),
                                                  AgentTurn(tool="read_evidence", arguments={"id": "evidence:evidence_tr_foreign"}),
                                                  AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  official_input=True, knowledge_gate=True, knowledge_gate_mode="format_only")
        [output] = runner.run([self.row])
        self.assertEqual(output["trace2env"]["knowledge_gate_mode"], "format_only")
        calls = [c for c in llm.calls if c["role"] == "runtime_agent_turn"]
        brief = json.loads(json.loads(calls[0]["user"])["conversation"][0]["content"])
        self.assertTrue(brief["package_applicability"]["abstained"])  # the same-family item exists but is withheld
        self.assertEqual([hit["id"] for hit in brief["similar_turns"]], ["evidence:evidence_tr_foreign"])
        self.assertEqual(brief["similar_turns"][0]["applicability"]["disposition"], "sanitized")
        tool_results = [json.loads(m["content"]) for c in calls[1:] for m in json.loads(c["user"])["conversation"] if m.get("role") == "tool"]
        same = next(r for r in tool_results if r["id"] == "evidence:evidence_tr_same")
        self.assertIn("withheld", same)
        self.assertNotIn("text", same)
        self.assertEqual(same["applicability"]["disposition"], "rejected")
        foreign = next(r for r in tool_results if r["id"] == "evidence:evidence_tr_foreign")
        self.assertIn("root@<host>:/app#", foreign["text"])
        record = output["trace2env"]["knowledge_gate"]
        self.assertEqual(record["mode"], "format_only")
        self.assertTrue(record["abstained"])
        self.assertEqual({d["id"]: d["disposition"] for d in record["decisions"]}["evidence:evidence_tr_same"], "rejected")
        # The default mode is unchanged: the same-family item is shown.
        llm2 = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_agent_turn": [AgentTurn(final=submission)]})
        runner2 = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm2, agent_llm=StructuredAgentLLM(llm2),
                                   official_input=True, knowledge_gate=True)
        [output2] = runner2.run([self.row])
        self.assertEqual(output2["trace2env"]["knowledge_gate_mode"], "full")
        self.assertFalse(output2["trace2env"]["knowledge_gate"]["abstained"])

    def test_gate_off_reproduces_v51(self):
        output, calls = self.run_row(knowledge_gate=False)
        self.assertFalse(output["trace2env"]["knowledge_gate_enabled"])
        self.assertIsNone(output["trace2env"]["knowledge_gate"])
        first = json.loads(calls[0]["user"])
        brief = json.loads(first["conversation"][0]["content"])
        self.assertNotIn("package_applicability", brief)
        self.assertTrue(all("applicability" not in hit for hit in brief["similar_turns"]))
        self.assertNotIn("Package applicability", calls[0]["system"])
        second = json.loads(calls[1]["user"])
        tool_result = json.loads(next(m["content"] for m in second["conversation"] if m.get("role") == "tool"))
        self.assertEqual(tool_result["text"], FOREIGN_LISTING)
        self.assertNotIn("applicability", tool_result)


class JudgeTests(unittest.TestCase):
    """v5.3: the applicability judge proposes labels and anchors; the gate honours 'supporting' only with a verified anchor."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.package_dir = evidence_package(Path(self.temporary.name))
        self.row = rows_with_history(train_trajectory_id())[1]

    def run_with_judge(self, verdicts, *, judge_fails=False):
        from trace2env.models import ApplicabilityJudgement, ApplicabilityVerdict
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="ok", effects=[], citations=[], rationale="")
        responses = {"track_state": [StateTrackingResult()] * 3, "runtime_agent_turn": [AgentTurn(final=submission)]}
        if not judge_fails:
            responses["knowledge_applicability"] = [ApplicabilityJudgement(items=[ApplicabilityVerdict(**v) for v in verdicts])] * 4
        llm = ScriptedLLM(responses)
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  official_input=True, knowledge_gate=True, knowledge_judge=True)
        [output] = runner.run([self.row])
        calls = [c for c in llm.calls if c["role"] == "runtime_agent_turn"]
        brief = json.loads(json.loads(calls[0]["user"])["conversation"][0]["content"])
        judge_calls = [c for c in llm.calls if c["role"] == "knowledge_applicability"]
        return output, brief, judge_calls

    def test_verified_anchor_upgrades_and_hallucinated_anchor_does_not(self):
        # The foreign listing is rule-labelled format_only; the judge claims it is supporting with a made-up anchor,
        # and claims the same-family turn is supporting with a real anchor (shown in the episode's history).
        output, brief, judge_calls = self.run_with_judge([
            {"id": "evidence:evidence_tr_foreign", "label": "supporting", "anchors": ["vulnerable"], "reason": "same binary"},
            {"id": "evidence:evidence_tr_same", "label": "supporting", "anchors": ["warriors/stone.red"], "reason": "same warriors task"},
        ])
        labels = {hit["id"]: hit["applicability"] for hit in brief["similar_turns"]}
        self.assertEqual(labels["evidence:evidence_tr_same"]["label"], "supporting")
        self.assertEqual(labels["evidence:evidence_tr_same"]["judge"]["verified_anchors"], ["warriors/stone.red"])
        self.assertEqual(labels["evidence:evidence_tr_foreign"]["label"], "format_only")  # 'vulnerable' never appeared here
        self.assertEqual(labels["evidence:evidence_tr_foreign"]["disposition"], "sanitized")
        self.assertIn("no anchor was verified", labels["evidence:evidence_tr_foreign"]["reason"])
        self.assertEqual(len(judge_calls), 1)  # the brief's hits were judged in one batched call
        payload = json.loads(judge_calls[0]["user"])
        self.assertNotIn("g2-clear.red  paper.red  stone.red", payload["episode"]["transcript_tail"])  # the target is not shown
        self.assertEqual(sorted(c["id"] for c in payload["candidates"]), ["evidence_tr_foreign", "evidence_tr_same"])
        record = output["trace2env"]["knowledge_gate"]["judge"]
        self.assertEqual((record["enabled"], record["calls"], record["unverified_upgrades"]), (True, 1, 1))
        self.assertTrue(output["trace2env"]["knowledge_judge"])

    def test_judge_downgrade_is_honoured_and_failure_falls_back_to_rules(self):
        output, brief, _ = self.run_with_judge([
            {"id": "evidence:evidence_tr_same", "label": "format_only", "anchors": [], "reason": "different pmars version"},
            {"id": "evidence:evidence_tr_foreign", "label": "contradicted", "anchors": [], "reason": "listing conflicts"},
        ])
        labels = {hit["id"]: hit["applicability"] for hit in brief["similar_turns"]}
        self.assertEqual((labels["evidence:evidence_tr_same"]["label"], labels["evidence:evidence_tr_same"]["disposition"]), ("format_only", "sanitized"))
        self.assertEqual(labels["evidence:evidence_tr_same"]["judge"]["rule_label"], "supporting")
        self.assertEqual((labels["evidence:evidence_tr_foreign"]["label"], labels["evidence:evidence_tr_foreign"]["disposition"]), ("contradicted", "sanitized"))
        self.assertTrue(brief["package_applicability"]["abstained"])
        self.assertEqual(output["trace2env"]["knowledge_gate"]["judge"]["downgraded"], 1)
        output, brief, _ = self.run_with_judge([], judge_fails=True)  # no scripted answer: the call raises
        labels = {hit["id"]: hit["applicability"]["label"] for hit in brief["similar_turns"]}
        self.assertEqual(labels["evidence:evidence_tr_same"], "supporting")  # deterministic label stands
        self.assertEqual(len(output["trace2env"]["knowledge_gate"]["judge"]["errors"]), 1)


if __name__ == "__main__":
    unittest.main()


class JudgeAgreementTests(unittest.TestCase):
    """v5.3.1: when the rule says supporting and the judge agrees, unverifiable anchors do not downgrade the item."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inspector = PackageInspector(EnvironmentPackage(evidence_package(Path(self.tmp.name))))

    def tearDown(self):
        self.tmp.cleanup()

    def test_agreement_without_name_anchors_keeps_supporting(self):
        from trace2env.models import ApplicabilityJudgement, ApplicabilityVerdict
        llm = ScriptedLLM({"knowledge_applicability": [ApplicabilityJudgement(items=[
            ApplicabilityVerdict(id="evidence:evidence_tr_same", label="supporting", anchors=["pmars -h 2>&1 | head -30"], reason="same task")])]})
        context = EpisodeContext(shown_names={"warriors/stone.red", "warriors"}, cwd="/app", transcript="root@host:/app# cat warriors/stone.red\npmars -h 2>&1 | head -30")
        gate = KnowledgeGate(self.inspector, context, NormalizedAction(type="ls", arguments={"argv": ["warriors/"], "commands": ["ls warriors/", "cat warriors/stone.red"]}), judge_llm=llm)
        same = self.inspector._evidence_index()["by_id"]["evidence_tr_same"]
        decision = gate.decide_evidence(same)
        self.assertEqual(gate._rule_decision(same)[0], "supporting")
        self.assertEqual(decision.label, "supporting")
        self.assertEqual(decision.disposition, "shown")
        self.assertEqual(decision.judge["verified_anchors"], [])
        self.assertIn("judge agrees", decision.reason)
        self.assertEqual(gate.summary()["judge"]["downgraded"], 0)


SIGN_IN_PAGE = ("### Ran Playwright code\n```js\nawait page.goto('http://gitlab.example.com/users/sign_in');\n```\n### Page\n"
                "- Page URL: http://gitlab.example.com/users/sign_in\n- Page Title: Sign in · GitLab\n### Snapshot\n```yaml\n- textbox \"Username\" [ref=e19]\n```")
DASHBOARD_PAGE = ("### Ran Playwright code\n```js\nawait page.getByRole('button', { name: 'Sign in' }).click();\n```\n### Page\n"
                  "- Page URL: http://gitlab.example.com/\n- Page Title: Projects · Dashboard · GitLab\n### Snapshot\n```yaml\n- link \"a11y-webring.club\" [ref=e40]\n```")
FORUM_PAGE = ("### Page\n- Page URL: http://forum.example.com/f/books\n- Page Title: books\n### Snapshot\n```yaml\n- link \"Love story\" [ref=e7]\n```")


def web_evidence_package(root: Path) -> Path:
    """Three web transitions: navigate to the GitLab sign-in page and submit it (one episode), a forum page (another)."""
    config = ReconstructionConfig(environment_id="test.web", name="Synthetic web", construction_kind="authored")
    navigate = LocalTransitionEvidence(
        id="evidence_web_nav", episode_id="ep_web_a", transition_id="tr_web_nav",
        action=NormalizedAction(type="browser_navigate", arguments={"url": "http://gitlab.example.com/users/sign_in"}),
        observation_text=SIGN_IN_PAGE, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
        observation_facts=[Fact(subject="surface.page.url", predicate="equals", value="http://gitlab.example.com/users/sign_in")],
    )
    click = LocalTransitionEvidence(
        id="evidence_web_click", episode_id="ep_web_a", transition_id="tr_web_click",
        action=NormalizedAction(type="browser_click", arguments={"element": "Sign in button", "ref": "e31"}),
        observation_text=DASHBOARD_PAGE, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
        preconditions=[Fact(subject="surface.page.url", predicate="equals", value="http://gitlab.example.com/users/sign_in?proxy=webarena-gitlab-worker-7")],
        observation_facts=[Fact(subject="surface.page.url", predicate="equals", value="http://gitlab.example.com/")],
    )
    forum = LocalTransitionEvidence(
        id="evidence_web_forum", episode_id="ep_web_b", transition_id="tr_web_forum",
        action=NormalizedAction(type="browser_click", arguments={"element": "books forum link", "ref": "e9"}),
        observation_text=FORUM_PAGE, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
        observation_facts=[Fact(subject="surface.page.url", predicate="equals", value="http://forum.example.com/f/books")],
    )
    artifacts = ReconstructionArtifacts(
        episodes=[Episode(id="ep_web_a", source_id="src_web_a", metadata={"episode_id": "webarena:576:r1"},
                          events=[RawEvent(id="a1", content="navigate"), RawEvent(id="o1", content=SIGN_IN_PAGE),
                                  RawEvent(id="a2", content="click"), RawEvent(id="o2", content=DASHBOARD_PAGE)]),
                  Episode(id="ep_web_b", source_id="src_web_b", metadata={"episode_id": "webarena:613:r1"},
                          events=[RawEvent(id="a3", content="click"), RawEvent(id="o3", content=FORUM_PAGE)])],
        transitions=[TransitionSlice(id="tr_web_nav", episode_id="ep_web_a", action_event_ids=["a1"], observation_event_ids=["o1"]),
                     TransitionSlice(id="tr_web_click", episode_id="ep_web_a", action_event_ids=["a2"], observation_event_ids=["o2"]),
                     TransitionSlice(id="tr_web_forum", episode_id="ep_web_b", action_event_ids=["a3"], observation_event_ids=["o3"])],
        evidence=[navigate, click, forum], rules=[], renderers=[],
        action_schema=ActionSchema(actions=[
            ActionSpec(name="browser_navigate", description="Open a URL.", arguments={"url": ArgumentSpec(type="string")}),
            ActionSpec(name="browser_click", description="Click an element.", arguments={"element": ArgumentSpec(type="string"), "ref": ArgumentSpec(type="string")}),
        ]),
        state_schema=StateSchema(fields=[StateField(path="surface.page.url", type="string", default="", visibility="surface")]),
        notes=[],
    )
    return EnvironmentCompiler(config).compile(artifacts, root / "web-package")


class PageIdentityTests(unittest.TestCase):
    """v5.3.2: on a shared web instance, the same page is the same object; page identity is the supporting signal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inspector = PackageInspector(EnvironmentPackage(web_evidence_package(Path(self.tmp.name))))
        self.records = self.inspector._evidence_index()["by_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_normalize_page_drops_proxy_parameters_and_trailing_slashes(self):
        self.assertEqual(normalize_page("http://GitLab.example.com/users/sign_in?proxy=webarena-gitlab-worker-1#top"), "http://gitlab.example.com/users/sign_in")
        self.assertEqual(normalize_page("http://forum.example.com/"), "http://forum.example.com")
        self.assertIsNone(normalize_page(""))

    def test_context_records_the_current_page_and_the_visited_pages(self):
        history = ["### Turn 1\n**Action:**\n```text\nbrowser_navigate(url=\"http://gitlab.example.com/users/sign_in\")\n```", SIGN_IN_PAGE]
        current = "### Turn 2\n**Current State:**\n### Page\n- Page URL: http://gitlab.example.com/users/sign_in?proxy=webarena-gitlab-worker-3\n- Page Title: Sign in · GitLab\n\n**Action:**\n```text\nbrowser_click(element=\"Sign in\", ref=\"e31\")\n```"
        context = build_episode_context(history, current_text=current)
        self.assertEqual(context.current_page, "http://gitlab.example.com/users/sign_in")
        self.assertEqual(context.current_page_raw, "http://gitlab.example.com/users/sign_in?proxy=webarena-gitlab-worker-3")
        self.assertEqual(context.shown_pages, {"http://gitlab.example.com/users/sign_in"})
        self.assertIsNone(build_episode_context(history).current_page)

    def test_same_page_and_same_destination_are_supporting_only_in_pages_mode(self):
        context = EpisodeContext(current_page="http://gitlab.example.com/users/sign_in", current_page_raw="http://gitlab.example.com/users/sign_in?proxy=w")
        click = NormalizedAction(type="browser_click", arguments={"element": "Sign in button", "ref": "e31"})
        pages = KnowledgeGate(self.inspector, context, click, names="pages")
        label, disposition, reason, provenance, _ = pages._rule_decision(self.records["evidence_web_click"])
        self.assertEqual((label, disposition), ("supporting", "shown"))
        self.assertIn("same page 'http://gitlab.example.com/users/sign_in?proxy=w'", reason)
        self.assertEqual(provenance["task"], "576")
        label, _, reason, _, _ = pages._rule_decision(self.records["evidence_web_forum"])
        self.assertEqual(label, "format_only", reason)  # another page: no signal
        navigate = NormalizedAction(type="browser_navigate", arguments={"url": "http://gitlab.example.com/users/sign_in?proxy=webarena-gitlab-worker-9"})
        pages = KnowledgeGate(self.inspector, EpisodeContext(), navigate, names="pages")
        label, _, reason, _, _ = pages._rule_decision(self.records["evidence_web_nav"])
        self.assertEqual(label, "supporting", reason)
        self.assertIn("same destination", reason)
        paths = KnowledgeGate(self.inspector, context, click)  # v5.3.1 model: URLs are not names
        self.assertEqual(paths._rule_decision(self.records["evidence_web_click"])[0], "format_only")
        summary = KnowledgeGate(self.inspector, context, click, names="pages").summary()
        self.assertEqual(summary["names"], "pages")
        with self.assertRaises(ValueError):
            KnowledgeGate(self.inspector, context, click, names="urls")

    def test_page_before_falls_back_to_the_preceding_transition(self):
        # the click record's precondition is removed: its page before the action is the navigate record's page after
        click = self.records["evidence_web_click"].model_copy(update={"preconditions": []})
        self.records["evidence_web_click"] = click
        gate = KnowledgeGate(self.inspector, EpisodeContext(current_page="http://gitlab.example.com/users/sign_in", current_page_raw="x"),
                             NormalizedAction(type="browser_click", arguments={"ref": "e31"}), names="pages")
        self.assertEqual(gate._page_before(click), "http://gitlab.example.com/users/sign_in")
        self.assertEqual(gate._rule_decision(click)[0], "supporting")


JSON_SCREEN_A = """[  0] | android.widget.FrameLayout | res="com.example.notes:id/root" | bounds=[0,0][1080,2400]
[  1] | android.widget.TextView | text="Notes" res="com.example.notes:id/title" | bounds=[0,100][500,200]
[  2] | android.widget.Button | text="Add" res="com.example.notes:id/add" | bounds=[900,2200][1080,2300]
[  3] | android.widget.ListView | res="com.example.notes:id/list" | bounds=[0,200][1080,2200]
[  4] | android.widget.FrameLayout | res="com.android.systemui:id/status_bar" | bounds=[0,0][1080,80]"""
JSON_SCREEN_A2 = JSON_SCREEN_A.replace('text="Notes"', 'text="My notes"')  # same screen, different text
JSON_SCREEN_B = """[  0] | android.widget.FrameLayout | res="com.example.notes:id/root" | bounds=[0,0][1080,2400]
[  1] | android.widget.EditText | res="com.example.notes:id/editor" | bounds=[0,100][1080,2000]
[  2] | android.widget.Button | text="Save" res="com.example.notes:id/save" | bounds=[900,2200][1080,2300]
[  3] | android.widget.FrameLayout | res="com.android.systemui:id/status_bar" | bounds=[0,0][1080,80]"""
PHONE_SCREEN_A = """**Current Phone State:**
• **App:** Notes (com.example.notes)
1. FrameLayout: "com.example.notes:id/root" - (0,0,1080,2400)
2. TextView: "com.example.notes:id/title", "Notes" - (0,100,500,200)
3. Button: "com.example.notes:id/add", "Add" - (900,2200,1080,2300)
4. ListView: "com.example.notes:id/list" - (0,200,1080,2200)
5. FrameLayout: "com.android.systemui:id/status_bar" - (0,0,1080,80)"""
DROID_SCREEN = """**Current Phone State:**
• **App:** me.example.pos
• **Activity:** com.tns.NativeScriptActivity
• **State ID:** 08b900/a3807d

Accessibility Tree:
  - LinearLayout id="me.example.pos:id/action_bar_root" bounds=0,72,1080,1776"""


def android_evidence_package(root: Path) -> Path:
    """One json-style episode on the Notes list screen (open list -> tap Add -> editor) and one on the editor."""
    config = ReconstructionConfig(environment_id="test.android", name="Synthetic android", construction_kind="authored")
    open_notes = LocalTransitionEvidence(
        id="evidence_android_open", episode_id="ep_android_a", transition_id="tr_android_open",
        action=NormalizedAction(type="open_app", arguments={"app_name": "Notes"}),
        observation_text=JSON_SCREEN_A, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
    )
    tap_add = LocalTransitionEvidence(
        id="evidence_android_add", episode_id="ep_android_a", transition_id="tr_android_add",
        action=NormalizedAction(type="click", arguments={"index": 2}),
        observation_text=JSON_SCREEN_B, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
    )
    save = LocalTransitionEvidence(
        id="evidence_android_save", episode_id="ep_android_b", transition_id="tr_android_save",
        action=NormalizedAction(type="click", arguments={"index": 2}),
        observation_text=JSON_SCREEN_A, outcome=Outcome.SUCCESS, confidence=Confidence(value=0.9),
    )
    artifacts = ReconstructionArtifacts(
        episodes=[Episode(id="ep_android_a", source_id="src_android_a", metadata={"episode_id": "awb:android:7"},
                          events=[RawEvent(id="a1", content="open_app"), RawEvent(id="o1", content=JSON_SCREEN_A),
                                  RawEvent(id="a2", content="click"), RawEvent(id="o2", content=JSON_SCREEN_B)]),
                  Episode(id="ep_android_b", source_id="src_android_b", metadata={"episode_id": "awb:android:9"},
                          events=[RawEvent(id="a3", content="click"), RawEvent(id="o3", content=JSON_SCREEN_A)])],
        transitions=[TransitionSlice(id="tr_android_open", episode_id="ep_android_a", action_event_ids=["a1"], observation_event_ids=["o1"]),
                     TransitionSlice(id="tr_android_add", episode_id="ep_android_a", action_event_ids=["a2"], observation_event_ids=["o2"]),
                     TransitionSlice(id="tr_android_save", episode_id="ep_android_b", action_event_ids=["a3"], observation_event_ids=["o3"])],
        evidence=[open_notes, tap_add, save], rules=[], renderers=[],
        action_schema=ActionSchema(actions=[
            ActionSpec(name="open_app", description="Open an app.", arguments={"app_name": ArgumentSpec(type="string")}),
            ActionSpec(name="click", description="Tap an element.", arguments={"index": ArgumentSpec(type="integer")}),
        ]),
        state_schema=StateSchema(fields=[StateField(path="surface.screen", type="string", default="", visibility="surface")]),
        notes=[],
    )
    return EnvironmentCompiler(config).compile(artifacts, root / "android-package")


class ScreenIdentityTests(unittest.TestCase):
    """v5.3.3: on Android the same screen (app + on-screen elements, or a DroidBot state id) is the supporting signal."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.inspector = PackageInspector(EnvironmentPackage(android_evidence_package(Path(self.tmp.name))))
        self.records = self.inspector._evidence_index()["by_id"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_screen_signature_reads_the_three_benchmark_formats(self):
        a = screen_signature(JSON_SCREEN_A)
        self.assertEqual(a[0], "com.example.notes")  # the dominant non-system package names the app
        self.assertEqual(len(a[1]), 4)  # the status-bar id does not count
        self.assertIsNone(screen_signature('[ 0] | res="com.android.systemui:id/clock" | [ 1] res="com.android.systemui:id/battery" | [ 2] res="com.android.systemui:id/wifi_combo" | [ 3] res="com.google.android.apps.nexuslauncher:id/workspace"'))  # status bar + launcher only
        self.assertTrue(same_screen(a, screen_signature(JSON_SCREEN_A2)))  # texts differ, elements do not
        self.assertTrue(same_screen(a, screen_signature(PHONE_SCREEN_A)))  # the phone layout of the same screen
        self.assertFalse(same_screen(a, screen_signature(JSON_SCREEN_B)))  # the editor: other elements
        self.assertEqual(screen_signature(DROID_SCREEN), ("state", "08b900/a3807d"))
        self.assertIsNone(screen_signature("[  0] | android.widget.FrameLayout | bounds=[0,0][1080,2400]"))  # no resource ids
        self.assertIsNone(screen_signature(""))

    def test_context_takes_the_current_screen_from_the_state_section_only(self):
        current = "### Turn 2\n**Current State:**\n" + JSON_SCREEN_A + "\n\n**Action:**\n```json\n{\"action_type\": \"click\", \"index\": 2}\n```"
        context = build_episode_context([], current_text=current)
        self.assertTrue(same_screen(context.current_screen, screen_signature(JSON_SCREEN_A)))
        self.assertIsNone(build_episode_context([]).current_screen)

    def test_same_screen_and_same_opened_app_are_supporting_only_in_screens_mode(self):
        context = EpisodeContext(current_screen=screen_signature(JSON_SCREEN_A2))
        click = NormalizedAction(type="click", arguments={"index": 2})
        gate = KnowledgeGate(self.inspector, context, click, names="screens")
        label, disposition, reason, provenance, _ = gate._rule_decision(self.records["evidence_android_add"])
        self.assertEqual((label, disposition), ("supporting", "shown"))  # the list screen is where this tap happened
        self.assertIn("same screen (app com.example.notes", reason)
        self.assertIn("com.example.notes:id/", reason)  # the anchor names a resource id the transcript contains
        self.assertEqual(provenance["task"], "7")
        label, _, reason, _, _ = gate._rule_decision(self.records["evidence_android_save"])
        self.assertEqual(label, "format_only", reason)  # first transition of its episode: no screen before it is known
        label, _, reason, _, _ = KnowledgeGate(self.inspector, EpisodeContext(current_screen=screen_signature(JSON_SCREEN_B)), click,
                                               names="screens")._rule_decision(self.records["evidence_android_add"])
        self.assertEqual(label, "format_only", reason)  # another screen: no signal
        opened = NormalizedAction(type="open_app", arguments={"app_name": "notes"})
        label, _, reason, _, _ = KnowledgeGate(self.inspector, EpisodeContext(), opened, names="screens")._rule_decision(self.records["evidence_android_open"])
        self.assertEqual(label, "supporting", reason)
        self.assertIn("same app opened", reason)
        other = NormalizedAction(type="open_app", arguments={"app_name": "Clock"})
        self.assertEqual(KnowledgeGate(self.inspector, EpisodeContext(), other, names="screens")._rule_decision(self.records["evidence_android_open"])[0], "format_only")
        self.assertEqual(KnowledgeGate(self.inspector, context, click)._rule_decision(self.records["evidence_android_add"])[0], "format_only")  # paths model
        self.assertEqual(KnowledgeGate(self.inspector, context, click, names="pages")._rule_decision(self.records["evidence_android_add"])[0], "format_only")
        summary = gate.summary()
        self.assertEqual(summary["names"], "screens")
        self.assertIn("screen_matches", summary)
