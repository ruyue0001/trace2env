"""Online-harness features: history fidelity, wait handling, compact context, transcript scaffold, unknown semantics."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tests.test_agentworld import terminal_package, terminal_rows
from trace2env.agentworld import RESPONSE_MARKER, RESPONSE_TAG, case_from_row, clean_response_marker, inference_messages, parse_model_output
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.llm import ScriptedLLM, StructuredAgentLLM
from trace2env.memory import EpisodicMemory
from trace2env.models import (AgentTurn, EnvironmentState, MemoryEntry, NormalizedAction, ObservedTurn, Outcome,
                              StateTrackingResult, TransitionRule, TransitionSubmission)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.state_tracking import StateTracker
from trace2env.transcript import build_scaffold, initial_prompt_state, last_prompt, resolve_cwd
from trace2env.workspace import WorkspaceTools, clip_span, compact_action


class TranscriptTests(unittest.TestCase):
    def test_prompt_parsing(self):
        self.assertEqual(initial_prompt_state("root@abc123:/app#")["cwd"], "/app")
        self.assertEqual(initial_prompt_state("\nroot@abc123:/app# \n")["prompt"], "root@abc123:/app#")
        self.assertIsNone(initial_prompt_state("a\nb\nc\nroot@abc123:/app#"))
        idle = last_prompt("root@h:/app# ls\nfile\nroot@h:/app#")
        self.assertEqual((idle["cwd"], idle["idle_at_end"]), ("/app", "True"))
        busy = last_prompt("root@h:/app# python3 run.py\nworking...")
        self.assertEqual((busy["cwd"], busy["idle_at_end"]), ("/app", "False"))
        self.assertEqual(resolve_cwd("/app", "../tmp/x"), "/tmp/x")
        self.assertEqual(resolve_cwd("/app", None), "/root")

    def test_scaffold_echoes_commands_heredocs_and_cd(self):
        arguments = {"commands": ["cd /tmp", "cat > f.py << 'EOF'", "print(1)", "EOF", "python3 f.py"], "programs": ["cd", "cat", "python3"]}
        prompt = last_prompt("root@h:/app# ls\nroot@h:/app#")
        scaffold = build_scaffold(arguments, prompt, "/app")
        lines = scaffold["skeleton"].splitlines()
        self.assertEqual(lines[0], "root@h:/app# cd /tmp")
        self.assertEqual(lines[2], "root@h:/tmp# cat > f.py << 'EOF'")
        self.assertEqual(lines[3:5], ["> print(1)", "> EOF"])
        self.assertTrue(lines[5].startswith("<output of: the heredoc"))
        self.assertEqual(lines[6], "root@h:/tmp# python3 f.py")
        self.assertEqual(lines[-1], "<output of: python3 f.py>")
        self.assertTrue(any("returned prompt `root@h:/tmp#`" in note for note in scaffold["notes"]))
        self.assertEqual((scaffold["commands"], scaffold["cwd"]), (3, "/tmp"))
        self.assertTrue(any("cd /tmp" in note for note in scaffold["notes"]))
        self.assertIsNone(build_scaffold({"duration": 1.0, "keystrokes": [{"keystrokes": "", "duration": 1.0}]}, prompt, "/app"))
        unknown_host = build_scaffold({"command": "ls -la", "argv": ["-la"]}, None, "/app")
        self.assertTrue(unknown_host["skeleton"].startswith("<prompt for /app> ls -la"))
        self.assertIn("returned prompt", unknown_host["notes"][-2] + unknown_host["notes"][-1])
        self.assertNotIn("returned prompt", unknown_host["skeleton"])  # advisory note, not a skeleton line

    def test_control_keys_and_garbled_prompts(self):
        from trace2env.agentworld import terminal_keystrokes_action
        from trace2env.transcript import prompt_signature

        action = terminal_keystrokes_action([{"keystrokes": "C-c", "duration": 0.1}, {"keystrokes": "tmux kill-session -t w\n", "duration": 0.5}])
        self.assertEqual((action.type, action.arguments["commands"], action.arguments["programs"]), ("shell", ["C-c", "tmux kill-session -t w"], ["tmux"]))
        self.assertEqual(terminal_keystrokes_action([{"keystrokes": "C-c", "duration": 0.1}]).type, "keys")
        scaffold = build_scaffold(action.arguments, {"user": "root", "host": "h", "cwd": "/app", "sigil": "#"}, "/app")
        self.assertTrue(scaffold["skeleton"].splitlines()[0].startswith("<key press C-c"))
        self.assertEqual(scaffold["skeleton"].splitlines()[1], "root@h:/app# tmux kill-session -t w")
        signature = prompt_signature(["root@h:/app# ls\nroot@h:/app# cd x\nroot@h:/app/x#", "firoot@h:/app/x# chmod +x f\nroot@h:/app/x#"])
        self.assertEqual((signature["user"], signature["host"], signature["cwd"]), ("root", "h", "/app/x"))


class WaitHintTests(unittest.TestCase):
    def test_wait_hint_describes_the_pending_state(self):
        from trace2env.transcript import wait_hint

        typed = wait_hint("root@h:/app# ls\nfile\nroot@h:/app# rencrypt --help")
        self.assertEqual((typed["previous_capture_state"], typed["pending_command"]), ("command_typed_no_output_yet", "rencrypt --help"))
        idle = wait_hint("root@h:/app# ls\nfile\nroot@h:/app#")
        self.assertEqual((idle["previous_capture_state"], idle["pending_command"]), ("idle_prompt", None))
        running = wait_hint("root@h:/app# python3 train.py\nProgress: 89.9%")
        self.assertEqual(running["previous_capture_state"], "program_output_in_progress")
        self.assertTrue(running["previous_capture_tail"].endswith("Progress: 89.9%"))
        self.assertIn("none may be invented", running["guidance"])


class WorkspaceHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.package = EnvironmentPackage(terminal_package(root))
        self.memory = EpisodicMemory(root / "session.sqlite")
        self.long = "".join(f"line {i:04d} needle\n" for i in range(1200))  # ~20k chars
        self.memory.record(MemoryEntry(turn=1, kind="observed", action=NormalizedAction(type="cat", arguments={"command": "cat big", "argv": ["big"], "keystrokes": [{"keystrokes": "cat big\n", "duration": 0.5}]}), observation=self.long))
        self.memory.record(MemoryEntry(turn=2, kind="observed", action=NormalizedAction(type="pwd"), observation="root@h:/app# pwd\n/app\nroot@h:/app#"))
        self.tools = WorkspaceTools(self.package, PackageInspector(self.package), self.memory, EnvironmentState(session={"cwd": "/app"}),
                                    NormalizedAction(type="pwd"), features={"history", "compact", "unknown"},
                                    brief_action_type="pwd", turn_chars=1000, recall_full=1)

    def test_read_turn_pages_exact_spans(self):
        page = self.tools.call("read_turn", {"turn": 1, "offset": 500, "length": 100})
        self.assertEqual((page["text"], page["total_chars"], page["next_offset"]), (self.long[500:600], len(self.long), 600))
        self.assertIn("memory:1", self.tools.retrieved_ids)
        self.assertEqual(self.tools.call("read_turn", {"turn": 9})["error"], "No recorded turn 9")
        self.assertIn("read_turn", [tool.name for tool in self.tools.tools()])

    def test_recent_and_recall_keep_the_most_relevant_hits_in_full(self):
        recent = self.tools.call("recent_turns", {"limit": 2})
        self.assertEqual(recent[1]["observation"], "root@h:/app# pwd\n/app\nroot@h:/app#")  # newest, complete
        self.assertIn("read_turn(turn=1, offset=", recent[0]["observation"])  # older, head/tail clipped with a hint
        self.assertEqual(recent[0]["observation_chars"], len(self.long))
        self.assertNotIn("keystrokes", recent[0]["action"]["arguments"])
        hits = self.tools.call("recall", {"query": "needle", "limit": 5})
        self.assertEqual(hits[0]["observation"], self.long)  # best match in full
        clipped = clip_span("x" * 100, 40, turn=3)
        self.assertIn("read_turn(turn=3, offset=30)", clipped)
        self.assertTrue(clipped.startswith("x" * 30))

    def test_compact_inspect_and_unknown_state(self):
        slim = self.tools.call("inspect_action", {"action_type": "pwd"})
        self.assertIn("already in your brief", slim["note"])
        self.assertEqual([rule["id"] for rule in slim["rules"]], ["pwd_read"])
        self.assertIn("rule:pwd_read", self.tools.retrieved_ids)
        missing = self.tools.call("read_state", {"path": "world.files"})
        self.assertTrue(missing["missing"])
        self.assertIn("not observed", missing["meaning"])
        self.assertEqual(compact_action(NormalizedAction(type="shell", arguments={"commands": ["a"], "keystrokes": [{"keystrokes": "a\n", "duration": 2}]}))["arguments"],
                         {"commands": ["a"], "duration_total": 2.0})


class TrackerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = EnvironmentPackage(terminal_package(self.root))
        self.package.rules.append(TransitionRule(id="wait-noop", action_type="wait", description="A wait allows progress.",
                                                 renderer="terminal", confidence=0.99))

    def test_effect_free_template_free_rule_never_replaces_model_tracking(self):
        turn = ObservedTurn(turn=1, action=NormalizedAction(type="wait", arguments={"duration": 5.0}), observation="Done.\nroot@host:/app#")
        legacy = StateTracker(self.package, ScriptedLLM({"track_state": [StateTrackingResult()]}))
        _, steps = legacy.advance(EnvironmentState(session={"cwd": "/app"}, surface={"prompt": "root@host:/app#"}), [turn])
        self.assertEqual((steps[0].route, steps[0].rule_id), ("rule", "wait-noop"))
        llm = ScriptedLLM({"track_state": [StateTrackingResult()]})
        fixed = StateTracker(self.package, llm, features={"wait"})
        _, steps = fixed.advance(EnvironmentState(session={"cwd": "/app"}, surface={"prompt": "root@host:/app#"}), [turn])
        self.assertEqual(steps[0].route, "model")
        self.assertTrue(any("no effects and no template" in issue for issue in steps[0].issues))
        self.assertEqual(len(llm.calls), 1)

    def test_compact_tracker_moves_static_text_to_the_system_prompt(self):
        tracker = StateTracker(self.package, None, environment_prompt="Terminal world model.", features={"compact", "wait", "unknown"})
        system, payload = tracker._tracking_request(EnvironmentState(session={"cwd": "/app"}), [
            ObservedTurn(turn=1, action=NormalizedAction(type="pwd", arguments={"command": "pwd", "keystrokes": []}), observation="/app")])
        self.assertIn("# Declared state fields", system)
        self.assertIn("session.cwd (string)", system)
        self.assertIn("Terminal world model.", system)
        self.assertEqual(set(payload), {"current_state", "observed_turns", "instruction"})
        self.assertNotIn("keystrokes", payload["observed_turns"][0]["action"]["arguments"])
        legacy_system, legacy_payload = StateTracker(self.package, None)._tracking_request(EnvironmentState(), [])
        self.assertIn("state_schema", legacy_payload)

    def test_environment_prompt_is_bounded_by_default_and_whole_on_request(self):
        from trace2env.runtime import ENVIRONMENT_PROMPT_CHARS, environment_context

        tail = "\nFINAL TOOL EXAMPLE\n{'success': True, 'message': 'tail preserved'}"
        prompt = "x" * 24_000 + tail
        # Default: the AgentWorldBench bound of every reported run (24 000 characters).
        self.assertEqual(ENVIRONMENT_PROMPT_CHARS, 24_000)
        bounded = environment_context({"environment_prompt": prompt})
        self.assertTrue(bounded.endswith("x" * 24_000))
        self.assertNotIn(tail, bounded)
        tracker = StateTracker(self.package, None, environment_prompt=prompt, features={"compact"})
        bounded_system, _ = tracker._tracking_request(EnvironmentState(), [])
        self.assertNotIn(tail, bounded_system)
        # Opt-in (long-horizon callers): the prompt whole, tail included.
        whole = environment_context({"environment_prompt": prompt}, None)
        self.assertTrue(whole.endswith(prompt))
        tracker = StateTracker(self.package, None, environment_prompt=prompt, features={"compact"}, environment_prompt_chars=None)
        whole_system, _ = tracker._tracking_request(EnvironmentState(), [])
        self.assertIn(tail, whole_system)

    def test_initial_prompt_line_is_tracked_without_a_model(self):
        memory = EpisodicMemory(self.root / "m.sqlite")
        tracker = StateTracker(self.package, None, memory=memory, features={"compact"})
        result = tracker.track_initial_prompt(EnvironmentState(), "root@h:/work#", NormalizedAction(type="session.start"))
        state, step = result
        self.assertEqual((state.session["cwd"], step.route, step.rule_id), ("/work", "rule", "initial-prompt-line"))
        self.assertEqual(memory.count(), 1)
        self.assertIsNone(tracker.track_initial_prompt(EnvironmentState(), "lots\nof\nscreen\ntext\nroot@h:/work#", NormalizedAction(type="session.start")))
        self.assertIsNone(StateTracker(self.package, None, memory=memory).track_initial_prompt(EnvironmentState(), "root@h:/work#", NormalizedAction(type="session.start")))


class VacuousRuleCitationTests(unittest.TestCase):
    """Citing an effect-free, template-free rule while proposing effects is a citation, not an application."""

    def test_vacuous_rule_citation_does_not_reject_the_step(self):
        from trace2env.models import StateMutation, StepRequest
        from trace2env.runtime import RuntimeHarness

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = terminal_package(root)
            package = EnvironmentPackage(package_dir)
            package.rules.append(TransitionRule(id="wait-noop", action_type="wait", description="A wait allows progress.",
                                                renderer="terminal", confidence=0.99))
            package.action_schema.actions.append(__import__("trace2env.models", fromlist=["ActionSpec"]).ActionSpec(name="wait"))
            submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="done\nroot@host:/app#", rule_ids=["wait-noop"],
                                              effects=[StateMutation(op="set", path="session.cwd", value="/app")], citations=[], rationale="")
            for features, expect_route in (({"wait"}, "agent"), (set(), None)):
                llm = ScriptedLLM({"runtime_agent_turn": [AgentTurn(final=submission)]})
                harness = RuntimeHarness(package_dir, root / f"session-{len(features)}", package=package, agent_llm=StructuredAgentLLM(llm),
                                         trust_policy="schema_checked", features=features)
                request = StepRequest(action=NormalizedAction(type="wait", arguments={"duration": 1.0}))
                if expect_route:
                    result = harness.step(request)
                    self.assertEqual((result.route, result.plan.rule_ids, result.observation), ("agent", [], submission.observation))
                    self.assertIn("rule:wait-noop", result.citations)
                    self.assertNotIn("unsupported_effect", [issue.code for issue in result.verification.issues])
                else:
                    with self.assertRaisesRegex(RuntimeError, "absent from its cited rules"):
                        harness.step(request)


class WaitHintGatingTests(unittest.TestCase):
    """The pending-work hint for waits belongs to wait handling; the command skeleton needs the scaffold feature."""

    def test_wait_feature_alone_briefs_the_pending_work_but_no_skeleton(self):
        from trace2env.models import StepRequest
        from trace2env.runtime import RuntimeHarness

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = terminal_package(root)
            package = EnvironmentPackage(package_dir)
            package.action_schema.actions.append(__import__("trace2env.models", fromlist=["ActionSpec"]).ActionSpec(name="wait"))
            submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="usage: x\nroot@host:/app#", effects=[], citations=[], rationale="")
            for features, expect_hint, expect_skeleton in (({"wait"}, True, False), ({"scaffold"}, True, True), (set(), False, False)):
                llm = ScriptedLLM({"runtime_agent_turn": [AgentTurn(final=submission), AgentTurn(final=submission)]})
                harness = RuntimeHarness(package_dir, root / f"s{len(features)}{expect_skeleton}", package=package,
                                         agent_llm=StructuredAgentLLM(llm), trust_policy="schema_checked", features=features)
                harness.memory.record(MemoryEntry(turn=1, kind="observed", action=NormalizedAction(type="pwd"), observation="root@host:/app# rencrypt --help"))
                harness.step(StepRequest(action=NormalizedAction(type="wait", arguments={"duration": 1.0, "keystrokes": [{"keystrokes": "", "duration": 1.0}]})))
                brief = json.loads(json.loads(llm.calls[0]["user"])["conversation"][0]["content"])
                self.assertEqual("transcript_scaffold" in brief, expect_hint, features)
                if expect_hint:
                    self.assertEqual(brief["transcript_scaffold"]["pending_command"], "rencrypt --help")
                    self.assertIn("never\ninvent commands", llm.calls[0]["system"].replace("never invent", "never\ninvent"))
                harness2 = RuntimeHarness(package_dir, root / f"t{len(features)}{expect_skeleton}", package=package,
                                          agent_llm=StructuredAgentLLM(llm), trust_policy="schema_checked", features=features)
                harness2.memory.record(MemoryEntry(turn=1, kind="observed", action=NormalizedAction(type="pwd"), observation="root@host:/app# pwd\n/app\nroot@host:/app#"))
                harness2.step(StepRequest(action=NormalizedAction(type="cd", arguments={"argv": ["tests"], "command": "cd tests", "keystrokes": [{"keystrokes": "cd tests\n", "duration": 0.1}]})))
                brief2 = json.loads(json.loads(llm.calls[1]["user"])["conversation"][0]["content"])
                self.assertEqual("transcript_scaffold" in brief2 and "skeleton" in brief2["transcript_scaffold"], expect_skeleton, features)


class RunnerFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = terminal_package(self.root)
        self.rows = terminal_rows()

    def test_empty_observation_is_a_valid_prediction_and_the_brief_carries_the_new_sections(self):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="", effects=[], citations=[], rationale="nothing printed")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()], "runtime_agent_turn": [AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm), features="all")
        [output] = runner.run([self.rows[1]])  # turn 2: `cd tests`, no rule → agent
        self.assertEqual(output["gen"], f"<{RESPONSE_TAG}>\n\n</{RESPONSE_TAG}>")
        self.assertEqual(parse_model_output(output["gen"], RESPONSE_TAG), "")  # judged as an empty screen, not a failed row
        self.assertEqual(output["trace2env"]["features"], ["compact", "evidence", "history", "scaffold", "unknown", "wait"])
        self.assertEqual(AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir)).features, {"history", "wait", "compact", "unknown", "evidence"})
        tracking = output["trace2env"]["state_tracking"]
        self.assertEqual((tracking["rule_turns"], tracking["model_calls"]), (2, 0))  # turn 0 deterministic, turn 1 by the pwd rule
        agent_call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        payload = json.loads(agent_call["user"])
        self.assertEqual(list(payload)[:2], ["tools", "conversation"])  # static catalog first for prefix caching
        brief = json.loads(payload["conversation"][0]["content"])
        self.assertIn("transcript_scaffold", brief)
        self.assertTrue(brief["transcript_scaffold"]["skeleton"].startswith("root@host:/app# cd tests"))
        self.assertEqual(brief["known_files"]["count"], 0)
        self.assertIn("not observed", brief["state_semantics"])
        self.assertEqual([field["path"] for field in brief["state_fields"]], ["session.cwd", "surface.prompt"])
        self.assertEqual([entry["turn"] for entry in brief["recent_memory"]], [0, 1])
        self.assertIn("observation_chars", brief["recent_memory"][0])
        self.assertNotIn("keystrokes", json.dumps(brief["action"]))
        self.assertIn("read_turn", agent_call["system"])
        self.assertIn("read_turn", [tool["name"] for tool in payload["tools"]])

    def test_history_full_carries_every_observed_turn_verbatim(self):
        # Harness v4 information parity: tiny budgets and a one-turn window, yet the brief holds both observed
        # turns complete, without paging markers; the row records the setting.
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="ok", effects=[], citations=[], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult()], "runtime_agent_turn": [AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                  history_window=1, history_turn_chars=12, history_total_chars=12, history_full=True)
        [output] = runner.run([self.rows[1]])
        self.assertTrue(output["trace2env"]["history_full"])
        agent_call = next(call for call in llm.calls if call["role"] == "runtime_agent_turn")
        brief = json.loads(json.loads(agent_call["user"])["conversation"][0]["content"])
        self.assertEqual([entry["turn"] for entry in brief["recent_memory"]], [0, 1])
        for entry in brief["recent_memory"]:
            self.assertEqual(len(entry["observation"]), entry["observation_chars"])
            self.assertNotIn("read_turn(", entry["observation"])
        self.assertFalse(AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir)).history_full)  # v3 default

    def test_official_input_is_placed_once_verbatim_without_the_target(self):
        # Harness v5.1: the benchmark's own model input (inference_messages) reaches the model boundary exactly once,
        # unclipped: its system message in the system text, its turn messages as the brief's first block; the target
        # observation is absent; the budgeted recent_memory view is gone; memory ids stay citable.
        row = self.rows[1]  # turn 2 (`cd tests`, no rule -> agent route): one earlier turn in the history
        case = case_from_row(row)
        official = inference_messages(case)
        self.assertEqual([m["role"] for m in official], ["system", "user", "assistant", "user"])
        target = clean_response_marker(case.responses[case.turn_idx - 1])
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="ok", effects=[], citations=["memory:2"], rationale="")

        def twice(text):  # the brief is JSON inside the JSON request payload: a verbatim string appears escaped twice
            return json.dumps(json.dumps(text, ensure_ascii=False), ensure_ascii=False)[1:-1]

        for kwargs in ({}, {"state_tracking": False, "package_knowledge": False}):
            llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_agent_turn": [AgentTurn(final=submission)]})
            runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm),
                                      official_input=True, **kwargs)
            [output] = runner.run([row])
            self.assertTrue(output["trace2env"]["official_input"])
            self.assertEqual(output["trace2env"]["route"], "agent")
            call = next(c for c in llm.calls if c["role"] == "runtime_agent_turn")
            payload = json.loads(call["user"])
            brief = json.loads(payload["conversation"][0]["content"])
            self.assertEqual(list(brief)[0], "official_input")
            got = [{"role": m["role"], "content": m["content"]} for m in brief["official_input"]["messages"]]
            self.assertEqual(got, [m for m in official if m["role"] != "system"])  # verbatim, in order, complete
            self.assertEqual(call["system"].count(official[0]["content"]), 1)  # the system message exactly once
            for message in official[1:]:
                self.assertEqual(call["user"].count(twice(message["content"])), 1)  # each turn message exactly once
            self.assertNotIn(twice(RESPONSE_MARKER + "\n" + target), call["user"])  # the answer is not in the input
            self.assertNotIn(target, call["system"])
            self.assertNotIn("recent_memory", brief)
            self.assertNotIn("official_input", brief["context"])
            self.assertEqual([m.get("memory_id") for m in brief["official_input"]["messages"] if m["role"] == "assistant"], ["memory:2"])
            self.assertNotIn("recent_memory holds", call["system"])
            self.assertIn("Official input (harness v5.1)", call["system"])
            issues = [i.get("code") if isinstance(i, dict) else i for i in output["trace2env"].get("verification") or []]
            self.assertNotIn("uncited_reference", issues)  # memory:2 was shown through the official block
        # Off by default: the v3 brief is unchanged.
        llm = ScriptedLLM({"track_state": [StateTrackingResult()] * 3, "runtime_agent_turn": [AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm))
        [output] = runner.run([row])
        self.assertFalse(output["trace2env"]["official_input"])
        call = next(c for c in llm.calls if c["role"] == "runtime_agent_turn")
        brief = json.loads(json.loads(call["user"])["conversation"][0]["content"])
        self.assertIn("recent_memory", brief)
        self.assertNotIn("official_input", brief)

    def test_legacy_harness_still_fails_empty_observations(self):
        submission = TransitionSubmission(outcome=Outcome.SUCCESS, observation="", effects=[], citations=[], rationale="")
        llm = ScriptedLLM({"track_state": [StateTrackingResult(), StateTrackingResult()], "runtime_agent_turn": [AgentTurn(final=submission)]})
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm, agent_llm=StructuredAgentLLM(llm), features=set())
        [output] = runner.run([self.rows[1]])
        self.assertEqual(output["gen"], "")


class CliFeatureOptionTests(unittest.TestCase):
    """``--features`` and the history budgets are shared harness options: ``simulate`` honours them like ``awb-run``."""

    def setUp(self):
        from trace2env.cli import main

        self.main = main
        self.root = Path(tempfile.mkdtemp())
        self.package = self.root / "ledger"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["init-demo", "--output", str(self.package)]), 0)

    def _simulate(self, session: str, *extra: str) -> dict:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = self.main(["simulate", str(self.package), "--session", str(self.root / session),
                                "--action", '{"type":"withdraw","arguments":{"amount":25}}', *extra])
        self.assertEqual(status, 0)
        return json.loads(out.getvalue())

    def test_simulate_accepts_feature_sets(self):
        self.assertEqual(self._simulate("s-default")["observation"], "Withdrew 25. Balance: 75.0")
        self.assertEqual(self._simulate("s-none", "--features", "none")["observation"], "Withdrew 25. Balance: 75.0")
        self.assertEqual(self._simulate("s-list", "--features", "history,compact", "--history-turn-chars", "1000",
                                        "--history-total-chars", "5000")["observation"], "Withdrew 25. Balance: 75.0")
        with self.assertRaises(ValueError):
            self._simulate("s-bad", "--features", "telepathy")

    def test_default_feature_set_matches_runner(self):
        from trace2env.cli import _features, _resolved_features
        from trace2env.runtime import DEFAULT_FEATURES

        class Args:
            features = "default"
        self.assertEqual(_features(Args()), "default")
        self.assertEqual(_resolved_features(Args()), set(DEFAULT_FEATURES))
        Args.features = "none"
        self.assertEqual(_resolved_features(Args()), set())
        Args.features = "wait, unknown"
        self.assertEqual(_resolved_features(Args()), {"wait", "unknown"})


if __name__ == "__main__":
    unittest.main()


class TrackerRefusalTests(unittest.TestCase):
    def test_refused_or_invalid_tracking_window_is_skipped_not_fatal(self):
        from trace2env.llm import ProviderRefusal, StructuredOutputError
        from trace2env.memory import EpisodicMemory
        from trace2env.state_tracking import StateTracker

        class Refusing:
            def __init__(self, error):
                self.error = error

            def complete(self, **kwargs):
                raise self.error

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = EnvironmentPackage(terminal_package(root))
            for error in (ProviderRefusal("Provider refused role track_state: flagged"), StructuredOutputError("invalid after repairs")):
                memory = EpisodicMemory(root / f"{type(error).__name__}.sqlite")
                tracker = StateTracker(package, Refusing(error), memory=memory, features={"history", "wait", "compact", "unknown"})
                state = EnvironmentState(session={"cwd": "/app"})
                turns = [ObservedTurn(turn=1, action=NormalizedAction(type="cd", arguments={"argv": ["tests"]}), observation="root@h:/app# cd tests\nroot@h:/app/tests#")]
                after, steps = tracker.advance(state, turns)
                self.assertEqual(after.session["cwd"], "/app")  # state untouched
                self.assertEqual([step.route for step in steps], ["skipped"])
                self.assertIn("state left unchanged", steps[0].issues[-1])
                self.assertEqual([entry.turn for entry in memory.recent(5)], [1])  # history kept

    def test_truncated_tracking_answer_is_still_tracked_in_halves(self):
        from trace2env.llm import OutputTruncated
        from trace2env.memory import EpisodicMemory
        from trace2env.models import StateMutation
        from trace2env.state_tracking import StateTracker

        class Truncating:
            """Overflows on a two-turn window, answers a one-turn window."""

            def __init__(self):
                self.calls = 0

            def complete(self, *, system, user, response_model, role):
                self.calls += 1
                if user.count('"turn":') >= 2:
                    raise OutputTruncated("Structured output for role track_state was cut off at the output limit (max_tokens=8)")
                return StateTrackingResult(mutations=[StateMutation(op="set", path="session.cwd", value="/app/tests")])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = EnvironmentPackage(terminal_package(root))
            memory = EpisodicMemory(root / "m.sqlite")
            tracker = StateTracker(package, Truncating(), memory=memory, features={"history", "wait", "compact", "unknown"})
            turns = [ObservedTurn(turn=t, action=NormalizedAction(type="cd", arguments={"argv": ["tests"]}), observation="root@h:/app# cd tests\nroot@h:/app/tests#") for t in (1, 2)]
            after, steps = tracker.advance(EnvironmentState(session={"cwd": "/app"}), turns)
            self.assertEqual(after.session["cwd"], "/app/tests")  # halves were tracked, not dropped
            self.assertEqual(steps[-1].route, "model")
            self.assertTrue(any("tracked in halves" in issue for issue in steps[-1].issues))
            self.assertEqual([entry.turn for entry in memory.recent(5)], [1, 2])
