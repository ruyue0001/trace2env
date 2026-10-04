"""AgentWorldBench adapter, official-protocol port, run-time state tracking, and the prediction runner.

Rows here are synthetic and follow the published record format; they are not benchmark data.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import (
    RESPONSE_TAG,
    aggregate_scores,
    build_judge_messages,
    case_from_row,
    clean_response_marker,
    export_episodes,
    inference_messages,
    initial_observation,
    judge_rows,
    load_judge_system_prompt,
    load_rows,
    longest_records,
    normalize_action,
    observed_turns,
    parse_judge_output,
    parse_model_output,
    prompt_sections,
    split_of,
    strip_instruction_suffix,
    trace2env_diagnostics,
    trajectory_episode,
    wrap_prediction,
)
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.cli import main
from trace2env.compiler import EnvironmentCompiler
from trace2env.llm import ScriptedChatLLM, ScriptedLLM, StructuredAgentLLM
from trace2env.models import (
    ActionSchema,
    ActionSpec,
    AgentTurn,
    ArgumentSpec,
    EnvironmentState,
    NormalizedAction,
    ObservedTurn,
    Outcome,
    ReconstructionArtifacts,
    ReconstructionConfig,
    RenderContract,
    StateField,
    StateMutation,
    StateSchema,
    StateTrackingResult,
    TransitionRule,
    TransitionSubmission,
)
from trace2env.package import EnvironmentPackage
from trace2env.reconstruction import TraceReconstructor
from trace2env.state_tracking import StateTracker

SUFFIX = ("\n\nPlease analyze the input above and predict the realistic output/response that the environment "
          "would produce.\n\nFirst, think step by step. Then, provide the simulated environment observation wrapped "
          "strictly within the <predicted_observation></predicted_observation> tags.")
SYSTEM = "# Role and Objective\n\nYou are a **Terminal World Model** (synthetic test prompt)."


def keystrokes_prompt(turn: int, command: str, current_state: str | None = None) -> str:
    state = f"**Current State:**\n{current_state}\n\n" if current_state else ""
    body = json.dumps([{"keystrokes": command + "\n", "duration": 0.1}], indent=2)
    return f"### Turn {turn}\n{state}**Action:**\n```json\n{body}\n```"


def train_trajectory_id() -> str:
    return next(str(value) for value in range(1, 10_000) if split_of(str(value)) == "train")


def terminal_rows(trajectory_id: str | None = None) -> list[dict]:
    """A three-turn synthetic terminal trajectory as three benchmark records (turn 1, 2, and 3)."""
    trajectory_id = trajectory_id or train_trajectory_id()
    prompts = [keystrokes_prompt(1, "pwd", "root@host:/app#"), keystrokes_prompt(2, "cd tests"), keystrokes_prompt(3, "pwd")]
    responses = [
        "**Environment Observation:**\nroot@host:/app# pwd\n/app\nroot@host:/app#",
        "**Environment Observation:**\nroot@host:/app# cd tests\nroot@host:/app/tests#",
        "**Environment Observation:**\nroot@host:/app/tests# pwd\n/app/tests\nroot@host:/app/tests#",
    ]
    rows = []
    for turn in (1, 2, 3):
        history = prompts[:turn]
        rows.append({
            "task": "terminal", "id": int(trajectory_id),
            "prompt": history[:-1] + [history[-1] + SUFFIX], "response": responses[:turn],
            "current_prompt": prompts[turn - 1], "system_str": SYSTEM, "turn_idx": turn, "total_turns": 3,
        })
    return rows


def terminal_package(root: Path) -> Path:
    """Hand-authored fixture: a pwd rule with a template and a cd action without any rule."""
    config = ReconstructionConfig(environment_id="test.terminal", name="Synthetic terminal", construction_kind="authored")
    artifacts = ReconstructionArtifacts(
        episodes=[], transitions=[], evidence=[],
        action_schema=ActionSchema(actions=[
            ActionSpec(name="pwd", description="Print the working directory."),
            ActionSpec(name="cd", description="Change directory.", arguments={"argv": ArgumentSpec(type="array")}),
        ]),
        state_schema=StateSchema(fields=[
            StateField(path="session.cwd", type="string", default="/app", visibility="session"),
            StateField(path="surface.prompt", type="string", default="root@host:/app#", visibility="surface"),
        ]),
        rules=[TransitionRule(
            id="pwd_read", action_type="pwd", description="pwd echoes the command and prints the working directory.",
            renderer="terminal",
            observation_template="{state_before.surface.prompt} pwd\n{state_after.session.cwd}\n{state_after.surface.prompt}",
        )],
        renderers=[RenderContract(id="terminal", action_types=["pwd", "cd"], required_fields=["state_after.surface.prompt"],
                                  instructions="Echo the command after the prompt, then output, then the next prompt.")],
    )
    return EnvironmentCompiler(config).compile(artifacts, root / "terminal-package")


JUDGE_OUTPUT = """The prompt line matches; the listing is missing a line.
<final_evaluation>
```json
{
    "strengths": ["Prompt format matches"],
    "weaknesses": ["One line is missing"],
    "scores": {"format": 5, "factuality": 3, "consistency": 4, "realism": 4, "quality": 3}
}
```
</final_evaluation>"""


class DroidBotActionTests(unittest.TestCase):
    """The benchmark's DroidBot-style Android commands normalize to their verb with the target element as arguments."""

    def test_element_commands(self):
        action = normalize_action("android", "**Action:**\n```text\ntouch <button id=3 bound_box=108,1133,972,1266>PRIJAVI SE</button>\n```")
        self.assertEqual(action.type, "touch")
        self.assertEqual(action.arguments, {"element": "button", "id": "3", "bound_box": "108,1133,972,1266", "text": "PRIJAVI SE"})
        action = normalize_action("android", "**Action:**\n```text\nset_text <input bound_box=108,864,972,1039>Password</input> dummy_user_input\n```")
        self.assertEqual(action.type, "set_text")
        self.assertEqual(action.arguments["input"], "dummy_user_input")
        self.assertEqual(action.arguments["text"], "Password")
        action = normalize_action("android", "**Action:**\n```text\nscroll up <scrollbar bound_box=0,72,1080,1776></scrollbar>\n```")
        self.assertEqual((action.type, action.arguments["direction"], action.arguments["element"]), ("scroll", "up", "scrollbar"))
        action = normalize_action("android", "**Action:**\n```text\nunselect <checkbox id=7 alt='Show password' bound_box=1,2,3,4>Show</checkbox>\n```")
        self.assertEqual((action.type, action.arguments["alt"]), ("unselect", "Show password"))

    def test_intent_and_kill_app(self):
        action = normalize_action("android", "**Action:**\n```text\nintent am start com.alko.app/com.alko.app.StartActivity\n```")
        self.assertEqual((action.type, action.arguments), ("intent", {"command": "am start com.alko.app/com.alko.app.StartActivity"}))
        self.assertEqual(normalize_action("android", "**Action:**\n```text\nkill_app\n```").arguments, {})
        self.assertEqual(normalize_action("android", "**Action:**\n```text\nkill_app\n```").type, "kill_app")
        # other free text is still the task's opaque action
        self.assertEqual(normalize_action("android", "**Action:**\n```text\nsomething else entirely\n```").type, "android.action")


class RecordTests(unittest.TestCase):
    def test_case_follows_the_official_inference_layout(self):
        rows = terminal_rows()
        case = case_from_row(rows[2])
        self.assertEqual(case.turn_idx, 3)
        self.assertEqual(case.ground_truth, rows[2]["response"][-1])
        self.assertEqual(case.action.type, "pwd")
        messages = inference_messages(case)
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], SYSTEM)
        self.assertTrue(messages[-1]["content"].endswith(SUFFIX))
        self.assertEqual(messages[2]["content"], rows[2]["response"][0])
        turns = observed_turns(case)
        self.assertEqual([t.turn for t in turns], [1, 2])
        self.assertEqual(turns[1].action.type, "cd")
        self.assertEqual(turns[1].action.arguments["argv"], ["tests"])
        self.assertEqual(turns[0].observation, "root@host:/app# pwd\n/app\nroot@host:/app#")
        self.assertEqual(initial_observation(case), "root@host:/app#")
        self.assertEqual(strip_instruction_suffix(rows[0]["prompt"][0]), rows[0]["current_prompt"])

    def test_prompt_sections_and_markers(self):
        sections = prompt_sections("### Turn 4\n**Task Instruction:**\nOpen mail\n\n**Current State:**\nrow one\nrow two\n\n**Action:**\n```text\npyautogui.click(1, 2)\n```" + SUFFIX)
        self.assertEqual(sections["turn"], 4)
        self.assertEqual(sections["task_instruction"], "Open mail")
        self.assertEqual(sections["current_state"], "row one\nrow two")
        self.assertTrue(sections["action"].startswith("```text"))
        self.assertEqual(clean_response_marker("**Environment Observation:**\nok"), "ok")
        self.assertEqual(clean_response_marker("plain"), "plain")

    def test_action_normalization_across_domains(self):
        terminal = normalize_action("terminal", keystrokes_prompt(1, "ls -la /tmp"))
        self.assertEqual((terminal.type, terminal.arguments["argv"]), ("ls", ["-la", "/tmp"]))
        compound = normalize_action("terminal", keystrokes_prompt(1, "cd x && make"))
        self.assertEqual(compound.type, "shell")
        assignment = normalize_action("terminal", keystrokes_prompt(1, "FOO=1 python3 run.py"))
        self.assertEqual(assignment.type, "python3")
        empty = normalize_action("terminal", "**Action:**\n```json\n[{\"keystrokes\": \"\", \"duration\": 1.0}]\n```")
        self.assertEqual((empty.type, empty.arguments["duration"]), ("wait", 1.0))
        control = normalize_action("terminal", "**Action:**\n```json\n[{\"keystrokes\": \"C-c\", \"duration\": 0.1}]\n```")
        self.assertEqual(control.type, "keys")
        mcp = normalize_action("mcp", "**Action:**\n```json\n{\"name\": \"directory_tree\", \"arguments\": {\"path\": \"/app\"}}\n```")
        self.assertEqual((mcp.type, mcp.arguments), ("directory_tree", {"path": "/app"}))
        swe = normalize_action("swe", "**Action:**\n```json\n{\"name\": \"TodoWrite\", \"arguments\": \"{\\\"todos\\\": []}\"}\n```")
        self.assertEqual(swe.arguments, {"todos": []})
        android_json = normalize_action("android", "**Action:**\n```json\n{\"action_type\": \"navigate_back\"}\n```")
        self.assertEqual(android_json.type, "navigate_back")
        android_xml = normalize_action("android", "**Action:**\n```xml\n<invoke name=\"phone\">\n<parameter name=\"action\">click</parameter>\n<parameter name=\"index\">46</parameter>\n</invoke>\n```")
        self.assertEqual((android_xml.type, android_xml.arguments), ("phone.click", {"index": 46}))
        web = normalize_action("web", "**Action:**\n```text\nbrowser_navigate(url=\"http://example.com\")\n```")
        self.assertEqual((web.type, web.arguments), ("browser_navigate", {"url": "http://example.com"}))
        os_click = normalize_action("os", "**Action:**\n```text\npyautogui.click(129, 392)\n```")
        self.assertEqual((os_click.type, os_click.arguments), ("pyautogui.click", {"args": [129, 392]}))
        agent = normalize_action("os", "**Action:**\n```text\nAgent.click(coordinates=[1222, 422])\n```")
        self.assertEqual(agent.arguments, {"coordinates": [1222, 422]})
        code = normalize_action("os", "**Action:**\n```python\nimport os\nos.remove('x')\nprint_result()\n```")
        self.assertEqual((code.type, code.arguments["calls"]), ("code", ["os.remove", "print_result"]))
        unknown = normalize_action("search", "**Action:**\nfree text")
        self.assertEqual((unknown.type, unknown.arguments["text"]), ("search.action", "free text"))

    def test_split_is_stable_per_trajectory_and_longest_record_wins(self):
        rows = terminal_rows()
        self.assertEqual({split_of(str(row["id"])) for row in rows}, {"train"})
        self.assertEqual([row["turn_idx"] for row in longest_records(rows)], [3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal_test.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            self.assertEqual(len(load_rows([directory])), 3)
            self.assertEqual(len(load_rows([path])), 3)


class OutputAndJudgeTests(unittest.TestCase):
    def test_prediction_extraction_mirrors_the_official_parser(self):
        gen = "<think>draft <predicted_observation>fake</predicted_observation></think>\n<predicted_observation>\nreal\n</predicted_observation>"
        self.assertEqual(parse_model_output(gen, RESPONSE_TAG), "real")
        self.assertEqual(parse_model_output("<predicted_observation>open ended", RESPONSE_TAG), "open ended")
        self.assertEqual(parse_model_output("no tags at all", RESPONSE_TAG), "no tags at all")
        self.assertEqual(parse_model_output("", RESPONSE_TAG), "No output")
        self.assertEqual(parse_model_output(wrap_prediction("x\ny"), RESPONSE_TAG), "x\ny")

    def test_judge_messages_use_official_prompts_and_context(self):
        row = terminal_rows()[2]
        messages = build_judge_messages(row, wrap_prediction("root@host:/app/tests# pwd\n/app/tests\nroot@host:/app/tests#"))
        self.assertEqual(messages[0]["content"], load_judge_system_prompt("terminal"))
        self.assertTrue(messages[0]["content"].startswith("# Role and Objective"))
        user = messages[1]["content"]
        self.assertIn("# Context (Historical Interactions):", user)
        self.assertIn(row["prompt"][0], user)
        self.assertIn("# Current Turn:\n\n" + row["current_prompt"], user)
        self.assertNotIn("**Environment Observation:**", user.split("**Ground Truth (Real Output):**")[1])
        self.assertIn("<final_evaluation></final_evaluation>", user)

    def test_judge_output_parsing_and_failure_modes(self):
        parsed = parse_judge_output(JUDGE_OUTPUT)
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["scores"], {"format": 5, "factuality": 3, "consistency": 4, "realism": 4, "quality": 3})
        self.assertAlmostEqual(parsed["total_score"], 3.8)
        self.assertEqual(parsed["weaknesses"], ["One line is missing"])
        clamped = parse_judge_output('{"scores": {"format": "7/5", "factuality": 2.6, "consistency": 0, "realism": 4, "quality": 1}}')
        self.assertEqual(clamped["scores"], {"format": 5, "factuality": 3, "consistency": 0, "realism": 4, "quality": 1})
        self.assertAlmostEqual(clamped["total_score"], 13 / 4)
        self.assertFalse(parse_judge_output("no json here")["success"])
        self.assertFalse(parse_judge_output("")["success"])
        self.assertFalse(parse_judge_output('{"strengths": []}')["success"])

    def test_judge_rows_and_aggregation_match_the_official_scale(self):
        rows = terminal_rows()
        rows[0]["gen"] = wrap_prediction("root@host:/app# pwd\n/app\nroot@host:/app#")
        rows[1]["gen"] = ""
        rows[2]["gen"] = wrap_prediction("something")
        broken = "<final_evaluation>not json</final_evaluation>"
        llm = ScriptedChatLLM({"agentworld_judge": [JUDGE_OUTPUT, broken], "agentworld_judge_retry1": [broken],
                               "agentworld_judge_retry2": [JUDGE_OUTPUT.replace('"format": 5', '"format": 1')]})
        judged = judge_rows(rows, llm)
        self.assertEqual(judged[0]["failed"], 0.0)
        self.assertEqual(judged[0]["format"], 5)
        self.assertEqual(judged[1]["error_message"], "No model generation")
        self.assertEqual(judged[2]["format"], 1)
        self.assertEqual([call["role"] for call in llm.calls],
                         ["agentworld_judge", "agentworld_judge", "agentworld_judge_retry1", "agentworld_judge_retry2"])
        summary = aggregate_scores(judged)
        domain = summary["domains"]["terminal"]
        self.assertEqual((domain["total"], domain["valid"], domain["failed"]), (3, 2, 1))
        self.assertAlmostEqual(domain["scores"]["format"], 50.0)  # mean(5, 1) = 3 -> (3 - 1) / 4 * 100
        self.assertAlmostEqual(domain["scores"]["total_score"], (3.4 - 1) / 4 * 100)  # mean(3.8, 3.0) = 3.4
        self.assertAlmostEqual(summary["overall"], (3.4 - 1) / 4 * 100)
        self.assertEqual(summary["failed"], 1)


class EpisodeExportTests(unittest.TestCase):
    def test_export_roundtrips_through_the_trace_adapters(self):
        rows = terminal_rows()
        episode = trajectory_episode(rows[2])
        kinds = [event["kind"] for event in episode["events"]]
        self.assertEqual(kinds, ["state", "action", "observation", "action", "observation", "action", "observation"])
        self.assertEqual(episode["events"][1]["content"]["type"], "pwd")
        with tempfile.TemporaryDirectory() as directory:
            paths, manifest = export_episodes(rows, directory)
            self.assertEqual(len(paths), 1)
            digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            assignment = manifest.assignments[f"src_{digest}"]
            self.assertEqual((assignment.split, assignment.trajectory_group), ("train", episode["trajectory_group"]))
            loaded = load_raw_traces(paths[0])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].metadata["trajectory_group"], episode["trajectory_group"])
            transitions = segment_transitions(loaded[0])
            self.assertEqual(len(transitions), 3)
            self.assertTrue(all(len(t.observation_event_ids) == 1 for t in transitions))
            recon = TraceReconstructor(ScriptedLLM({}), ReconstructionConfig(environment_id="awb.terminal", name="T"),
                                       Path(directory) / "work")
            episodes, slices = recon.ingest(paths, manifest)
            self.assertEqual((len(episodes), len(slices)), (1, 3))
            self.assertEqual(export_episodes(rows, Path(directory) / "none", split="test")[0], [])


class StateTrackerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = EnvironmentPackage(terminal_package(self.root))
        self.turns = observed_turns(case_from_row(terminal_rows()[2]))

    def test_rules_apply_when_their_rendering_matches_and_models_fill_the_rest(self):
        llm = ScriptedLLM({"track_state": [StateTrackingResult(
            mutations=[StateMutation(op="set", path="session.cwd", value="/app/tests"),
                       StateMutation(op="set", path="surface.prompt", value="root@host:/app/tests#"),
                       StateMutation(op="set", path="world.unknown", value=1),
                       StateMutation(op="set", path="session.cwd", value={"$action_arg": "argv.0"})],
            uncertain_paths=["session.env"])]})
        tracker = StateTracker(self.package, llm, window=8)
        state, steps = tracker.advance(EnvironmentState(session={"cwd": "/app"}, surface={"prompt": "root@host:/app#"}), self.turns)
        self.assertEqual([step.route for step in steps], ["rule", "model"])
        self.assertEqual(steps[0].rule_id, "pwd_read")
        self.assertEqual(steps[1].applied, 2)
        self.assertEqual(len(steps[1].dropped), 2)
        self.assertIn("uncertain: session.env", steps[1].issues)
        self.assertEqual((state.session["cwd"], state.surface["prompt"], state.step), ("/app/tests", "root@host:/app/tests#", 2))
        payload = json.loads(llm.calls[0]["user"])
        self.assertEqual([turn["turn"] for turn in payload["observed_turns"]], [2])
        self.assertEqual(payload["current_state"]["session"]["cwd"], "/app")

    def test_rule_that_contradicts_the_observation_is_not_trusted(self):
        observed = [self.turns[0].model_copy(update={"observation": "root@host:/app# pwd\n/somewhere/else\nroot@host:/app#"})]
        tracker = StateTracker(self.package, None)
        state, steps = tracker.advance(EnvironmentState(session={"cwd": "/app"}, surface={"prompt": "root@host:/app#"}), observed)
        self.assertEqual(steps[0].route, "skipped")
        self.assertTrue(any("predicted a different observation" in issue for issue in steps[0].issues))
        self.assertEqual(state.step, 1)

    def test_windows_flush_before_a_rule_and_at_the_end(self):
        model_turn = self.turns[1]
        turns = [model_turn.model_copy(update={"turn": 1}), model_turn.model_copy(update={"turn": 2}),
                 self.turns[0].model_copy(update={"turn": 3}), model_turn.model_copy(update={"turn": 4})]
        results = [StateTrackingResult()] * 3
        llm = ScriptedLLM({"track_state": list(results)})
        tracker = StateTracker(self.package, llm, window=5)
        _, steps = tracker.advance(EnvironmentState(session={"cwd": "/app"}, surface={"prompt": "root@host:/app#"}), turns)
        self.assertEqual([(step.route, step.turns) for step in steps], [("model", [1, 2]), ("rule", [3]), ("model", [4])])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package_dir = terminal_package(self.root)
        self.rows = terminal_rows()

    def test_prompting_mode_reproduces_official_inference(self):
        llm = ScriptedChatLLM({"agentworld_infer": ["<think>hmm</think>\n<predicted_observation>\nout\n</predicted_observation>"] * 3})
        outputs = AgentWorldRunner(mode="prompting", chat_llm=llm).run(self.rows)
        self.assertEqual(parse_model_output(outputs[2]["gen"], RESPONSE_TAG), "out")
        self.assertEqual(outputs[2]["trace2env"]["route"], "prompting")
        self.assertEqual(llm.calls[2]["messages"], inference_messages(case_from_row(self.rows[2])))

    def test_agentic_mode_reconstructs_state_once_per_trajectory_and_routes_each_turn(self):
        submission = TransitionSubmission(
            outcome=Outcome.SUCCESS, observation="root@host:/app# cd tests\nroot@host:/app/tests#",
            effects=[StateMutation(op="set", path="session.cwd", value="/app/tests"),
                     StateMutation(op="set", path="surface.prompt", value="root@host:/app/tests#")],
            citations=["memory:2", "note:missing"], rationale="cd changes the prompt path.")
        llm = ScriptedLLM({
            "track_state": [StateTrackingResult(),
                            StateTrackingResult(mutations=[StateMutation(op="set", path="session.cwd", value="/app/tests"),
                                                           StateMutation(op="set", path="surface.prompt", value="root@host:/app/tests#")])],
            "runtime_agent_turn": [AgentTurn(tool="recall", arguments={"query": "pwd"}), AgentTurn(final=submission)],
        })
        runner = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), llm=llm,
                                  agent_llm=StructuredAgentLLM(llm), history_window=2, features=set())  # baseline harness
        outputs = runner.run(list(reversed(self.rows)))  # input order is preserved regardless of trajectory grouping
        by_turn = {row["turn_idx"]: row for row in outputs}
        self.assertEqual([row["turn_idx"] for row in outputs], [3, 2, 1])
        self.assertEqual(parse_model_output(by_turn[1]["gen"], RESPONSE_TAG), "root@host:/app# pwd\n/app\nroot@host:/app#")
        self.assertEqual(by_turn[1]["trace2env"]["route"], "rule")
        agentic = by_turn[2]["trace2env"]
        self.assertEqual((agentic["route"], agentic["tool_calls"]), ("agent", ["recall"]))
        self.assertIn("model_proposed_effects", agentic["verification"])
        self.assertIn("uncited_reference", agentic["verification"])  # note:missing was never retrieved
        self.assertEqual(agentic["citations"], ["memory:2", "note:missing"])
        self.assertEqual(parse_model_output(by_turn[2]["gen"], RESPONSE_TAG), submission.observation)
        self.assertEqual(parse_model_output(by_turn[3]["gen"], RESPONSE_TAG), clean_response_marker(self.rows[2]["response"][-1]))
        self.assertEqual(by_turn[3]["trace2env"]["rule_ids"], ["pwd_read"])
        tracking = by_turn[3]["trace2env"]["state_tracking"]
        self.assertEqual((tracking["rule_turns"], tracking["model_turns"], tracking["model_calls"]), (1, 2, 2))
        roles = [call["role"] for call in llm.calls]
        self.assertEqual(roles.count("track_state"), 2)
        agent_calls = [call for call in llm.calls if call["role"] == "runtime_agent_turn"]
        brief = json.loads(json.loads(agent_calls[0]["user"])["conversation"][0]["content"])
        self.assertEqual(brief["state_summary"]["session"]["cwd"], "/app")
        self.assertEqual([entry["turn"] for entry in brief["recent_memory"]], [0, 1])  # initial screen and the pwd turn
        self.assertIsNone(brief["retrieved"]["applicable_rule"])
        self.assertIn("Terminal World Model", agent_calls[0]["system"])
        recall_result = json.loads(agent_calls[1]["user"])["conversation"][-1]
        self.assertEqual(recall_result["role"], "tool")
        self.assertIn("root@host:/app# pwd", recall_result["content"])
        self.assertEqual(trace2env_diagnostics(outputs)["routes"], {"rule": 2, "agent": 1})

    def test_agentic_mode_without_a_model_only_executes_rules(self):
        outputs = AgentWorldRunner(mode="agentic", package_dir=str(self.package_dir), features=set()).run(self.rows)
        self.assertEqual(outputs[0]["trace2env"]["route"], "rule")
        self.assertEqual((outputs[1]["gen"], outputs[1]["trace2env"]["route"]), ("", "error"))
        self.assertIn("NoApplicableRule", outputs[1]["trace2env"]["harness_error"])
        # Turn 3 still renders from the (untracked) state: cwd never changed, so the prediction is wrong but present.
        self.assertEqual(outputs[2]["trace2env"]["state_tracking"]["skipped_turns"], 2)
        self.assertEqual(parse_model_output(outputs[2]["gen"], RESPONSE_TAG), "root@host:/app# pwd\n/app\nroot@host:/app#")

    def test_cli_commands_run_end_to_end_without_a_network(self):
        data = self.root / "terminal_test.jsonl"
        data.write_text("\n".join(json.dumps(row) for row in self.rows) + "\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["awb-export", str(data), "--output", str(self.root / "episodes")]), 0)
        self.assertEqual(json.loads(out.getvalue())["trajectories"], 1)
        self.assertTrue((self.root / "episodes" / "split_manifest.json").exists())
        predictions = self.root / "predictions.jsonl"
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["awb-run", str(data), "--output", str(predictions), "--mode", "agentic",
                                   "--package", str(self.package_dir), "--no-model"]), 0)
        self.assertEqual(json.loads(out.getvalue())["routes"], {"rule": 2, "error": 1})
        judged = self.root / "judged.jsonl"
        llm = ScriptedChatLLM({"agentworld_judge": [JUDGE_OUTPUT, JUDGE_OUTPUT]})
        with patch("trace2env.cli._chat_llm", return_value=llm), contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["awb-judge", "--predictions", str(predictions), "--output", str(judged),
                                   "--summary", str(self.root / "summary.json")]), 0)
        self.assertIn("Terminal (2/3 valid, 1 failed)", out.getvalue())
        self.assertIn("total_score: 70.00", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["awb-score", "--predictions", str(judged)]), 0)
        self.assertIn("Overall: 70.00", out.getvalue())
        self.assertIn("Trace2Env routes: error=1, rule=2", out.getvalue())


if __name__ == "__main__":
    unittest.main()
