#!/usr/bin/env python3
"""Independent check of the exported EnvScaler evaluation rows against the raw rollout files. No model call.

    python work/exp-envscaler/verify_rows.py --data work/envscaler --experiment work/exp-envscaler \
        --output work/exp-envscaler/rows_verification.json

The exporter is not trusted here: what should be in the rows is re-derived from the collection's own files
(``benchmark/``, ``reserve/``, ``env_defs/``) with plain JSON, and compared with what the rows hold. Exit status is
non-zero when any check finds a violation.

  selection      the evaluated (task, turn) set is exactly the one the recorded rule (``unique_turns.json``: level and
                 treatment of repeated calls) gives when recomputed from the raw rollouts; for the study's rule
                 (environment, same-answer) that is: no call to a tool the environment lacks, and one row per distinct
                 (call, answer) of an environment, the occurrence with the longest history (then the last task by name)
  report         ``unique_turns.json`` accounts for every turn, and each dropped turn has the reason the raw data gives
  unique         no two rows of an environment evaluate the same call with the same answer
  complete       every distinct (call, answer) of the held-out rollouts is evaluated by a row: nothing unique is lost
                 (checked when the rule promises it: same-answer at level episode or environment)
  identity       ids, environment, split, task, turn counts, list lengths, no probe marker
  history        every turn of a row's history is the raw turn: same tool, same arguments, and the recorded
                 ``observation`` text (never the parsed ``observation_raw``, which shows later writes)
  target         the evaluated turn's action and ground truth are the raw turn's, and the instruction suffix is official
  initial_state  no prompt shows the initial database, matching the official MCP prompt layout
  no_leak        no ground-truth field name and no agent-side ``chat_with_user`` call anywhere in a row
  system_prompt  one system prompt per environment; its tool definitions are the definition file's, in order; its
                 few-shot examples are each the first call of a tool in the construction rollouts, and none of them is
                 a (call, answer) of a held-out rollout
  oracle         executing the environment's own source on the raw state before the turn returns the row's ground
                 truth (up to values the environment generates)
  generated      the values the scorer accepts by shape are the ones the environment generates: the source is run again
                 with other ids and another clock; a token of the ground truth the scorer relaxes must change, and a
                 token that changes and that the model was never shown must be relaxed (a computed date must not be)
  scorer         the row's ground truth, given as the prediction, scores as an exact match
  shards         the shards partition the rows and keep each trajectory in one shard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.agentworld import RESPONSE_MARKER, action_block, prompt_sections, wrap_prediction  # noqa: E402
from trace2env.atif import BENCHMARK_SUFFIX  # noqa: E402
import datetime  # noqa: E402

import trace2env.envscaler_oracle as oracle_module  # noqa: E402
from trace2env.envscaler_oracle import EnvironmentOracle  # noqa: E402
from trace2env.envscaler_scoring import generated_tokens, row_context, score_row  # noqa: E402

AGENT_SIDE = {"chat_with_user"}
# A second execution with other ids (seed) and another clock shows which tokens of an observation are generated.
DEFAULT_CLOCK, OTHER_CLOCK = oracle_module.CLOCK, datetime.datetime(2031, 7, 9, 3, 4, 5)
TOKEN = re.compile(r"[0-9A-Za-z][0-9A-Za-z:.\-]*")
GROUND_TRUTH_NAMES = ("state_before", "state_after", "state_diff", "sigma_t", "observation_raw", "O_tplus1", "S_tplus1",
                      "Delta_t", "n_delta", "obs_success", "init_config", "final_state", "checklist")
EXAMPLE = re.compile(r"^## Example: (.+?)\n\*\*Action:\*\*\n```json\n(.*?)\n```\n\*\*Environment Observation:\*\*\n(.*?)\n\n(?=## Example: |\Z)", re.S | re.M)


def call_key(action: dict) -> str:
    return json.dumps({"name": action.get("name"), "arguments": action.get("arguments") or {}}, sort_keys=True, ensure_ascii=False)


def raw_turns(rollout: dict) -> list[dict]:
    """The environment tool calls of a raw rollout, numbered from 1 (the agent's own closing call is not one)."""
    steps = [step for step in rollout["trajectory"] if step["action"]["name"] not in AGENT_SIDE]
    return [{"turn": index, "action": step["action"], "observation": step["observation"], "artifact": bool(step.get("error")),
             "state_before": step["state_before"], "stale_raw": isinstance(step.get("observation_raw"), dict)
             and str(step["observation_raw"]) != step["observation"]} for index, step in enumerate(steps, start=1)]


def load_rollouts(directory: Path) -> dict[str, dict]:
    return {rollout["task_id"]: rollout for rollout in (json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json")))}


class Checks:
    def __init__(self) -> None:
        self.checked: Counter[str] = Counter()
        self.violations: dict[str, list[str]] = defaultdict(list)

    def expect(self, name: str, condition: bool, detail: str) -> None:
        self.checked[name] += 1
        if not condition:
            self.violations[name].append(detail)


def verify_environment(env: str, data: Path, experiment: Path, rows_name: str) -> tuple[dict, Checks]:
    checks = Checks()
    benchmark_dir = next(path for path in sorted((data / "benchmark").iterdir()) if path.name.split("__")[0] == env)
    held_out, construction = load_rollouts(benchmark_dir), load_rollouts(data / "reserve" / env)
    metadata = json.loads((data / "env_defs" / f"{env}_metadata.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (experiment / env / "awb" / rows_name).read_text(encoding="utf-8").splitlines()]
    turns = {task: raw_turns(rollout) for task, rollout in held_out.items()}

    # selection, recomputed from the raw rollouts under the rule the export recorded (none when it filtered nothing)
    report_path = experiment / env / "unique_turns.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))[env] if report_path.is_file() else None
    level, repeated = (report["level"], report["repeated_calls"]) if report else ("none", None)
    everything_raw = [(task, item) for task, items in turns.items() for item in items]
    if level == "none":
        expected = {(task, item["turn"]) for task, item in everything_raw}
    else:
        def keep_one(candidates: list, key) -> list:
            best: dict = {}
            for task, item in candidates:
                if key(task, item) not in best or (item["turn"], task) > (best[key(task, item)][1]["turn"], best[key(task, item)][0]):
                    best[key(task, item)] = (task, item)
            return list(best.values())

        survivors = [(task, item) for task, item in everything_raw if not item["artifact"]]
        survivors = keep_one(survivors, (lambda task, item: (task, call_key(item["action"]), item["observation"])) if repeated == "same-answer"
                             else (lambda task, item: (task, call_key(item["action"]))))
        if level in ("environment", "answer"):
            survivors = keep_one(survivors, lambda task, item: (call_key(item["action"]), item["observation"]))
        if level == "answer":
            survivors = keep_one(survivors, lambda task, item: (item["action"]["name"], item["observation"]))
        expected = {(task, item["turn"]) for task, item in survivors}
    evaluated = [(row["benchmark_task"], row["turn_idx"]) for row in rows]
    checks.expect("selection", set(evaluated) == expected,
                  f"rows and rule ({level}, {repeated}) disagree: only in rows {sorted(set(evaluated) - expected)[:5]}, only in rule {sorted(expected - set(evaluated))[:5]}")
    checks.expect("selection", len(evaluated) == len(set(evaluated)), "a (task, turn) has more than one row")
    valid = [(task, turn) for task, turn in evaluated if task in turns and 1 <= turn <= len(turns[task])]
    pair_of = lambda task, turn: (call_key(turns[task][turn - 1]["action"]), turns[task][turn - 1]["observation"])  # noqa: E731
    if level in ("environment", "answer"):
        pairs = Counter(pair_of(task, turn) for task, turn in valid)
        checks.expect("unique", all(count == 1 for count in pairs.values()), f"{sum(1 for c in pairs.values() if c > 1)} (call, answer) pairs have several rows in the environment")
    elif level == "episode":
        pairs = Counter((task,) + pair_of(task, turn) for task, turn in valid)
        checks.expect("unique", all(count == 1 for count in pairs.values()), "a (call, answer) has several rows within one episode")
    if repeated == "same-answer" and level in ("episode", "environment"):
        # This rule promises that nothing unique is lost: every distinct (call, answer) the environment gave keeps a row.
        promised = {(call_key(item["action"]), item["observation"]) for _, item in everything_raw if not item["artifact"]}
        checks.expect("complete", {pair_of(task, turn) for task, turn in valid} == promised,
                      f"{len(promised - {pair_of(task, turn) for task, turn in valid})} distinct (call, answer) pairs of the raw rollouts have no row")
    if level != "none":
        checks.expect("selection", not any(turns[task][turn - 1]["artifact"] for task, turn in valid), "a call to a tool the environment lacks is evaluated")
    winners = {(call_key(item["action"]), item["observation"]) for _, item in everything_raw if not item["artifact"]}

    # the exporter's own report must agree with the raw data
    total = sum(len(items) for items in turns.values())
    for item in ([] if report else [None]):
        checks.expect("report", len(rows) == total, f"no filter report, yet {len(rows)} rows for {total} raw turns")
    report = report or {"turns": total, "kept": len(rows), "dropped": []}
    checks.expect("report", report["turns"] == total and report["kept"] == len(rows) and len(report["dropped"]) == total - len(rows),
                  f"report counts {report['turns']}/{report['kept']}/{len(report['dropped'])} vs raw {total}/{len(rows)}/{total - len(rows)}")
    for item in report["dropped"]:
        raw = turns[item["benchmark_task"]][item["turn"] - 1]
        key = (call_key(raw["action"]), raw["observation"])
        if item["reason"] == "loader_artifact":
            ok = raw["artifact"]
        else:
            kept = item["kept"]
            kept_raw = turns[kept["benchmark_task"]][kept["turn"] - 1]
            same = (call_key(kept_raw["action"]), kept_raw["observation"]) == key
            within = kept["benchmark_task"] == item["benchmark_task"] and kept["turn"] > item["turn"]
            if item["reason"] == "repeated_action":  # the same call later in the episode; under same-answer also the same answer
                ok = within and call_key(kept_raw["action"]) == call_key(raw["action"]) and (same or repeated != "same-answer")
            elif item["reason"] == "same_call_and_answer_elsewhere":
                ok = same
            else:  # same_tool_and_answer_elsewhere
                ok = kept_raw["action"]["name"] == raw["action"]["name"] and kept_raw["observation"] == raw["observation"]
        checks.expect("report", ok, f"{item['benchmark_task']} turn {item['turn']}: reason {item['reason']} is not what the raw data shows")

    definitions = [tool["function"]["name"] for tool in metadata["tools"]]
    oracle, other_oracle = EnvironmentOracle(metadata), EnvironmentOracle(metadata, seed=1)
    system_prompts = {row["system_str"] for row in rows}
    checks.expect("system_prompt", len(system_prompts) == 1, f"{len(system_prompts)} different system prompts in one environment")
    stale_used = 0
    for row in rows:
        task, turn = row["benchmark_task"], row["turn_idx"]
        where = f"{task} turn {turn}"
        raw = turns.get(task)
        checks.expect("identity", raw is not None, f"{where}: not a held-out task")
        if raw is None:
            continue
        checks.expect("identity", row["id"] == f"envscaler-{task}" and row["env_id"] == env and row["task"] == "mcp" and row["source"] == "envscaler"
                      and row["split"] == "test" and not row.get("probe") and held_out[task]["env_id"] == env, f"{where}: identity fields")
        checks.expect("identity", row["total_turns"] == len(raw) and 1 <= turn <= len(raw) and len(row["prompt"]) == turn == len(row["response"]),
                      f"{where}: turn counts ({row['total_turns']} vs {len(raw)} raw turns, {len(row['prompt'])} prompts)")
        for index in range(1, min(turn, len(raw)) + 1):
            prompt, source = row["prompt"][index - 1], raw[index - 1]
            try:
                action = json.loads(action_block(prompt)[1])
            except json.JSONDecodeError:
                action = None
            name = "target" if index == turn else "history"
            checks.expect(name, action == {"name": source["action"]["name"], "arguments": source["action"].get("arguments") or {}},
                          f"{where}: action of turn {index} differs from the raw rollout")
            checks.expect(name, row["response"][index - 1] == f"{RESPONSE_MARKER}\n{source['observation']}",
                          f"{where}: observation of turn {index} is not the recorded observation text")
            checks.expect(name, prompt.startswith(f"### Turn {index}\n"), f"{where}: turn {index} is not labelled as such")
            stale_used += source["stale_raw"]
            sections = prompt_sections(prompt)
            checks.expect("initial_state", "current_state" not in sections, f"{where}: turn {index} shows a state")
        checks.expect("target", row["prompt"][-1] == row["current_prompt"] + BENCHMARK_SUFFIX, f"{where}: current_prompt + official suffix is not the last prompt")
        # The text the model is shown and the row's own keys (not the JSON-encoded row, where quotes inside text are escaped).
        text = "\n".join([*row["prompt"], *row["response"], row["current_prompt"]])
        leaked = [name for name in GROUND_TRUTH_NAMES if f'"{name}"' in text or f"'{name}'" in text or name in row]
        checks.expect("no_leak", not leaked and "chat_with_user" not in text and "chat_with_user" not in row["system_str"],
                      f"{where}: {leaked or 'chat_with_user'} appears in the row")
        result = oracle.step(raw[turn - 1]["state_before"], raw[turn - 1]["action"])
        truth = row["response"][-1][len(RESPONSE_MARKER) + 1:]
        agrees = result["observation"] is not None and score_row({**row, "gen": wrap_prediction(result["observation"])})["match"]
        if result["observation"] is not None and not generated_tokens(truth, row_context(row)):
            agrees = agrees and result["observation"] == truth  # nothing generated: the source must return the very text
        checks.expect("oracle", bool(agrees), (f"{where}: the environment source cannot execute this call ({result['raised']}); the recorded {truth[:40]!r} is "
                                               "the collector's answer, not the environment's") if result["raised"] else
                      f"{where}: the environment source returns {str(result['observation'])[:80]!r}, the row says {truth[:80]!r}")
        if result["observation"] is not None:
            oracle_module.CLOCK = OTHER_CLOCK
            try:
                again = other_oracle.step(raw[turn - 1]["state_before"], raw[turn - 1]["action"])["observation"]
            finally:
                oracle_module.CLOCK = DEFAULT_CLOCK
            first, second, recorded = (TOKEN.findall(text) for text in (result["observation"], str(again), truth))
            aligned = len(first) == len(second) == len(recorded)
            varying = {recorded[index] for index in range(len(first)) if first[index] != second[index]} if aligned else set()
            context = row_context(row)
            relaxed = {match.group(0) for match in generated_tokens(truth, context, row.get("generated_shapes"))}
            fixed = sorted(token for token in relaxed if not any(token in item for item in varying))
            unrelaxed = sorted(item for item in varying if item not in context and not any(token in item for token in relaxed))
            checks.expect("generated", aligned and not fixed and not unrelaxed,
                          f"{where}: " + ("the two executions differ in structure" if not aligned else
                                          f"accepted by shape but computed, not generated: {fixed}" if fixed else
                                          f"generated but compared exactly: {unrelaxed}"))
        scored = score_row({**row, "gen": wrap_prediction(truth)})
        checks.expect("scorer", scored["match"] and scored["match_strict"] and scored["text_match"] and scored["outcome_match"],
                      f"{where}: the ground truth does not score as an exact match of itself")

    # system prompt: tool definitions and few-shot examples
    system = next(iter(system_prompts))
    body = system[:system.find("# Few-shot Examples")] if "# Few-shot Examples" in system else system
    checks.expect("system_prompt", re.findall(r"^### \d+\. (.+)$", body, re.M) == definitions, "tool definitions differ from the definition file")
    first_calls: dict[str, tuple[str, str]] = {}
    for task in sorted(construction):  # the exporter names construction episodes after their task; file order is task order
        for item in raw_turns(construction[task]):
            first_calls.setdefault(item["action"]["name"], (call_key(item["action"]), item["observation"]))
    examples = EXAMPLE.findall(system[system.find("# Few-shot Examples"):]) if "# Few-shot Examples" in system else []
    checks.expect("system_prompt", bool(examples), "no few-shot examples")
    held_out_pairs = {(call_key(item["action"]), item["observation"]) for items in turns.values() for item in items}
    for name, action, observation in examples:
        pair = (call_key(json.loads(action)), observation)
        checks.expect("system_prompt", first_calls.get(name) == pair, f"example {name} is not the first call of that tool in the construction rollouts")
        checks.expect("system_prompt", pair not in held_out_pairs, f"example {name} is a (call, answer) of a held-out rollout")
    checks.expect("system_prompt", [name for name, _, _ in examples] == [name for name in definitions if name in first_calls]
                  + sorted(name for name in first_calls if name not in definitions), "examples are not one per used tool in definition order")

    # shards
    shard_rows = [json.loads(line) for path in sorted((experiment / env / "awb" / "shards").glob("shard*.jsonl"))
                  for line in path.read_text(encoding="utf-8").splitlines()]
    everything = [json.loads(line) for line in (experiment / env / "awb" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    identify = lambda item: (item["id"], item["turn_idx"], item.get("probe"))  # noqa: E731
    checks.expect("shards", Counter(map(identify, shard_rows)) == Counter(map(identify, everything)), "the shards are not a partition of rows.jsonl")
    homes = defaultdict(set)
    for path in sorted((experiment / env / "awb" / "shards").glob("shard*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            homes[json.loads(line)["id"]].add(path.name)
    checks.expect("shards", all(len(files) == 1 for files in homes.values()), "a trajectory is split over several shards")
    checks.expect("shards", [identify(item) for item in everything if not item.get("probe")] == [identify(item) for item in rows],
                  f"rows.jsonl does not hold {rows_name}")
    summary = {"rows": len(rows), "raw_turns": total, "distinct_call_answer_pairs": len(winners), "few_shot_examples": len(examples),
               "history_turns_whose_parsed_observation_is_stale": stale_used}
    return summary, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="work/envscaler")
    parser.add_argument("--experiment", default="work/exp-envscaler")
    parser.add_argument("--rows-name", default="rows-recorded.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, experiment = Path(args.data), Path(args.experiment)
    report: dict[str, dict] = {}
    names: list[str] = []
    for benchmark_dir in sorted(path for path in (data / "benchmark").iterdir() if path.is_dir()):
        env = benchmark_dir.name.split("__")[0]
        if not (experiment / env / "awb" / args.rows_name).is_file():
            continue
        summary, checks = verify_environment(env, data, experiment, args.rows_name)
        report[env] = {**summary, "checks": {name: {"checked": checks.checked[name], "violations": len(checks.violations[name]),
                                                     "first": checks.violations[name][:5]} for name in checks.checked}}
        names = list(dict.fromkeys(names + list(checks.checked)))
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'check':<15}" + "".join(f"{env:>22}" for env in report))
    for name in names:
        print(f"{name:<15}" + "".join(f"{report[env]['checks'].get(name, {}).get('checked', 0):>14} ok" + (f" {report[env]['checks'][name]['violations']:>3}!!"
              if report[env]["checks"].get(name, {}).get("violations") else "      ") for env in report))
    bad = sum(item["violations"] for env in report for item in report[env]["checks"].values())
    print("rows verified:", {env: report[env]["rows"] for env in report}, "| violations:", bad)
    for env in report:
        for name, item in report[env]["checks"].items():
            for detail in item["first"]:
                print(f"VIOLATION {env} {name}: {detail}")
    if bad or not report:
        sys.exit(1)


if __name__ == "__main__":
    main()
