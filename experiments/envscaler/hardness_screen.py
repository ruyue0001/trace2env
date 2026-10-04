#!/usr/bin/env python3
"""How much of an EnvScaler environment's turn-wise test can be answered without modelling the episode? No model call.

    python work/exp-envscaler/hardness_screen.py --data work/envscaler --output work/exp-envscaler/hardness_screen.json
    python work/exp-envscaler/hardness_screen.py --data OLD --scored env_151_rl=work/exp-envscaler/env_151_rl/awb/exact-prompting.jsonl

A screen for choosing environments *before* paying for predictions. Every recorded tool call of an environment
(benchmark and reserve together: selection comes before splitting) is given to four predictors that know nothing
about the episode's own course, and a call one of them answers exactly is *cheap*:

  initial_state   the environment's own source run on the episode's INITIAL database: a perfect world model with no
                  memory of the episode. It answers every read of an untouched record, every fixed confirmation, every
                  rule; it fails exactly where the answer depends on an earlier write of the episode.
  repeat_last     the answer the identical call got earlier in the same episode
  majority        the tool's most frequent answer in the OTHER episodes of the environment
  template        the tool's most frequent answer in the other episodes with the call's argument values blanked out,
                  filled with this call's arguments (fixed wording, echoed ids and quantities)

What no predictor answers is what a world model has to track: the *hard* share. The screen also reports what the
oracle-based predictor cannot see, because the source knows every rule: how often the environment rejects, how many
rejection branches the rollouts reach, and how much confirmation / rejection wording the docstrings do not state.
`--scored ENV=FILE` (an `envscaler-score --scored` file) cross-tabulates a measured system with the screen.

Evaluation-side: it EXECUTES `env_class_code` from `env_defs` (as `envscaler-probe` does). Run it on trusted definitions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_difficulty import in_docstring, reached, rejection_branches  # noqa: E402
from trace2env.envscaler import generated_shapes, iter_rollouts, rollout_turns  # noqa: E402
from trace2env.envscaler_oracle import EnvironmentOracle, literal, outcome_signature  # noqa: E402
from trace2env.envscaler_scoring import UNPARSED, kind_of, parse_observation, values_equal  # noqa: E402

PREDICTORS = ("initial_state", "repeat_last", "majority", "template")


def same(truth: str, predicted: str | None, shapes: list[str] | None) -> bool:
    if predicted is None:
        return False
    expected, actual = parse_observation(truth), parse_observation(predicted)
    if expected is UNPARSED or actual is UNPARSED:
        return truth.strip() == predicted.strip()
    return values_equal(expected, actual, "", shapes=shapes)


def blanked(text: str, arguments: dict[str, Any]) -> str:
    """The answer with the call's argument values replaced by their names (longest first, whole tokens only)."""
    values = sorted(((str(value), name) for name, value in arguments.items() if isinstance(value, (str, int, float)) and str(value)),
                    key=lambda item: -len(item[0]))
    for value, name in values:
        text = re.sub(rf"(?<![0-9A-Za-z]){re.escape(value)}(?![0-9A-Za-z])", f"⟦{name}⟧", text)
    return text


def filled(template: str, arguments: dict[str, Any]) -> str:
    for name, value in arguments.items():
        template = template.replace(f"⟦{name}⟧", str(value))
    return template


def screen_environment(env: str, data: Path) -> dict[str, Any]:
    metadata = json.loads((data / "env_defs" / f"{env}_metadata.json").read_text(encoding="utf-8"))
    docstrings = {tool["function"]["name"]: tool["function"].get("description", "") for tool in metadata["tools"]}
    shapes = generated_shapes(str(metadata.get("env_class_code") or ""))
    oracle = EnvironmentOracle(metadata)
    branches = rejection_branches(metadata)
    directories = [path for split in ("benchmark", "reserve") for path in sorted((data / split).glob(f"{env}*")) if path.is_dir()]
    rollouts = [rollout for _, rollout in iter_rollouts(directories)] if directories else []
    calls: list[dict[str, Any]] = []
    for rollout in rollouts:
        steps = {step.get("step"): step for step in rollout["trajectory"]}
        seen: dict[str, str] = {}
        for index, turn in enumerate(rollout_turns(rollout), start=1):
            arguments = turn["action"].arguments
            action = {"name": turn["name"], "arguments": arguments}
            key = json.dumps(action, sort_keys=True)
            replay = oracle.step(rollout["init_state"], action)
            value = literal(turn["output"])
            calls.append({"task": rollout["task_id"], "turn": index, "tool": turn["name"], "arguments": arguments, "truth": turn["output"],
                          "kind": kind_of(parse_observation(turn["output"])), "initial_state": replay["observation"], "repeat_last": seen.get(key),
                          "template_of_truth": blanked(turn["output"], arguments),
                          "signature": outcome_signature(value, action, steps[turn["step"]]["state_before"]),
                          "error": str(value.get("error")) if isinstance(value, dict) and value.get("success") is False else None})
            seen[key] = turn["output"]
    answers: dict[str, Counter] = defaultdict(Counter)       # tool -> answer text -> episodes' calls
    templates: dict[str, Counter] = defaultdict(Counter)
    by_task: dict[tuple[str, str], Counter] = defaultdict(Counter)
    templates_by_task: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for call in calls:
        answers[call["tool"]][call["truth"]] += 1
        templates[call["tool"]][call["template_of_truth"]] += 1
        by_task[(call["tool"], call["task"])][call["truth"]] += 1
        templates_by_task[(call["tool"], call["task"])][call["template_of_truth"]] += 1
    solved, kinds, hard_by_tool, hard_by_kind, tools = Counter(), Counter(), Counter(), Counter(), Counter()
    undocumented, non_data, reached_branches = 0, 0, set()
    rows = []
    for call in calls:
        others = answers[call["tool"]] - by_task[(call["tool"], call["task"])]           # leave this episode out
        other_templates = templates[call["tool"]] - templates_by_task[(call["tool"], call["task"])]
        call["majority"] = others.most_common(1)[0][0] if others else None
        call["template"] = filled(other_templates.most_common(1)[0][0], call["arguments"]) if other_templates else None
        hits = [name for name in PREDICTORS if same(call["truth"], call[name], shapes)]
        kinds[call["kind"]] += 1
        tools[call["tool"]] += 1
        for name in hits:
            solved[name] += 1
        solved["any"] += bool(hits)
        if not hits:
            hard_by_tool[call["tool"]] += 1
            hard_by_kind[call["kind"]] += 1
        if call["kind"] != "data":
            non_data += 1
            undocumented += not in_docstring(call["signature"][2], docstrings.get(call["tool"], ""))
        if call["error"]:
            reached_branches |= reached(branches, call["tool"], call["error"])
        rows.append({"task": call["task"], "turn": call["turn"], "tool": call["tool"], "kind": call["kind"], "cheap": hits})
    total = len(calls)

    def share(count: int, of: int = total) -> float | None:
        return round(100 * count / of, 1) if of else None

    return {"rollouts": len(rollouts), "calls": total, "kinds": dict(kinds),
            "answered_percent": {**{name: share(solved[name]) for name in PREDICTORS}, "any": share(solved["any"])},
            "hard_calls": total - solved["any"], "hard_percent": share(total - solved["any"]),
            "hard_by_kind": dict(hard_by_kind),
            "hard_by_tool": {tool: f"{count} of {tools[tool]}" for tool, count in hard_by_tool.most_common()},
            "needs_the_episodes_earlier_writes_percent": share(total - solved["initial_state"]),
            "rejections_percent": share(kinds["error"]), "rejection_branches": {"in_source": len(branches), "reached": len(reached_branches)},
            "wording_not_in_docstring_percent_of_non_reads": share(undocumented, non_data),
            "generated_shapes": shapes, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "work" / "envscaler")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--with-rows", action="store_true", help="keep the per-call verdicts in the output (large)")
    parser.add_argument("--scored", action="append", default=[], metavar="ENV=FILE",
                        help="an `envscaler-score --scored` file of a measured system on ENV's rows: accuracy on cheap vs hard rows")
    args = parser.parse_args()
    environments = sorted({path.name.split("__")[0] for split in ("benchmark", "reserve") for path in (args.data / split).glob("*") if path.is_dir()})
    report = {env: screen_environment(env, args.data) for env in environments}
    for item in args.scored:
        env, _, path = item.partition("=")
        measured = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            measured[(str(row["benchmark_task"]), int(row["turn_idx"]))] = bool(row["exact"]["match"])
        table: dict[str, list[int]] = {"cheap": [0, 0], "hard": [0, 0]}
        for row in report[env]["rows"]:
            key = (row["task"], row["turn"])
            if key in measured:
                cell = table["cheap" if row["cheap"] else "hard"]
                cell[0] += 1
                cell[1] += measured[key]
        report[env]["measured"] = {name: {"rows": rows, "exact": exact, "exact_percent": round(100 * exact / rows, 1) if rows else None}
                                   for name, (rows, exact) in table.items()}
    header = (f"{'environment':<12}{'calls':>6}{'init':>7}{'repeat':>7}{'major.':>7}{'templ.':>7}{'ANY':>7}{'HARD':>7}{'tracking':>9}"
              f"{'reject.':>8}{'branches':>10}{'undoc.':>8}")
    print(header + "\n" + "-" * len(header))
    for env, entry in report.items():
        answered = entry["answered_percent"]
        print(f"{env:<12}{entry['calls']:>6}{answered['initial_state']:>7}{answered['repeat_last']:>7}{answered['majority']:>7}{answered['template']:>7}"
              f"{answered['any']:>7}{entry['hard_percent']:>7}{entry['needs_the_episodes_earlier_writes_percent']:>9}{entry['rejections_percent']:>8}"
              f"{str(entry['rejection_branches']['reached']) + '/' + str(entry['rejection_branches']['in_source']):>10}"
              f"{entry['wording_not_in_docstring_percent_of_non_reads']:>8}")
        if "measured" in entry:
            print(f"{'':<12}measured system: " + ", ".join(f"{name} rows {cell['exact']}/{cell['rows']} = {cell['exact_percent']}%"
                                                           for name, cell in entry["measured"].items()))
    print("\ncolumns: % of calls answered by each know-nothing predictor; HARD = answered by none; tracking = the call on the initial\n"
          "database gives another answer; reject. = % rejections; branches = rejection branches reached / in source; undoc. = % of\n"
          "confirmations and rejections whose wording no docstring states")
    if args.output:
        if not args.with_rows:
            report = {env: {key: value for key, value in entry.items() if key != "rows"} for env, entry in report.items()}
        args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
