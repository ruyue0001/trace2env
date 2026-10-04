#!/usr/bin/env python3
"""How hard is the turn-wise test? Deterministic analysis of the exported EnvScaler rows; no model call.

    python work/exp-envscaler/analyze_difficulty.py --data work/envscaler --experiment work/exp-envscaler \
        --output work/exp-envscaler/difficulty.json

Evaluation-side: it reads the environment source (statically, with ``ast``; nothing is executed here) and the
ground truth. For each environment it reports, separately for recorded turns and probes:

* what a row asks for, by the shape of its true response: ``data`` (a read: the answer is in the database the
  prompt shows, after the episode's earlier writes), ``message`` (a confirmation; 81 of them in the collection
  changed nothing), or ``error`` (a rejection); and for the last two, where the wording can come from:
  the tool's docstring, the few-shot example of the tool, or an earlier turn of the same episode (all in the
  prompt), none of these, and whether the construction traces show it (what a package built from them could know);
* coverage of the environment's rejection branches (every ``return {"success": False, "error": ...}`` in the
  source) by recorded evaluation turns, construction turns, and probes;
* what two predictors that know nothing score under ``envscaler-score``: a constant generic success, and,
  for probes, the observation the recorded call produced at that turn.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.agentworld import clean_response_marker, wrap_prediction  # noqa: E402
from trace2env.envscaler import iter_rollouts, rollout_turns  # noqa: E402
from trace2env.envscaler_oracle import literal, outcome_signature  # noqa: E402
from trace2env.envscaler_scoring import aggregate, score_rows  # noqa: E402


def in_docstring(wording: str, docstring: str) -> bool:
    """Every static fragment of two or more words appears in the docstring (and there is at least one)."""
    fragments = [part.strip(" .,:;'\"") for part in wording.split("{}")]
    fragments = [part for part in fragments if len(part.split()) >= 2]
    return bool(fragments) and all(part.lower() in docstring.lower() for part in fragments)


def rejection_branches(metadata: dict) -> set[tuple[str, str]]:
    """``(tool, error template)`` for every rejection the source can return; f-string fields become ``{}``."""
    branches = set()
    for tool, details in metadata["env_func_details"].items():
        for node in ast.walk(ast.parse(textwrap.dedent(details["source_code"]))):
            if not (isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)):
                continue
            keys = [key.value if isinstance(key, ast.Constant) else None for key in node.value.keys]
            if "success" not in keys or "error" not in keys:
                continue
            flag, error = node.value.values[keys.index("success")], node.value.values[keys.index("error")]
            if not (isinstance(flag, ast.Constant) and flag.value is False):
                continue
            if isinstance(error, ast.Constant):
                branches.add((tool, str(error.value)))
            elif isinstance(error, ast.JoinedStr):
                branches.add((tool, "".join(part.value if isinstance(part, ast.Constant) else "{}" for part in error.values)))
            else:
                branches.add((tool, "{}"))
    return branches


def reached(branches: set[tuple[str, str]], tool: str, message: str) -> set[tuple[str, str]]:
    hits = set()
    for branch_tool, template in branches:
        if branch_tool != tool:
            continue
        pattern = "^" + ".*".join(re.escape(part) for part in template.split("{}")) + "$"
        if re.match(pattern, message, re.S):
            hits.add((branch_tool, template))
    return hits


def observed(rollouts: list[dict]) -> list[dict]:
    """One entry per recorded tool call: signature, error text, and position in its episode."""
    entries = []
    for rollout in rollouts:
        steps = {step.get("step"): step for step in rollout["trajectory"]}
        for index, turn in enumerate(rollout_turns(rollout), start=1):
            value = literal(turn["output"])
            action = {"name": turn["name"], "arguments": turn["action"].arguments}
            signature = outcome_signature(value, action, steps[turn["step"]]["state_before"])
            entries.append({"task": rollout["task_id"], "turn": index, "signature": signature,
                            "error": str(value.get("error")) if isinstance(value, dict) and value.get("success") is False else None,
                            "earlier_write": any(steps[earlier["step"]]["state_diff"] for earlier in rollout_turns(rollout)[:index - 1])})
    return entries


def taxonomy(entries: list[dict], history: dict[str, list[dict]], docstrings: dict[str, str], construction: set,
             examples: set) -> dict:
    counts: Counter[str] = Counter()
    for entry in entries:
        tool, kind, wording = entry["signature"]
        counts["rows"] += 1
        counts[f"{kind}"] += 1
        if kind == "data":
            counts["data_after_an_earlier_write"] += bool(entry.get("earlier_write"))
        if kind in ("message", "error"):
            documented = in_docstring(wording, docstrings.get(tool, ""))
            earlier = any(item["signature"] == entry["signature"] and item["turn"] < entry["turn"] for item in history.get(entry["task"], []))
            shown = entry["signature"] in examples
            counts[f"{kind}_wording_in_docstring"] += documented
            counts[f"{kind}_wording_in_few_shot_examples"] += shown
            counts[f"{kind}_wording_earlier_in_episode"] += earlier
            counts[f"{kind}_wording_not_in_prompt"] += not documented and not earlier and not shown
            counts[f"{kind}_wording_not_in_prompt_but_in_construction_traces"] += (not documented and not earlier and not shown
                                                                                   and entry["signature"] in construction)
    counts["rows"] += 0
    counts["needs_more_than_the_prompt"] = counts["message_wording_not_in_prompt"] + counts["error_wording_not_in_prompt"]
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="work/envscaler")
    parser.add_argument("--experiment", default="work/exp-envscaler")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, experiment = Path(args.data), Path(args.experiment)
    report: dict[str, dict] = {}
    for benchmark_dir in sorted(path for path in (data / "benchmark").iterdir() if path.is_dir()):
        env = benchmark_dir.name.split("__")[0]
        metadata = json.loads((data / "env_defs" / f"{env}_metadata.json").read_text(encoding="utf-8"))
        docstrings = {tool["function"]["name"]: tool["function"].get("description", "") for tool in metadata["tools"]}
        branches = rejection_branches(metadata)
        every_turn = observed([rollout for _, rollout in iter_rollouts([benchmark_dir])])
        reserve_dir = data / "reserve" / env  # an environment may have no reserve rollouts
        construction = observed([rollout for _, rollout in iter_rollouts([reserve_dir])] if reserve_dir.is_dir() else [])
        construction_signatures = {entry["signature"] for entry in construction}
        # The rows' few-shot section shows the first construction call of each tool (envscaler.few_shot_examples).
        first_calls: dict[str, tuple] = {}
        for entry in construction:
            first_calls.setdefault(entry["signature"][0], entry["signature"])
        example_signatures = set(first_calls.values())
        history: dict[str, list[dict]] = {}
        for entry in every_turn:
            history.setdefault(entry["task"], []).append(entry)  # a row's history is complete, whichever turns are evaluated
        probes = []
        for path in sorted((experiment / env / "ground_truth" / "probes").glob("*.json")):
            for item in json.loads(path.read_text(encoding="utf-8")):
                signature = (item["signature"]["tool"], item["signature"]["kind"], item["signature"]["wording"])
                value = literal(item["observation"])
                probes.append({"task": path.stem, "turn": item["turn"], "signature": signature,
                               "error": str(value.get("error")) if item["outcome_label"] == "rejected" and isinstance(value, dict) else None})

        def coverage(entries: list[dict]) -> set:
            return {branch for entry in entries if entry["error"] is not None for branch in reached(branches, entry["signature"][0], entry["error"])}

        rows = [json.loads(line) for line in (experiment / env / "awb" / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        evaluated = {(row["benchmark_task"], row["turn_idx"]) for row in rows if not row.get("probe")}
        held_out = [entry for entry in every_turn if (entry["task"], entry["turn"]) in evaluated]  # the rows the export kept
        recorded_answer = {(row["id"], row["turn_idx"]): clean_response_marker(row["response"][row["turn_idx"] - 1])
                           for row in rows if not row.get("probe")}
        generic = "{'success': True, 'message': 'Operation completed successfully.'}"
        baselines = {
            "constant_generic_success": aggregate(score_rows({**row, "gen": wrap_prediction(generic)} for row in rows)),
            "answer_of_the_recorded_call": aggregate(score_rows({**row, "gen": wrap_prediction(recorded_answer[(row["id"], row["turn_idx"])])}
                                                                for row in rows if row.get("probe"))),
        }
        report[env] = {
            "recorded_rows": taxonomy(held_out, history, docstrings, construction_signatures, example_signatures),
            "probe_rows": taxonomy(probes, history, docstrings, construction_signatures, example_signatures),
            "rejection_branches": {
                "in_source": len(branches),
                "wording_in_docstring": sum(1 for tool, template in branches if in_docstring(template, docstrings.get(tool, ""))),
                "reached_by_recorded_evaluation_turns": len(coverage(held_out)),
                "reached_by_construction_turns": len(coverage(construction)),
                "reached_by_probes": len(coverage(probes)),
                "reached_by_probes_and_construction_turns": len(coverage(probes) & coverage(construction)),
            },
            "know_nothing_predictors": {name: {group: {metric: summary[group][metric] for metric in ("rows", "outcome_match", "match")}
                                               for group in ("set:recorded", "set:probe") if group in summary}
                                        for name, summary in baselines.items()},
        }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    header = f"{'environment':<12}{'set':<10}{'rows':>6}{'data':>6}{'message':>9}{'error':>7}{'beyond prompt':>15}{'always-success outcome%':>25}"
    print(header)
    for env, item in report.items():
        for name, key in (("recorded", "recorded_rows"), ("probe", "probe_rows")):
            counts = item[key]
            if not counts["rows"]:
                continue  # no probes were exported
            constant = item["know_nothing_predictors"]["constant_generic_success"][f"set:{name}"]["outcome_match"]
            print(f"{env:<12}{name:<10}{counts['rows']:>6}{counts.get('data', 0):>6}{counts.get('message', 0):>9}{counts.get('error', 0):>7}"
                  f"{counts['needs_more_than_the_prompt']:>15}{constant:>25.1f}")
        branches_report = item["rejection_branches"]
        print(f"{'':<12}rejection branches: {branches_report['in_source']} in source, {branches_report['wording_in_docstring']} worded in a docstring; "
              f"reached by recorded evaluation turns {branches_report['reached_by_recorded_evaluation_turns']}, by construction turns "
              f"{branches_report['reached_by_construction_turns']}, by probes {branches_report['reached_by_probes']} "
              f"(of which also in construction: {branches_report['reached_by_probes_and_construction_turns']})")


if __name__ == "__main__":
    main()
