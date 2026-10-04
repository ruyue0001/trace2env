#!/usr/bin/env python3
"""How far does the official AgentWorldBench judge suit these rows? Controlled predictions with a known verdict.

    python work/exp-envscaler/calibrate_judge.py build  --env env_151_rl
    work/exp-envscaler/run_judge_official.sh env_151_rl judge-calibration        # the original eval.py; about 13 judge calls
    python work/exp-envscaler/calibrate_judge.py report --env env_151_rl --output work/exp-envscaler/judge_calibration.json

`build` calls no model: every "prediction" is a held-out row's own ground truth, unchanged or changed in one way
whose exact verdict is known (the scorer's verdict is recorded next to it). The judge then grades them as it would
grade a system, and `report` puts the two verdicts side by side. The question is not whether the judge agrees on
right answers, but how many points a verifiably wrong answer keeps.

The cases name rows of `env_151_rl` by trajectory and turn, and `build` fails when a row no longer has the ground
truth a case was written for.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.envscaler_scoring import score_row  # noqa: E402

LABEL = "judge-calibration"
MARKER = "**Environment Observation:**\n"
DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")
SHARDS = 4


def _wrong_field(value: dict[str, Any]) -> dict[str, Any]:
    assert value["data"][0]["enrollment_status"] == "enrolled"
    value["data"][0]["enrollment_status"] = "completed"
    return value


def _reversed_records(value: dict[str, Any]) -> dict[str, Any]:
    assert len(value["data"]) == 2
    value["data"].reverse()
    return value


def _extra_record(value: dict[str, Any]) -> dict[str, Any]:
    record = dict(value["data"][0], enrollment_id="E-99999999-aaaa-4bbb-8ccc-dddddddddddd", trial_id="CT404")
    value["data"].append(record)
    return value


def _fresh_uuid(value: dict[str, Any]) -> dict[str, Any]:
    value["enrollment_id"] = "7d1f0c52-9b3e-4a6f-8c21-5e0a9d4b7f13"
    return value


def _replace(answer: dict[str, Any]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return lambda value: answer


def _identity(value: dict[str, Any]) -> dict[str, Any]:
    return value


# (trajectory id, turn, ground truth the case was written for, [(label, change, rendered as JSON)])
CASES: list[tuple[str, int, str, list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]], bool]]]] = [
    ("envscaler-env_151_rl-task_12", 3, "'data': [{'enrollment_id': 'E-3cabfd84", [
        ("read: exact ground truth", _identity, False),
        ("read: same values rendered as JSON", _identity, True),
        ("read: ONE FIELD VALUE WRONG (status enrolled -> completed)", _wrong_field, False),
        ("read: the two records in reverse order", _reversed_records, False),
        ("read: a hallucinated third record", _extra_record, False),
    ]),
    ("envscaler-env_151_rl-task_12", 4, "'message': 'Enrollment status updated successfully.'", [
        ("message: exact ground truth", _identity, False),
        ("message: right outcome, different wording", _replace({"success": True, "message": "Enrollment updated."}), False),
        ("message: WRONG OUTCOME (rejects an accepted call)", _replace({"success": False, "error": "Enrollment not found."}), False),
    ]),
    ("envscaler-env_151_rl-task_12", 9, "'message': 'Participant enrolled in trial.', 'enrollment_id': '4ac08a88", [
        ("generated id: a different fresh uuid (unpredictable by design)", _fresh_uuid, False),
        ("generated id: WRONG OUTCOME (what gpt-5.6 answered in the smoke)",
         _replace({"success": False, "error": "Participant does not have valid consent."}), False),
    ]),
    ("envscaler-env_151_rl-task_19", 7, "{'success': False, 'error': 'User account not found'}", [
        ("error: exact ground truth", _identity, False),
        ("error: right outcome, different wording (gpt-5.6's smoke answer)",
         _replace({"success": False, "error": "User account with username 'jon.mcculloch' not found."}), False),
        ("error: WRONG OUTCOME (invents an account)",
         _replace({"success": True, "data": {"account_id": "UA-1f2e3d4c-5b6a-4789-8abc-def012345678", "participant_id": "P001",
                                             "username": "jon.mcculloch", "hashed_password": "h$9f8e7d6c",
                                             "account_status": "active"}}), False),
    ]),
]


def awb_dir(experiment: Path, env: str) -> Path:
    return experiment / env / "awb"


def build(experiment: Path, env: str) -> int:
    directory = awb_dir(experiment, env)
    rows = {(row["id"], row["turn_idx"]): row for row in map(json.loads, (directory / "rows.jsonl").open(encoding="utf-8"))}
    predictions = []
    for trajectory, turn, expected, variants in CASES:
        row = rows.get((trajectory, turn))
        if row is None:
            sys.exit(f"{trajectory} turn {turn} is not an evaluated row of {env}")
        truth = row["response"][-1]
        if not truth.startswith(MARKER) or expected not in truth:
            sys.exit(f"{trajectory} turn {turn}: the ground truth is not the one this case was written for:\n{truth}")
        for label, change, as_json in variants:
            value = change(copy.deepcopy(ast.literal_eval(truth[len(MARKER):])))
            text = json.dumps(value) if as_json else repr(value)
            prediction = dict(row, gen=f"<predicted_observation>\n{text}\n</predicted_observation>", calibration_label=label)
            verdict = score_row(prediction)
            prediction["calibration_exact"] = {"match": verdict["match"], "outcome_match": verdict["outcome_match"]}
            predictions.append(prediction)
    for shard in range(SHARDS):
        with (directory / f"pred-{LABEL}-shard{shard}.jsonl").open("w", encoding="utf-8") as handle:
            for prediction in predictions[shard::SHARDS]:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    with (directory / f"pred-{LABEL}.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    for prediction in predictions:
        exact = prediction["calibration_exact"]
        print(f"match={str(exact['match']):5} outcome={str(exact['outcome_match']):5} {prediction['calibration_label']}")
    print(f"{len(predictions)} controlled predictions -> {directory}/pred-{LABEL}*.jsonl")
    return 0


def percent(raw: float) -> float:
    return round((raw - 1) / 4 * 100, 2)


def relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


def report(experiment: Path, env: str, judged_path: Path | None, output: Path) -> int:
    judged_path = judged_path or awb_dir(experiment, env) / f"official-judged-{LABEL}.jsonl"
    judged = {row["calibration_label"]: row for row in map(json.loads, judged_path.open(encoding="utf-8"))}
    cases = []
    for trajectory, turn, _expected, variants in CASES:
        for label, _change, _as_json in variants:
            row = judged.get(label)
            if row is None or row.get("failed"):
                sys.exit(f"no valid judgment for: {label}")
            cases.append({"row": f"{trajectory} turn {turn}", "prediction": label, "exact": row["calibration_exact"],
                          "judge": {**{name: row[name] for name in DIMENSIONS}, "total_score": row["total_score"],
                                    "total_0_100": percent(row["total_score"])}})
    groups = {
        "exact match (the judge should give 100)": [case for case in cases if case["exact"]["match"]],
        "right outcome, wrong content (exact match gives 0)": [case for case in cases
                                                              if case["exact"]["outcome_match"] and not case["exact"]["match"]],
        "wrong outcome (exact match gives 0)": [case for case in cases if not case["exact"]["outcome_match"]],
    }
    summary = {name: {"cases": len(members), "judge_total_0_100": sorted(case["judge"]["total_0_100"] for case in members)}
               for name, members in groups.items()}
    result = {
        "what": "Predictions with a known verdict, judged by the original AgentWorldBench eval.py (one judge pass, temperature 0.6: "
                "a re-run moves single cases by a few points).",
        "environment": env, "judged_file": relative(judged_path), "cases": cases, "summary": summary,
        "overall_0_100": round(sum(case["judge"]["total_0_100"] for case in cases) / len(cases), 2),
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for case in cases:
        exact = "match" if case["exact"]["match"] else ("outcome only" if case["exact"]["outcome_match"] else "wrong outcome")
        print(f"{case['judge']['total_0_100']:6.1f}  {exact:13}  {case['prediction']}")
    print(f"-> {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "report"))
    parser.add_argument("--experiment", type=Path, default=ROOT / "work" / "exp-envscaler")
    parser.add_argument("--env", default="env_151_rl")
    parser.add_argument("--judged", type=Path, help="judged file (default: <env>/awb/official-judged-judge-calibration.jsonl)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.command == "build":
        return build(args.experiment, args.env)
    return report(args.experiment, args.env, args.judged, args.output or args.experiment / "judge_calibration.json")


if __name__ == "__main__":
    raise SystemExit(main())
