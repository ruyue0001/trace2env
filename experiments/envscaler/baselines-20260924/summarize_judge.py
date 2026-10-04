"""Summarize official GPT-5.2 judge results for one backbone's six prediction sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
backbone = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v41-flash"
if backbone not in {"deepseek-v41-flash", "gpt56sol"}:
    raise SystemExit(f"unknown backbone: {backbone}")
plan = json.loads((root / "plan.json").read_text())
runs = []
aggregates = {}
for env in plan["environment_order"]:
    directory = root / "rows" / env
    for baseline, label in (
        ("envpack_prompting", f"envpack-prompting-{backbone}"),
        ("harness_only_v51", f"harness-only-v51-{backbone}"),
    ):
        paths = {name: directory / f"official-{name}-{label}.{suffix}" for name, suffix in (
            ("judged", "jsonl"), ("score", "json"), ("judge-run", "json"))}
        if not all(path.is_file() for path in paths.values()):
            runs.append({"environment": env, "baseline": baseline, "status": "pending"})
            continue
        judged = [json.loads(line) for line in paths["judged"].read_text().splitlines()]
        score = json.loads(paths["score"].read_text())
        record = json.loads(paths["judge-run"].read_text())
        expected = plan["environments"][env]["rows"]
        if len(judged) != expected or score["total"] != expected or record["rows"] != expected:
            raise ValueError(f"Judge row count mismatch: {env}/{label}")
        valid = [row for row in judged if row.get("failed") == 0.0]
        if score["valid"] != len(valid) or score["failed"] != expected - len(valid):
            raise ValueError(f"Judge validity mismatch: {env}/{label}")
        if len(valid) != expected:
            runs.append({"environment": env, "baseline": baseline, "status": "incomplete",
                         "rows": expected, "valid": len(valid), "failed": expected - len(valid)})
            continue
        overall = (sum(float(row["total_score"]) for row in valid) / len(valid) - 1) * 25
        if abs(overall - score["overall"]) > 1e-6:
            raise ValueError(f"Judge score mismatch: {env}/{label}")
        runs.append({"environment": env, "baseline": baseline, "status": "complete",
                     "rows": expected, "overall": score["overall"], "judge_model": record["judge_model"]})
        entry = aggregates.setdefault(baseline, {"rows": 0, "total_score_sum": 0.0, "environments": []})
        entry["rows"] += expected
        entry["total_score_sum"] += sum(float(row["total_score"]) for row in valid)
        entry["environments"].append(env)
for baseline, entry in aggregates.items():
    entry["overall"] = (entry.pop("total_score_sum") / entry["rows"] - 1) * 25
    entry["complete"] = len(entry["environments"]) == len(plan["environment_order"])
output = root / ("judge-summary.json" if backbone == "deepseek-v41-flash" else f"judge-summary-{backbone}.json")
output.write_text(json.dumps({"backbone": backbone, "runs": runs, "aggregates": aggregates}, indent=2) + "\n")
for row in runs:
    print(f"{row['environment']:11} {row['baseline']:19} " +
          (f"{row['overall']:.2f} ({row['rows']} valid)" if row['status'] == 'complete' else row['status'].upper()))
