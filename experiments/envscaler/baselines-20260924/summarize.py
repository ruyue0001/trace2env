"""Summarize the frozen 3 x 2 x 2 EnvScaler exact-score matrix."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
plan = json.loads((root / "plan.json").read_text())
runs = []
aggregate = defaultdict(lambda: {"rows": 0, "exact": 0, "strict": 0, "outcome": 0, "parsed": 0, "environments": []})
for env in plan["environment_order"]:
    for backbone in plan["backbone_order"]:
        for baseline, prefix in (("envpack_prompting", "envpack-prompting"), ("harness_only_v51", "harness-only-v51")):
            label = f"{prefix}-{backbone}"
            directory = root / "rows" / env
            exact_path = directory / f"exact-{label}.json"
            scored_path = directory / f"exact-{label}.jsonl"
            run_path = directory / f"run-{label}.json"
            if not all(path.is_file() for path in (exact_path, scored_path, run_path)):
                runs.append({"environment": env, "backbone": backbone, "baseline": baseline, "status": "pending"})
                continue
            summary = json.loads(exact_path.read_text())["all"]
            run = json.loads(run_path.read_text())
            scored = [json.loads(line) for line in scored_path.read_text().splitlines()]
            expected = plan["environments"][env]["rows"]
            if summary["rows"] != expected or run["rows"] != expected or len(scored) != expected:
                raise ValueError(f"row count mismatch in {env}/{label}")
            counts = {key: sum(bool(row["exact"][field]) for row in scored)
                      for key, field in (("exact", "match"), ("strict", "match_strict"),
                                         ("outcome", "outcome_match"), ("parsed", "parsed"))}
            runs.append({"environment": env, "backbone": backbone, "baseline": baseline, "status": "complete",
                         "rows": expected, **counts, "exact_percent": round(100 * counts["exact"] / expected, 2)})
            total = aggregate[(backbone, baseline)]
            total["rows"] += expected
            for key, value in counts.items():
                total[key] += value
            total["environments"].append(env)
aggregates = [{"backbone": backbone, "baseline": baseline, **values,
               "complete": len(values["environments"]) == len(plan["environment_order"])}
              for (backbone, baseline), values in aggregate.items()]
(root / "summary.json").write_text(json.dumps({"runs": runs, "aggregates": aggregates}, indent=2) + "\n")
for row in runs:
    print(f"{row['environment']:11} {row['backbone']:18} {row['baseline']:19} "
          + (f"{row['exact']}/{row['rows']} ({row['exact_percent']:.2f}%)" if row['status'] == 'complete' else 'PENDING'))
