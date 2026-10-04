#!/usr/bin/env python3
"""Overlap between construction (reserve/) and evaluation (benchmark/) tasks of each EnvScaler environment.

    python work/exp-envscaler/check_overlap.py --data work/envscaler --output work/exp-envscaler/overlap.json

Tasks of one environment share its tools by design; what must not be shared is the task itself. For every
(benchmark task, reserve task) pair this reports the token Jaccard of the two instructions, the Jaccard of the
record ids in the two initial databases, the share of the benchmark task's exact (tool, arguments) calls that
the reserve task also made, and whether the two tool-name sequences are identical. A pair is flagged when the
instructions overlap by 0.6 or more, the databases are equal, or the tool sequences are identical. Scenarios
reuse id conventions (``P001``, ``ING1``) with different record contents, so id and call overlap are reported
but not flagged: they are a retrieval hazard (right-looking records from another database), not a leak.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.envscaler import database, iter_rollouts, rollout_turns  # noqa: E402

INSTRUCTION_THRESHOLD = 0.6


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def tasks(directory: Path, instructions: dict[str, str]) -> dict[str, dict]:
    out = {}
    if not directory.is_dir():  # an environment may have no reserve rollouts
        return out
    for _, rollout in iter_rollouts([directory]):
        turns = rollout_turns(rollout)
        state = database(rollout.get("init_state")) or {}
        out[rollout["task_id"]] = {
            "instruction": tokens(instructions.get(rollout["task_id"], "")),
            "ids": {key for records in state.values() if isinstance(records, dict) for key in records},
            "calls": {(turn["name"], json.dumps(turn["action"].arguments, sort_keys=True)) for turn in turns},
            "sequence": [turn["name"] for turn in turns],
            "database": json.dumps(state, sort_keys=True),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="work/envscaler")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = Path(args.data)
    report = {"threshold": {"instruction_jaccard": INSTRUCTION_THRESHOLD}, "environments": {}}
    for benchmark_dir in sorted((data / "benchmark").iterdir()):
        if not benchmark_dir.is_dir():
            continue
        env = benchmark_dir.name.split("__")[0]
        scenarios = json.loads((data / "env_defs" / f"{env}_scenarios.json").read_text(encoding="utf-8"))
        instructions = {item["task_id"]: item.get("task", "") for item in scenarios}
        benchmark, reserve = tasks(benchmark_dir, instructions), tasks(data / "reserve" / env, instructions)
        pairs = []
        for (b_id, b), (r_id, r) in itertools.product(benchmark.items(), reserve.items()):
            pair = {"benchmark": b_id, "reserve": r_id, "instruction_jaccard": round(jaccard(b["instruction"], r["instruction"]), 3),
                    "record_id_jaccard": round(jaccard(b["ids"], r["ids"]), 3),
                    "shared_call_share": round(len(b["calls"] & r["calls"]) / max(1, len(b["calls"])), 3),
                    "same_tool_sequence": b["sequence"] == r["sequence"], "same_database": b["database"] == r["database"]}
            pair["flagged"] = (pair["instruction_jaccard"] >= INSTRUCTION_THRESHOLD or pair["same_tool_sequence"] or pair["same_database"])
            pairs.append(pair)
        report["environments"][env] = {
            "benchmark_tasks": len(benchmark), "reserve_tasks": len(reserve), "shared_task_ids": sorted(set(benchmark) & set(reserve)),
            # No pairs when an environment has no reserve rollouts: nothing can overlap.
            "max_instruction_jaccard": max((p["instruction_jaccard"] for p in pairs), default=0.0),
            "max_record_id_jaccard": max((p["record_id_jaccard"] for p in pairs), default=0.0),
            "max_shared_call_share": max((p["shared_call_share"] for p in pairs), default=0.0),
            "flagged": [p for p in pairs if p["flagged"]],
            "closest": sorted(pairs, key=lambda p: -p["instruction_jaccard"])[:3],
        }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {env: {"flagged": len(item["flagged"]), "shared_task_ids": len(item["shared_task_ids"]),
                     "max_instruction_jaccard": item["max_instruction_jaccard"]} for env, item in report["environments"].items()}
    print(json.dumps(summary))
    if any(item["flagged"] or item["shared_task_ids"] for item in report["environments"].values()):
        sys.exit("construction and evaluation tasks overlap; see the flagged pairs in " + args.output)


if __name__ == "__main__":
    main()
