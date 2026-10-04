#!/usr/bin/env python3
"""Exact match and the official judge on the same predictions, row by row.

    python work/exp-envscaler/compare_metrics.py env_151_rl prompting

Reads `<env>/awb/exact-<label>.jsonl` (`envscaler-score --scored`) and `<env>/awb/official-judged-<label>.jsonl`
(the original AgentWorldBench `eval.py judge`), joins them by trajectory and turn, and writes
`<env>/awb/metrics-<label>.json`. The reported group is `reads:excluded`, as in `envscaler-score`: the rows whose
true response is not `data` (a read that succeeds is answered from the state the prompt shows). For that group
and, for reference, for all rows and per response kind: both metrics, the judge's scores grouped by the exact
verdict, every exact miss with the judge's score, and every exact match the judge marked down. No model call.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality", "total_score")
VERDICTS = ("match", "outcome only", "wrong outcome")


def load(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percent(raw: float) -> float:
    return (raw - 1) / 4 * 100


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float]:
    if not total:
        return [0.0, 0.0]
    share = successes / total
    centre = share + z * z / (2 * total)
    spread = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total))
    return [round(100 * (centre - spread) / (1 + z * z / total), 1), round(100 * (centre + spread) / (1 + z * z / total), 1)]


def verdict_of(exact: dict[str, Any]) -> str:
    return "match" if exact["match"] else ("outcome only" if exact["outcome_match"] else "wrong outcome")


def judge_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    valid = [row for row in rows if not row["judge"].get("failed")]
    summary: dict[str, Any] = {"rows": len(rows), "valid": len(valid)}
    for name in DIMENSIONS:
        values = [percent(row["judge"][name]) for row in valid]
        summary[name] = round(sum(values) / len(values), 2) if values else None
    totals = sorted(round(percent(row["judge"]["total_score"]), 1) for row in valid)
    summary["total_min"], summary["total_max"] = (totals[0], totals[-1]) if totals else (None, None)
    summary["rows_at_100"] = sum(1 for total in totals if total == 100.0)
    return summary


def exact_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    matched = sum(1 for row in rows if row["exact"]["match"])
    outcome = sum(1 for row in rows if row["exact"]["outcome_match"])
    return {"rows": len(rows), "match": matched, "match_percent": round(100 * matched / len(rows), 2) if rows else None,
            "match_wilson_95": wilson(matched, len(rows)), "outcome_match": outcome,
            "outcome_match_percent": round(100 * outcome / len(rows), 2) if rows else None}


def action_of(row: dict[str, Any]) -> Any:
    text = row["current_prompt"].split("**Action:**", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except ValueError:
        return text.strip()


def listed(row: dict[str, Any]) -> dict[str, Any]:
    judge = row["judge"]
    return {"row": f"{row['id']} turn {row['turn_idx']}", "kind": row["exact"]["kind"], "exact": verdict_of(row["exact"]),
            "action": action_of(row), "truth": row["response"][-1].split("\n", 1)[-1],
            "predicted": judge.get("extracted_output"),
            "judge": {**{name: judge[name] for name in DIMENSIONS}, "total_0_100": round(percent(judge["total_score"]), 1)},
            "judge_weaknesses": judge.get("weaknesses")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("env")
    parser.add_argument("label")
    parser.add_argument("--experiment", type=Path, default=ROOT / "work" / "exp-envscaler")
    args = parser.parse_args()
    directory = args.experiment / args.env / "awb"
    scored = load(directory / f"exact-{args.label}.jsonl")
    judged = {(row["id"], row["turn_idx"]): row for row in load(directory / f"official-judged-{args.label}.jsonl")}
    if len(judged) != len(scored) or any((row["id"], row["turn_idx"]) not in judged for row in scored):
        sys.exit(f"the scored and the judged file do not hold the same rows ({len(scored)} vs {len(judged)})")
    for row in scored:
        other = judged[(row["id"], row["turn_idx"])]
        if other["gen"] != row["gen"]:
            sys.exit(f"{row['id']} turn {row['turn_idx']}: the judged prediction is not the scored prediction")
        row["judge"] = {key: other.get(key) for key in (*DIMENSIONS, "failed", "weaknesses", "extracted_output")}
    kinds = sorted({row["exact"]["kind"] for row in scored})
    without_reads = [row for row in scored if row["exact"]["kind"] != "data"]
    result = {
        "environment": args.env, "label": args.label,
        "exact": {"all": exact_summary(scored), "reads:excluded": exact_summary(without_reads),
                  **{f"kind:{kind}": exact_summary(r for r in scored if r["exact"]["kind"] == kind) for kind in kinds}},
        "official_judge_0_100": {"all": judge_summary(scored), "reads:excluded": judge_summary(without_reads),
                                 **{f"kind:{kind}": judge_summary(r for r in scored if r["exact"]["kind"] == kind) for kind in kinds},
                                 **{f"reads:excluded|exact:{verdict}": judge_summary(r for r in without_reads
                                                                                     if verdict_of(r["exact"]) == verdict)
                                    for verdict in VERDICTS}},
        "exact_misses": [listed(row) for row in without_reads if not row["exact"]["match"]],
        "exact_matches_the_judge_marked_down": [listed(row) for row in without_reads
                                                if row["exact"]["match"] and not row["judge"].get("failed")
                                                and row["judge"]["total_score"] < 5],
        "exact_misses_among_reads": [listed(row) for row in scored if row["exact"]["kind"] == "data" and not row["exact"]["match"]],
    }
    output = directory / f"metrics-{args.label}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    exact, judge = result["exact"]["reads:excluded"], result["official_judge_0_100"]
    print(f"reads excluded ({exact['rows']} of {len(scored)} rows)")
    print(f"exact match {exact['match']}/{exact['rows']} = {exact['match_percent']} (95% {exact['match_wilson_95']}), "
          f"outcome {exact['outcome_match_percent']}")
    print("official judge  " + "  ".join(f"{name} {judge['reads:excluded'][name]}" for name in DIMENSIONS))
    for verdict in VERDICTS:
        group = judge[f"reads:excluded|exact:{verdict}"]
        print(f"  judge total where exact says {verdict:13}: {group['total_score']} over {group['valid']} rows "
              f"(min {group['total_min']}, max {group['total_max']}, at 100: {group['rows_at_100']})")
    print(f"-> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
