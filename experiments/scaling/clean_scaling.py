#!/usr/bin/env python3
"""Trace-scaling curve with construction-task benchmark trajectories removed.

Each scaling package is built from a prefix of the seeded task order. Benchmark trajectories that are the same TB2 task
as one of a package's construction traces (confirmed by hand from shared task files; see
work/exp-v1_20_r=1/awb/overlap_audit.py) are removed for that package, and a common clean set (rows of trajectories
matching none of the 36 tasks) is used for the comparable curve.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
V120 = ROOT / "work/exp-v1_20_r=1"
TASK_OF = {  # benchmark trajectory id -> construction task it is an instance of
    104971650552388: "write-compressor", 210448203382431: "largest-eigenval", 177623154441707: "dna-assembly",
    126048951621424: "winning-avg-corewars", 272823365904851: "git-multibranch", 30043326384016: "hf-model-inference",
    208104038662149: "hf-model-inference", 255969445734706: "prove-plus-comm", 77632686965927: "prove-plus-comm",
    111876641555338: "distribution-search",
    248745653937994: "schemelike-metacircular-eval", 277725593326837: "fix-git", 234242807083239: "path-tracing-reverse",
    121730716066482: "path-tracing-reverse", 234090921269454: "nginx-request-logging", 9428045888654: "crack-7z-hash",
    19240998661436: "log-summary-date-ranges",
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(v):
    return (v - 1) / 4 * 100


def main():
    manifest = json.load(open(Path(__file__).with_name("manifest.json")))
    order = manifest["order_of_the_20"] + manifest["order_of_the_16_extra"]
    prompting = {(r["id"], r["turn_idx"]): r for r in load(V120 / "awb/judged-prompting.jsonl")}
    union_clean = {k for k in prompting if k[0] not in TASK_OF}
    print(f"{'package':12s}{'tasks':>6}{'own-contam traj/rows':>22}{'all rows':>10}{'own-clean':>11}{'union-clean':>13}{'prompting(union)':>18}{'diff ± se':>16}")
    for name, n in [("v1_1_r=1", 1), ("v1_1s2_r=1", None), ("v1_5_r=1", 5), ("v1_10_r=1", 10), ("v1_15_r=1", 15), ("v1_20_r=1", 20), ("v1_25_r=1", 25), ("v1_30_r=1", 30), ("v1_36_r=1", 36)]:
        tasks = set(order[:n]) if n else set(manifest["order_of_the_20_seed2"][:1])
        path = V120 / "awb/judged-v3.jsonl" if name == "v1_20_r=1" else Path(__file__).with_name(name) / "awb/judged.jsonl"
        rows = {(r["id"], r["turn_idx"]): r for r in load(path) if not r.get("failed")}
        contam = {k for k in rows if TASK_OF.get(k[0]) in tasks}
        own_clean = [k for k in rows if k not in contam]
        uc = [k for k in rows if k in union_clean]
        d = [norm(rows[k]["total_score"]) - norm(prompting[k]["total_score"]) for k in uc if k in prompting and not prompting[k].get("failed")]
        print(f"{name:12s}{len(tasks):>6}{len({k[0] for k in contam}):>12}/{len(contam):<9}{mean(norm(rows[k]['total_score']) for k in rows):>10.2f}"
              f"{mean(norm(rows[k]['total_score']) for k in own_clean):>11.2f}{mean(norm(rows[k]['total_score']) for k in uc):>13.2f}"
              f"{mean(norm(prompting[k]['total_score']) for k in uc):>18.2f}{mean(d):>+10.2f} ± {pstdev(d) / len(d) ** 0.5:.2f}")
    print(f"union-clean rows: {len(union_clean)} of {len(prompting)} ({len({k[0] for k in union_clean})} trajectories); contaminated trajectories overall: {len(TASK_OF)}")


if __name__ == "__main__":
    main()
