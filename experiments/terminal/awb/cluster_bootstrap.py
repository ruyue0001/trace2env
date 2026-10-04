#!/usr/bin/env python3
"""Trajectory-clustered inference for the main terminal comparisons (existing judged predictions only).

Rows of the same AgentWorldBench trajectory share a prefix and are not independent, so the resampling unit is the
trajectory (76 in the terminal split; 2–5 judged rows each). For each comparison A vs B:

* estimand   = mean over rows of (score_A − score_B), 0–100 scale (identical to the row-level number in the reports),
               plus the unweighted mean over trajectories of the per-trajectory mean difference;
* bootstrap  = resample trajectories with replacement (with a fixed seed), recompute both estimands, percentile 95 % CI;
* sign-flip  = paired cluster permutation test: flip the sign of every row difference of a trajectory jointly, two-sided
               Monte-Carlo p-value for the row-mean estimand;
* W / T / L  = trajectories whose mean difference is > 0 / = 0 / < 0.

    python cluster_bootstrap.py --resamples 10000 --flips 20000 --seed 20260921 --out report-cluster-bootstrap.json

Rows failed in either system are dropped for that comparison. The subset "non-construction" removes the 10 trajectories
whose task is a construction task of v1_20_r=1 (construction_task_split.py); "clean-36" removes the 17 trajectories whose
task is in any of the 36 scaling-package tasks (work/exp-scaling/clean_scaling.py) and is used for v1_36 vs v1_20.
"""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "work/exp-scaling"))
from construction_task_split import CONSTRUCTION_TASK_TRAJECTORIES  # noqa: E402
from clean_scaling import TASK_OF as SCALING_TASK_OF  # noqa: E402

FILES = {
    "prompting": HERE / "judged-prompting.jsonl",
    "prompting+rag": HERE / "judged-prompting+rag.jsonl",
    "harness_only_hv3": HERE / "judged-harness_only_hv3.jsonl",
    "schema_only_hv3": HERE / "judged-schema_only_hv3.jsonl",
    "examples_only": HERE / "judged-examples_only.jsonl",
    "trace2env_v3 (v1_20_r=1)": HERE / "judged-v3.jsonl",
    "workspace_single_shot_hv3": HERE / "judged-single_shot_hv3.jsonl",
    "trace2env_no_state_hv3": HERE / "judged-no_state_hv3.jsonl",
    "v1_36_r=1": ROOT / "work/exp-scaling/v1_36_r=1/awb/judged.jsonl",
}
COMPARISONS = [  # (A, B): statistic is A − B
    ("trace2env_v3 (v1_20_r=1)", "prompting"),
    ("trace2env_v3 (v1_20_r=1)", "prompting+rag"),
    ("harness_only_hv3", "prompting"),
    ("schema_only_hv3", "harness_only_hv3"),
    ("examples_only", "schema_only_hv3"),
    ("trace2env_v3 (v1_20_r=1)", "examples_only"),
    ("workspace_single_shot_hv3", "trace2env_v3 (v1_20_r=1)"),
    ("trace2env_no_state_hv3", "trace2env_v3 (v1_20_r=1)"),
    ("v1_36_r=1", "trace2env_v3 (v1_20_r=1)"),
]
SUBSETS = {
    "all": lambda tid: True,
    "non-construction": lambda tid: tid not in CONSTRUCTION_TASK_TRAJECTORIES,
    "clean-36": lambda tid: tid not in SCALING_TASK_OF,
}


def load(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if not r.get("failed"):
                out[(r["id"], r["turn_idx"])] = (r["total_score"] - 1) / 4 * 100
    return out


def analyse(a, b, keep, rng, resamples, flips):
    diffs = defaultdict(list)
    for k in a:
        if k in b and keep(k[0]):
            diffs[k[0]].append(a[k] - b[k])
    tids = sorted(diffs)
    sums = [sum(diffs[t]) for t in tids]
    counts = [len(diffs[t]) for t in tids]
    tmeans = [s / c for s, c in zip(sums, counts)]
    n_rows, n_traj = sum(counts), len(tids)
    obs_row = sum(sums) / n_rows
    obs_traj = mean(tmeans)
    boot_row, boot_traj = [], []
    for _ in range(resamples):
        idx = [rng.randrange(n_traj) for _ in range(n_traj)]
        boot_row.append(sum(sums[i] for i in idx) / sum(counts[i] for i in idx))
        boot_traj.append(sum(tmeans[i] for i in idx) / n_traj)
    boot_row.sort()
    boot_traj.sort()

    def ci(v):
        return (v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))])

    extreme = 0
    for _ in range(flips):
        stat = sum(s if rng.random() < 0.5 else -s for s in sums) / n_rows
        if abs(stat) >= abs(obs_row) - 1e-12:
            extreme += 1
    p = (extreme + 1) / (flips + 1)
    return {
        "rows": n_rows, "trajectories": n_traj,
        "mean_diff_rows": round(obs_row, 3), "ci95_rows": [round(x, 3) for x in ci(boot_row)],
        "mean_diff_trajectory_avg": round(obs_traj, 3), "ci95_trajectory_avg": [round(x, 3) for x in ci(boot_traj)],
        "trajectory_wins": sum(1 for m in tmeans if m > 1e-9), "trajectory_ties": sum(1 for m in tmeans if abs(m) <= 1e-9),
        "trajectory_losses": sum(1 for m in tmeans if m < -1e-9),
        "sign_flip_p_two_sided": round(p, 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--flips", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", default=str(HERE / "report-cluster-bootstrap.json"))
    args = ap.parse_args()
    systems = {name: load(path) for name, path in FILES.items()}
    report = {"method": "paired bootstrap over trajectories (percentile 95% CI) and cluster sign-flip test; estimand = row-mean "
                        "difference (A − B, 0–100) and trajectory-averaged difference", "seed": args.seed,
              "resamples": args.resamples, "sign_flips": args.flips, "python": sys.version.split()[0],
              "files": {k: str(v.relative_to(ROOT)) for k, v in FILES.items()}, "results": []}
    for subset, keep in SUBSETS.items():
        for a, b in COMPARISONS:
            if subset == "clean-36" and a != "v1_36_r=1":
                continue
            rng = random.Random(f"{args.seed}|{subset}|{a}|{b}")  # one recorded seed, one independent stream per analysis
            res = analyse(systems[a], systems[b], keep, rng, args.resamples, args.flips)
            res.update({"subset": subset, "A": a, "B": b})
            report["results"].append(res)
    json.dump(report, open(args.out, "w"), indent=1)
    for subset in SUBSETS:
        rows = [r for r in report["results"] if r["subset"] == subset]
        if not rows:
            continue
        print(f"\n### subset = {subset}")
        print("| A − B | rows | traj | mean diff (rows) | 95% CI (traj bootstrap) | traj-avg diff [CI] | traj W/T/L | sign-flip p |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['A']} − {r['B']} | {r['rows']} | {r['trajectories']} | {r['mean_diff_rows']:+.2f} | [{r['ci95_rows'][0]:+.2f}, {r['ci95_rows'][1]:+.2f}] | "
                  f"{r['mean_diff_trajectory_avg']:+.2f} [{r['ci95_trajectory_avg'][0]:+.2f}, {r['ci95_trajectory_avg'][1]:+.2f}] | "
                  f"{r['trajectory_wins']} / {r['trajectory_ties']} / {r['trajectory_losses']} | {r['sign_flip_p_two_sided']:.4f} |")
    print("\nwritten:", args.out)


if __name__ == "__main__":
    main()
