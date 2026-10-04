#!/usr/bin/env python3
"""Three-way comparison on identical AgentWorldBench rows: prompting baseline, legacy harness (clean run), v2 harness.

    python work/exp-v1_20_r=1/awb/compare_v2.py [--v2 awb/judged-v2.jsonl] [--out awb/report-v2-compare.json]

Paired statistics use only rows judged validly for every system. Scores are (raw-1)/4*100 per the official scorer.
"""
import argparse
import json
from collections import Counter, defaultdict
from statistics import mean, pstdev

A = "work/exp-v1_20_r=1/awb"
DIMS = ["format", "factuality", "consistency", "realism", "quality"]


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def norm(value):
    return (value - 1) / 4 * 100


def paired(rows_a, rows_b, keys):
    diffs = [rows_a[k]["total_score"] - rows_b[k]["total_score"] for k in keys]
    out = {"rows": len(keys), "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
           "losses": sum(d < 0 for d in diffs), "mean_diff_0_100": round(mean(diffs) / 4 * 100, 2) if diffs else None,
           "se_0_100": round(pstdev(diffs) / len(diffs) ** 0.5 / 4 * 100, 2) if len(diffs) > 1 else None,
           "per_dimension_0_100": {d: round(mean(rows_a[k][d] - rows_b[k][d] for k in keys) / 4 * 100, 2) for d in DIMS} if keys else {}}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", default=f"{A}/judged-v2.jsonl")
    parser.add_argument("--legacy", default=f"{A}/judged-agentic-clean.jsonl")
    parser.add_argument("--prompting", default=f"{A}/judged-prompting.jsonl")
    parser.add_argument("--out", default=f"{A}/report-v2-compare.json")
    args = parser.parse_args()
    systems = {"prompting": load(args.prompting), "legacy": load(args.legacy), "v2": load(args.v2)}
    overlap = json.load(open("work/pilot-tb2/validation/awb_overlap.json"))["trajectories"]
    keyed = {name: {(r["id"], r["turn_idx"]): r for r in rows} for name, rows in systems.items()}
    report = {"systems": {}, "paired": {}, "by_match_level": {}, "by_turn_position": {}, "per_trajectory": [], "failures": {}}

    for name, rows in systems.items():
        valid = [r for r in rows if not r.get("failed") and r.get("total_score") is not None]
        t = [r.get("trace2env") or {} for r in rows]
        entry = {"rows": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
                 "scores_0_100": {d: round(norm(mean(r[d] for r in valid)), 2) for d in DIMS},
                 "total_0_100": round(norm(mean(r["total_score"] for r in valid)), 2),
                 "routes": dict(Counter(x.get("route") for x in t)),
                 "latency_s_per_row": round(mean(x.get("latency_seconds") or 0 for x in t), 1)}
        if name != "prompting":
            entry["tool_calls_per_row"] = round(mean(len(x.get("tool_calls") or []) for x in t), 2)
            entry["tool_usage"] = dict(Counter(c for x in t for c in (x.get("tool_calls") or []) if isinstance(c, str)).most_common(10))
            entry["citations_per_row"] = round(mean(len(x.get("citations") or []) for x in t), 2)
            entry["features"] = dict(Counter(",".join(sorted(x.get("features") or [])) for x in t))
            by_traj = {}
            for r in rows:
                st = (r.get("trace2env") or {}).get("state_tracking") or {}
                if st and (r["id"] not in by_traj or r["turn_idx"] > by_traj[r["id"]][0]):
                    by_traj[r["id"]] = (r["turn_idx"], st)
            agg = Counter()
            for _, st in by_traj.values():
                for k, v in st.items():
                    if isinstance(v, (int, float)):
                        agg[k] += v
            entry["state_tracking_last_row_per_trajectory"] = dict(agg)
        report["systems"][name] = entry

    keys = [k for k in keyed["v2"] if k in keyed["prompting"] and k in keyed["legacy"]
            and not any(keyed[s][k].get("failed") or keyed[s][k].get("total_score") is None for s in systems)]
    report["paired"] = {"v2_vs_prompting": paired(keyed["v2"], keyed["prompting"], keys),
                        "v2_vs_legacy": paired(keyed["v2"], keyed["legacy"], keys),
                        "legacy_vs_prompting": paired(keyed["legacy"], keyed["prompting"], keys)}

    groups = defaultdict(list)
    for k in keys:
        groups[str(overlap.get(str(k[0]), {}).get("level") or "none")].append(k)
    for level, ks in sorted(groups.items()):
        report["by_match_level"][level] = {
            "rows": len(ks), "trajectories": len({k[0] for k in ks}),
            **{name: round(norm(mean(keyed[name][k]["total_score"] for k in ks)), 2) for name in systems},
            "v2_vs_prompting": paired(keyed["v2"], keyed["prompting"], ks),
            "v2_vs_legacy": paired(keyed["v2"], keyed["legacy"], ks)}

    def position(k):
        turn = k[1]
        return "turn 1" if turn == 1 else "turns 2-10" if turn <= 10 else "turns 11-30" if turn <= 30 else "turns 31+"
    pos = defaultdict(list)
    for k in keys:
        pos[position(k)].append(k)
    for name_pos, ks in sorted(pos.items(), key=lambda kv: min(k[1] for k in kv[1])):
        report["by_turn_position"][name_pos] = {
            "rows": len(ks), **{name: round(norm(mean(keyed[name][k]["total_score"] for k in ks)), 2) for name in systems},
            "v2_vs_prompting": paired(keyed["v2"], keyed["prompting"], ks)}

    per = defaultdict(lambda: defaultdict(list))
    for k in keys:
        for name in systems:
            per[k[0]][name].append(keyed[name][k]["total_score"])
    for tid, v in sorted(per.items(), key=lambda kv: mean(kv[1]["v2"]) - mean(kv[1]["prompting"]), reverse=True):
        ov = overlap.get(str(tid), {})
        report["per_trajectory"].append({
            "trajectory": tid, "rows": len(v["v2"]), "tb2_task_match": ov.get("tb2_task"), "match_level": ov.get("level"),
            **{name: round(norm(mean(v[name])), 1) for name in systems},
            "v2_minus_prompting": round((mean(v["v2"]) - mean(v["prompting"])) / 4 * 100, 1),
            "v2_minus_legacy": round((mean(v["v2"]) - mean(v["legacy"])) / 4 * 100, 1)})

    report["failures"] = {name: [{"id": r["id"], "turn": r["turn_idx"],
                                  "error": r.get("error_message") or (r.get("trace2env") or {}).get("harness_error")}
                                 for r in rows if r.get("failed")] for name, rows in systems.items()}
    json.dump(report, open(args.out, "w"), indent=1, default=str)

    for name, e in report["systems"].items():
        print(f"{name:<10} rows={e['rows']} valid={e['valid']} failed={e['failed']} total={e['total_0_100']} {e['scores_0_100']}")
    for name, p in report["paired"].items():
        print(f"{name}: {p['wins']}/{p['ties']}/{p['losses']} mean {p['mean_diff_0_100']} ± {p['se_0_100']} dims {p['per_dimension_0_100']}")
    for level, g in report["by_match_level"].items():
        print(f"match={level:<9} rows={g['rows']:>3} traj={g['trajectories']:>2} prompting={g['prompting']} legacy={g['legacy']} v2={g['v2']} "
              f"v2-prompting={g['v2_vs_prompting']['mean_diff_0_100']} ± {g['v2_vs_prompting']['se_0_100']}")
    for name_pos, g in report["by_turn_position"].items():
        print(f"{name_pos:<11} rows={g['rows']:>3} prompting={g['prompting']} legacy={g['legacy']} v2={g['v2']} "
              f"v2-prompting={g['v2_vs_prompting']['mean_diff_0_100']} ± {g['v2_vs_prompting']['se_0_100']}")
    won = sum(1 for t in report["per_trajectory"] if t["v2_minus_prompting"] > 0)
    lost = sum(1 for t in report["per_trajectory"] if t["v2_minus_prompting"] < 0)
    print(f"trajectories v2>prompting: {won}, v2<prompting: {lost}, of {len(report['per_trajectory'])}")
    print("report written to", args.out)


if __name__ == "__main__":
    main()
