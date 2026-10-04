#!/usr/bin/env python3
"""Compare any number of judged prediction sets on identical AgentWorldBench rows.

    python work/exp-v1_20_r=1/awb/compare_systems.py --system prompting=awb/judged-prompting.jsonl \
        --system v2=awb/judged-v2.jsonl --system schema_only=awb/judged-schema_only.jsonl ... \
        --baseline prompting --reference v2 --out awb/report-ablations.json

For every system: the five judged dimensions and total (official (raw-1)/4*100 scaling), valid/failed rows,
routes, latency, tool calls, and — from the run's call logs (calls-<label>-shard*.jsonl) — model calls, tokens,
and cost at OpenRouter's list prices. Paired statistics (wins/ties/losses, mean ± se) against the baseline and
the reference on the rows valid for every system; breakdowns by benchmark-overlap level, turn position, and
action kind; a per-trajectory table; and the failures.
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from statistics import mean, pstdev

A = "work/exp-v1_20_r=1/awb"
DIMS = ["format", "factuality", "consistency", "realism", "quality"]
# OpenRouter list prices (USD per token) on 2026-09-19 for openai/gpt-5.6-sol.
PRICE = {"prompt": 2e-6, "completion": 1e-5, "cached": 2e-7}


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def norm(value):
    return (value - 1) / 4 * 100


def paired(rows_a, rows_b, keys):
    diffs = [rows_a[k]["total_score"] - rows_b[k]["total_score"] for k in keys]
    return {"rows": len(keys), "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs),
            "mean_diff_0_100": round(mean(diffs) / 4 * 100, 2) if diffs else None,
            "se_0_100": round(pstdev(diffs) / len(diffs) ** 0.5 / 4 * 100, 2) if len(diffs) > 1 else None,
            "per_dimension_0_100": {d: round(mean(rows_a[k][d] - rows_b[k][d] for k in keys) / 4 * 100, 2) for d in DIMS} if keys else {}}


def action_kind(row):
    match = re.search(r"```json\s*(\[.*?\])\s*```", row.get("current_prompt", ""), re.S)
    try:
        actions = json.loads(match.group(1))
    except Exception:  # noqa: BLE001
        return "unparsed"
    keys = [a.get("keystrokes", "") for a in actions]
    if all(k == "" for k in keys):
        return "wait"
    if any("<<" in k for k in keys):
        return "heredoc"
    if any(re.fullmatch(r"\s*(C-[a-z]|Escape|Enter|Up|Down)\s*", k) for k in keys):
        return "keys"
    if sum(k.count("\n") for k in keys) > 1 or len(keys) > 1:
        return "batch"
    return "single"


def call_log_stats(label):
    calls = []
    for path in sorted(glob.glob(f"{A}/calls-{label}-shard*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                calls.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not calls:
        return None
    per_role = defaultdict(lambda: Counter())
    for call in calls:
        usage = call.get("usage") or {}
        role = call.get("role") or "?"
        per_role[role]["calls"] += 1
        per_role[role]["cache_hits"] += bool(call.get("cache_hit"))
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            per_role[role][key] += int(usage.get(key) or 0)
        per_role[role]["seconds"] += float(call.get("elapsed_s") or 0)
    out = {}
    total = Counter()
    for role, counter in per_role.items():
        cached = counter["cached_tokens"]
        cost = (counter["prompt_tokens"] - cached) * PRICE["prompt"] + cached * PRICE["cached"] + counter["completion_tokens"] * PRICE["completion"]
        out[role] = {**{k: int(v) if k != "seconds" else round(v, 1) for k, v in counter.items()}, "cost_usd": round(cost, 2)}
        for key in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens"):
            total[key] += counter[key]
        total["cost_usd"] += cost
    out["total"] = {k: (round(v, 2) if k == "cost_usd" else int(v)) for k, v in total.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, metavar="LABEL=JUDGED.jsonl")
    parser.add_argument("--baseline", default="prompting")
    parser.add_argument("--reference", default="v2")
    parser.add_argument("--out", default=f"{A}/report-ablations.json")
    args = parser.parse_args()
    systems = {}
    for item in args.system:
        label, _, path = item.partition("=")
        systems[label] = load(path)
    overlap = json.load(open("work/pilot-tb2/validation/awb_overlap.json"))["trajectories"]
    keyed = {name: {(r["id"], r["turn_idx"]): r for r in rows} for name, rows in systems.items()}
    report = {"systems": {}, "paired": {}, "by_match_level": {}, "by_turn_position": {}, "by_action_kind": {},
              "per_trajectory": [], "failures": {}}

    for name, rows in systems.items():
        valid = [r for r in rows if not r.get("failed") and r.get("total_score") is not None]
        t = [r.get("trace2env") or {} for r in rows]
        entry = {"rows": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
                 "scores_0_100": {d: round(norm(mean(r[d] for r in valid)), 2) for d in DIMS} if valid else {},
                 "total_0_100": round(norm(mean(r["total_score"] for r in valid)), 2) if valid else None,
                 "routes": dict(Counter(x.get("route") for x in t)),
                 "latency_s_per_row": round(mean(x.get("latency_seconds") or 0 for x in t), 1),
                 "latency_s_median": round(sorted(x.get("latency_seconds") or 0 for x in t)[len(t) // 2], 1)}
        if any(x.get("mode") == "agentic" for x in t):
            entry["tool_calls_per_row"] = round(mean(len(x.get("tool_calls") or []) for x in t), 2)
            entry["tool_usage"] = dict(Counter(c for x in t for c in (x.get("tool_calls") or []) if isinstance(c, str)).most_common(12))
            entry["citations_per_row"] = round(mean(len(x.get("citations") or []) for x in t), 2)
            entry["prediction_mode"] = dict(Counter(x.get("prediction_mode") or "agent" for x in t))
            entry["harness_errors"] = dict(Counter(str(x.get("harness_error"))[:80] for x in t if x.get("harness_error")).most_common(6))
            usage = Counter()
            for x in t:
                for k, v in (x.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] += v
            entry["agent_usage_from_rows"] = dict(usage)
        stats = call_log_stats(name)
        if stats:
            entry["call_log"] = stats
            entry["cost_usd_per_row"] = round(stats["total"]["cost_usd"] / len(rows), 3)
            entry["model_calls_per_row"] = round(stats["total"]["calls"] / len(rows), 2)
        elif any(isinstance(x.get("usage"), dict) and x["usage"] for x in t):
            # Chat-mode rows (prompting, prompting_rag) carry the provider's usage per call, including OpenRouter's cost.
            usage = Counter()
            for x in t:
                for k, v in (x.get("usage") or {}).items():
                    if isinstance(v, (int, float)):
                        usage[k] += v
            rows_with = sum(1 for x in t if x.get("usage"))
            listed = ((usage["prompt_tokens"] - usage.get("cached_tokens", 0)) * PRICE["prompt"] + usage.get("cached_tokens", 0) * PRICE["cached"]
                      + usage["completion_tokens"] * PRICE["completion"])
            entry["row_usage"] = {**{k: int(v) for k, v in usage.items() if k != "cost_usd"}, "rows_with_usage": rows_with,
                                  "cost_usd_reported": round(usage.get("cost_usd", 0.0), 2), "cost_usd_list_price": round(listed, 2)}
            entry["cost_usd_per_row"] = round((usage.get("cost_usd") or listed) / len(rows), 3)
            entry["model_calls_per_row"] = round(rows_with / len(rows), 2)
            entry["prompt_chars_per_row"] = round(mean(x.get("prompt_chars") or 0 for x in t))
        report["systems"][name] = entry

    keys = [k for k in keyed[args.reference] if all(k in keyed[s] for s in systems)
            and not any(keyed[s][k].get("failed") or keyed[s][k].get("total_score") is None for s in systems)]
    for name in systems:
        if name in (args.baseline,):
            continue
        report["paired"][name] = {f"vs_{args.baseline}": paired(keyed[name], keyed[args.baseline], keys)}
        if name != args.reference and args.reference in systems:
            report["paired"][name][f"vs_{args.reference}"] = paired(keyed[name], keyed[args.reference], keys)

    def breakdown(grouper, target):
        groups = defaultdict(list)
        for k in keys:
            groups[grouper(k)].append(k)
        for label, ks in groups.items():
            target[label] = {"rows": len(ks), **{name: round(norm(mean(keyed[name][k]["total_score"] for k in ks)), 2) for name in systems},
                             **{f"{name}_vs_{args.baseline}": paired(keyed[name], keyed[args.baseline], ks)["mean_diff_0_100"]
                                for name in systems if name != args.baseline}}

    breakdown(lambda k: str(overlap.get(str(k[0]), {}).get("level") or "none"), report["by_match_level"])
    breakdown(lambda k: "turn 1" if k[1] == 1 else "turns 2-10" if k[1] <= 10 else "turns 11-30" if k[1] <= 30 else "turns 31+",
              report["by_turn_position"])
    breakdown(lambda k: action_kind(keyed[args.reference][k]), report["by_action_kind"])

    per = defaultdict(lambda: defaultdict(list))
    for k in keys:
        for name in systems:
            per[k[0]][name].append(keyed[name][k]["total_score"])
    for tid, v in sorted(per.items(), key=lambda kv: kv[0]):
        ov = overlap.get(str(tid), {})
        row = {"trajectory": tid, "rows": len(v[args.reference]), "tb2_task_match": ov.get("tb2_task"), "match_level": ov.get("level"),
               **{name: round(norm(mean(v[name])), 1) for name in systems}}
        report["per_trajectory"].append(row)
    report["failures"] = {name: [{"id": r["id"], "turn": r["turn_idx"], "error": r.get("error_message") or (r.get("trace2env") or {}).get("harness_error")}
                                 for r in rows if r.get("failed")] for name, rows in systems.items()}
    report["paired_rows"] = len(keys)
    json.dump(report, open(args.out, "w"), indent=1, default=str)

    print(f"{'system':<14}{'rows':>5}{'valid':>6}{'fail':>5}{'format':>8}{'fact':>7}{'cons':>7}{'real':>7}{'qual':>7}{'TOTAL':>8}{'s/row':>7}{'tools':>7}{'calls':>7}{'$/row':>7}")
    for name, e in report["systems"].items():
        s = e["scores_0_100"]
        print(f"{name:<14}{e['rows']:>5}{e['valid']:>6}{e['failed']:>5}{s.get('format', 0):>8.1f}{s.get('factuality', 0):>7.1f}{s.get('consistency', 0):>7.1f}"
              f"{s.get('realism', 0):>7.1f}{s.get('quality', 0):>7.1f}{e['total_0_100'] or 0:>8.2f}{e['latency_s_per_row']:>7.1f}"
              f"{e.get('tool_calls_per_row', 0):>7.2f}{e.get('model_calls_per_row', 0):>7.2f}{e.get('cost_usd_per_row', 0):>7.3f}")
    print(f"paired rows: {len(keys)}")
    for name, comparisons in report["paired"].items():
        for against, p in comparisons.items():
            print(f"  {name} {against}: {p['wins']}/{p['ties']}/{p['losses']} mean {p['mean_diff_0_100']:+.2f} ± {p['se_0_100']} dims {p['per_dimension_0_100']}")
    for title, table in (("match level", report["by_match_level"]), ("turn position", report["by_turn_position"]), ("action kind", report["by_action_kind"])):
        print(f"--- by {title}")
        for label, g in sorted(table.items(), key=lambda kv: -kv[1]["rows"]):
            print(f"  {label:<12} rows={g['rows']:>3} " + " ".join(f"{name}={g[name]:.1f}" for name in systems))
    won = {name: sum(1 for t in report["per_trajectory"] if t[name] > t[args.baseline]) for name in systems if name != args.baseline}
    lost = {name: sum(1 for t in report["per_trajectory"] if t[name] < t[args.baseline]) for name in systems if name != args.baseline}
    print(f"trajectories won/lost vs {args.baseline}: " + ", ".join(f"{n} {won[n]}/{lost[n]}" for n in won) + f" of {len(report['per_trajectory'])}")
    print("report written to", args.out)


if __name__ == "__main__":
    main()
