#!/usr/bin/env python3
"""Backbone transfer: the four terminal systems under a second backbone versus the same four under GPT-5.6-Sol.

    python work/exp-v1_20_r=1/backbones/compare_backbones.py --backbone deepseek-v4.1-flash --out .../report-backbones.json

Per backbone and system: judged dimensions and total (official scaling), valid / failed rows, routes, latency, tool
calls per row, model calls / tokens / cost from the run's call logs (structured roles) and from the rows' recorded
usage (free-text prompting roles), paired wins / ties / losses versus that backbone's prompting baseline, and the
trajectory-clustered bootstrap (same seed and procedure as awb/cluster_bootstrap.py). Then the prompting → Trace2Env
improvement of the two backbones side by side, absolute and relative.
"""
import argparse, glob, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[3]
AWB = ROOT / "work/exp-v1_20_r=1/awb"
BACKBONES = ROOT / "work/exp-v1_20_r=1/backbones"
sys.path.insert(0, str(AWB))
from cluster_bootstrap import analyse  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
PRICE = {  # OpenRouter list prices, USD per token, read from the models endpoint (GPT 2026-09-19, others 2026-09-21)
    "gpt-5.6-sol": {"prompt": 2e-6, "completion": 1e-5, "cached": 2e-7},
    "kimi-k2.6": {"prompt": 0.95e-6, "completion": 4e-6, "cached": 0.95e-6},
    "deepseek-v4.1-flash": {"prompt": 0.15e-6, "completion": 0.6e-6, "cached": 0.003e-6},
    "deepseek-v4-pro": {"prompt": 0.422e-6, "completion": 0.845e-6, "cached": 0.035e-6},
}
SYSTEMS = ["prompting", "prompting+rag", "harness_only_hv3", "v3"]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(v):
    return (v - 1) / 4 * 100


def call_logs(directory, label, price):
    per_role = defaultdict(Counter)
    for path in sorted(glob.glob(str(directory / f"calls-{label}-shard*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = c.get("usage") or {}
            r = per_role[c.get("role") or "?"]
            r["calls"] += 1
            r["cache_hits"] += bool(c.get("cache_hit"))
            r["repaired_attempts"] += len(c.get("attempts") or [])
            for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                r[k] += int(u.get(k) or 0)
            r["seconds"] += float(c.get("elapsed_s") or 0)
    out, total = {}, Counter()
    for role, r in per_role.items():
        cached = r["cached_tokens"]
        cost = (r["prompt_tokens"] - cached) * price["prompt"] + cached * price["cached"] + r["completion_tokens"] * price["completion"]
        out[role] = {**{k: (round(v, 1) if k == "seconds" else int(v)) for k, v in r.items()}, "cost_usd_list": round(cost, 2)}
        for k in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "repaired_attempts"):
            total[k] += r[k]
        total["cost_usd_list"] += cost
    out["total"] = {k: (round(v, 2) if k == "cost_usd_list" else int(v)) for k, v in total.items()}
    return out if per_role else None


def system_entry(directory, label, price, backbone):
    rows = load(directory / f"judged-{label}.jsonl")
    valid = [r for r in rows if not r.get("failed")]
    e = {"backbone": backbone, "system": label, "rows": len(rows), "valid_rows": len(valid), "failed_rows": len(rows) - len(valid),
         "error_rate": round((len(rows) - len(valid)) / len(rows), 4)}
    e.update({d: round(mean(norm(r[d]) for r in valid), 2) for d in DIMS})
    e["total"] = round(mean(norm(r["total_score"]) for r in valid), 2)
    infos = [r.get("trace2env") or {} for r in rows]
    e["routes"] = dict(Counter(i.get("route") or "n/a" for i in infos))
    e["latency_s_per_row"] = round(mean(i.get("latency_seconds") or 0 for i in infos), 1)
    e["tool_calls_per_row"] = round(mean(len(i.get("tool_calls") or []) for i in infos), 2)
    e["harness_errors"] = sum(1 for i in infos if i.get("harness_error"))
    logs = call_logs(directory, label, price)
    if logs:
        e["call_log"] = logs
        e["model_calls_per_row"] = round(logs["total"]["calls"] / len(rows), 2)
        e["tokens_per_row"] = {"prompt": round(logs["total"]["prompt_tokens"] / len(rows)), "completion": round(logs["total"]["completion_tokens"] / len(rows))}
        e["cost_usd_per_row_list"] = round(logs["total"]["cost_usd_list"] / len(rows), 4)
    usage = Counter()
    reported = 0.0
    for i in infos:
        u = i.get("usage") or {}
        for k in ("prompt_tokens", "completion_tokens"):
            usage[k] += int(u.get(k) or 0)
        reported += float(u.get("cost_usd") or 0)
    if usage["prompt_tokens"] and not logs:  # free-text roles record usage on the row
        e["model_calls_per_row"] = 1.0
        e["tokens_per_row"] = {"prompt": round(usage["prompt_tokens"] / len(rows)), "completion": round(usage["completion_tokens"] / len(rows))}
        e["cost_usd_per_row_list"] = round((usage["prompt_tokens"] * price["prompt"] + usage["completion_tokens"] * price["completion"]) / len(rows), 4)
        if reported:
            e["cost_usd_per_row_reported"] = round(reported / len(rows), 4)
    return e, {(r["id"], r["turn_idx"]): r for r in valid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="deepseek-v4.1-flash", help="slug of the directory under backbones/ (also the price key)")
    ap.add_argument("--price", default=None, help="price key when the directory is an archived run (e.g. deepseek-v4-pro for deepseek-v4-pro-run2-v3.1); default: the longest PRICE key that prefixes --backbone")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--flips", type=int, default=20000)
    args = ap.parse_args()
    backbone_dirs = {"gpt-5.6-sol": AWB, args.backbone: BACKBONES / args.backbone}
    price_key = args.price or max((k for k in PRICE if args.backbone.startswith(k)), key=len, default=args.backbone)
    PRICE[args.backbone] = PRICE[price_key]
    args.out = args.out or str(BACKBONES / args.backbone / "report-backbones.json")
    report = {"seed": args.seed, "resamples": args.resamples, "sign_flips": args.flips, "systems": {}, "paired": {}, "transfer": {}}
    scores = {}
    for backbone, directory in backbone_dirs.items():
        for label in SYSTEMS:
            if not (directory / f"judged-{label}.jsonl").exists():
                continue
            entry, rows = system_entry(directory, label, PRICE[backbone], backbone)
            report["systems"][f"{backbone}/{label}"] = entry
            scores[(backbone, label)] = {k: norm(r["total_score"]) for k, r in rows.items()}
            scores[(backbone, label, "dims")] = {k: {d: norm(r[d]) for d in DIMS} for k, r in rows.items()}
    for backbone in backbone_dirs:
        base = scores.get((backbone, "prompting"))
        if not base:
            continue
        for label in SYSTEMS[1:]:
            a = scores.get((backbone, label))
            if not a:
                continue
            keys = [k for k in a if k in base]
            d = [a[k] - base[k] for k in keys]
            rng = random.Random(f"{args.seed}|all|{backbone}/{label}|{backbone}/prompting")
            boot = analyse(a, base, lambda tid: True, rng, args.resamples, args.flips)
            report["paired"][f"{backbone}: {label} - prompting"] = {
                "rows": len(keys), "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d),
                "mean_diff": round(mean(d), 2), "se_rows": round(pstdev(d) / len(d) ** 0.5, 2),
                "per_dimension": {dim: round(mean(scores[(backbone, label, "dims")][k][dim] - scores[(backbone, "prompting", "dims")][k][dim] for k in keys), 2) for dim in DIMS},
                "trajectory_bootstrap": boot}
        # v3 across backbones on the same rows
    for label in SYSTEMS:
        g, k = scores.get(("gpt-5.6-sol", label)), scores.get((args.backbone, label))
        if g and k:
            keys = [x for x in k if x in g]
            d = [k[x] - g[x] for x in keys]
            rng = random.Random(f"{args.seed}|all|{args.backbone}/{label}|gpt/{label}")
            report["paired"][f"{label}: {args.backbone} - gpt-5.6-sol"] = {"rows": len(keys), "wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d),
                                                                     "mean_diff": round(mean(d), 2), "se_rows": round(pstdev(d) / len(d) ** 0.5, 2),
                                                                     "trajectory_bootstrap": analyse(k, g, lambda tid: True, rng, args.resamples, args.flips)}
    for backbone in backbone_dirs:
        p, v = report["systems"].get(f"{backbone}/prompting"), report["systems"].get(f"{backbone}/v3")
        if p and v:
            report["transfer"][backbone] = {"prompting": p["total"], "trace2env_v3": v["total"], "absolute_gain": round(v["total"] - p["total"], 2),
                                            "relative_gain_pct": round((v["total"] - p["total"]) / p["total"] * 100, 1),
                                            "per_dimension_gain": {d: round(v[d] - p[d], 2) for d in DIMS}}
    json.dump(report, open(args.out, "w"), indent=1)
    print("| backbone / system | valid | " + " | ".join(DIMS) + " | total | vs prompting (rows) | W/T/L | traj 95% CI | traj W/T/L | calls/row | tokens/row (p/c) | $/row | s/row |")
    print("|---|---|" + "---|" * len(DIMS) + "---|---|---|---|---|---|---|---|---|")
    for key, e in report["systems"].items():
        b, label = key.split("/", 1)
        pr = report["paired"].get(f"{b}: {label} - prompting")
        tb = pr["trajectory_bootstrap"] if pr else None
        print(f"| {key} | {e['valid_rows']}/{e['rows']} | " + " | ".join(f"{e[d]:.1f}" for d in DIMS) + f" | **{e['total']:.2f}** | "
              + (f"{pr['mean_diff']:+.2f} ± {pr['se_rows']:.2f}" if pr else "—") + " | " + (f"{pr['wins']}/{pr['ties']}/{pr['losses']}" if pr else "—") + " | "
              + (f"[{tb['ci95_rows'][0]:+.2f}, {tb['ci95_rows'][1]:+.2f}]" if tb else "—") + " | "
              + (f"{tb['trajectory_wins']}/{tb['trajectory_ties']}/{tb['trajectory_losses']}" if tb else "—")
              + f" | {e.get('model_calls_per_row', '—')} | {e.get('tokens_per_row', {}).get('prompt', '—')}/{e.get('tokens_per_row', {}).get('completion', '—')} | "
              + f"{e.get('cost_usd_per_row_reported', e.get('cost_usd_per_row_list', '—'))} | {e['latency_s_per_row']} |")
    print("\ntransfer:", json.dumps(report["transfer"], indent=1))
    for k, v in report["paired"].items():
        if f"{args.backbone} - gpt" in k:
            tb = v["trajectory_bootstrap"]
            print(f"{k}: {v['mean_diff']:+.2f} ± {v['se_rows']:.2f} rows W/T/L {v['wins']}/{v['ties']}/{v['losses']}; traj CI [{tb['ci95_rows'][0]:+.2f}, {tb['ci95_rows'][1]:+.2f}] p={tb['sign_flip_p_two_sided']}")
    print("written:", args.out)


if __name__ == "__main__":
    main()
