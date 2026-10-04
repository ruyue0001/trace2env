#!/usr/bin/env python3
"""Final evaluation on all 354 rows of the AgentWorldBench terminal split: prompting, harness_only_v51 and the frozen
trace2env_v531 for one backbone directory (`backbones/<slug>/full354/`), with the 42 rows of the 10 construction
trajectories reported separately from the other 312, paired row differences with standard errors and
trajectory-clustered bootstraps (76 clusters), gate statistics, audit summaries and resources.

    python work/exp-v1_20_r=1/backbones/full354_report.py --backbone gpt-5.6-sol
    python work/exp-v1_20_r=1/backbones/full354_report.py --backbone deepseek-flash-official
"""
import argparse, glob, json, random, re, sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(EXP / "awb"))
sys.path.insert(0, str(EXP / "backbones"))
from cluster_bootstrap import analyse  # noqa: E402
from compare_backbones import PRICE  # noqa: E402
from construction_task_split import CONSTRUCTION_TASK_TRAJECTORIES  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
SYSTEMS = ["prompting", "prompting_unescaped", "harness_only_v51", "trace2env_v531"]
FILES = {"prompting": "prompting", "prompting_unescaped": "prompting-unescaped", "harness_only_v51": "harness_only_v51", "trace2env_v531": "trace2env_v531"}
ENT = re.compile(r"&(lt|gt|amp|quot|#\d+|#x[0-9a-fA-F]+);")


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(r["id"], r["turn_idx"]): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 2), "se": round(pstdev(d) / len(d) ** 0.5, 2) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|full354|{name}"), 10000, 20000)
        out["ci"] = f"[{boot['ci95_rows'][0]:+.2f}, {boot['ci95_rows'][1]:+.2f}]"
        out["traj_wtl"] = f"{boot['trajectory_wins']}/{boot['trajectory_ties']}/{boot['trajectory_losses']}"
    return out


def fmt(p):
    if not p:
        return "—"
    return f"{p['mean']:+.2f} ± {p['se']:.2f} ({p['wtl']}" + (f"; traj {p['ci']} {p['traj_wtl']}" if "ci" in p else "") + ")"


def usage_cost(u, price):
    return (u.get("prompt_tokens", 0) - u.get("cached_tokens", 0)) * price["prompt"] + u.get("cached_tokens", 0) * price["cached"] + u.get("completion_tokens", 0) * price["completion"]


def resources(directory, label, rows, price):
    infos = [(r.get("trace2env") or {}) for r in rows.values()]
    tool_calls = [len([t for t in (i.get("tool_calls") or []) if t != "_usage"]) for i in infos]
    latency = [i.get("latency_seconds") or 0 for i in infos]
    roles = {}
    for path in glob.glob(str(directory / f"calls-{label}-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = roles.setdefault(c.get("role"), {"calls": 0, "cache_hits": 0, "fresh_cost": [], "fresh_elapsed": [], "prompt": [], "errors": 0})
            r["calls"] += 1
            if c.get("error"):
                r["errors"] += 1
            if c.get("cache_hit"):
                r["cache_hits"] += 1
                continue
            u = c.get("usage") or {}
            r["fresh_elapsed"].append(float(c.get("elapsed_s") or 0))
            if u:
                r["fresh_cost"].append(usage_cost(u, price))
                r["prompt"].append(u.get("prompt_tokens") or 0)
    n = max(1, len(rows))
    per_role = {role: {"calls": r["calls"], "cache_hits": r["cache_hits"], "errors": r["errors"],
                       "cost_per_row": round((mean(r["fresh_cost"]) * r["calls"] if r["fresh_cost"] else 0.0) / n, 3),
                       "time_per_row": round((mean(r["fresh_elapsed"]) * r["calls"] if r["fresh_elapsed"] else 0.0) / n),
                       "prompt_mean": round(mean(r["prompt"])) if r["prompt"] else None, "prompt_max": max(r["prompt"]) if r["prompt"] else None}
                for role, r in roles.items()}
    return {"tool_calls_per_row": round(mean(tool_calls), 2) if tool_calls else 0, "budget_rows": sum(c >= 8 for c in tool_calls),
            "latency_s_per_row": round(mean(latency)) if latency else None, "cost_per_row_total": round(sum(v["cost_per_row"] for v in per_role.values()), 3), "roles": per_role}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--rows", default=str(EXP / "backbones/full354_rows.jsonl"))
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--exclude", default=str(EXP / "backbones/eval30_rows.jsonl"),
                    help="rows to exclude in the robustness table (the 30 development rows the gate was tuned on)")
    args = ap.parse_args()
    keys = sorted(keyed(args.rows))
    overlap = [k for k in keys if k[0] in CONSTRUCTION_TASK_TRAJECTORIES]
    rest = [k for k in keys if k[0] not in CONSTRUCTION_TASK_TRAJECTORIES]
    H = EXP / "backbones" / args.backbone / "full354"
    price = PRICE.get(args.backbone) or (PRICE.get("deepseek-v4.1-flash") if args.backbone.startswith("deepseek-flash") else None) or PRICE["gpt-5.6-sol"]
    if args.backbone.startswith("deepseek-flash-official"):
        print("note: costs use the OpenRouter DeepSeek-endpoint list price as an approximation of the official API's price")
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; rows {len(keys)} ({len(overlap)} construction-overlap in {len({k[0] for k in overlap})} trajectories, {len(rest)} other); trajectories {len({k[0] for k in keys})}\n")
    print("| system | valid | " + " | ".join(DIMS) + " | **total (354)** | 312 non-overlap | 42 overlap |")
    print("|---|---|" + "---|" * len(DIMS) + "---|---|---|")
    for name in SYSTEMS:
        path = H / f"judged-{FILES[name]}.jsonl"
        if not path.exists():
            print(f"| {name} | missing | | | | | | | | |")
            continue
        rows = keyed(path)
        rows_by[name] = {k: rows[k] for k in keys if k in rows}
        valid = {k: r for k, r in rows_by[name].items() if not r.get("failed")}
        scores[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
        dims = {d: mean(norm(r[d]) for r in valid.values()) for d in DIMS}
        print(f"| {name} | {len(valid)}/{len(keys)} | " + " | ".join(f"{dims[d]:.2f}" for d in DIMS) + f" | **{mean(scores[name].values()):.2f}** | {mean(scores[name][k] for k in rest if k in scores[name]):.2f} | {mean(scores[name][k] for k in overlap if k in scores[name]):.2f} |")
    print("\nPaired differences (rows): all 354 / 312 non-overlap / 42 overlap")
    print("| comparison | all | non-overlap | overlap |"); print("|---|---|---|---|")
    for a, b in (("trace2env_v531", "harness_only_v51"), ("trace2env_v531", "prompting"), ("harness_only_v51", "prompting"),
                 ("trace2env_v531", "prompting_unescaped"), ("harness_only_v51", "prompting_unescaped"), ("prompting_unescaped", "prompting")):
        if a in scores and b in scores:
            print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keys, args.seed, a + b))} | {fmt(paired(scores[a], scores[b], rest, args.seed, a + b + 'r'))} | {fmt(paired(scores[a], scores[b], overlap, args.seed, a + b + 'o'))} |")
    if args.exclude and Path(args.exclude).exists():
        dev = set(keyed(args.exclude))
        keep = [k for k in keys if k not in dev]
        keep_rest = [k for k in rest if k not in dev]
        print(f"\nRobustness: excluding the {len(dev)} development rows ({len(keep)} rows, {len(keep_rest)} non-overlap)")
        print("| system | total (non-dev) | non-overlap non-dev |"); print("|---|---|---|")
        for name in SYSTEMS:
            if name in scores:
                print(f"| {name} | {mean(scores[name][k] for k in keep if k in scores[name]):.2f} | {mean(scores[name][k] for k in keep_rest if k in scores[name]):.2f} |")
        print("| comparison | non-dev rows | non-overlap non-dev rows |"); print("|---|---|---|")
        for a, b in (("trace2env_v531", "harness_only_v51"), ("trace2env_v531", "prompting"), ("trace2env_v531", "prompting_unescaped"), ("harness_only_v51", "prompting")):
            if a in scores and b in scores:
                print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keep, args.seed, a + b + 'nd'))} | {fmt(paired(scores[a], scores[b], keep_rest, args.seed, a + b + 'ndr'))} |")
    if "trace2env_v531" in rows_by and "harness_only_v51" in rows_by:
        common = [k for k in rest if k in scores["trace2env_v531"] and k in scores["harness_only_v51"]]
        print("\nDimension change trace2env_v531 − harness_only_v51 (non-overlap rows): " + ", ".join(
            f"{d} {mean(norm(rows_by['trace2env_v531'][k][d]) - norm(rows_by['harness_only_v51'][k][d]) for k in common):+.2f}" for d in DIMS))
    if "prompting" in rows_by:
        escaped = [k for k in keys if ENT.search(rows_by["prompting"][k].get("gen") or "")]
        clean = [k for k in rest if k not in escaped]
        print(f"\nPrompting outputs with HTML entities: {len(escaped)} of {len(keys)}; artifact-free non-overlap rows: {len(clean)}")
        for a, b in (("trace2env_v531", "prompting"), ("harness_only_v51", "prompting"), ("trace2env_v531", "harness_only_v51")):
            if a in scores and b in scores:
                print(f"  {a} − {b}: artifact-free non-overlap {fmt(paired(scores[a], scores[b], clean, args.seed, a + b + 'c'))} | escaped non-overlap {fmt(paired(scores[a], scores[b], [k for k in escaped if k in rest], args.seed, a + b + 'e'))}")
    if "trace2env_v531" in rows_by:
        gates = {k: ((r.get("trace2env") or {}).get("knowledge_gate") or {}) for k, r in rows_by["trace2env_v531"].items()}
        sup = [k for k, g in gates.items() if g.get("supporting")]
        labels = sum((Counter(g.get("labels", {})) for g in gates.values()), Counter())
        print(f"\nGate (trace2env_v531): rows {len(gates)}, abstained {sum(1 for g in gates.values() if g.get('abstained'))}, rows with supporting items {len(sup)} "
              f"(overlap {sum(1 for k in sup if k in overlap)}, non-overlap {sum(1 for k in sup if k in rest)}), items per row {mean(g.get('items', 0) for g in gates.values()):.1f}, labels {dict(labels)}")
        print("  non-overlap supporting rows: " + "; ".join(f"{k[0]}/{k[1]}: {len(gates[k]['supporting'])}" for k in sup if k in rest))
        for label in ("audit-trace2env_v531.json", "gate-audit-trace2env_v531.json", "audit-harness_only_v51.json"):
            p = H / label
            if p.exists():
                a = json.load(open(p, encoding="utf-8"))
                print(f"  {label}: " + (f"rows_with_problems {a.get('rows_with_problems')}, rows_with_agent_calls {a.get('rows_with_agent_calls')}, max_rounds {a.get('max_rounds')}, max_prompt_tokens {a.get('max_prompt_tokens')}" if "rows_with_problems" in a else f"problems {a.get('problems')}, items {a.get('package_items_seen')}, supporting {a.get('supporting_items')}, sanitized {a.get('sanitized_items')}"))
    print("\nRoutes and resources:")
    for name in ("harness_only_v51", "trace2env_v531"):
        if name in rows_by:
            routes = Counter((x.get("trace2env") or {}).get("route") for x in rows_by[name].values())
            r = resources(H, FILES[name], rows_by[name], price)
            print(f"  {name}: routes {dict(routes)}, failed {sum(1 for x in rows_by[name].values() if x.get('failed'))}, tool calls/row {r['tool_calls_per_row']}, rows at budget {r['budget_rows']}, latency {r['latency_s_per_row']} s/row, $ {r['cost_per_row_total']}/row; roles {r['roles']}")


if __name__ == "__main__":
    main()
