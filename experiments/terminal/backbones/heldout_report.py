#!/usr/bin/env python3
"""Held-out validation report: trace2env_v531 (the v5.3.1 gate, no judge) and harness_only_v51 on the 100 held-out rows
(heldout_rows.jsonl: rows outside the 25 development trajectories, seed 20260923) against the existing all-row
prompting and v3 references of the same backbone; construction-overlap rows (14) reported separately from the other 86.

    python work/exp-v1_20_r=1/backbones/heldout_report.py --backbone gpt-5.6-sol --reference awb
    python work/exp-v1_20_r=1/backbones/heldout_report.py --backbone deepseek-v4.1-flash --reference backbones/deepseek-v4.1-flash

Scores: official scaling, valid rows, paired row differences with standard error, row W/T/L, trajectory-clustered
bootstrap (50 clusters). Resources per row from the call logs (fresh-equivalent per role: agent, tracking) and the rows'
tool calls and latency. Gate statistics and audit summaries for trace2env_v531. One run per system.
"""
import argparse, glob, json, random, sys
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


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(r["id"], r["turn_idx"]): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 1), "se": round(pstdev(d) / len(d) ** 0.5, 1) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|heldout|{name}"), 10000, 20000)
        out["ci"] = f"[{boot['ci95_rows'][0]:+.1f}, {boot['ci95_rows'][1]:+.1f}]"
        out["traj_wtl"] = f"{boot['trajectory_wins']}/{boot['trajectory_ties']}/{boot['trajectory_losses']}"
    return out


def fmt(p):
    if not p:
        return "—"
    return f"{p['mean']:+.1f} ± {p['se']:.1f} ({p['wtl']}" + (f"; traj {p['ci']} {p['traj_wtl']}" if "ci" in p else "") + ")"


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
            r = roles.setdefault(c.get("role"), {"calls": 0, "fresh_cost": [], "fresh_elapsed": [], "prompt": [], "completion": []})
            r["calls"] += 1
            if c.get("cache_hit"):
                continue
            u = c.get("usage") or {}
            r["fresh_elapsed"].append(float(c.get("elapsed_s") or 0))
            if u:
                r["fresh_cost"].append(usage_cost(u, price))
                r["prompt"].append(u.get("prompt_tokens") or 0)
                r["completion"].append(u.get("completion_tokens") or 0)
    n = max(1, len(rows))
    per_role = {}
    for role, r in roles.items():
        fresh_equiv = (mean(r["fresh_cost"]) * r["calls"] if r["fresh_cost"] else 0.0) / n
        per_role[role] = {"calls": r["calls"], "cache_hits": r["calls"] - len(r["fresh_elapsed"]), "cost_per_row": round(fresh_equiv, 3),
                          "time_per_row": round((mean(r["fresh_elapsed"]) * r["calls"] if r["fresh_elapsed"] else 0.0) / n),
                          "prompt_mean": round(mean(r["prompt"])) if r["prompt"] else None, "prompt_max": max(r["prompt"]) if r["prompt"] else None,
                          "completion_mean": round(mean(r["completion"])) if r["completion"] else None}
    agent = per_role.get("runtime_agent_turn", {})
    return {"tool_calls_per_row": round(mean(tool_calls), 2) if tool_calls else 0, "budget_rows": sum(c >= 8 for c in tool_calls),
            "latency_s_per_row": round(mean(latency)) if latency else None, "cost_per_row_total": round(sum(v["cost_per_row"] for v in per_role.values()), 3),
            "agent": agent, "roles": per_role}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--rows", default=str(EXP / "backbones/heldout_rows.jsonl"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    keys = sorted(keyed(args.rows))
    overlap = [k for k in keys if k[0] in CONSTRUCTION_TASK_TRAJECTORIES]
    rest = [k for k in keys if k[0] not in CONSTRUCTION_TASK_TRAJECTORIES]
    H, R = EXP / "backbones" / args.backbone / "heldout", EXP / args.reference
    price = (PRICE.get(args.backbone) or PRICE.get(args.backbone.replace("-pinned", ""))
             or (PRICE.get("deepseek-v4.1-flash") if args.backbone.startswith("deepseek-flash") else None) or PRICE["gpt-5.6-sol"])
    if args.backbone.startswith("deepseek-flash-official"):
        print("note: costs use the OpenRouter DeepSeek-endpoint list price as an approximation of the official API's price")
    systems = {"prompting": (R, "prompting"), "prompting_unescaped": (H, "prompting-unescaped"), "v3": (R, "v3"),
               "harness_only_v51": (H, "harness_only_v51"), "trace2env_v531": (H, "trace2env_v531")}
    if args.backbone.endswith("-pinned"):  # the unpinned runs of the same backbone, for comparison
        U = EXP / "backbones" / args.backbone.replace("-pinned", "") / "heldout"
        systems["harness_only_v51_unpinned"] = (U, "harness_only_v51")
        systems["trace2env_v531_unpinned"] = (U, "trace2env_v531")
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; held-out rows {len(keys)} ({len(overlap)} construction-overlap, {len(rest)} other); trajectories {len({k[0] for k in keys})}\n")
    print("| system | valid | " + " | ".join(DIMS) + " | total (all) | total (non-overlap) | total (overlap) |")
    print("|---|---|" + "---|" * len(DIMS) + "---|---|---|")
    for name, (directory, label) in systems.items():
        path = directory / f"judged-{label}.jsonl"
        if not path.exists():
            print(f"| {name} | missing | | | | | | | | |")
            continue
        rows = keyed(path)
        rows_by[name] = {k: rows[k] for k in keys if k in rows}
        valid = {k: r for k, r in rows_by[name].items() if not r.get("failed")}
        scores[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
        dims = {d: mean(norm(r[d]) for r in valid.values()) for d in DIMS}
        total = mean(scores[name].values())
        rest_mean = mean(scores[name][k] for k in rest if k in scores[name])
        over_mean = mean(scores[name][k] for k in overlap if k in scores[name]) if overlap else float("nan")
        print(f"| {name} | {len(valid)}/{len(keys)} | " + " | ".join(f"{dims[d]:.1f}" for d in DIMS) + f" | **{total:.1f}** | {rest_mean:.1f} | {over_mean:.1f} |")
    print("\nPaired differences (rows): all / non-overlap / overlap")
    print("| comparison | all | non-overlap | overlap |"); print("|---|---|---|---|")
    for a, b in (("trace2env_v531", "harness_only_v51"), ("trace2env_v531", "prompting"), ("harness_only_v51", "prompting"),
                 ("trace2env_v531", "prompting_unescaped"), ("harness_only_v51", "prompting_unescaped"), ("prompting_unescaped", "prompting"),
                 ("trace2env_v531", "v3"), ("v3", "prompting"),
                 ("trace2env_v531_unpinned", "harness_only_v51"), ("harness_only_v51", "harness_only_v51_unpinned"), ("trace2env_v531", "trace2env_v531_unpinned")):
        if a in scores and b in scores:
            print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keys, args.seed, a + b))} | {fmt(paired(scores[a], scores[b], rest, args.seed, a + b + 'r'))} | {fmt(paired(scores[a], scores[b], overlap, args.seed, a + b + 'o'))} |")
    if "prompting" in rows_by:
        import re
        ent = re.compile(r"&(lt|gt|amp|quot|#\d+|#x[0-9a-fA-F]+);")
        escaped = [k for k in keys if ent.search(rows_by["prompting"][k].get("gen") or "")]
        clean = [k for k in rest if k not in escaped]
        print(f"\nPrompting outputs with HTML entities: {len(escaped)} of {len(keys)} rows; artifact-free non-overlap rows: {len(clean)}")
        print("| comparison | artifact-free non-overlap rows | escaped non-overlap rows |"); print("|---|---|---|")
        for a, b in (("trace2env_v531", "prompting"), ("harness_only_v51", "prompting"), ("trace2env_v531", "harness_only_v51"), ("trace2env_v531", "prompting_unescaped"), ("harness_only_v51", "prompting_unescaped")):
            if a in scores and b in scores:
                print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], clean, args.seed, a + b + 'c'))} | {fmt(paired(scores[a], scores[b], [k for k in escaped if k in rest], args.seed, a + b + 'e'))} |")
    if "trace2env_v531" in rows_by and "harness_only_v51" in rows_by:
        common = [k for k in rest if k in scores["trace2env_v531"] and k in scores["harness_only_v51"]]
        print("\nDimension change trace2env_v531 − harness_only_v51 (non-overlap rows): " + ", ".join(
            f"{d} {mean(norm(rows_by['trace2env_v531'][k][d]) - norm(rows_by['harness_only_v51'][k][d]) for k in common):+.1f}" for d in DIMS))
    if "trace2env_v531" in rows_by:
        gates = {k: ((r.get("trace2env") or {}).get("knowledge_gate") or {}) for k, r in rows_by["trace2env_v531"].items()}
        sup = [k for k, g in gates.items() if g.get("supporting")]
        labels = sum((Counter(g.get("labels", {})) for g in gates.values()), Counter())
        dispositions = sum((Counter(g.get("dispositions", {})) for g in gates.values()), Counter())
        print(f"\nGate (trace2env_v531): rows {len(gates)}, abstained {sum(1 for g in gates.values() if g.get('abstained'))}, rows with supporting items {len(sup)} "
              f"(overlap {sum(1 for k in sup if k in overlap)}, non-overlap {sum(1 for k in sup if k in rest)}), items per row {mean(g.get('items', 0) for g in gates.values()):.1f}, "
              f"labels {dict(labels)}, dispositions {dict(dispositions)}, masked tokens {sum(g.get('masked_tokens', 0) for g in gates.values())}")
        print("supporting rows: " + "; ".join(f"{k[0]}/{k[1]}: {len(gates[k]['supporting'])}{' (overlap)' if k in overlap else ''}" for k in sup))
        for name in ("trace2env_v531", "harness_only_v51"):
            if name in scores:
                s = scores[name]
                print(f"{name} by gate decision: supporting rows {mean(s[k] for k in sup if k in s) if sup else float('nan'):.1f} (n={len([k for k in sup if k in s])}), "
                      f"abstained rows {mean(s[k] for k in keys if k in s and k not in sup):.1f}")
        for label in ("audit-trace2env_v531.json", "gate-audit-trace2env_v531.json", "audit-harness_only_v51.json"):
            p = H / label
            if p.exists():
                a = json.load(open(p, encoding="utf-8"))
                print(f"{label}: " + (f"rows_with_problems {a.get('rows_with_problems')}, max_prompt_tokens {a.get('max_prompt_tokens')}, max_rounds {a.get('max_rounds')}" if "rows_with_problems" in a else f"problems {a.get('problems')}, items {a.get('package_items_seen')}, supporting {a.get('supporting_items')}, sanitized {a.get('sanitized_items')}"))
    print("\nResources (fresh-equivalent per row; reference systems from row usage are not listed):")
    print("| system | tool calls/row | rows at budget | latency s/row | $ per row | roles (calls, cache hits, $/row, s/row, prompt mean/max, completion mean) |"); print("|---|---|---|---|---|---|")
    for name in ("harness_only_v51", "trace2env_v531"):
        if name in rows_by:
            r = resources(H, name, rows_by[name], price)
            roles = "; ".join(f"{k}: {v['calls']}, {v['cache_hits']}, ${v['cost_per_row']}, {v['time_per_row']} s, {v['prompt_mean']}/{v['prompt_max']}, {v['completion_mean']}" for k, v in sorted(r["roles"].items()))
            print(f"| {name} | {r['tool_calls_per_row']} | {r['budget_rows']} | {r['latency_s_per_row']} | {r['cost_per_row_total']} | {roles} |")
            routes = Counter((x.get("trace2env") or {}).get("route") for x in rows_by[name].values())
            print(f"  {name} routes {dict(routes)}; failed rows {sum(1 for x in rows_by[name].values() if x.get('failed'))}")
    print("\nPer-row totals:")
    names = [n for n in systems if n in scores]
    print("| row | overlap | v531 gate | " + " | ".join(names) + " |"); print("|---|---|---|" + "---|" * len(names))
    for k in keys:
        g = gates.get(k, {}) if "trace2env_v531" in rows_by else {}
        tag = f"sup {len(g.get('supporting', []))}" if g.get("supporting") else ("abstain" if g else "")
        print(f"| {k[0]}/{k[1]} | {'yes' if k in overlap else ''} | {tag} | " + " | ".join(f"{scores[n][k]:.0f}" if k in scores[n] else "fail" for n in names) + " |")


if __name__ == "__main__":
    main()
