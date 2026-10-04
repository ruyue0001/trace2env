#!/usr/bin/env python3
"""Android cross-fit report: prompting and trace2env_v533 (later harness_only_v51, prompting_rag, envpack_prompting)
on all 200 rows of the AgentWorldBench android split for one backbone directory
(`work/exp-android-cv/backbones/<slug>/full200/`). Every row was predicted with the package of its sub-source built
from the trajectories outside its fold (k=5, `package_map.json`). Rows are stratified by sub-source (json = AndroidWorld
style, phone = phone-tool XML, unparsed = DroidBot), by fold, by whether the row's current screen is a screen recorded in
its out-of-fold corpus (`row_strata.json`, the screen-identity signal) and by whether its app appears in another fold;
paired row differences with standard errors and trajectory-clustered bootstraps (92 clusters); gate statistics; resources.

    python work/exp-android-cv/full200_report.py --backbone gpt-5.6-sol
"""
import argparse, glob, json, random, sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "work/exp-android-cv"
TERM = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(TERM / "awb"))
sys.path.insert(0, str(TERM / "backbones"))
from cluster_bootstrap import analyse  # noqa: E402
from compare_backbones import PRICE  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
SYSTEMS = ["prompting", "prompting_rag", "envpack_prompting", "harness_only_v51", "trace2env_v531", "trace2env_v533"]
STYLES = ["json", "phone", "unparsed"]


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(str(r["id"]), int(r["turn_idx"])): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())} if Path(path).exists() else {}


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 2), "se": round(pstdev(d) / len(d) ** 0.5, 2) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|android|{name}"), 10000, 20000)
        out["ci"] = f"[{boot['ci95_rows'][0]:+.2f}, {boot['ci95_rows'][1]:+.2f}]"
        out["traj_wtl"] = f"{boot['trajectory_wins']}/{boot['trajectory_ties']}/{boot['trajectory_losses']}"
    return out


def fmt(p):
    if not p:
        return "—"
    return f"{p['mean']:+.2f} ± {p['se']:.2f} (n={p['rows']}; {p['wtl']}" + (f"; traj {p['ci']} {p['traj_wtl']}" if "ci" in p else "") + ")"


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
                continue
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
                       "cost_per_row": round((mean(r["fresh_cost"]) * (r["calls"] - r["errors"]) if r["fresh_cost"] else 0.0) / n, 3),
                       "time_per_row": round((mean(r["fresh_elapsed"]) * (r["calls"] - r["errors"]) if r["fresh_elapsed"] else 0.0) / n),
                       "prompt_mean": round(mean(r["prompt"])) if r["prompt"] else None, "prompt_max": max(r["prompt"]) if r["prompt"] else None}
                for role, r in roles.items()}
    return {"tool_calls_per_row": round(mean(tool_calls), 2) if tool_calls else 0, "budget_rows": sum(c >= 8 for c in tool_calls),
            "latency_s_per_row": round(mean(latency)) if latency else None, "cost_per_row_total": round(sum(v["cost_per_row"] for v in per_role.values()), 3), "roles": per_role}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--rows", default=str(EXP / "full200_rows.jsonl"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    rows_in = keyed(args.rows)
    keys = sorted(rows_in)
    strata_in = json.load(open(EXP / "row_strata.json"))
    info = {k: strata_in[f"{k[0]}|{k[1]}"] for k in keys}
    groups = {}
    for style in STYLES:
        groups[style] = [k for k in keys if info[k]["style"] == style]
    groups["screen match"] = [k for k in keys if info[k]["screen_match"]]
    groups["no screen match"] = [k for k in keys if not info[k]["screen_match"]]
    groups["app covered"] = [k for k in keys if info[k]["app_covered"]]
    groups["app not covered"] = [k for k in keys if not info[k]["app_covered"]]
    folds = {f"fold {f}": [k for k in keys if info[k]["fold"] == f] for f in range(5)}
    H = EXP / "backbones" / args.backbone / "full200"
    price = PRICE.get(args.backbone) or PRICE["gpt-5.6-sol"]
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; rows {len(keys)} in {len({k[0] for k in keys})} trajectories; strata (rows): "
          + ", ".join(f"{n} {len(v)}" for n, v in groups.items()) + "; folds (rows): " + ", ".join(f"{n} {len(v)}" for n, v in folds.items()) + "\n")
    names = list(groups)
    print("| system | valid | " + " | ".join(DIMS) + " | **total (200)** | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(DIMS) + "---|" + "---|" * len(names))
    for name in SYSTEMS:
        path = H / f"judged-{name}.jsonl"
        if not path.exists():
            continue
        rows = keyed(path)
        rows_by[name] = {k: rows[k] for k in keys if k in rows}
        valid = {k: r for k, r in rows_by[name].items() if not r.get("failed")}
        scores[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
        dims = {d: mean(norm(r[d]) for r in valid.values()) for d in DIMS}

        def m(ks):
            v = [scores[name][k] for k in ks if k in scores[name]]
            return f"{mean(v):.2f}" if v else "—"
        print(f"| {name} | {len(valid)}/{len(keys)} | " + " | ".join(f"{dims[d]:.2f}" for d in DIMS) + f" | **{m(keys)}** | " + " | ".join(m(groups[n]) for n in names) + " |")
    print("\nPer fold (total score)")
    print("| system | " + " | ".join(folds) + " |"); print("|---|" + "---|" * len(folds))
    for name in scores:
        print(f"| {name} | " + " | ".join(f"{mean([scores[name][k] for k in ks if k in scores[name]]):.2f}" if any(k in scores[name] for k in ks) else "—" for ks in folds.values()) + " |")
    pairs = (("trace2env_v533", "prompting"), ("trace2env_v533", "harness_only_v51"), ("harness_only_v51", "prompting"),
             ("trace2env_v533", "trace2env_v531"), ("trace2env_v531", "prompting"),
             ("envpack_prompting", "prompting"), ("envpack_prompting", "prompting_rag"), ("envpack_prompting", "harness_only_v51"),
             ("envpack_prompting", "trace2env_v533"), ("prompting_rag", "prompting"), ("prompting_rag", "trace2env_v533"))
    print("\nPaired differences (rows): all / by sub-source / by screen match")
    cols = ["all"] + STYLES + ["screen match", "no screen match", "app covered"]
    print("| comparison | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
    for a, b in pairs:
        if a in scores and b in scores:
            print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keys, args.seed, a + b))} | "
                  + " | ".join(fmt(paired(scores[a], scores[b], groups[c], args.seed, a + b + c)) for c in cols[1:]) + " |")
    for gate_system in ("trace2env_v531", "trace2env_v533"):
        if gate_system in rows_by:
            gates = {k: ((r.get("trace2env") or {}).get("knowledge_gate") or {}) for k, r in rows_by[gate_system].items()}
            sup = [k for k, g in gates.items() if g.get("supporting")]
            labels = sum((Counter(g.get("labels", {})) for g in gates.values()), Counter())
            print(f"\nGate ({gate_system}): rows {len(gates)}, screen matches {sum(g.get('screen_matches', 0) for g in gates.values())}, abstained {sum(1 for g in gates.values() if g.get('abstained'))}, "
                  f"rows with supporting items {len(sup)} (screen-match rows {sum(1 for k in sup if info[k]['screen_match'])}, other {sum(1 for k in sup if not info[k]['screen_match'])}), "
                  f"format documented on {sum(1 for g in gates.values() if g.get('format_documented'))} rows, items per row {mean(g.get('items', 0) for g in gates.values()):.1f}, labels {dict(labels)}")
            for label in (f"gate-audit-{gate_system}.json", f"audit-{gate_system}.json"):
                p = H / label
                if p.exists():
                    a = json.load(open(p, encoding="utf-8"))
                    print(f"  {label}: " + (f"rows_with_problems {a.get('rows_with_problems')} of {a.get('rows_with_agent_calls')}" if "rows_with_problems" in a
                                            else f"problems {a.get('problems')} of {a.get('package_items_seen')} items seen, labels {a.get('labels_at_boundary')}"))
    print("\nPrompting-style systems (one call per row): tokens from the rows' recorded usage")
    for name in ("prompting", "prompting_rag", "envpack_prompting"):
        if name in rows_by:
            us = [(r.get("trace2env") or {}).get("usage") or {} for r in rows_by[name].values()]
            pt = [u["prompt_tokens"] for u in us if u.get("prompt_tokens")]; ct = [u.get("completion_tokens") or 0 for u in us if u.get("prompt_tokens")]
            errs = sum(1 for r in rows_by[name].values() if (r.get("trace2env") or {}).get("error") or not (r.get("gen") or "").strip())
            extra = ""
            if name == "envpack_prompting":
                ms = [(r.get("trace2env") or {}).get("envpack") or {} for r in rows_by[name].values()]
                extra = f"; block chars median {median(m.get('block_chars', 0) for m in ms):.0f}, evidence per row mean {mean(len(m.get('evidence', [])) for m in ms):.1f}, rules mean {mean(len(m.get('rules', [])) for m in ms):.1f}"
            print(f"  {name}: errors {errs}" + (f", prompt tokens mean {round(mean(pt))} max {max(pt)}, completion mean {round(mean(ct))} over {len(pt)} rows" if pt else ", no usage recorded") + extra)
    print("\nRoutes and resources:")
    for name in ("harness_only_v51", "trace2env_v531", "trace2env_v533"):
        if name in rows_by:
            routes = Counter((x.get("trace2env") or {}).get("route") for x in rows_by[name].values())
            r = resources(H, name, rows_by[name], price)
            print(f"  {name}: routes {dict(routes)}, failed {sum(1 for x in rows_by[name].values() if x.get('failed'))}, tool calls/row {r['tool_calls_per_row']}, rows at budget {r['budget_rows']}, "
                  f"latency {r['latency_s_per_row']} s/row, fresh-equivalent cost ${r['cost_per_row_total']}/row, roles {json.dumps(r['roles'])}")


if __name__ == "__main__":
    main()
