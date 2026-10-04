#!/usr/bin/env python3
"""Web evaluation report: prompting, prompting_rag, envpack_prompting, harness_only_v51, trace2env_v531 and the adopted
trace2env_v532 (web system of record, 2026-09-23) on all 200 rows of the
AgentWorldBench web split for one backbone directory (`work/exp-web-v1_50/backbones/<slug>/full200/`). Rows are
stratified by their trajectory's link to the construction tasks (`construction_overlap.json`: the benchmark
trajectories whose WebArena task we ran verbatim, those on the same template, template-only evidence, and the
unlinked rest) and by site; paired row differences with standard errors and trajectory-clustered bootstraps
(118 clusters); gate statistics; audits; resources.

    python work/exp-web-v1_50/full200_report.py --backbone gpt-5.6-sol
"""
import argparse, glob, json, random, re, sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "work/exp-web-v1_50"
TERM = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(TERM / "awb"))
sys.path.insert(0, str(TERM / "backbones"))
from cluster_bootstrap import analyse  # noqa: E402
from compare_backbones import PRICE  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
SYSTEMS = ["prompting", "prompting_unescaped", "prompting_rag", "envpack_prompting", "harness_only_v51", "trace2env_v531", "trace2env_v532"]
FILES = {"prompting": "prompting", "prompting_unescaped": "prompting-unescaped", "harness_only_v51": "harness_only_v51", "trace2env_v531": "trace2env_v531", "trace2env_v532": "trace2env_v532", "prompting_rag": "prompting_rag", "envpack_prompting": "envpack_prompting"}
ENT = re.compile(r"&(lt|gt|amp|quot|#\d+|#x[0-9a-fA-F]+);")
SITES = {"gitlab.example.com": "gitlab", "magento-admin.example.com": "admin", "forum.example.com": "forum", "magento-store.example.com": "store"}


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(r["id"], r["turn_idx"]): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())} if Path(path).exists() else {}


def site_of(row):
    text = "\n".join(str(p) for p in row.get("prompt", [])) + str(row.get("current_prompt", ""))
    for host, name in SITES.items():
        if host in text:
            return name
    return "other"


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 2), "se": round(pstdev(d) / len(d) ** 0.5, 2) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|full200|{name}"), 10000, 20000)
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
            if c.get("error"):  # a failed call (e.g. the interrupted DeepSeek attempt's 402s) has no usage and no cost
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
    overlap = json.load(open(EXP / "construction_overlap.json"))
    strata = {"exact task": set(overlap.get("exact_task", [])), "same template": set(overlap.get("same_template", [])),
              "template only": set(overlap.get("template_only", []))}
    linked = set().union(*strata.values())
    stratum_keys = {name: [k for k in keys if str(k[0]) in {str(t) for t in trajs}] for name, trajs in strata.items()}
    stratum_keys["unlinked"] = [k for k in keys if str(k[0]) not in {str(t) for t in linked}]
    site_keys = {}
    for k in keys:
        site_keys.setdefault(site_of(rows_in[k]), []).append(k)
    H = EXP / "backbones" / args.backbone / "full200"
    price = PRICE.get(args.backbone) or PRICE["gpt-5.6-sol"]
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; rows {len(keys)} in {len({k[0] for k in keys})} trajectories; strata (rows): "
          + ", ".join(f"{n} {len(v)}" for n, v in stratum_keys.items()) + "; sites (rows): " + ", ".join(f"{s} {len(v)}" for s, v in sorted(site_keys.items())) + "\n")
    names = list(stratum_keys) + sorted(site_keys)
    print("| system | valid | " + " | ".join(DIMS) + " | **total (200)** | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(DIMS) + "---|" + "---|" * len(names))
    for name in SYSTEMS:
        path = H / f"judged-{FILES[name]}.jsonl"
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
        print(f"| {name} | {len(valid)}/{len(keys)} | " + " | ".join(f"{dims[d]:.2f}" for d in DIMS) + f" | **{m(keys)}** | "
              + " | ".join(m(stratum_keys[n]) if n in stratum_keys else m(site_keys[n]) for n in names) + " |")
    pairs = (("trace2env_v531", "harness_only_v51"), ("trace2env_v531", "prompting"), ("harness_only_v51", "prompting"),
             ("trace2env_v532", "trace2env_v531"), ("trace2env_v532", "harness_only_v51"), ("trace2env_v532", "prompting"),
             ("envpack_prompting", "prompting"), ("envpack_prompting", "prompting_rag"), ("envpack_prompting", "harness_only_v51"),
             ("envpack_prompting", "trace2env_v532"), ("envpack_prompting", "trace2env_v531"), ("prompting_rag", "prompting"),
             ("prompting_rag", "trace2env_v532"),
             ("trace2env_v531", "prompting_unescaped"), ("harness_only_v51", "prompting_unescaped"), ("prompting_unescaped", "prompting"))
    print("\nPaired differences (rows): all / unlinked / linked to construction tasks")
    linked_keys = [k for k in keys if k not in set(stratum_keys["unlinked"])]
    print("| comparison | all 200 | unlinked | linked (exact + template) | exact task |"); print("|---|---|---|---|---|")
    for a, b in pairs:
        if a in scores and b in scores:
            print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keys, args.seed, a + b))} | {fmt(paired(scores[a], scores[b], stratum_keys['unlinked'], args.seed, a + b + 'u'))} | "
                  f"{fmt(paired(scores[a], scores[b], linked_keys, args.seed, a + b + 'l'))} | {fmt(paired(scores[a], scores[b], stratum_keys['exact task'], args.seed, a + b + 'e'))} |")
    print("\nBy site (rows)")
    print("| comparison | " + " | ".join(sorted(site_keys)) + " |"); print("|---|" + "---|" * len(site_keys))
    for a, b in pairs[:6]:
        if a in scores and b in scores:
            print(f"| {a} − {b} | " + " | ".join(fmt(paired(scores[a], scores[b], site_keys[s], args.seed, a + b + s)) for s in sorted(site_keys)) + " |")
    if "trace2env_v531" in rows_by and "harness_only_v51" in rows_by:
        common = [k for k in stratum_keys["unlinked"] if k in scores["trace2env_v531"] and k in scores["harness_only_v51"]]
        if common:
            print("\nDimension change trace2env_v531 − harness_only_v51 (unlinked rows): " + ", ".join(
                f"{d} {mean(norm(rows_by['trace2env_v531'][k][d]) - norm(rows_by['harness_only_v51'][k][d]) for k in common):+.2f}" for d in DIMS))
    if "prompting" in rows_by:
        escaped = [k for k in keys if k in rows_by["prompting"] and ENT.search(rows_by["prompting"][k].get("gen") or "")]
        print(f"\nPrompting outputs with HTML entities: {len(escaped)} of {len(keys)}")
    for gate_system in ("trace2env_v531", "trace2env_v532"):
      if gate_system in rows_by:
        gates = {k: ((r.get("trace2env") or {}).get("knowledge_gate") or {}) for k, r in rows_by[gate_system].items()}
        sup = [k for k, g in gates.items() if g.get("supporting")]
        labels = sum((Counter(g.get("labels", {})) for g in gates.values()), Counter())
        print(f"\nGate ({gate_system}): rows {len(gates)}, page matches {sum(g.get('page_matches', 0) for g in gates.values())}, abstained {sum(1 for g in gates.values() if g.get('abstained'))}, rows with supporting items {len(sup)} "
              f"(linked {sum(1 for k in sup if k in set(linked_keys))}, unlinked {sum(1 for k in sup if k in set(stratum_keys['unlinked']))}), "
              f"format documented on {sum(1 for g in gates.values() if g.get('format_documented'))} rows, items per row {mean(g.get('items', 0) for g in gates.values()):.1f}, labels {dict(labels)}")
        for label in (f"audit-{gate_system}.json", f"gate-audit-{gate_system}.json", "audit-harness_only_v51.json"):
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
            if name == "prompting_rag":
                hits = [len(((r.get("trace2env") or {}).get("rag") or {}).get("hits", [])) for r in rows_by[name].values()]
                extra = f"; retrieval hits per row mean {mean(hits):.2f}, rows with none {sum(1 for h in hits if h == 0)}"
            if name == "envpack_prompting":
                ms = [(r.get("trace2env") or {}).get("envpack") or {} for r in rows_by[name].values()]
                extra = f"; block chars median {median(m.get('block_chars', 0) for m in ms):.0f}, evidence per row mean {mean(len(m.get('evidence', [])) for m in ms):.1f}, rules mean {mean(len(m.get('rules', [])) for m in ms):.1f}"
            print(f"  {name}: errors {errs}" + (f", prompt tokens mean {round(mean(pt))} max {max(pt)}, completion mean {round(mean(ct))} over {len(pt)} rows" if pt else ", no usage recorded") + extra)
    print("\nRoutes and resources:")
    for name in ("harness_only_v51", "trace2env_v531", "trace2env_v532"):
        if name in rows_by:
            routes = Counter((x.get("trace2env") or {}).get("route") for x in rows_by[name].values())
            r = resources(H, FILES[name], rows_by[name], price)
            print(f"  {name}: routes {dict(routes)}, failed {sum(1 for x in rows_by[name].values() if x.get('failed'))}, tool calls/row {r['tool_calls_per_row']}, rows at budget {r['budget_rows']}, "
                  f"latency {r['latency_s_per_row']} s/row, fresh-equivalent cost ${r['cost_per_row_total']}/row, roles {json.dumps(r['roles'])}")


if __name__ == "__main__":
    main()
