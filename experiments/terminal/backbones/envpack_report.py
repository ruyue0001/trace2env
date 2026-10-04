#!/usr/bin/env python3
"""`envpack_prompting` on the 354 terminal rows: the non-agentic package baseline against direct prompting,
prompting+rag (fixed top-5 raw trace turns), harness_only_v51 and trace2env_v531 for one backbone directory
(`backbones/<slug>/full354/`; prompting+rag from `awb/judged-prompting+rag.jsonl`). Five dimensions and total,
valid/error counts, paired row differences with standard errors and trajectory-clustered bootstraps (76 clusters),
the 42 construction-overlap rows apart from the other 312, token usage per row from the call logs, and what the
package block contained (items per kind, evidence and rule frequencies), so any prompt can be reconstructed from
`trace2env.envpack` plus the call log.

    python work/exp-v1_20_r=1/backbones/envpack_report.py --backbone gpt-5.6-sol
"""
import argparse, glob, json, random, sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(EXP / "awb"))
sys.path.insert(0, str(EXP / "backbones"))
from cluster_bootstrap import analyse  # noqa: E402
from compare_backbones import PRICE  # noqa: E402
from construction_task_split import CONSTRUCTION_TASK_TRAJECTORIES  # noqa: E402

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
SYSTEMS = ["prompting", "prompting_rag", "harness_only_v51", "trace2env_v531", "envpack_prompting"]


def norm(v):
    return (v - 1) / 4 * 100


def keyed(path):
    return {(r["id"], r["turn_idx"]): r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())} if Path(path).exists() else {}


def paired(a, b, keys, seed, name):
    common = [k for k in keys if k in a and k in b]
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    out = {"rows": len(common), "mean": round(mean(d), 2), "se": round(pstdev(d) / len(d) ** 0.5, 2) if len(d) > 1 else 0.0,
           "wtl": f"{sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)}"}
    if len({k[0] for k in common}) >= 3:
        boot = analyse({k: a[k] for k in common}, {k: b[k] for k in common}, lambda tid: True, random.Random(f"{seed}|envpack|{name}"), 10000, 20000)
        out["ci"] = f"[{boot['ci95_rows'][0]:+.2f}, {boot['ci95_rows'][1]:+.2f}]"
        out["traj_wtl"] = f"{boot['trajectory_wins']}/{boot['trajectory_ties']}/{boot['trajectory_losses']}"
    return out


def fmt(p):
    if not p:
        return "—"
    return f"{p['mean']:+.2f} ± {p['se']:.2f} ({p['wtl']}" + (f"; traj {p['ci']} {p['traj_wtl']}" if "ci" in p else "") + ")"


def usage_cost(u, price):
    return (u.get("prompt_tokens", 0) - u.get("cached_tokens", 0)) * price["prompt"] + u.get("cached_tokens", 0) * price["cached"] + u.get("completion_tokens", 0) * price["completion"]


def tokens_from_rows(rows):
    """Token usage recorded on the prediction rows themselves (prompting-style systems record `usage` per row)."""
    prompt, completion, n = [], [], 0
    for r in rows.values():
        u = (r.get("trace2env") or {}).get("usage") or {}
        if u.get("prompt_tokens"):
            prompt.append(u["prompt_tokens"]); completion.append(u.get("completion_tokens") or 0); n += 1
    return {"rows_with_usage": n, "prompt_mean": round(mean(prompt)) if prompt else None, "prompt_max": max(prompt) if prompt else None,
            "completion_mean": round(mean(completion)) if completion else None, "prompt_total": sum(prompt), "completion_total": sum(completion)}


def tokens_from_calls(directory, label):
    prompt, completion, calls, errors, hits = [], [], 0, 0, 0
    for path in glob.glob(str(directory / f"calls-{label}-*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            calls += 1
            if c.get("error"):
                errors += 1
                continue
            if c.get("cache_hit"):
                hits += 1
            u = c.get("usage") or {}
            if u.get("prompt_tokens"):
                prompt.append(u["prompt_tokens"]); completion.append(u.get("completion_tokens") or 0)
    return {"calls": calls, "errors": errors, "cache_hits": hits, "prompt_mean": round(mean(prompt)) if prompt else None,
            "prompt_max": max(prompt) if prompt else None, "completion_mean": round(mean(completion)) if completion else None,
            "prompt_total": sum(prompt), "completion_total": sum(completion)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--rows", default=str(EXP / "backbones" / "full354_rows.jsonl"))
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()
    keys = sorted(keyed(args.rows))
    overlap = [k for k in keys if k[0] in CONSTRUCTION_TASK_TRAJECTORIES]
    rest = [k for k in keys if k[0] not in CONSTRUCTION_TASK_TRAJECTORIES]
    H = EXP / "backbones" / args.backbone / "full354"
    rag = H / "judged-prompting_rag.jsonl"
    if not rag.exists():
        rag = EXP / "awb" / "judged-prompting+rag.jsonl"  # the 2026-09-20 GPT run, same rows and judge
    files = {"prompting": H / "judged-prompting.jsonl", "prompting_rag": rag,
             "harness_only_v51": H / "judged-harness_only_v51.jsonl", "trace2env_v531": H / "judged-trace2env_v531.jsonl",
             "envpack_prompting": H / "judged-envpack_prompting.jsonl"}
    price = PRICE.get(args.backbone) or PRICE["gpt-5.6-sol"]
    scores, rows_by = {}, {}
    print(f"backbone {args.backbone}; rows {len(keys)} ({len(overlap)} construction-overlap in {len({k[0] for k in overlap})} trajectories, {len(rest)} other); prompting_rag from {rag}\n")
    print("| system | valid | errors | " + " | ".join(DIMS) + " | **total (354)** | 312 non-overlap | 42 overlap |")
    print("|---|---|---|" + "---|" * len(DIMS) + "---|---|---|")
    for name in SYSTEMS:
        rows = keyed(files[name])
        if not rows:
            print(f"| {name} | missing | | | | | | | | | |")
            continue
        rows_by[name] = {k: rows[k] for k in keys if k in rows}
        valid = {k: r for k, r in rows_by[name].items() if not r.get("failed")}
        errors = sum(1 for r in rows_by[name].values() if (r.get("trace2env") or {}).get("error") or not (r.get("gen") or "").strip())
        scores[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
        dims = {d: mean(norm(r[d]) for r in valid.values()) for d in DIMS}
        def m(ks):
            v = [scores[name][k] for k in ks if k in scores[name]]
            return f"{mean(v):.2f}" if v else "—"
        print(f"| {name} | {len(valid)}/{len(rows_by[name])} | {errors} | " + " | ".join(f"{dims[d]:.2f}" for d in DIMS)
              + f" | **{m(keys)}** | {m(rest)} | {m(overlap)} |")
    pairs = (("envpack_prompting", "prompting"), ("envpack_prompting", "prompting_rag"), ("envpack_prompting", "harness_only_v51"),
             ("envpack_prompting", "trace2env_v531"), ("trace2env_v531", "harness_only_v51"), ("prompting_rag", "prompting"))
    print("\nPaired row differences (mean ± SE, wins/ties/losses; trajectory-clustered bootstrap 95% CI)")
    print("| comparison | all 354 | 312 non-overlap | 42 overlap |"); print("|---|---|---|---|")
    for a, b in pairs:
        if a in scores and b in scores:
            print(f"| {a} − {b} | {fmt(paired(scores[a], scores[b], keys, args.seed, a + b))} | {fmt(paired(scores[a], scores[b], rest, args.seed, a + b + 'r'))} | {fmt(paired(scores[a], scores[b], overlap, args.seed, a + b + 'o'))} |")
    if "envpack_prompting" in rows_by and "prompting" in rows_by:
        common = [k for k in rest if k in scores["envpack_prompting"] and k in scores["prompting"]]
        print("\nDimension change envpack_prompting − prompting (non-overlap rows): " + ", ".join(
            f"{d} {mean(norm(rows_by['envpack_prompting'][k][d]) - norm(rows_by['prompting'][k][d]) for k in common):+.2f}" for d in DIMS))
    print("\nToken usage per row (from the rows' recorded usage; call logs in brackets)")
    for name in ("prompting", "envpack_prompting", "prompting_rag"):
        if name in rows_by:
            t = tokens_from_rows(rows_by[name])
            chars = [(r.get("trace2env") or {}).get("prompt_chars") for r in rows_by[name].values()]
            if name == "prompting_rag":
                hits = [len(((r.get("trace2env") or {}).get("rag") or {}).get("hits", [])) for r in rows_by[name].values()]
                print(f"  prompting_rag retrieval: hits per row mean {mean(hits):.2f}, rows with none {sum(1 for h in hits if h == 0)}")
            chars = [c for c in chars if c]
            cost = sum(usage_cost((r.get("trace2env") or {}).get("usage") or {}, price) for r in rows_by[name].values()) / max(1, len(rows_by[name]))
            if t["rows_with_usage"]:
                print(f"  {name}: prompt tokens mean {t['prompt_mean']} max {t['prompt_max']}, completion mean {t['completion_mean']}, totals {t['prompt_total']} / {t['completion_total']} "
                      f"over {t['rows_with_usage']} rows, list-price cost ${cost:.3f}/row" + (f"; prompt chars mean {round(mean(chars))}" if chars else ""))
            else:
                print(f"  {name}: no per-row usage recorded by that run" + (f"; prompt chars mean {round(mean(chars))} max {max(chars)}" if chars else ""))
    if "envpack_prompting" in rows_by:
        manifests = [(r.get("trace2env") or {}).get("envpack") or {} for r in rows_by["envpack_prompting"].values()]
        counts = {kind: [len(m.get(kind, [])) for m in manifests] for kind in ("rules", "renderers", "invariants", "notes", "demonstrations", "evidence")}
        print("\nPackage block contents per row (envpack_prompting): " + ", ".join(f"{k} mean {mean(v):.1f} max {max(v)}" for k, v in counts.items())
              + f"; block chars median {median(m.get('block_chars', 0) for m in manifests):.0f} max {max(m.get('block_chars', 0) for m in manifests)}"
              + f"; rows with no evidence {sum(1 for m in manifests if not m.get('evidence'))}; distinct evidence ids {len({e for m in manifests for e in m.get('evidence', [])})}")
        top = Counter(e for m in manifests for e in m.get("evidence", [])).most_common(5)
        print("  most placed evidence: " + ", ".join(f"{e} ({n})" for e, n in top))
        actions = Counter(m.get("canonical_action_type") for m in manifests)
        print("  canonical actions: " + ", ".join(f"{a} {n}" for a, n in actions.most_common(8)))
        # Reconstruction: the block is a deterministic function of (package, row, settings); recompute it for a sample of rows and
        # compare with the manifests recorded at prediction time (ids and size), so any prompt can be rebuilt exactly.
        try:
            sys.path.insert(0, str(ROOT / "src"))
            from trace2env.agentworld import case_from_row
            from trace2env.envpack import envpack_view
            from trace2env.package import EnvironmentPackage, PackageInspector
            inspector = PackageInspector(EnvironmentPackage(str(EXP / "packages" / "v1_20_r=1")))
            sample = list(rows_by["envpack_prompting"].values())[::max(1, len(rows_by["envpack_prompting"]) // 40)]
            exact = 0
            for r in sample:
                m = (r.get("trace2env") or {}).get("envpack") or {}
                block, rebuilt = envpack_view(inspector, case_from_row(r), top_k=m.get("top_k", 6), evidence_chars=m.get("evidence_chars", 3000))
                exact += all(rebuilt.get(k) == m.get(k) for k in ("rules", "renderers", "notes", "demonstrations", "evidence", "query")) and len(block) == m.get("block_chars")
            print(f"  reconstruction check: {exact} of {len(sample)} sampled rows rebuild the identical block (same ids, same size) from the package, the row and the recorded settings")
        except Exception as exc:  # noqa: BLE001
            print(f"  reconstruction check skipped: {exc}")
        print("  reconstruction: each row's `trace2env.envpack` lists the ids placed and the settings; the prompt is the official input plus envpack_view(package, row, settings).")
    if "envpack_prompting" in rows_by:
        rr = [(r.get("trace2env") or {}).get("route") for r in rows_by["envpack_prompting"].values()]
        print(f"\nRoutes (envpack_prompting): {dict(Counter(rr))}; rows with a recorded error: {sum(1 for r in rows_by['envpack_prompting'].values() if (r.get('trace2env') or {}).get('error'))}")


if __name__ == "__main__":
    main()
