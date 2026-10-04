#!/usr/bin/env python3
"""Trace-scaling table: package statistics, construction cost, evaluation scores, paired differences, inference cost,
as a function of the number of construction traces and turns.

    python work/exp-scaling/analyze_scaling.py [--out work/exp-scaling/report.json]
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

DIMS = ["format", "factuality", "consistency", "realism", "quality"]
PRICE = {"prompt": 2e-6, "completion": 1e-5, "cached": 2e-7}
V120 = Path("work/exp-v1_20_r=1")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(v):
    return (v - 1) / 4 * 100


def package_stats(package):
    root = Path(package)
    counts = {"executable_rules": len(load(root / "rules" / "index.jsonl")), "candidate_rules": len(load(root / "candidates" / "rules.jsonl")),
              "contracts": len(json.load(open(root / "renderer" / "contracts.json"))),
              "invariants": len(json.load(open(root / "invariants.json"))),
              "notes": len(load(root / "knowledge" / "notes.jsonl")), "demonstrations": len(load(root / "demonstrations" / "transitions.jsonl")),
              "evidence": len(load(root / "evidence" / "local_transitions.jsonl")),
              "actions": len(json.load(open(root / "action_schema.json"))["actions"]),
              "state_fields": len(json.load(open(root / "state_schema.json"))["fields"]),
              "rejected": len(json.load(open(root / "exceptions" / "rejected.json"))), "unresolved": len(json.load(open(root / "exceptions" / "unresolved.json")))}
    counts["templated_rules"] = sum(1 for r in load(root / "rules" / "index.jsonl") if r.get("observation_template"))
    return counts


def construction_cost(workspace):
    """Model calls, tokens, and list-price cost over every stage version that fed the current package."""
    total = Counter()
    state = json.load(open(Path(workspace) / "state.json"))["current"]
    for stage, version in state.items():
        inputs = Path(workspace) / "stages" / stage / version / "inputs.json"
        if not inputs.exists():
            continue
        usage = (json.load(open(inputs)).get("usage") or {})
        for key in ("calls", "prompt_tokens", "completion_tokens", "cached_tokens", "cost_usd"):
            total[key] += usage.get(key, 0) or 0
        for path in glob.glob(str(Path(workspace) / "stages" / stage / version / "calls.jsonl")):
            for line in open(path):
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                u = c.get("usage") or {}
                total["log_calls"] += 1
                total["log_prompt"] += int(u.get("prompt_tokens") or 0)
                total["log_completion"] += int(u.get("completion_tokens") or 0)
                total["log_cached"] += int(u.get("cached_tokens") or 0)
    listed = (total["log_prompt"] - total["log_cached"]) * PRICE["prompt"] + total["log_cached"] * PRICE["cached"] + total["log_completion"] * PRICE["completion"]
    return {"model_calls": int(total["log_calls"]), "prompt_tokens": int(total["log_prompt"]), "completion_tokens": int(total["log_completion"]),
            "cost_usd_list": round(listed, 2), "cost_usd_recorded": round(total["cost_usd"], 2)}


def inference_cost(awb_dir):
    total = Counter()
    for path in glob.glob(f"{awb_dir}/calls-*shard*.jsonl"):
        for line in open(path):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = c.get("usage") or {}
            total["calls"] += 1
            total["prompt"] += int(u.get("prompt_tokens") or 0)
            total["completion"] += int(u.get("completion_tokens") or 0)
            total["cached"] += int(u.get("cached_tokens") or 0)
    listed = (total["prompt"] - total["cached"]) * PRICE["prompt"] + total["cached"] * PRICE["cached"] + total["completion"] * PRICE["completion"]
    return {"model_calls": int(total["calls"]), "prompt_tokens": int(total["prompt"]), "completion_tokens": int(total["completion"]), "cost_usd_list": round(listed, 2)}


def paired(a, b, keys):
    d = [a[k]["total_score"] - b[k]["total_score"] for k in keys]
    return {"wins": sum(x > 0 for x in d), "ties": sum(x == 0 for x in d), "losses": sum(x < 0 for x in d),
            "mean": round(mean(d) / 4 * 100, 2), "se": round(pstdev(d) / len(d) ** 0.5 / 4 * 100, 2),
            "dims": {dim: round(mean(a[k][dim] - b[k][dim] for k in keys) / 4 * 100, 2) for dim in DIMS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="work/exp-scaling/report.json")
    args = parser.parse_args()
    manifest = json.load(open("work/exp-scaling/manifest.json"))
    prompting = {(r["id"], r["turn_idx"]): r for r in load(V120 / "awb" / "judged-prompting.jsonl")}
    reference = {(r["id"], r["turn_idx"]): r for r in load(V120 / "awb" / "judged-v3.jsonl")}  # v1_20_r=1 under harness v3
    rows = []
    for name, spec in manifest["packages"].items():
        entry = {"package": name, "traces": spec["trajectories"], "turns": spec["turns"], "tasks": spec["tasks"]}
        package = Path(spec["package"])
        if not (package / "manifest.json").exists():
            entry["status"] = "package missing"
            rows.append(entry)
            continue
        entry["package_stats"] = package_stats(package)
        workspace = V120 if spec["reused_existing_package"] else Path("work/exp-scaling") / name
        entry["construction"] = construction_cost(workspace)
        judged_path = (V120 / "awb" / "judged-v3.jsonl") if spec["reused_existing_package"] else Path("work/exp-scaling") / name / "awb" / "judged.jsonl"
        if not judged_path.exists():
            entry["status"] = "not evaluated"
            rows.append(entry)
            continue
        judged = {(r["id"], r["turn_idx"]): r for r in load(judged_path)}
        valid = [r for r in judged.values() if not r.get("failed")]
        entry["scores"] = {d: round(norm(mean(r[d] for r in valid)), 2) for d in DIMS}
        entry["total"] = round(norm(mean(r["total_score"] for r in valid)), 2)
        entry["valid"] = len(valid)
        keys = [k for k in judged if k in prompting and k in reference and not judged[k].get("failed") and not reference[k].get("failed")]
        entry["vs_prompting"] = paired(judged, prompting, keys)
        entry["vs_v1_20"] = paired(judged, reference, keys) if not spec["reused_existing_package"] else None
        t = [r.get("trace2env") or {} for r in judged.values()]
        entry["routes"] = dict(Counter(x.get("route") for x in t))
        entry["latency_s_per_row"] = round(mean(x.get("latency_seconds") or 0 for x in t), 1)
        entry["tool_calls_per_row"] = round(mean(len(x.get("tool_calls") or []) for x in t), 2)
        entry["inference"] = inference_cost(str(V120 / "awb").replace("awb", "awb") if spec["reused_existing_package"] else str(Path("work/exp-scaling") / name / "awb"))
        if spec["reused_existing_package"]:
            entry["inference"] = inference_cost_v3()
        rows.append(entry)
    json.dump({"packages": rows, "rows_evaluated": 354}, open(args.out, "w"), indent=1)
    print(f"{'package':<10}{'traces':>7}{'turns':>7}{'evid':>6}{'rules':>7}{'contr':>6}{'notes':>6}{'demos':>6}{'build$':>8}{'TOTAL':>8}{'vs prompt':>14}{'vs v1_20':>14}{'infer$/row':>11}{'s/row':>7}")
    for e in rows:
        if "total" not in e:
            print(f"{e['package']:<10}{e['traces']:>7}{e['turns']:>7}  {e.get('status')}")
            continue
        ps, c = e["package_stats"], e["construction"]
        vp, vr = e["vs_prompting"], e["vs_v1_20"]
        print(f"{e['package']:<10}{e['traces']:>7}{e['turns']:>7}{ps['evidence']:>6}{ps['executable_rules']:>3}/{ps['candidate_rules']:<3}{ps['contracts']:>6}{ps['notes']:>6}{ps['demonstrations']:>6}"
              f"{c['cost_usd_list']:>8.2f}{e['total']:>8.2f}{vp['mean']:>+8.2f}±{vp['se']:<5}{(f'{vr['mean']:+.2f}±{vr['se']}' if vr else '—'):>14}{e['inference']['cost_usd_list']/354:>11.3f}{e['latency_s_per_row']:>7.1f}")
    print("report written to", args.out)


def inference_cost_v3():
    return inference_cost(str(V120 / "awb")) if False else _v3_cost()


def _v3_cost():
    total = Counter()
    for path in glob.glob(str(V120 / "awb" / "calls-v3-shard*.jsonl")):
        for line in open(path):
            try:
                c = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = c.get("usage") or {}
            total["calls"] += 1
            total["prompt"] += int(u.get("prompt_tokens") or 0)
            total["completion"] += int(u.get("completion_tokens") or 0)
            total["cached"] += int(u.get("cached_tokens") or 0)
    listed = (total["prompt"] - total["cached"]) * PRICE["prompt"] + total["cached"] * PRICE["cached"] + total["completion"] * PRICE["completion"]
    return {"model_calls": int(total["calls"]), "prompt_tokens": int(total["prompt"]), "completion_tokens": int(total["completion"]), "cost_usd_list": round(listed, 2)}


if __name__ == "__main__":
    main()
