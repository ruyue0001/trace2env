#!/usr/bin/env python3
"""Official prompting baselines across backbones (and the Trace2Env systems they are compared with), with paired and
trajectory-clustered statistics.  python compare_prompting_baselines.py > backbones/prompting-baselines.md"""
import json, random, re, sys
from pathlib import Path
from statistics import mean, pstdev
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "work/exp-v1_20_r=1/awb"))
from cluster_bootstrap import analyse  # noqa: E402
DIMS = ["format", "factuality", "consistency", "realism", "quality"]
ENT = re.compile(r"&(gt|lt|amp|quot);")
FILES = {
    "GPT-5.6-Sol prompting": ROOT / "work/exp-v1_20_r=1/awb/judged-prompting.jsonl",
    "DeepSeek V4.1-Flash prompting (low)": ROOT / "work/exp-v1_20_r=1/backbones/deepseek-v4.1-flash/judged-prompting.jsonl",
    "DeepSeek V4-Pro prompting (low)": ROOT / "work/exp-v1_20_r=1/backbones/deepseek-v4-pro/judged-prompting.jsonl",
    "DeepSeek V4-Pro prompting (thinking default, 65k cap)": ROOT / "work/exp-v1_20_r=1/backbones/deepseek-v4-pro-think/judged-prompting.jsonl",
    "GPT-5.6-Sol Trace2Env v3": ROOT / "work/exp-v1_20_r=1/awb/judged-v3.jsonl",
    "DeepSeek V4.1-Flash Trace2Env v3 (low)": ROOT / "work/exp-v1_20_r=1/backbones/deepseek-v4.1-flash/judged-v3.jsonl",
}
PRICE = {"GPT-5.6-Sol": (2e-6, 1e-5), "DeepSeek V4.1-Flash": (0.15e-6, 0.6e-6), "DeepSeek V4-Pro": (0.422e-6, 0.845e-6)}
norm = lambda v: (v - 1) / 4 * 100  # noqa: E731
sets = {}
print("| System | valid rows | " + " | ".join(DIMS) + " | total | outputs with HTML entities | reasoning tok/row | latency s/row | $/row |")
print("|---|---|" + "---|" * len(DIMS) + "---|---|---|---|---|")
for name, path in FILES.items():
    if not path.exists():
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    valid = {(r["id"], r["turn_idx"]): r for r in rows if not r.get("failed")}
    sets[name] = {k: norm(r["total_score"]) for k, r in valid.items()}
    ent = sum(1 for r in valid.values() if ENT.search(r.get("extracted_output") or ""))
    us = [(r.get("trace2env") or {}).get("usage") or {} for r in rows]
    rt = mean(u.get("reasoning_tokens") or 0 for u in us)
    lat = mean((r.get("trace2env") or {}).get("latency_seconds") or 0 for r in rows)
    price = next(p for k, p in PRICE.items() if name.startswith(k))
    cost = mean((u.get("prompt_tokens") or 0) * price[0] + (u.get("completion_tokens") or 0) * price[1] for u in us) if any(u.get("prompt_tokens") for u in us) else float("nan")
    print(f"| {name} | {len(valid)}/{len(rows)} | " + " | ".join(f"{mean(norm(r[d]) for r in valid.values()):.1f}" for d in DIMS)
          + f" | **{mean(sets[name].values()):.2f}** | {ent} | {rt:.0f} | {lat:.0f} | {cost:.4f} |")
print("\nPaired comparisons (rows valid in both; row mean ± se; trajectory-clustered 95 % CI; trajectory W/T/L; sign-flip p):")
names = list(sets)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
for a, b in pairs:
    keys = [k for k in sets[a] if k in sets[b]]
    d = [sets[a][k] - sets[b][k] for k in keys]
    res = analyse(sets[a], sets[b], lambda t: True, random.Random(f"20260921|{a}|{b}"), 10000, 20000)
    print(f"- {a} − {b}: {mean(d):+.2f} ± {pstdev(d)/len(d)**0.5:.2f} (n={len(keys)}); CI [{res['ci95_rows'][0]:+.2f}, {res['ci95_rows'][1]:+.2f}]; "
          f"traj {res['trajectory_wins']}/{res['trajectory_ties']}/{res['trajectory_losses']}; p={res['sign_flip_p_two_sided']}")
