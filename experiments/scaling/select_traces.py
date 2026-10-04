#!/usr/bin/env python3
"""Nested construction sets for the trace-scaling experiment (seeded order over the v1_20_r=1 traces).

Eligibility is inherited from work/exp-v1_20_r=1/manifest.json: one finished, exception-free, reward-1.0 trial per task
after excluding the 25 tasks with a strong AgentWorldBench overlap (36 tasks). The 20 tasks already used by v1_20_r=1
are shuffled once with a fixed seed; v1_1 / v1_5 / v1_10 are prefixes of that order, v1_20 is all 20 (the existing
package), v1_36 is all 36 eligible tasks (the 20 plus the 16 remaining ones, appended in seeded order).
"""
import json
import random
from pathlib import Path

SEED = 20260920
base = json.load(open("work/exp-v1_20_r=1/manifest.json"))
selected = {item["task"]: item for item in base["selected"]}
pool = {item["task"]: item for item in base["candidate_pool"]}
assert set(selected) <= set(pool) and len(pool) == 36
order20 = sorted(selected)
random.Random(SEED).shuffle(order20)
extra = sorted(set(pool) - set(selected))
random.Random(SEED + 1).shuffle(extra)
order36 = order20 + extra
sets = {"v1_1_r=1": order20[:1], "v1_5_r=1": order20[:5], "v1_10_r=1": order20[:10], "v1_15_r=1": order20[:15], "v1_20_r=1": order20,
        "v1_25_r=1": order36[:25], "v1_30_r=1": order36[:30], "v1_36_r=1": order36}
# A second 1-trace package from a different seed (a different trace and a different induction sample), to see how much a
# single-trace point moves; the first seed whose leading task differs from v1_1's is used.
SEED2 = SEED + 1
while True:
    order20_s2 = sorted(selected)
    random.Random(SEED2).shuffle(order20_s2)
    if order20_s2[0] != order20[0]:
        break
    SEED2 += 1
sets["v1_1s2_r=1"] = order20_s2[:1]


def entry(task):
    item = pool[task]
    return {"task": task, "trial": item["trial"], "turns": item["turns"], "reward": item["reward"],
            "collection_episode": f"work/tb2/episodes/{task}__{item['trial']}.json", "trajectory_path": item["trajectory_path"],
            "trial_dir": item["trial_dir"], "in_v1_20": task in selected}


existing = json.load(open("work/exp-scaling/manifest.json")) if Path("work/exp-scaling/manifest.json").exists() else {}
manifest = {
    "experiment": "trace-scaling", "created": "2026-09-20", "seed": SEED, "seed2_for_v1_1s2": SEED2,
    "eligibility": base["sampling"], "excluded_overlap_tasks": sorted(base["excluded_overlap_tasks"]),
    "order_of_the_20": order20, "order_of_the_16_extra": extra, "order_of_the_20_seed2": order20_s2,
    "packages": {},
}
for name, tasks in sets.items():
    if name in (existing.get("packages") or {}):
        manifest["packages"][name] = existing["packages"][name]  # keep prepared records (digests, origins) untouched
        continue
    manifest["packages"][name] = {"tasks": tasks, "trajectories": len(tasks), "turns": sum(pool[t]["turns"] for t in tasks),
                                  "traces": [entry(t) for t in tasks],
                                  "package": ("work/exp-v1_20_r=1/packages/v1_20_r=1" if name == "v1_20_r=1" else f"work/exp-scaling/{name}/packages/{name}"),
                                  "reused_existing_package": name == "v1_20_r=1"}
for key in ("trace_preparation",):
    if key in existing:
        manifest[key] = existing[key]
Path("work/exp-scaling/manifest.json").write_text(json.dumps(manifest, indent=1))
for name, spec in manifest["packages"].items():
    print(f"{name}: {spec['trajectories']} traces, {spec['turns']} turns: {', '.join(spec['tasks'][:6])}{' ...' if len(spec['tasks']) > 6 else ''}")
