#!/usr/bin/env python3
"""The held-out validation set: rows of the AgentWorldBench terminal split whose trajectories are outside the 30-row
development set (its 25 trajectories are excluded entirely, other turns included), sampled at the row level with a
fixed seed and stratified by construction-overlap status (proportion preserved). Writes heldout_rows.jsonl and one
shard per trajectory (heldout_shards/traj_<id>.jsonl) for the per-trajectory runner. Only ids and turn indices are
handled here; row contents are not printed.

    python work/exp-v1_20_r=1/backbones/make_heldout_rows.py --n 100 --seed 20260923
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "work/exp-v1_20_r=1"
sys.path.insert(0, str(EXP / "awb"))
from construction_task_split import CONSTRUCTION_TASK_TRAJECTORIES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--benchmark", default=str(ROOT / "work/agentworldbench/terminal_test.jsonl"))
    ap.add_argument("--dev", default=str(EXP / "backbones/eval30_rows.jsonl"))
    ap.add_argument("--out", default=str(EXP / "backbones/heldout_rows.jsonl"))
    ap.add_argument("--shards", default=str(EXP / "backbones/heldout_shards"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.benchmark, encoding="utf-8") if l.strip()]
    dev_traj = {json.loads(l)["id"] for l in open(args.dev, encoding="utf-8") if l.strip()}
    pool = [r for r in rows if r["id"] not in dev_traj]
    overlap = [r for r in pool if r["id"] in CONSTRUCTION_TASK_TRAJECTORIES]
    clean = [r for r in pool if r["id"] not in CONSTRUCTION_TASK_TRAJECTORIES]
    n_overlap = round(args.n * len(overlap) / len(pool))
    rng = random.Random(args.seed)
    chosen = rng.sample(sorted(overlap, key=lambda r: (r["id"], r["turn_idx"])), n_overlap) + rng.sample(sorted(clean, key=lambda r: (r["id"], r["turn_idx"])), args.n - n_overlap)
    chosen.sort(key=lambda r: (r["id"], r["turn_idx"]))
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    shards = Path(args.shards)
    shards.mkdir(parents=True, exist_ok=True)
    for old in shards.glob("traj_*.jsonl"):
        old.unlink()
    by_traj = {}
    for r in chosen:
        by_traj.setdefault(r["id"], []).append(r)
    for tid, rs in by_traj.items():
        (shards / f"traj_{tid}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
    per = Counter(r["id"] for r in chosen)
    print(f"benchmark {len(rows)} rows / {len({r['id'] for r in rows})} trajectories; development trajectories excluded {len(dev_traj)}; "
          f"pool {len(pool)} rows / {len({r['id'] for r in pool})} trajectories ({len(overlap)} overlap rows in {len({r['id'] for r in overlap})} trajectories)")
    print(f"sampled {len(chosen)} rows (seed {args.seed}): {n_overlap} overlap + {len(chosen) - n_overlap} other, {len(by_traj)} trajectories, "
          f"rows per trajectory max {max(per.values())}, turn_idx mean {sum(r['turn_idx'] for r in chosen) / len(chosen):.1f} max {max(r['turn_idx'] for r in chosen)}")
    print(f"written {args.out} and {len(by_traj)} shards under {shards}")


if __name__ == "__main__":
    main()
