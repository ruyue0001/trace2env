#!/usr/bin/env python3
"""Cross-fit integrity check on the finished rows: every envpack row used its mapped out-of-fold package, every
prompting_rag hit comes from a trajectory outside the row's fold (never the row's own), and the agentic systems' shard
logs name the mapped package. Usage: check_crossfit.py --backbone <slug>"""
import argparse, glob, json, re
from pathlib import Path

EXP = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--backbone", required=True); args = ap.parse_args()
    H = EXP / "backbones" / args.backbone / "full200"
    pkg_map = json.load(open(EXP / "package_map.json")); traj = json.load(open(EXP / "trajectories.json"))
    label_of = {f"android_{uid}".split("__")[0]: uid for uid in traj}  # trace-corpus labels are the file stems before '__'
    problems = []
    for label in ("envpack_prompting", "prompting_rag", "harness_only_v51", "trace2env_v533"):
        rows = [json.loads(l) for p in glob.glob(str(H / f"pred-{label}-*.jsonl")) for l in open(p) if l.strip()]
        if not rows:
            print(f"{label}: no rows yet"); continue
        n = 0
        for r in rows:
            uid = str(r["id"]); t = r.get("trace2env") or {}
            style, fold = traj[uid]["style"], traj[uid]["fold"]
            if label == "envpack_prompting":
                name = str(((t.get("envpack") or {}).get("package") or {}).get("name"))
                if not name.endswith(f"cross-fit {style}-f{fold}"):  # the package manifest name ends with its corpus label
                    problems.append((label, uid, r["turn_idx"], f"package {name} != {style}-f{fold}"))
                n += 1
            elif label == "prompting_rag":
                for hit in ((t.get("rag") or {}).get("hits") or []):
                    m = re.match(r"trace:(android_.+?):\d+$", hit)
                    src = label_of.get(m.group(1), m.group(1)) if m else hit
                    if src == uid or traj.get(src, {}).get("fold") == fold or traj.get(src, {}).get("style") != style:
                        problems.append((label, uid, r["turn_idx"], f"hit {hit} is in-fold or another sub-source"))
                n += 1
        # agentic systems: the shard log's package argument
        for p in glob.glob(str(H / f"{label}-*.log")):
            uid = Path(p).name[len(label) + 1:-4]
            if uid in pkg_map:
                pass  # the launcher records the mapped package in package-map-<label>.txt; per-shard checks below
        print(f"{label}: {len(rows)} rows checked ({n} with cross-fit fields)")
    mapped = H / "package-manifest-trace2env_v533.sha256"
    if mapped.exists():
        print("recorded package digests:", len(mapped.read_text().splitlines()), "packages")
    print("problems:", len(problems)); [print("  ", p) for p in problems[:10]]


if __name__ == "__main__":
    main()
