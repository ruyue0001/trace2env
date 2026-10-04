#!/usr/bin/env python3
"""Re-split the benchmark rows by whether the trajectory's task is one of the 20 construction tasks.

The list below was confirmed by hand from overlap_audit.py (shared task files such as decomp.c, sequences.fasta,
warriors/*.red, plus_comm.v, dist.npy, eigen.py, app.py + sentiment_model, /git/project). Prints per-system totals and
paired differences for construction-task rows, all other rows, and the manifest's overlap levels with the
construction-task trajectories removed.
"""
import json
from statistics import mean, pstdev
from mine_cases import SYSTEMS, load, norm

CONSTRUCTION_TASK_TRAJECTORIES = {
    104971650552388: "write-compressor", 210448203382431: "largest-eigenval", 177623154441707: "dna-assembly",
    126048951621424: "winning-avg-corewars", 272823365904851: "git-multibranch", 30043326384016: "hf-model-inference",
    208104038662149: "hf-model-inference", 255969445734706: "prove-plus-comm", 77632686965927: "prove-plus-comm",
    111876641555338: "distribution-search",
}


def paired(a, b, keys):
    d = [norm(a[k]["total_score"]) - norm(b[k]["total_score"]) for k in keys]
    return f"{mean(d):+.2f} ± {pstdev(d) / len(d) ** 0.5:.2f} (W/T/L {sum(x > 0 for x in d)}/{sum(x == 0 for x in d)}/{sum(x < 0 for x in d)})"


def main():
    systems = {k: {(r["id"], r["turn_idx"]): r for r in load(v)} for k, v in SYSTEMS.items()}
    keys = [k for k in systems["v3"] if all(k in s and not s[k].get("failed") for s in systems.values())]
    rep = json.load(open("report-all-systems.json"))
    level = {int(t["trajectory"]): t.get("match_level") for t in rep["per_trajectory"]}
    groups = {
        "construction-task (10 traj)": [k for k in keys if k[0] in CONSTRUCTION_TASK_TRAJECTORIES],
        "all other rows": [k for k in keys if k[0] not in CONSTRUCTION_TASK_TRAJECTORIES],
    }
    for lv in ("none", "probable", "strong"):
        groups[f"level={lv} (manifest)"] = [k for k in keys if level.get(k[0]) == lv]
        groups[f"level={lv} minus construction-task"] = [k for k in keys if level.get(k[0]) == lv and k[0] not in CONSTRUCTION_TASK_TRAJECTORIES]
    order = ["prompting", "rag", "harness", "schema", "structure", "examples", "nostate", "sshot", "raw", "v2", "v3"]
    print(f"{'group':40s}{'rows':>5} " + "".join(f"{n:>9}" for n in order) + "   v3-prompting            examples-schema         v3-examples")
    for name, ks in groups.items():
        if not ks:
            continue
        traj = len({k[0] for k in ks})
        totals = "".join(f"{mean(norm(systems[n][k]['total_score']) for k in ks):9.1f}" for n in order)
        print(f"{name:40s}{len(ks):>5} {totals}   {paired(systems['v3'], systems['prompting'], ks)}   {paired(systems['examples'], systems['schema'], ks)}   {paired(systems['v3'], systems['examples'], ks)}   [{traj} traj]")
    print("\nconstruction-task trajectories, v3 vs prompting per trajectory:")
    for t, task in CONSTRUCTION_TASK_TRAJECTORIES.items():
        ks = [k for k in keys if k[0] == t]
        print(f"  {t:>16} {task:22s} rows={len(ks)} manifest={level.get(t)} " + " ".join(f"{n}={mean(norm(systems[n][k]['total_score']) for k in ks):.0f}" for n in ("prompting", "rag", "harness", "schema", "examples", "v3")))


if __name__ == "__main__":
    main()
