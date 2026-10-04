#!/usr/bin/env python3
"""Lexical audit of benchmark trajectories against the 20 construction traces.

The manifest's overlap labels came from instruction files and workspace paths; tasks without distinctive files were
missed. Here every benchmark trajectory (its prompts, real observations, and system text) is compared with every
construction episode by the number of shared rare tokens (tokens in at most 2 construction episodes and at most 3
benchmark trajectories). Prints the top matches per trajectory; the final same-task list is confirmed by hand.
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict

TOK = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{5,}")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eps = {}
    for f in glob.glob(os.path.join(root, "traces/reconstruction/*.json")):
        eps[os.path.basename(f).split("__")[0]] = set(t.lower() for t in TOK.findall(open(f, encoding="utf-8").read()))
    traj, rows_per = defaultdict(str), Counter()
    for line in open(os.path.join(root, "awb/judged-v3.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows_per[r["id"]] += 1
        if len(traj[r["id"]]) < 400000:
            traj[r["id"]] += " ".join(r["prompt"]) + " " + " ".join(r["response"]) + " " + r["system_str"]
    btok = {k: set(t.lower() for t in TOK.findall(v)) for k, v in traj.items()}
    df_c, df_b = Counter(), Counter()
    for s in eps.values():
        df_c.update(s)
    for s in btok.values():
        df_b.update(s)
    rep = json.load(open(os.path.join(root, "awb/report-all-systems.json")))
    level = {int(t["trajectory"]): (t.get("match_level"), t.get("tb2_task_match")) for t in rep["per_trajectory"]}
    out = []
    for b, bs in btok.items():
        scores = []
        for c, cs in eps.items():
            shared = [t for t in bs & cs if df_c[t] <= 2 and df_b[t] <= 3]
            files = [t for t in shared if "." in t and not t.endswith((".deb", ".gz", ".so"))]
            scores.append((len(shared), len(files), c, sorted(files)[:5]))
        scores.sort(reverse=True)
        out.append((b, rows_per[b], level.get(b), scores[:2]))
    out.sort(key=lambda x: -x[3][0][0])
    for b, n, lv, top in out:
        print(f"{b:>16} rows={n:3d} {str(lv):48s} " + " | ".join(f"{c}:{s}/{f} {files}" for s, f, c, files in top))


if __name__ == "__main__":
    main()
