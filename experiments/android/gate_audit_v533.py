#!/usr/bin/env python3
"""Knowledge-gate audit for a cross-fitted run: the terminal audit script takes one package, so it is run once per
package over the trajectories mapped to it (their calls, rows and judged rows), and the 15 reports are merged.

    python work/exp-android-cv/gate_audit_v533.py --backbone gpt-5.6-sol [--label trace2env_v533]
"""
import argparse, json, os, subprocess, sys, tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "work/exp-android-cv"
AUDIT = ROOT / "work/exp-v1_20_r=1/backbones/knowledge_gate_audit.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--label", default="trace2env_v533")
    args = ap.parse_args()
    H = EXP / "backbones" / args.backbone / "full200"
    package_map = json.load(open(EXP / "package_map.json"))
    by_package = defaultdict(list)
    for uid, pkg in package_map.items():
        by_package[pkg].append(uid)
    rows = [json.loads(l) for l in open(EXP / "full200_rows.jsonl", encoding="utf-8") if l.strip()]
    judged_path = H / f"judged-{args.label}.jsonl"
    judged = [json.loads(l) for l in open(judged_path, encoding="utf-8") if l.strip()] if judged_path.exists() else []
    merged = {"agent_calls": 0, "package_items_seen": 0, "sanitized_items": 0, "supporting_items": 0, "judged_items_at_boundary": 0,
              "labels_at_boundary": Counter(), "dispositions_at_boundary": Counter(), "problems": 0, "problem_examples": [],
              "rows": {"rows": 0, "rows_with_gate_record": 0, "abstained_rows": 0, "rows_with_supporting": 0, "items_total": 0,
                       "labels": Counter(), "dispositions": Counter(), "masked_tokens_total": 0}, "per_package": {}}
    with tempfile.TemporaryDirectory(prefix="gate-audit-") as tmp:
        for pkg, uids in sorted(by_package.items()):
            d = Path(tmp) / pkg.replace("/", "_"); d.mkdir()
            for uid in uids:
                src = H / f"calls-{args.label}-{uid}.jsonl"
                if src.exists():
                    os.symlink(src, d / src.name)
            with open(d / "rows.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    if r["id"] in uids:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            with open(d / "judged.jsonl", "w", encoding="utf-8") as f:
                for r in judged:
                    if str(r["id"]) in uids:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out = d / "report.json"
            cmd = [sys.executable, str(AUDIT), "--rows", str(d / "rows.jsonl"), "--calls", str(d / f"calls-{args.label}-*.jsonl"),
                   "--judged", str(d / "judged.jsonl"), "--package", str(EXP / pkg), "--out", str(out)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                print(f"{pkg}: audit failed: {proc.stderr[-500:]}")
                continue
            rep = json.load(open(out))
            merged["per_package"][pkg] = {k: rep.get(k) for k in ("agent_calls", "package_items_seen", "supporting_items", "problems")}
            for k in ("agent_calls", "package_items_seen", "sanitized_items", "supporting_items", "judged_items_at_boundary", "problems"):
                merged[k] += rep.get(k) or 0
            merged["labels_at_boundary"].update(rep.get("labels_at_boundary") or {})
            merged["dispositions_at_boundary"].update(rep.get("dispositions_at_boundary") or {})
            merged["problem_examples"].extend(rep.get("problem_examples") or [])
            rs = rep.get("rows") or {}
            for k in ("rows", "rows_with_gate_record", "abstained_rows", "rows_with_supporting", "masked_tokens_total"):
                merged["rows"][k] += rs.get(k) or 0
            merged["rows"]["items_total"] += round((rs.get("items_per_row") or 0) * (rs.get("rows_with_gate_record") or 0))
            merged["rows"]["labels"].update(rs.get("labels") or {})
            merged["rows"]["dispositions"].update(rs.get("dispositions") or {})
    merged["rows"]["items_per_row"] = round(merged["rows"]["items_total"] / max(1, merged["rows"]["rows_with_gate_record"]), 1)
    merged["problem_examples"] = merged["problem_examples"][:12]
    for k in ("labels_at_boundary", "dispositions_at_boundary"):
        merged[k] = dict(merged[k])
    merged["rows"]["labels"] = dict(merged["rows"]["labels"]); merged["rows"]["dispositions"] = dict(merged["rows"]["dispositions"])
    out = H / f"gate-audit-{args.label}.json"
    out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in merged.items() if k != "per_package"}, indent=1)); print("written", out)


if __name__ == "__main__":
    main()
