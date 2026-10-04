#!/usr/bin/env python3
"""Recover a world-model agent's transcript for one benchmark row from the call logs (no model calls).

    python case_transcript.py --label v3 --out DIR ID:TURN [...]

Matches runtime_agent_turn calls by the brief's context.turn and raw_action keystrokes; keeps the longest conversation
(the final call holds the brief, every tool call and result, and the submission).
"""
import argparse, glob, json, os
from mine_cases import SYSTEMS, load, keystrokes

CALL_LABELS = {"harness": "harness_only_hv3", "schema": "schema_only_hv3", "nostate": "no_state_hv3", "examples": "examples_only",
               "structure": "structure_only", "sshot": "single_shot_hv3", "raw": "raw_traces", "v2": "v2", "v3": "v3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("keys", nargs="+")
    args = ap.parse_args()
    keys = {(int(k.split(":")[0]), int(k.split(":")[1])) for k in args.keys}
    rows = {(r["id"], r["turn_idx"]): r for r in load(SYSTEMS[args.label]) if (r["id"], r["turn_idx"]) in keys}
    want = {}
    for key, r in rows.items():
        want.setdefault((key[1], keystrokes(r["current_prompt"])), []).append(key)
    best = {}
    pattern = f"calls-{CALL_LABELS[args.label]}-shard*.jsonl"
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if '"runtime_agent_turn"' not in line[:200] and '"runtime_single_shot"' not in line[:200]:
                    continue
                d = json.loads(line)
                if d["role"] not in ("runtime_agent_turn", "runtime_single_shot"):
                    continue
                try:
                    user = json.loads(d["user"])
                    conv = user.get("conversation") or [{"role": "user", "content": d["user"]}]
                    brief = json.loads(conv[0]["content"]) if isinstance(conv[0]["content"], str) else conv[0]["content"]
                    ctx = brief.get("context", {})
                    turn, raw = ctx.get("turn"), ctx.get("raw_action", "")
                except Exception:
                    continue
                k = (turn, keystrokes(raw))
                if k not in want:
                    continue
                for key in want[k]:
                    r = rows[key]
                    # disambiguate trajectories that share (turn, action) by the initial screen text
                    init = (r["prompt"][0] if r["turn_idx"] > 0 else r["current_prompt"])
                    mem = json.dumps(brief.get("recent_memory", []))[:4000]
                    screen = ctx.get("current_input", {}).get("current_state", "")
                    if screen and screen not in init and screen not in json.dumps(r["prompt"]):
                        continue
                    entry = {"path": path, "sequence": d["sequence"], "conversation": conv, "response": d["response"],
                             "usage": d.get("usage"), "elapsed_s": d.get("elapsed_s"), "conv_chars": len(d["user"])}
                    if key not in best or entry["conv_chars"] > best[key]["conv_chars"]:
                        best[key] = entry
    os.makedirs(args.out, exist_ok=True)
    for key, entry in best.items():
        path = os.path.join(args.out, f"{key[0]}_{key[1]}_{args.label}_transcript.json")
        json.dump(entry, open(path, "w"), indent=1)
        brief = json.loads(entry["conversation"][0]["content"])
        print(f"\n=== {key} {args.label}: {len(entry['conversation'])} messages, {entry['conv_chars']} chars ({entry['path']} seq {entry['sequence']})")
        print("brief keys:", list(brief))
        print("state_summary:", json.dumps(brief.get("state_summary"))[:700])
        for k in ("similar_turns", "knowledge", "retrieved", "inspection", "evidence", "demonstrations", "trace_hits", "hints"):
            if k in brief:
                v = brief[k]
                ids = [x.get("id") for x in v] if isinstance(v, list) and v and isinstance(v[0], dict) else None
                print(f"{k}: {ids if ids else json.dumps(v)[:400]}")
        for m in entry["conversation"][1:]:
            c = m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
            print(f"[{m['role']}] {c[:700]!r}")
        print("FINAL:", json.dumps(entry["response"])[:1500])
    missing = keys - set(best)
    if missing:
        print("no transcript found for", missing)


if __name__ == "__main__":
    main()
