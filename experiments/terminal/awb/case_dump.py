#!/usr/bin/env python3
"""Dump one judged row across all systems: truth, preceding turns, each system's prediction, judge dims and remarks.

    python case_dump.py --out DIR ID:TURN [ID:TURN ...]
"""
import argparse, json, os, re
from mine_cases import SYSTEMS, DIMS, load, norm, keystrokes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--digest", action="store_true", help="print a compact digest")
    args = ap.parse_args()
    keys = [(int(k.split(":")[0]), int(k.split(":")[1])) for k in args.keys]
    systems = {}
    for name, path in SYSTEMS.items():
        systems[name] = {(r["id"], r["turn_idx"]): r for r in load(path) if (r["id"], r["turn_idx"]) in set(keys)}
    os.makedirs(args.out, exist_ok=True)
    for key in keys:
        base = systems["v3"][key]
        case = {"id": key[0], "turn": key[1], "total_turns": base["total_turns"], "system_str": base["system_str"],
                "history": [{"turn": i, "action": keystrokes(p), "prompt": p, "observation": r}
                            for i, (p, r) in enumerate(zip(base["prompt"][:-1], base["response"][:-1]))],
                "current_prompt": base["current_prompt"], "current_keystrokes": keystrokes(base["current_prompt"]),
                "truth": base["response"][-1], "systems": {}}
        for name, rows in systems.items():
            r = rows.get(key)
            if not r:
                continue
            case["systems"][name] = {"score": norm(r["total_score"]) if not r.get("failed") else None,
                                     "dims": {d: r.get(d) for d in DIMS}, "prediction": r.get("extracted_output") or r.get("gen"),
                                     "strengths": r.get("strengths"), "weaknesses": r.get("weaknesses"), "trace2env": r.get("trace2env")}
        path = os.path.join(args.out, f"{key[0]}_{key[1]}.json")
        json.dump(case, open(path, "w"), indent=1)
        if args.digest:
            print(f"\n{'#'*100}\n# {key[0]} turn {key[1]}/{case['total_turns']}   action: {case['current_keystrokes'][:300]!r}")
            print(f"# history actions: {[h['action'][:60] for h in case['history']]}")
            print(f"--- TRUTH ({len(case['truth'])} chars):\n{case['truth'][:900]}")
            for name in ("prompting", "rag", "harness", "schema", "examples", "nostate", "v3"):
                s = case["systems"].get(name)
                if not s:
                    continue
                print(f"--- {name} {s['score']:.0f} {s['dims']} :: {(s['prediction'] or '')[:500]!r}")
                if s["weaknesses"]:
                    print(f"    judge weaknesses: {' | '.join(w[:160] for w in s['weaknesses'][:2])}")


if __name__ == "__main__":
    main()
