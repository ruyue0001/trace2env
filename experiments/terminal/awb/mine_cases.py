#!/usr/bin/env python3
"""Screen judged rows for qualitative case studies: join every system by (id, turn_idx), tag each row by
what the turn needs (same slice logic as analyze_structure.py), and list rows that fit a requested pattern.

    python mine_cases.py --pattern ladder      # prompting wrong, +rag wrong, harness_only partial, v3 right
    python mine_cases.py --pattern failure     # v3 clearly below prompting
"""
import argparse, json, re, sys
from statistics import mean

SYSTEMS = {  # label -> judged file
    "prompting": "judged-prompting.jsonl", "rag": "judged-prompting+rag.jsonl",
    "harness": "judged-harness_only_hv3.jsonl", "schema": "judged-schema_only_hv3.jsonl",
    "structure": "judged-structure_only.jsonl", "examples": "judged-examples_only.jsonl",
    "nostate": "judged-no_state_hv3.jsonl", "sshot": "judged-single_shot_hv3.jsonl",
    "raw": "judged-raw_traces.jsonl", "v2": "judged-v2.jsonl", "v3": "judged-v3.jsonl",
}
DIMS = ["format", "factuality", "consistency", "realism", "quality"]
WRITE_RE = re.compile(r"(?:cat\s*>+\s*|tee\s+(?:-a\s+)?|touch\s+|mkdir\s+(?:-p\s+)?|cp\s+\S+\s+|mv\s+\S+\s+|>\s*|>>\s*)([\w./~-]+)")
READ_PROGRAMS = ("cat", "head", "tail", "less", "more", "ls", "python3", "python", "bash", "sh", "node", "wc", "grep", "diff", "cd", "stat",
                 "file", "sha256sum", "md5sum", "chmod", "chown", "rm", "make", "gcc", "g++", "javac", "java", "cargo", "go", "sed", "awk", "sort")
ERROR_RE = re.compile(r"No such file or directory|command not found|cannot access|not a directory|Permission denied|ModuleNotFoundError|"
                      r"FileNotFoundError|No module named|is not recognized|does not exist|not found", re.I)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(v):
    return (v - 1) / 4 * 100


def actions(prompt):
    m = re.search(r"```json\s*(\[.*?\])\s*```", prompt, re.S)
    try:
        return json.loads(m.group(1)) if m else []
    except json.JSONDecodeError:
        return []


def keystrokes(prompt):
    return "".join(a.get("keystrokes", "") for a in actions(prompt))


def kind(prompt):
    acts = actions(prompt)
    text = "".join(a.get("keystrokes", "") for a in acts)
    if not text.strip():
        return "wait"
    if "<<" in text and "EOF" in text.upper():
        return "heredoc"
    if len([a for a in acts if a.get("keystrokes", "").strip()]) > 1:
        return "batch"
    if text.strip().startswith("C-") or "\x03" in text:
        return "keys"
    return "single"


def tags_for(row):
    history = [keystrokes(p) for p in row["prompt"][:-1]]
    current = keystrokes(row["current_prompt"])
    written = set()
    for text in history:
        for m in WRITE_RE.finditer(text):
            name = m.group(1).strip("'\"")
            if name and not name.startswith("-") and name != "/dev/null":
                written.add(name.rsplit("/", 1)[-1])
    tokens = set(re.findall(r"[\w./~-]+", current))
    out = set()
    if any(n in tokens for n in written) and any(current.strip().startswith(p) or f" {p} " in f" {current} " for p in READ_PROGRAMS):
        out.add("state_read")
    if ERROR_RE.search(row["response"][-1]):
        out.add("state_error")
    if any(re.search(r"(^|[;&|]\s*)cd\s", t) for t in history):
        out.add("after_cd")
    if row["turn_idx"] >= 31:
        out.add("long_horizon")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="ladder", choices=["ladder", "failure", "all"])
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    systems = {k: {(r["id"], r["turn_idx"]): r for r in load(v)} for k, v in SYSTEMS.items()}
    keys = [k for k in systems["v3"] if all(k in s for s in systems.values())]
    rows = []
    for k in keys:
        sc = {name: (norm(s[k]["total_score"]) if not s[k].get("failed") else None) for name, s in systems.items()}
        fact = {name: (s[k]["factuality"] if not s[k].get("failed") else None) for name, s in systems.items()}
        r = systems["v3"][k]
        info = r.get("trace2env") or {}
        cites = info.get("citations") or []
        rows.append({"id": k[0], "turn": k[1], "total_turns": r["total_turns"], "kind": kind(r["current_prompt"]),
                     "tags": sorted(tags_for(r)), "scores": sc, "fact": fact,
                     "keys": keystrokes(r["current_prompt"])[:90].replace("\n", "⏎"),
                     "truth_len": len(r["response"][-1]), "v3_cites": cites, "v3_tools": info.get("tool_calls"),
                     "v3_route": info.get("route")})
    def ok(x):
        s = x["scores"]
        if any(s[n] is None for n in ("prompting", "rag", "harness", "v3")):
            return False
        if args.pattern == "ladder":
            return s["prompting"] <= 45 and s["rag"] <= 55 and s["harness"] < s["v3"] and s["v3"] >= 70 and s["v3"] - s["prompting"] >= 25
        if args.pattern == "failure":
            return s["v3"] <= 40 and s["prompting"] - s["v3"] >= 25
        return True
    picked = [x for x in rows if ok(x)]
    key = (lambda x: -(x["scores"]["v3"] - x["scores"]["prompting"])) if args.pattern != "failure" else (lambda x: -(x["scores"]["prompting"] - x["scores"]["v3"]))
    picked.sort(key=key)
    print(f"{len(picked)} rows match pattern {args.pattern} of {len(rows)} joined rows")
    names = list(SYSTEMS)
    print("id turn/total kind tags | " + " ".join(f"{n:>6}" for n in names) + " | v3 cites | keys")
    for x in picked[: args.limit]:
        cites = {}
        for c in x["v3_cites"]:
            cites[c.split(":")[0]] = cites.get(c.split(":")[0], 0) + 1
        print(f"{x['id']} {x['turn']}/{x['total_turns']} {x['kind']:7s} {','.join(x['tags']) or '-':38s} | "
              + " ".join(f"{(x['scores'][n] if x['scores'][n] is not None else -1):6.0f}" for n in names)
              + f" | {cites} {x['v3_tools']} | {x['keys']}")
    json.dump(picked, open(f"cases-{args.pattern}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
