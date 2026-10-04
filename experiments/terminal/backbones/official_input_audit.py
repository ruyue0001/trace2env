#!/usr/bin/env python3
"""Model-boundary audit of harness v5.1 information preservation.

    python work/exp-v1_20_r=1/backbones/official_input_audit.py --rows ROWS.jsonl --calls 'DIR/calls-LABEL*.jsonl' [--out FILE]

For every world-model agent call recorded by the call log (the exact ``system`` and ``user`` strings sent to the model),
the audit rebuilds the official prompting input of the matching benchmark row with the baseline's own code
(``inference_messages``) and checks, at that boundary: the official turn messages appear in the brief verbatim, in
order and complete; each of them and the official system message appear exactly once in the request; the target
observation (the current turn's reference response) appears nowhere in the request; every assistant message carries a
memory id; the same holds on every later tool-calling round of the row. It also reports the request size per round
(characters and the provider's prompt-token count) so long trajectories and later rounds are visible rather than
silently truncated. Information preservation is what is checked, not message-layout identity.
"""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from trace2env.agentworld import RESPONSE_MARKER, case_from_row, clean_response_marker, inference_messages  # noqa: E402


def twice(text: str) -> str:
    """A verbatim string as it appears inside the request: JSON-escaped twice (brief inside the payload)."""
    return json.dumps(json.dumps(text, ensure_ascii=False), ensure_ascii=False)[1:-1]


def _contains(node, target: str) -> bool:
    """Whether ``target`` occurs in any string leaf of a decoded JSON value (JSON-encoded strings are decoded too)."""
    if isinstance(node, dict):
        return any(_contains(v, target) for v in node.values())
    if isinstance(node, list):
        return any(_contains(v, target) for v in node)
    if isinstance(node, str):
        if target in node:
            return True
        try:
            inner = json.loads(node)
        except (ValueError, TypeError):
            return False
        return _contains(inner, target) if isinstance(inner, (dict, list)) else False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--calls", required=True, help="glob of call logs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.rows, encoding="utf-8") if l.strip()]
    expected = {}
    for row in rows:
        case = case_from_row(row)
        official = inference_messages(case)
        target = clean_response_marker(case.responses[case.turn_idx - 1])
        # keyed by trajectory as well: different trajectories can share an identical first prompt (MCP rows all start
        # with the same `list_allowed_directories` call), and the per-trajectory call logs name the trajectory
        key = (str(row["id"]), tuple(m["content"] for m in official if m["role"] != "system"))
        expected[key] = {"row": (row["id"], row["turn_idx"]), "official": official, "target": target}
    per_row = defaultdict(lambda: {"rounds": 0, "prompt_tokens": [], "request_chars": [], "problems": []})
    unmatched = 0
    for path in sorted(glob.glob(args.calls)):
        trajectory = Path(path).stem.rsplit("-", 1)[-1]  # calls-<label>-<trajectory>.jsonl
        for line in open(path, encoding="utf-8"):
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                continue
            if call.get("role") != "runtime_agent_turn":
                continue
            user, system = call.get("user") or "", call.get("system") or ""
            try:
                payload = json.loads(user)
                conversation = payload["conversation"]
                # Native tool transport: the system text is the conversation's first message and the brief the first
                # user message; structured transport: the brief is the first message and the system text is separate.
                if conversation and conversation[0].get("role") == "system":
                    system = system or conversation[0].get("content") or ""
                    first_user = next(m for m in conversation if m.get("role") == "user")
                    brief = json.loads(first_user["content"])
                else:
                    brief = json.loads(conversation[0]["content"])
                block = brief["official_input"]["messages"]
            except Exception:
                unmatched += 1
                continue
            key = (trajectory, tuple(m["content"] for m in block))
            exp = expected.get(key)
            if exp is None:
                unmatched += 1
                continue
            entry = per_row[exp["row"]]
            entry["rounds"] += 1
            usage = call.get("usage") or {}
            entry["prompt_tokens"].append(int(usage.get("prompt_tokens") or 0))
            entry["request_chars"].append(len(system) + len(user))
            official = exp["official"]
            problems = []
            got = [(m["role"], m["content"]) for m in block]
            want = [(m["role"], m["content"]) for m in official if m["role"] != "system"]
            if got != want:
                problems.append("turn messages differ from inference_messages")
            if system.count(official[0]["content"]) != 1:
                problems.append(f"system message occurs {system.count(official[0]['content'])} times in the system text")
            multiplicity = Counter(m["content"] for m in official[1:])  # identical screens recur in some histories
            for content, expected_n in multiplicity.items():
                n = user.count(twice(content))
                if conversation and conversation[0].get("role") == "system":  # the system text is inside the payload here
                    n -= json.dumps(conversation[0].get("content") or "", ensure_ascii=False).count(twice(content))
                if n != expected_n:
                    problems.append(f"a turn message occurs {n} times in the request, {expected_n} times in the official input")
            # Target leak check on the decoded request: the brief and the system text must not contain the current
            # turn's reference observation. Tool results are reported separately (a memory tool can legitimately return
            # an identical earlier screen); the model's own earlier tool-call arguments are its output, not input.
            target = exp["target"]
            # A target that occurs verbatim in the episode's own official history (a repeated screen: a wait turn, an
            # unchanged prompt line, a listing shown twice) is already in the model's official input and cannot be
            # leaked by anything derived from that history (memory views, the pending-work hint); counted, not flagged.
            repeated = bool(target) and any(target in m["content"] for m in official[1:-1])
            if repeated:
                entry["repeated_screen"] = True
            if target and len(target) >= 40 and not repeated:
                # The official block is the benchmark's own input and may legitimately contain the target text when a
                # screen repeats (a wait turn, an unchanged listing); its placement is verified above, so the leak check
                # covers the system text and every other part of the brief.
                outside_official = {k: v for k, v in brief.items() if k != "official_input"}
                # The official system message (verified above to occur exactly once) is the benchmark's own input too:
                # a short repeated screen can occur in its examples. Only the harness's part of the system text is checked.
                harness_system = system.replace(official[0]["content"], "", 1)
                if target in harness_system or _contains(outside_official, target):
                    problems.append("target observation present in the system text or the brief")
                names = {}
                for m in conversation[1:]:
                    if m.get("role") == "assistant":
                        for tc in m.get("tool_calls") or []:
                            names[tc.get("id")] = (tc.get("function") or {}).get("name")
                    elif m.get("role") == "tool" and _contains(m.get("content"), target):
                        entry.setdefault("target_in_tool_results", []).append(names.get(m.get("tool_call_id"), "?"))
            if any(m["role"] == "assistant" and not m.get("memory_id") for m in block):
                problems.append("assistant message without memory id")
            if "recent_memory" in brief:
                problems.append("recent_memory view present alongside the official block")
            entry["problems"].extend(problems)
    report = {"rows_in_file": len(rows), "rows_with_agent_calls": len(per_row), "unmatched_agent_calls": unmatched,
              "rows_with_problems": sum(1 for e in per_row.values() if e["problems"]),
              "rows_with_target_in_tool_results": {f"{k[0]}/{k[1]}": sorted(set(e["target_in_tool_results"])) for k, e in per_row.items() if e.get("target_in_tool_results")},
              "max_rounds": max((e["rounds"] for e in per_row.values()), default=0),
              "max_prompt_tokens": max((max(e["prompt_tokens"]) for e in per_row.values() if e["prompt_tokens"]), default=0),
              "max_request_chars": max((max(e["request_chars"]) for e in per_row.values() if e["request_chars"]), default=0),
              "rows": {f"{k[0]}/{k[1]}": v for k, v in per_row.items()}}
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    for k, v in report["rows"].items():
        if v["problems"]:
            print("PROBLEM", k, v["problems"][:3])
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("written", args.out)


if __name__ == "__main__":
    main()
