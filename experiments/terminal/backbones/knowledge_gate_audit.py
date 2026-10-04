#!/usr/bin/env python3
"""Offline audit of harness v5.2 (compatibility-gated package knowledge) from a run's call logs and judged rows.

    python work/exp-v1_20_r=1/backbones/knowledge_gate_audit.py --rows ROWS.jsonl \
        --calls 'DIR/calls-trace2env_v52-*.jsonl' [--judged DIR/judged-trace2env_v52.jsonl] [--out FILE]

At the model boundary (the recorded requests), every package item the agent saw — the brief's similar turns,
demonstrations and notes, and the results of read_evidence / search_knowledge / inspect_action on later rounds — must
carry an applicability decision (label, provenance, reason, disposition). Sanitized items must show no raw container
values: no hostname after `user@`, no UUID or long hex id, no long-listing row with a bare size, and no file name from
the evidence episode's own inventory that this episode's transcript never showed. Supporting items must name shared
tokens that really occur in this episode's transcript or current action. The rows' recorded `knowledge_gate` decisions
give the label, disposition, abstention and masking statistics. The raw package is read only to recompute inventories.
"""
import argparse, glob, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from trace2env.knowledge_gate import GENERIC_NAMES, HOST_PROMPT, LONG_LISTING, UUID, build_episode_context, specific_tokens  # noqa: E402

LABELS = {"supporting", "uncertain", "format_only", "contradicted", "consistent", "untested"}
RAW_HOST = re.compile(r"[A-Za-z0-9_-]+@(?!<host>)[A-Za-z0-9][A-Za-z0-9.-]*")


def walk_items(node, found):
    """Collect every dict that looks like a package item view (has an id with a known kind prefix)."""
    if isinstance(node, dict):
        ident = node.get("id")
        if isinstance(ident, str) and ident.split(":")[0] in {"evidence", "demo", "demonstration", "note"}:
            found.append(node)
        for value in node.values():
            walk_items(value, found)
    elif isinstance(node, list):
        for value in node:
            walk_items(value, found)


def item_texts(item):
    for key in ("text", "snippet", "observation", "observation_head", "statement"):
        if isinstance(item.get(key), str):
            yield key, item[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--judged", default=None)
    ap.add_argument("--package", default=str(ROOT / "work/exp-v1_20_r=1/packages/v1_20_r=1"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # Evidence inventories from the raw package (read only), for the "foreign name" check.
    from trace2env.package import EnvironmentPackage, PackageInspector
    from trace2env.knowledge_gate import EpisodeContext, KnowledgeGate
    inspector = PackageInspector(EnvironmentPackage(args.package))
    inventories = KnowledgeGate(inspector, EpisodeContext(), None)._inventories()
    records = inspector._evidence_index()["by_id"]
    problems, seen_items, sanitized_items, supporting_items, judged_items = [], 0, 0, 0, 0
    label_counts, disposition_counts = Counter(), Counter()
    calls = 0
    for path in sorted(glob.glob(args.calls)):
        for line in open(path, encoding="utf-8"):
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                continue
            if call.get("role") != "runtime_agent_turn":
                continue
            calls += 1
            try:
                payload = json.loads(call["user"])
                conversation = payload["conversation"]
                if conversation and conversation[0].get("role") == "system":  # native tool transport
                    brief = json.loads(next(m for m in conversation if m.get("role") == "user")["content"])
                else:
                    brief = json.loads(conversation[0]["content"])
            except Exception:
                problems.append((path, "unreadable payload"))
                continue
            official = brief.get("official_input", {}).get("messages", [])
            context = build_episode_context([m["content"] for m in official], None)
            action_tokens = specific_tokens(json.dumps(brief.get("action", {})))
            known = context.shown_names | action_tokens
            # A name that occurs verbatim in the episode's own input (a `mkdir test-repo` operand, a name the tracked
            # state carried) is the episode's, not foreign, whatever the audit's history-only name extraction says.
            own_text = "\n".join(m["content"] for m in official) + json.dumps(brief.get("action", {}), ensure_ascii=False)
            views = []
            walk_items({k: v for k, v in brief.items() if k != "official_input"}, views)
            names = {}
            for message in conversation[1:]:
                if message.get("role") == "assistant":
                    for tc in message.get("tool_calls") or []:
                        names[tc.get("id")] = (tc.get("function") or {}).get("name")
                elif message.get("role") == "tool":
                    try:
                        result = json.loads(message.get("content") or "null")
                    except Exception:
                        continue
                    if names.get(message.get("tool_call_id")) in {"read_evidence", "search_knowledge", "inspect_action"}:
                        walk_items(result, views)
            for item in views:
                ident = item["id"]
                if not any(isinstance(item.get(k), str) for k in ("text", "snippet", "observation", "observation_head", "statement")):
                    continue  # a bare reference (neighbour pointer, artifact id, demo pointer), not content shown
                seen_items += 1
                decision = item.get("applicability")
                if not isinstance(decision, dict) or decision.get("label") not in LABELS or "disposition" not in decision or "reason" not in decision:
                    problems.append((ident, "item shown without a complete applicability decision"))
                    continue
                label_counts[decision["label"]] += 1
                disposition_counts[decision["disposition"]] += 1
                judged_items += 1 if decision.get("judge") else 0
                if decision["disposition"] == "sanitized":
                    sanitized_items += 1
                    evidence_id = ident.split(":", 1)[1]
                    if evidence_id.startswith("demo_"):
                        evidence_id = evidence_id[len("demo_"):]
                    record = records.get(evidence_id)
                    foreign = set()
                    if record is not None:
                        foreign = {t for t in inventories.get(record.episode_id, set())
                                   if t not in known and t.lower() not in GENERIC_NAMES and len(t) >= 5 and t not in own_text}
                    for key, text in item_texts(item):
                        if RAW_HOST.search(text):
                            problems.append((ident, f"raw hostname in sanitized {key}"))
                        if UUID.search(text):
                            problems.append((ident, f"uuid in sanitized {key}"))
                        for row in text.split("\n"):
                            if LONG_LISTING.match(row.rstrip()) and not re.search(r"\s<n>\s", row):
                                problems.append((ident, f"long-listing row with a bare size in sanitized {key}"))
                                break
                        leaked = [t for t in foreign if re.search(r"(?<![\w./-])" + re.escape(t) + r"(?![\w./-])", text)]
                        if leaked:
                            problems.append((ident, f"foreign names {leaked[:3]} in sanitized {key}"))
                elif decision["label"] == "supporting":
                    supporting_items += 1
                    judge = decision.get("judge") or {}
                    if judge.get("verified_anchors"):
                        transcript = "\n".join(m["content"] for m in official) + json.dumps(brief.get("action", {}))
                        if not all(a in transcript for a in judge["verified_anchors"]):
                            problems.append((ident, "judge anchor not found in this episode's transcript or action"))
                    else:
                        named = re.findall(r"'([^']+)'", decision.get("reason", ""))
                        # names the gate took from the tracked state carry the absolute prefix (`app/warriors/paper.red`)
                        # while the transcript shows the relative form; the basename decides whether it is the episode's own
                        if named and not any(n in known or n in own_text or n.rsplit("/", 1)[-1] in own_text for n in named):
                            problems.append((ident, "supporting reason names nothing this episode showed"))
    rows_stats = None
    if args.judged and Path(args.judged).exists():
        rows = [json.loads(l) for l in open(args.judged, encoding="utf-8") if l.strip()]
        gates = [(r.get("trace2env") or {}).get("knowledge_gate") for r in rows]
        gates = [g for g in gates if g]
        rows_stats = {
            "rows": len(rows), "rows_with_gate_record": len(gates),
            "abstained_rows": sum(1 for g in gates if g.get("abstained")),
            "rows_with_supporting": sum(1 for g in gates if g.get("supporting")),
            "items_per_row": round(sum(g.get("items", 0) for g in gates) / max(1, len(gates)), 1),
            "labels": dict(sum((Counter(g.get("labels", {})) for g in gates), Counter())),
            "dispositions": dict(sum((Counter(g.get("dispositions", {})) for g in gates), Counter())),
            "masked_tokens_total": sum(g.get("masked_tokens", 0) for g in gates),
        }
    report = {"agent_calls": calls, "package_items_seen": seen_items, "sanitized_items": sanitized_items, "supporting_items": supporting_items,
              "labels_at_boundary": dict(label_counts), "dispositions_at_boundary": dict(disposition_counts),
              "judged_items_at_boundary": judged_items,
              "problems": len(problems), "problem_examples": [f"{i}: {p}" for i, p in problems[:12]], "rows": rows_stats}
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("written", args.out)


if __name__ == "__main__":
    main()
