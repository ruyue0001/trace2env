"""``envpack_prompting``: the reconstructed package as a compact, deterministic, non-agentic prompt block.

The baseline isolates the value of the offline package from the agentic runtime: one free-text prediction call per
benchmark row over the official input (system prompt, full preceding history, current action) plus a fixed view of the
package appended to the system prompt — the current action's schema, the state schema, the executable rules,
renderer contracts and notes for the action, its demonstrations, and a fixed top-k of package evidence retrieved
lexically for the current action through the package's own knowledge catalog (the same FTS index and settings for
every row). No agent loop, no episodic memory, no tools, no state tracking, no session; the manifest of what was placed
in the prompt is recorded per row so any prompt can be reconstructed exactly.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .agentworld import AgentWorldCase

_TOKEN = re.compile(r"[A-Za-z0-9_./-]+")
_ACTION_BLOCK = re.compile(r"```json\s*(\[.*?\])\s*```", re.S)


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters omitted]"


def action_query(case: AgentWorldCase, limit: int = 12) -> str:
    """The fixed lexical query for the current action: the first ``limit`` distinct command words of the typed
    keystrokes (terminal), else the action type and its argument words; no flags, no normalization."""
    match = _ACTION_BLOCK.search(case.current_prompt)
    typed = ""
    if match:
        try:
            typed = "".join(str(item.get("keystrokes", "")) for item in json.loads(match.group(1)) if isinstance(item, dict))
        except (json.JSONDecodeError, AttributeError):
            typed = ""
    if not typed.strip():
        typed = f"{case.action.type} " + json.dumps(case.action.arguments, ensure_ascii=False)
    tokens: list[str] = []
    for token in _TOKEN.findall(typed):
        token = token.strip("./")
        if token and not token.startswith("-") and token not in tokens:
            tokens.append(token)
    return " ".join(tokens[:limit])


def _rule_lines(rule: dict[str, Any]) -> str:
    parts = [f"- rule `{rule.get('id')}` (confidence {rule.get('confidence')}, priority {rule.get('priority')}, outcome {rule.get('outcome')}): {rule.get('description', '')}"]
    if rule.get("scope"):
        parts.append(f"  scope: {json.dumps(rule['scope'], ensure_ascii=False)}")
    if rule.get("condition"):
        parts.append(f"  condition: {json.dumps(rule['condition'], ensure_ascii=False)}")
    if rule.get("effects"):
        parts.append(f"  effects: {json.dumps(rule['effects'], ensure_ascii=False)}")
    if rule.get("observation_template"):
        parts.append(f"  observation template: {json.dumps(rule['observation_template'], ensure_ascii=False)}")
    return "\n".join(parts)


def envpack_view(inspector: Any, case: AgentWorldCase, *, top_k: int = 6, evidence_chars: int = 3000, rule_limit: int = 12,
                 demo_limit: int = 4, demo_chars: int = 1500, state_fields: int = 80) -> tuple[str, dict[str, Any]]:
    """The package block for one row and the manifest of everything it contains (ids, counts, query, sizes)."""
    package = inspector.package
    view = inspector.inspect_action(case.action.type, limit=rule_limit)
    canonical = view["canonical_action_type"]
    query = action_query(case)
    hits = inspector.search_knowledge(query, kinds=["evidence"], limit=top_k) if query and top_k > 0 else []
    evidence = []
    for hit in hits:
        page = inspector.evidence_view(hit["ref_id"], length=evidence_chars)
        if page is not None:
            evidence.append(page)
    demonstrations = view["demonstrations"][:demo_limit]
    metadata = package.manifest.metadata or {}
    lines = [
        "\n\n---\n\n# Reconstructed environment package (offline reference)\n",
        f"Package `{package.manifest.name}` (environment `{package.manifest.environment_id}`, {metadata.get('construction_kind', 'reconstructed')}) "
        "was reconstructed offline from recorded sessions of this environment on other tasks. It is reference material: its "
        "rules and contracts describe how the environment responds to actions, its recorded turns show real output formats "
        "and program behaviour. File names, values and contents in it belong to those sessions, not to the current one: "
        "predict the current session's observation from its own history and state, and use the package for behaviour and "
        "format only.\n",
        f"\n## Current action `{canonical}` (schema)\n",
    ]
    if view["action_specs"]:
        for spec in view["action_specs"]:
            arguments = {name: {k: v for k, v in (arg or {}).items() if k in ("type", "required", "description") and v not in (None, "", False)}
                         for name, arg in (spec.get("arguments") or {}).items()}
            lines.append(f"- `{spec['name']}`: {spec.get('description', '')}\n  arguments: {json.dumps(arguments, ensure_ascii=False)}"
                         + (f"\n  aliases: {spec['aliases']}" if spec.get("aliases") else "") + "\n")
    else:
        lines.append(f"- `{canonical}` has no schema entry in the package.\n")
    names = sorted(spec.name for spec in package.action_schema.actions)
    lines.append(f"\nKnown actions ({len(names)}): {', '.join(names)}\n")
    fields = list(package.state_schema.fields)[:state_fields]
    lines.append(f"\n## State schema ({len(package.state_schema.fields)} fields" + (f", first {state_fields} shown" if len(package.state_schema.fields) > state_fields else "") + ")\n")
    for field in fields:
        lines.append(f"- `{field.path}` ({field.type}): {field.description or ''}\n")
    lines.append(f"\n## Rules for `{canonical}` ({len(view['rules'])} executable)\n")
    lines.extend(_rule_lines(rule) + "\n" for rule in view["rules"])
    if view["renderers"]:
        lines.append(f"\n## Renderer contracts ({len(view['renderers'])})\n")
        for contract in view["renderers"]:
            lines.append(f"- contract `{contract.get('id')}`: template {json.dumps(contract.get('template'), ensure_ascii=False)}; "
                         f"required fields {contract.get('required_fields')}; {contract.get('instructions', '')}\n")
    if view["invariants"]:
        lines.append(f"\n## Invariants ({len(view['invariants'])})\n")
        for invariant in view["invariants"]:
            lines.append(f"- `{invariant.get('id')}`: {invariant.get('description', '')} {json.dumps(invariant.get('condition'), ensure_ascii=False) if invariant.get('condition') else ''}\n")
    if view["notes"]:
        lines.append(f"\n## Notes ({len(view['notes'])})\n")
        for note in view["notes"]:
            lines.append(f"- [{note.get('kind')}] {note.get('statement', '')}\n")
    if demonstrations:
        lines.append(f"\n## Demonstrations for `{canonical}` ({len(demonstrations)})\n")
        for demo in demonstrations:
            action = demo.get("action") or {}
            lines.append(f"\n### Demonstration `{demo.get('id')}` (outcome {demo.get('outcome')})\n**Action:** `{action.get('type')}` "
                         f"{json.dumps(action.get('arguments', {}), ensure_ascii=False)[:600]}\n**Observation:**\n```text\n{_clip(str(demo.get('observation', '')), demo_chars)}\n```\n")
    lines.append(f"\n## Retrieved package evidence (top {top_k} by lexical retrieval for query {json.dumps(query)}; {len(evidence)} found)\n")
    for number, page in enumerate(evidence, start=1):
        arguments = page.get("action_arguments") or {}
        lines.append(f"\n### Evidence {number}: `{page['id']}` (recorded session `{page['episode']}`, turn {page['turn']}, action `{page['action_type']}`)\n"
                     f"**Action arguments:** {json.dumps(arguments, ensure_ascii=False)[:600]}\n**Observation ({page['total_chars']} chars):**\n```text\n{page['text']}"
                     + (f"\n[... {page['total_chars'] - page['length']} more characters omitted]" if page.get("next_offset") is not None else "") + "\n```\n")
    block = "".join(lines)
    manifest = {
        "canonical_action_type": canonical, "query": query, "top_k": top_k, "evidence_chars": evidence_chars,
        "rule_limit": rule_limit, "demo_limit": demo_limit, "demo_chars": demo_chars, "state_fields_shown": len(fields),
        "action_specs": [spec["name"] for spec in view["action_specs"]], "rules": [rule["id"] for rule in view["rules"]],
        "renderers": [contract["id"] for contract in view["renderers"]], "invariants": [item["id"] for item in view["invariants"]],
        "notes": [note["id"] for note in view["notes"]], "demonstrations": [demo["id"] for demo in demonstrations],
        "evidence": [page["id"] for page in evidence], "block_chars": len(block),
        "package": {"name": package.manifest.name, "environment_id": package.manifest.environment_id},
    }
    return block, manifest
