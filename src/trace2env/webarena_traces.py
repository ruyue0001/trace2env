"""WebArena trace collection support: task loading, the task agent's brief, and episode export.

The AgentWorldBench web split was recorded on WebArena's four self-hosted sites through Playwright MCP
(``browser_navigate`` / ``browser_click`` / ``browser_type`` ...), with the tool's text result as the observation.
``scripts/collect_webarena.py`` reproduces that setup on our own instance; this module holds the parts that do not
touch a browser or a model, so they can be tested: resolving the public task list against the benchmark's
hostnames, the agent brief (site, credentials, stop protocol), folding ``browser_snapshot`` calls into the preceding
action's observation the way the benchmark rows do, and the shared episode shape that ``reconstruct`` ingests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import SplitAssignment, SplitManifest

# The benchmark's hostnames for WebArena's sites, with the accounts WebArena ships (its own harness logs in through
# stored cookies; the benchmark's agents typed the credentials themselves, and so does ours).
SITES: dict[str, dict[str, Any]] = {
    "gitlab": {"placeholder": "__GITLAB__", "host": "gitlab.example.com", "name": "GitLab",
               "login_url": "http://gitlab.example.com/users/sign_in", "username": "byteblaze", "password": "hello1234",
               "container_port": 8023},
    "shopping": {"placeholder": "__SHOPPING__", "host": "magento-store.example.com", "name": "One Stop Market (Magento storefront)",
                 "login_url": "http://magento-store.example.com/customer/account/login/", "username": "emma.lopez@gmail.com",
                 "password": "Password.1", "container_port": 7770},
    "shopping_admin": {"placeholder": "__SHOPPING_ADMIN__", "host": "magento-admin.example.com", "name": "Magento admin panel",
                       "login_url": "http://magento-admin.example.com/admin", "username": "admin", "password": "admin1234",
                       "container_port": 7780},
    "reddit": {"placeholder": "__REDDIT__", "host": "forum.example.com", "name": "Postmill forum",
               "login_url": "http://forum.example.com/login", "username": "MarvelsGrantMan136", "password": "test1234",
               "container_port": 9999},
}
OTHER_PLACEHOLDERS = {"__MAP__": "http://map.example.com", "__WIKIPEDIA__": "http://wikipedia.example.com"}

# The tools the benchmark's agents used, plus the ones an agent needs to read a page and back out (no hover: the
# benchmark's agents never hovered, they clicked links or navigated), each with the arguments the benchmark's calls
# carry. The schemas shown to the agent are pruned to these keys so the recorded action text matches the benchmark's
# (`browser_click(element=..., ref=...)`, never `doubleClick=false, button="left", modifiers=[]`).
AGENT_TOOLS = ["browser_navigate", "browser_click", "browser_type", "browser_fill_form", "browser_press_key",
               "browser_select_option", "browser_evaluate", "browser_snapshot", "browser_navigate_back", "browser_wait_for"]
TOOL_ARGUMENTS: dict[str, list[str]] = {
    "browser_navigate": ["url"], "browser_click": ["element", "ref"], "browser_type": ["element", "ref", "text", "submit", "slowly"],
    "browser_fill_form": ["fields"], "browser_press_key": ["key"], "browser_select_option": ["element", "ref", "values"],
    "browser_evaluate": ["function", "element", "ref"], "browser_snapshot": [], "browser_navigate_back": [],
    "browser_wait_for": ["time", "text", "textGone"],
}


def prune_tool_schema(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """The tool's input schema restricted to the benchmark's argument names."""
    allowed = TOOL_ARGUMENTS.get(name)
    if allowed is None:
        return schema
    properties = {k: v for k, v in (schema.get("properties") or {}).items() if k in allowed}
    pruned = {**schema, "properties": properties}
    if "required" in schema:
        pruned["required"] = [k for k in schema["required"] if k in allowed]
    return pruned
FINAL_MARKER = "FINAL ANSWER:"
DONE_MARKER = "DONE"
MESSAGE_CHARS = 2000


def resolve_url(url: str) -> str:
    """WebArena placeholders → the benchmark's hosts (``__SHOPPING_ADMIN__`` is the ``/admin`` URL, as in WebArena)."""
    for site in SITES.values():
        base = f"http://{site['host']}" + ("/admin" if site["placeholder"] == "__SHOPPING_ADMIN__" else "")
        url = url.replace(site["placeholder"], base)
    for placeholder, replacement in OTHER_PLACEHOLDERS.items():
        url = url.replace(placeholder, replacement)
    return url


def load_tasks(task_list: str | Path, task_ids: list[int] | None = None) -> list[dict[str, Any]]:
    """WebArena's ``test.raw.json`` entries (optionally only ``task_ids``), start URLs resolved to the benchmark hosts."""
    tasks = json.loads(Path(task_list).read_text(encoding="utf-8"))
    wanted = set(task_ids) if task_ids is not None else None
    selected = []
    for task in tasks:
        if wanted is not None and task["task_id"] not in wanted:
            continue
        selected.append({**task, "start_url": " |OR| ".join(resolve_url(u) for u in str(task["start_url"]).split(" |OR| "))})
    if wanted is not None:
        missing = wanted - {t["task_id"] for t in selected}
        if missing:
            raise ValueError(f"Unknown WebArena task ids: {sorted(missing)}")
        selected.sort(key=lambda t: task_ids.index(t["task_id"]))
    return selected


def site_of_task(task: dict[str, Any]) -> str:
    sites = [s for s in task.get("sites", []) if s in SITES]
    if not sites:
        raise ValueError(f"Task {task.get('task_id')} runs on {task.get('sites')}, none of which is hosted")
    return sites[0]


def agent_system_prompt(task: dict[str, Any]) -> str:
    """The task agent's brief: one site, its credentials, the tools, and the stop protocol."""
    lines = ["You are an autonomous web agent. You control a real browser through the Playwright MCP tools "
             "(browser_navigate, browser_click, browser_type, browser_fill_form, browser_press_key, browser_select_option, "
             "browser_evaluate, browser_snapshot, browser_navigate_back, browser_wait_for) and must complete "
             "the user's task on the website below by interacting with it step by step.",
             "", "Websites available in this session (the browser starts logged out; log in yourself when the task needs an account):"]
    for key in task.get("sites", []):
        site = SITES.get(key)
        if site is None:
            continue
        lines.append(f"- {site['name']}: http://{site['host']} — username `{site['username']}`, password `{site['password']}` "
                     f"(login page {site['login_url']})")
    lines += ["", f"Start URL: {task['start_url']}",
              "",
              "Rules:",
              "- Issue exactly one tool call per turn and read its result before deciding the next step. After an action that "
              "changes the page (typing, filling a form, clicking something that navigates or opens a panel), call "
              "browser_snapshot to observe the page before choosing the next action; a tool result without a snapshot never "
              "shows you the current page.",
              "- Element references (ref=eN) are only valid for the snapshot they came from.",
              "- Stay on the websites listed above. Do not open other sites.",
              f"- When the task asks a question, finish with a message that starts with `{FINAL_MARKER}` followed by the answer. "
              f"When the task asks for an action, finish with a message that starts with `{DONE_MARKER}` once the action is complete. "
              "If the task cannot be completed, finish with `FINAL ANSWER: N/A` and say why.",
              "- Do not ask the user questions; there is no user to answer."]
    return "\n".join(lines)


def call_string(name: str, arguments: dict[str, Any]) -> str:
    """The benchmark's action text, e.g. ``browser_click(element="Sign In link", ref="e12")``."""
    parts = []
    for key, value in arguments.items():
        parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return f"{name}({', '.join(parts)})"


def fold_snapshots(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold ``browser_snapshot`` calls into the preceding action's observation.

    The benchmark rows never show a snapshot action: when an agent called ``browser_snapshot`` after an action, the
    row's observation for that action is the snapshot's text (the last environment output before the next action).
    A snapshot with no preceding action (a look at the blank start page) is dropped; the raw log keeps it.
    """
    folded: list[dict[str, Any]] = []
    for turn in turns:
        if turn["name"] == "browser_snapshot":
            if not folded:
                continue  # a snapshot of the blank start page before any action carries nothing
            previous = folded[-1]
            previous["output"] = turn["output"]
            previous.setdefault("folded_call_ids", []).append(turn["call_id"])
            continue
        folded.append({**turn})
    for index, turn in enumerate(folded, start=1):
        turn["turn"] = index
    return folded


def build_episode(task: dict[str, Any], turns: list[dict[str, Any]], *, run: str, agent: str, outcome: dict[str, Any],
                  metrics: dict[str, Any] | None = None, fold: bool = True) -> dict[str, Any]:
    """The shared episode shape (``mcp_traces._events``): instruction, then action / observation pairs keyed by call id."""
    site = site_of_task(task)
    kept = fold_snapshots(turns) if fold else [{**t, "turn": i} for i, t in enumerate(turns, start=1)]
    events: list[dict[str, Any]] = [{"actor": "user", "kind": "message", "content": task["intent"],
                                     "metadata": {"turn": 0, "section": "task_instruction"}}]
    for turn in kept:
        action = {"type": turn["name"], "arguments": dict(turn["arguments"]), "raw": call_string(turn["name"], turn["arguments"])}
        metadata = {"turn": turn["turn"], "call_id": turn["call_id"], "tool_name": turn["name"],
                    "agent_message": (turn.get("message") or "")[:MESSAGE_CHARS],
                    "raw_action": json.dumps({"name": turn["name"], "arguments": turn["arguments"]}, ensure_ascii=False)}
        if turn.get("folded_call_ids"):
            metadata["folded_snapshot_call_ids"] = list(turn["folded_call_ids"])
        events.append({"actor": "agent", "kind": "action", "content": action, "metadata": metadata, "call_id": turn["call_id"]})
        events.append({"actor": "environment", "kind": "observation", "content": turn["output"],
                       "metadata": {"turn": turn["turn"], "call_id": turn["call_id"]}, "call_id": turn["call_id"]})
    group = f"webarena:{task['task_id']}"
    return {
        "episode_id": f"webarena:{task['task_id']}:{run}",
        "task": "web",
        "source": "webarena-own",
        "site": site,
        "benchmark_task": str(task["task_id"]),
        "template_id": task.get("intent_template_id"),
        "trajectory_id": f"webarena:{task['task_id']}:{run}",
        "trajectory_group": group,
        "split": "train",
        "agent": agent,
        "instruction": task["intent"],
        "start_url": task["start_url"],
        "turns": len(kept),
        "raw_turns": len(turns),
        "outcome": outcome,
        "metrics": metrics or {},
        "tool_names": sorted({t["name"] for t in kept}),
        "events": events,
    }


def split_manifest_for(paths: list[Path]) -> SplitManifest:
    """Every collected episode is construction data (``train``), grouped by its WebArena task."""
    assignments: dict[str, SplitAssignment] = {}
    for path in paths:
        episode = json.loads(path.read_text(encoding="utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assignments[f"src_{digest}"] = SplitAssignment(split="train", trajectory_group=episode["trajectory_group"])
    return SplitManifest(assignments=assignments)


def final_answer(text: str) -> tuple[str | None, str | None]:
    """(kind, answer) when an assistant message ends the task, else (None, None)."""
    stripped = (text or "").strip()
    for line in stripped.splitlines():
        line = line.strip()
        if line.upper().startswith(FINAL_MARKER):
            return "answer", line[len(FINAL_MARKER):].strip()
        if line.upper().startswith(DONE_MARKER) and len(line) <= 200:
            return "done", line[len(DONE_MARKER):].strip(" .:-")
    return None, None
