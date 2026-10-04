"""MCP trace adapters: MCPMark logs and Toolathlon run directories → Trace2Env episodes and rows.

Two public trajectory releases feed the AgentWorldBench ``mcp`` domain:

* **MCPMark** (``Jakumetsu/mcpmark-trajectory-log``): one folder per task run with ``meta.json``
  and ``messages.json`` in the OpenAI Responses item format (``function_call`` /
  ``function_call_output`` items keyed by ``call_id``). Every tool output is the raw MCP text
  envelope the server returned.
* **Toolathlon-Verified** (``hkust-nlp/Toolathlon-Verified_Trajectories``): one directory per task
  and run with ``traj_log.json`` whose ``messages`` are OpenAI chat messages (assistant
  ``tool_calls`` answered by ``tool`` messages) and whose ``tool_calls.tools`` carries the tool
  definitions the agent saw.

Both become the episode shape ``awb-export`` and ``atif-export`` write (a ``user`` instruction
event, then alternating ``action`` / ``observation`` events) so ``reconstruct`` ingests them
unchanged. Observations stay verbatim: the MCP envelope string is the evidence. Action names are
normalized to the benchmark's naming (tool name for MCPMark; ``<server>-<tool>`` for Toolathlon,
whose newer harness writes ``<server>_<tool>``) while ``raw`` keeps the recorded call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from trace2env.agentworld import RESPONSE_MARKER, load_system_prompt_template, split_of
from trace2env.atif import BENCHMARK_SUFFIX
from trace2env.models import NormalizedAction, SplitAssignment, SplitManifest
from trace2env.storage import read_json, write_json

SOURCES = ("mcpmark", "toolathlon")
MESSAGE_CHARS = 2000
# Toolathlon local tools that manage the agent's own context (not environment interactions).
CONTEXT_TOOL_PREFIXES = ("local_search_overlong", "local_view_overlong", "local_check_context",
                         "local_manage_context", "local_history")


def _try_json(text: Any) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _arguments(value: Any) -> dict[str, Any]:
    """Same coercion ``agentworld.normalize_action`` applies to benchmark tool calls."""
    if isinstance(value, str):
        if not value.strip():
            return {}
        parsed = _try_json(value)
        value = parsed if parsed is not None else {"raw": value}
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {"value": value}


def _text(content: Any) -> str:
    """A message's text: strings verbatim, content-part lists joined on their text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in content]
        return "\n".join(part for part in parts if part)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


# ─── MCPMark ──────────────────────────────────────────────────────────────────

def mcpmark_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Environment interactions from a Responses item list, in call order, each with its output."""
    turns: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    message = ""
    batch = 0
    for index, item in enumerate(messages):
        kind = item.get("type")
        if kind == "function_call":
            if not pending:
                batch += 1
            turn = {"call_id": item.get("call_id"), "name": item.get("name"), "arguments": item.get("arguments", ""),
                    "output": None, "message": message, "index": index, "batch": batch}
            pending[str(item.get("call_id"))] = turn
            turns.append(turn)
        elif kind == "function_call_output":
            turn = pending.pop(str(item.get("call_id")), None)
            if turn is not None:
                turn["output"] = item.get("output", "")
            if not pending:
                message = ""
        elif item.get("role") == "assistant":
            message = _text(item.get("content"))
    for turn in turns:
        turn["batch_size"] = sum(1 for other in turns if other["batch"] == turn["batch"])
    return [turn for turn in turns if turn["output"] is not None]


def mcpmark_episode(directory: str | Path, *, split: str | None = None) -> dict[str, Any]:
    """An MCPMark task run (``meta.json`` + ``messages.json``) as an episode JSON."""
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    messages = read_json(directory / "messages.json")
    instruction = next((_text(item.get("content")) for item in messages if item.get("role") == "user"), "")
    turns = mcpmark_turns(messages)
    task = str(meta.get("task_name", directory.name))
    service = str(meta.get("mcp", directory.parent.parent.name.split("__")[-1]))
    family = task.split("__", 1)[0]
    result = meta.get("execution_result") or {}
    group = f"mcpmark:{service}:{family}"
    events = _events(instruction, [
        {**turn, "action": NormalizedAction(type=str(turn["name"]), arguments=_arguments(turn["arguments"]),
                                            raw={"name": turn["name"], "arguments": turn["arguments"]})}
        for turn in turns
    ])
    return {
        "episode_id": f"mcpmark:{service}:{task}:{meta.get('model_name')}:{directory.parent.name}",
        "task": "mcp",
        "source": "mcpmark",
        "service": service,
        "benchmark_task": task,
        "trajectory_id": f"mcpmark:{service}:{task}:{directory.parent.name}",
        "trajectory_group": group,
        "split": split or split_of(group),
        "agent": meta.get("model_name"),
        "instruction": instruction,
        "turns": len(turns),
        "outcome": {"solved": bool(result.get("success")), "harness_error": result.get("error_message"),
                    "verification_error": result.get("verification_error")},
        "metrics": {**(meta.get("token_usage") or {}), "turn_count": meta.get("turn_count"),
                    "agent_execution_time": meta.get("agent_execution_time")},
        "tool_names": sorted({str(turn["name"]) for turn in turns}),
        "tool_definitions": None,
        "source_path": str(directory.resolve()),
        "events": events,
    }


# ─── Toolathlon ───────────────────────────────────────────────────────────────

def toolathlon_tool_name(name: str, servers: Iterable[str]) -> str:
    """Benchmark-style ``<server>-<tool>`` for the harness's ``<server>_<tool>`` (``local_x`` → ``local-x``)."""
    for server in sorted(servers, key=len, reverse=True):
        prefix = server.replace("-", "_") + "_"
        if name.startswith(prefix):
            return f"{server}-{name[len(prefix):]}"
    if name.startswith("local_"):
        return "local-" + name[len("local_"):]
    return name


def toolathlon_turns(log: dict[str, Any]) -> list[dict[str, Any]]:
    """Environment interactions from a ``traj_log.json``: each tool call paired with its ``tool`` reply."""
    servers = list((log.get("config") or {}).get("needed_mcp_servers") or [])
    turns: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(log.get("messages") or []):
        role = item.get("role")
        if role == "assistant":
            calls = item.get("tool_calls") or []
            message = _text(item.get("content"))
            for position, call in enumerate(calls):
                function = call.get("function") or {}
                name = str(function.get("name", ""))
                turn = {"call_id": call.get("id"), "name": name, "type": toolathlon_tool_name(name, servers),
                        "arguments": function.get("arguments", ""), "output": None, "message": message,
                        "index": index, "batch_size": len(calls), "batch_index": position,
                        "context_tool": name.startswith(CONTEXT_TOOL_PREFIXES)}
                pending[str(call.get("id"))] = turn
                turns.append(turn)
        elif role == "tool":
            turn = pending.pop(str(item.get("tool_call_id")), None)
            if turn is not None:
                turn["output"] = _text(item.get("content"))
    return [turn for turn in turns if turn["output"] is not None and not turn["context_tool"]]


def toolathlon_tool_definitions(log: dict[str, Any]) -> list[dict[str, Any]]:
    """The OpenAI-style function definitions the agent saw, with benchmark-style names added."""
    servers = list((log.get("config") or {}).get("needed_mcp_servers") or [])
    definitions = []
    for tool in ((log.get("tool_calls") or {}).get("tools") or []):
        function = dict(tool.get("function") or {})
        name = str(function.get("name", ""))
        definitions.append({"name": toolathlon_tool_name(name, servers), "harness_name": name,
                            "description": function.get("description", ""), "parameters": function.get("parameters")})
    return definitions


def toolathlon_episode(directory: str | Path, *, split: str | None = None) -> dict[str, Any]:
    """A Toolathlon task-run directory (``traj_log.json`` + ``eval_res.json``) as an episode JSON."""
    directory = Path(directory)
    log = read_json(directory / "traj_log.json")
    config = log.get("config") or {}
    task = str(config.get("task_dir", directory.name)).split("/")[-1]
    run = directory.parent.parent.name.rsplit("_", 1)[-1]
    instruction = next((_text(item.get("content")) for item in log.get("messages") or [] if item.get("role") == "user"), "")
    turns = toolathlon_turns(log)
    verdict = read_json(directory / "eval_res.json") if (directory / "eval_res.json").is_file() else {}
    group = f"toolathlon:{task}"
    events = _events(instruction, [
        {**turn, "action": NormalizedAction(type=turn["type"], arguments=_arguments(turn["arguments"]),
                                            raw={"name": turn["name"], "arguments": turn["arguments"]})}
        for turn in turns
    ])
    stats = log.get("key_stats") or {}
    return {
        "episode_id": f"toolathlon:{task}:{directory.parent.parent.name}",
        "task": "mcp",
        "source": "toolathlon",
        "service": ",".join(config.get("needed_mcp_servers") or []),
        "benchmark_task": task,
        "trajectory_id": f"toolathlon:{task}:{run}",
        "trajectory_group": group,
        "split": split or split_of(group),
        "agent": directory.parent.parent.name.rsplit("_", 1)[0],
        "instruction": instruction,
        "turns": len(turns),
        "outcome": {"solved": verdict.get("pass"), "harness_error": None if log.get("status") == "success" else log.get("status"),
                    "details": (verdict.get("details") or "")[:500]},
        "metrics": {key: stats.get(key) for key in ("interaction_turns", "tool_calls", "agent_llm_requests", "total_tokens",
                                                     "input_tokens", "output_tokens", "cached_input_tokens")},
        "tool_names": sorted({turn["type"] for turn in turns}),
        "tool_definitions": toolathlon_tool_definitions(log),
        "source_path": str(directory.resolve()),
        "events": events,
    }


# ─── Shared episode shape ─────────────────────────────────────────────────────

def _events(instruction: str, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if instruction:
        events.append({"actor": "user", "kind": "message", "content": instruction,
                       "metadata": {"turn": 0, "section": "task_instruction"}})
    for index, turn in enumerate(turns, start=1):
        metadata = {"turn": index, "call_id": turn["call_id"], "tool_name": turn["name"],
                    "batch_size": turn.get("batch_size", 1), "agent_message": (turn.get("message") or "")[:MESSAGE_CHARS],
                    "raw_action": json.dumps({"name": turn["name"], "arguments": turn["arguments"]}, ensure_ascii=False)}
        events.append({"actor": "agent", "kind": "action", "content": turn["action"].model_dump(mode="json"),
                       "metadata": metadata, "call_id": turn["call_id"]})
        events.append({"actor": "environment", "kind": "observation", "content": turn["output"],
                       "metadata": {"turn": index, "call_id": turn["call_id"]}, "call_id": turn["call_id"]})
    return events


def load_episode(source: str, directory: str | Path, *, split: str | None = None) -> dict[str, Any]:
    if source == "mcpmark":
        return mcpmark_episode(directory, split=split)
    if source == "toolathlon":
        return toolathlon_episode(directory, split=split)
    raise ValueError(f"Unknown trace source {source!r}; expected one of {SOURCES}")


def export_manifest(manifest_path: str | Path, output_dir: str | Path, *, root: str | Path | None = None
                    ) -> tuple[list[Path], SplitManifest, list[dict[str, Any]]]:
    """Write one episode per manifest entry plus the split manifest; tool definitions go to ``tools/``.

    The sample manifest lists ``entries`` of ``{"source", "dir", "split"}`` (``dir`` relative to
    ``root``, default the manifest's parent). Tool definitions are written once per Toolathlon
    task under ``<output>/tools/<task>.json`` and referenced from the episode, so the episode files
    stay small and induction payloads are not inflated by schemas the model already sees.
    """
    manifest = read_json(Path(manifest_path))
    base = Path(root) if root is not None else Path(manifest_path).parent
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    assignments: dict[str, SplitAssignment] = {}
    summaries: list[dict[str, Any]] = []
    for entry in manifest.get("entries", []):
        directory = Path(entry["dir"])
        if not directory.is_absolute():
            directory = base / directory
        episode = load_episode(entry["source"], directory, split=entry.get("split"))
        if not episode["turns"]:
            summaries.append({"source": entry["source"], "task": episode["benchmark_task"], "turns": 0, "skipped": "no environment turns"})
            continue
        if episode["tool_definitions"] is not None:
            tools_path = output / "tools" / f"{episode['source']}__{episode['benchmark_task']}.json"
            write_json(tools_path, episode["tool_definitions"])
            episode["tool_definitions"] = str(tools_path.relative_to(output))
        name = f"{episode['source']}__{episode['benchmark_task']}__{episode['trajectory_id'].rsplit(':', 1)[-1]}.json"
        path = output / name
        write_json(path, episode)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assignments[f"src_{digest}"] = SplitAssignment(split=episode["split"], trajectory_group=episode["trajectory_group"])
        paths.append(path)
        summaries.append({"source": episode["source"], "service": episode["service"], "task": episode["benchmark_task"],
                          "split": episode["split"], "group": episode["trajectory_group"], "turns": episode["turns"],
                          "solved": episode["outcome"].get("solved"), "harness_error": episode["outcome"].get("harness_error"),
                          "tools": len(episode["tool_names"]), "path": str(path)})
    return paths, SplitManifest(assignments=assignments), summaries


# ─── Benchmark-format rows ────────────────────────────────────────────────────

def render_tool_definitions(definitions: list[dict[str, Any]] | None, tool_names: Iterable[str] = ()) -> str:
    """The ``### N. name`` / ``#### **Description**:`` / ``#### **Parameters**:`` layout of the official rows."""
    blocks = []
    if definitions:
        for index, tool in enumerate(definitions, start=1):
            schema = tool.get("parameters") or {}
            properties = schema.get("properties") or {} if isinstance(schema, dict) else {}
            required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
            lines = [f"### {index}. {tool['name']}", "#### **Description**:", str(tool.get('description', '')).strip(), "",
                     "#### **Parameters**:"]
            for key, spec in properties.items():
                kind = spec.get("type", "any") if isinstance(spec, dict) else "any"
                text = spec.get("description", "") if isinstance(spec, dict) else ""
                lines.append(f"- **{key}** `{kind}` ({'REQUIRED' if key in required else 'Optional'}): {text}")
            if not properties:
                lines.append("- (no parameters)")
            blocks.append("\n".join(lines))
    else:
        for index, name in enumerate(tool_names, start=1):
            blocks.append(f"### {index}. {name}\n#### **Description**:\n(definition not recorded in the trace)\n\n#### **Parameters**:\n- (unknown)")
    return "\n\n".join(blocks)


def _turn_prompt(index: int, action: dict[str, Any]) -> str:
    """The benchmark's action block, using the normalized (benchmark-style) tool name and typed arguments."""
    body = json.dumps({"name": action.get("type"), "arguments": action.get("arguments") or {}}, ensure_ascii=False, indent=2)
    return f"### Turn {index}\n**Action:**\n```json\n{body}\n```"


def episode_rows(episode: dict[str, Any], *, tool_definitions: list[dict[str, Any]] | None = None,
                 turns_per_trajectory: int | None = None, system_prompt: str | None = None) -> list[dict[str, Any]]:
    """AgentWorldBench-format records for an exported episode (mcp layout, official instruction suffix)."""
    actions = [event for event in episode["events"] if event["kind"] == "action"]
    observations = {event["metadata"]["turn"]: event["content"] for event in episode["events"] if event["kind"] == "observation"}
    prompts = [_turn_prompt(event["metadata"]["turn"], event["content"]) for event in actions]
    responses = [f"{RESPONSE_MARKER}\n{observations.get(event['metadata']['turn'], '')}" for event in actions]
    total = len(prompts)
    if not total:
        return []
    if turns_per_trajectory and turns_per_trajectory < total:
        chosen = sorted({max(1, round(i * total / turns_per_trajectory)) for i in range(1, turns_per_trajectory + 1)})
    else:
        chosen = list(range(1, total + 1))
    if system_prompt is None:
        template = load_system_prompt_template("mcp")
        system_prompt = template.replace("{tool_definitions}", render_tool_definitions(tool_definitions, episode.get("tool_names", [])))
    return [{
        "task": "mcp",
        "id": episode["trajectory_id"],
        "prompt": prompts[:turn - 1] + [prompts[turn - 1] + BENCHMARK_SUFFIX],
        "response": responses[:turn],
        "current_prompt": prompts[turn - 1],
        "system_str": system_prompt,
        "turn_idx": turn,
        "total_turns": total,
        "benchmark_task": episode.get("benchmark_task"),
        "source": episode.get("source"),
        "split": episode.get("split"),
    } for turn in chosen]
