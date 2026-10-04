#!/usr/bin/env python3
"""Collect WebArena traces through Playwright MCP with a task agent, in the AgentWorldBench web split's own format.

    export OPENROUTER_API_KEY=$(cat ~/.config/trace2env/openrouter_key)
    python scripts/collect_webarena.py --task-list work/webarena/test.raw.json --ids 105,106,576 \
        --output work/webarena/traces --mcp-cli scripts/webarena/node_modules/@playwright/mcp/cli.js \
        --node "$(command -v node)" --base-url https://openrouter.ai/api/v1 --model openai/gpt-5.6-sol \
        --api-key-env OPENROUTER_API_KEY

One Playwright MCP server per task (``--isolated``: a fresh, logged-out browser profile), the pinned version that
reproduces the benchmark's result text (``### Ran Playwright code`` / ``### Page`` / ``### Snapshot`` / ``### Events``:
@playwright/mcp 0.0.61–0.0.68; ``scripts/webarena/package.json`` pins 0.0.68). The benchmark's hostnames
(``gitlab.example.com`` ...) are mapped to the local reverse proxy with Chromium host-resolver rules, so recorded
URLs match the benchmark's. Every tool call and its verbatim text result is logged (``raw/``); the exported
episodes (``episodes/``) fold ``browser_snapshot`` calls into the preceding action's observation, as the benchmark
rows do, and ``split_manifest.json`` assigns every episode to ``train``. ``--scripted FILE`` replays fixed tool calls
without a model (tests).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.webarena_traces import (  # noqa: E402
    AGENT_TOOLS, SITES, agent_system_prompt, build_episode, final_answer, load_tasks, prune_tool_schema, site_of_task,
    split_manifest_for,
)

DEFAULT_MCP_CLI = ROOT / "scripts" / "webarena" / "node_modules" / "@playwright" / "mcp" / "cli.js"


class McpStdio:
    """A minimal MCP client over stdio (newline-delimited JSON-RPC)."""

    def __init__(self, cmd: list[str], cwd: str | None = None):
        self.process = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, bufsize=1, encoding="utf-8")
        self.counter = 0
        self.stderr_lines: list[str] = []
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self.stderr_lines.append(line.rstrip())

    def _send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None, timeout: float = 180.0) -> Any:
        self.counter += 1
        request_id = self.counter
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout
        assert self.process.stdout is not None
        while time.time() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Playwright MCP closed its stdout: " + "\n".join(self.stderr_lines[-5:]))
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"]))
                return message["result"]
        raise TimeoutError(f"{method} did not answer within {timeout} s")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def initialize(self) -> dict[str, Any]:
        info = self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                                           "clientInfo": {"name": "trace2env-webarena-collector", "version": "1"}})
        self.notify("notifications/initialized")
        return info

    def tools(self) -> list[dict[str, Any]]:
        return self.request("tools/list")["tools"]

    def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """(text result, is_error). Playwright errors come back as ``### Error`` text; transport errors are raised."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        return text, bool(result.get("isError"))

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except Exception:
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()


def mcp_config(path: Path, resolve_to: str | None) -> Path:
    """Write the server config: benchmark hostnames resolve to the local reverse proxy."""
    config: dict[str, Any] = {"browser": {"launchOptions": {"args": []}}}
    if resolve_to:
        rules = ", ".join(f"MAP {site['host']} {resolve_to}" for site in SITES.values())
        config["browser"]["launchOptions"]["args"].append(f"--host-resolver-rules={rules}")
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    return path


def openai_tools(mcp_tools: list[dict[str, Any]], allowed: list[str]) -> list[dict[str, Any]]:
    by_name = {t["name"]: t for t in mcp_tools}
    tools = []
    for name in allowed:
        tool = by_name.get(name)
        if tool is None:
            continue
        schema = prune_tool_schema(name, tool.get("inputSchema", {"type": "object", "properties": {}}))
        tools.append({"type": "function", "function": {"name": name, "description": tool.get("description", ""), "parameters": schema}})
    return tools


class ScriptedModel:
    """Replays fixed tool calls, then a final message (no API)."""

    def __init__(self, script: list[dict[str, Any]], final: str):
        self.script, self.final, self.index = script, final, 0

    def respond(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.index < len(self.script):
            call = self.script[self.index]
            self.index += 1
            message = {"role": "assistant", "content": call.get("message") or None,
                       "tool_calls": [{"id": f"scripted-{self.index}", "type": "function",
                                       "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])}}]}
        else:
            message = {"role": "assistant", "content": self.final}
        return message, {}


class OpenAIToolModel:
    def __init__(self, base_url: str, api_key: str, model: str, reasoning_effort: str | None, max_output_tokens: int,
                 provider_json: str | None = None):
        from openai import OpenAI
        from trace2env.llm import CLIENT_MAX_RETRIES, CLIENT_TIMEOUT_S
        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=CLIENT_MAX_RETRIES, timeout=CLIENT_TIMEOUT_S)
        self.model, self.reasoning_effort, self.max_output_tokens = model, reasoning_effort, max_output_tokens
        self.provider_json = json.loads(provider_json) if provider_json else None

    def respond(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        from trace2env.llm import call_with_retry
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            extra["reasoning"] = {"effort": self.reasoning_effort}
        if self.provider_json:
            extra["provider"] = self.provider_json

        def request():
            return self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto",
                                                       parallel_tool_calls=False, max_tokens=self.max_output_tokens, extra_body=extra)
        response = call_with_retry(request, role="webarena_task_agent")
        choice = response.choices[0]
        message: dict[str, Any] = {"role": "assistant", "content": choice.message.content}
        if choice.message.tool_calls:
            message["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                     for tc in choice.message.tool_calls]
        usage = response.usage.model_dump() if response.usage else {}
        usage["finish_reason"] = choice.finish_reason
        return message, usage


def model_view(messages: list[dict[str, Any]], keep_results: int) -> list[dict[str, Any]]:
    """The conversation the model sees: only the last ``keep_results`` tool results verbatim, older ones stubbed."""
    tool_indexes = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    stubbed = set(tool_indexes[:-keep_results]) if keep_results >= 0 else set()
    view = []
    for i, m in enumerate(messages):
        if i in stubbed:
            head = (m.get("content") or "").split("\n### Snapshot")[0]
            view.append({**m, "content": head[:600] + "\n[earlier page snapshot omitted to save context; call browser_snapshot to see the current page]"})
        else:
            view.append(m)
    return view


def run_task(task: dict[str, Any], *, model, mcp_cmd: list[str], cwd: Path, max_steps: int, keep_results: int, nudges: int = 2,
             log_path: Path | None = None) -> dict[str, Any]:
    started = time.time()
    server = McpStdio(mcp_cmd, cwd=str(cwd))
    turns: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    status, answer, error = "max_steps", None, None
    try:
        info = server.initialize()
        tools = openai_tools(server.tools(), AGENT_TOOLS)
        messages: list[dict[str, Any]] = [{"role": "system", "content": agent_system_prompt(task)},
                                          {"role": "user", "content": f"Task: {task['intent']}"}]
        nudged = 0
        while len(turns) < max_steps:
            message, usage = model.respond(model_view(messages, keep_results), tools)
            usages.append(usage)
            messages.append(message)
            if not message.get("tool_calls"):
                kind, value = final_answer(message.get("content") or "")
                if kind:
                    status, answer = kind, value
                    break
                if nudged >= nudges:
                    status = "no_final_answer"
                    break
                nudged += 1
                messages.append({"role": "user", "content": "Continue with exactly one tool call, or finish with `FINAL ANSWER: ...` / `DONE`."})
                continue
            for call in message["tool_calls"]:
                name = call["function"]["name"]
                try:
                    arguments = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {"_raw": call["function"]["arguments"]}
                try:
                    output, is_error = server.call(name, arguments)
                except Exception as exc:  # transport-level failure: recorded, the task continues
                    output, is_error = f"### Error\n{type(exc).__name__}: {exc}", True
                turn = {"turn": len(turns) + 1, "call_id": call["id"], "name": name, "arguments": arguments, "output": output,
                        "is_error": is_error, "message": message.get("content") or "", "elapsed_s": round(time.time() - started, 1)}
                turns.append(turn)
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
                if log_path is not None:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"task_id": task["task_id"], **turn}, ensure_ascii=False) + "\n")
                if len(turns) >= max_steps:
                    break
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        server.close()
    prompt_tokens = sum(int(u.get("prompt_tokens") or 0) for u in usages)
    completion_tokens = sum(int(u.get("completion_tokens") or 0) for u in usages)
    return {"task": task, "site": site_of_task(task), "server": info if "info" in locals() else None, "turns": turns, "messages": messages if "messages" in locals() else [],
            "status": status, "answer": answer, "error": error, "usages": usages,
            "metrics": {"model_calls": len(usages), "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                        "seconds": round(time.time() - started, 1)}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-list", required=True, help="WebArena test.raw.json")
    parser.add_argument("--ids", default=None, help="comma-separated task ids (default: every task of --selection)")
    parser.add_argument("--selection", default=None, help="collection_tasks.json from select_collection_tasks.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", default="r1", help="run label in episode ids")
    parser.add_argument("--node", default=os.environ.get("NODE", "node"))
    parser.add_argument("--mcp-cli", default=str(DEFAULT_MCP_CLI))
    parser.add_argument("--resolve-to", default="127.0.0.1", help="IP the benchmark hostnames resolve to ('' to leave DNS alone)")
    parser.add_argument("--no-sandbox", action="store_true", help="pass --no-sandbox to Playwright MCP")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--provider-json", default=None)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--keep-results", type=int, default=6, help="tool results shown verbatim to the agent (older ones stubbed)")
    parser.add_argument("--scripted", default=None, help="JSON {'calls': [{name, arguments}], 'final': str}: replay without a model")
    parser.add_argument("--force", action="store_true", help="re-run tasks whose episode already exists")
    parser.add_argument("--no-fold", action="store_true", help="keep browser_snapshot calls as turns of their own")
    args = parser.parse_args()

    ids: list[int] | None = None
    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    elif args.selection:
        ids = [int(t["task_id"]) for t in json.loads(Path(args.selection).read_text(encoding="utf-8"))["tasks"]]
    tasks = load_tasks(args.task_list, ids)
    output = Path(args.output)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "episodes").mkdir(parents=True, exist_ok=True)
    cwd = output / "mcp-cwd"
    cwd.mkdir(exist_ok=True)
    config_path = mcp_config(output / "playwright-mcp.config.json", args.resolve_to or None)
    mcp_cmd = [args.node, str(Path(args.mcp_cli).resolve()), "--headless", "--isolated", "--browser", "chromium",
               "--config", str(config_path.resolve())]
    if args.no_sandbox:
        mcp_cmd.append("--no-sandbox")
    if args.scripted:
        script = json.loads(Path(args.scripted).read_text(encoding="utf-8"))
        model: Any = ScriptedModel(script["calls"], script.get("final", "DONE"))
        agent = "scripted"
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"{args.api_key_env} is not set", file=sys.stderr)
            return 2
        model = OpenAIToolModel(args.base_url, api_key, args.model, args.reasoning_effort, args.max_output_tokens, args.provider_json)
        agent = args.model

    episode_paths: list[Path] = []
    for task in tasks:
        episode_path = output / "episodes" / f"webarena__{task['task_id']}__{args.run}.json"
        if episode_path.exists() and not args.force:
            print(f"task {task['task_id']}: episode exists, skipping")
            episode_paths.append(episode_path)
            continue
        print(f"task {task['task_id']} [{site_of_task(task)}]: {task['intent'][:90]}", flush=True)
        result = run_task(task, model=model, mcp_cmd=mcp_cmd, cwd=cwd, max_steps=args.max_steps, keep_results=args.keep_results,
                          log_path=output / "raw" / "calls.jsonl")
        raw_path = output / "raw" / f"task_{task['task_id']}_{args.run}.json"
        raw_path.write_text(json.dumps({k: v for k, v in result.items() if k != "task"} | {"task_id": task["task_id"], "intent": task["intent"]},
                                       indent=1, ensure_ascii=False), encoding="utf-8")
        outcome = {"status": result["status"], "answer": result["answer"], "error": result["error"]}
        episode = build_episode(task, result["turns"], run=args.run, agent=agent, outcome=outcome, metrics=result["metrics"],
                                fold=not args.no_fold)
        episode_path.write_text(json.dumps(episode, indent=1, ensure_ascii=False), encoding="utf-8")
        episode_paths.append(episode_path)
        summary = {"task_id": task["task_id"], "site": episode["site"], "status": result["status"], "answer": result["answer"],
                   "turns": episode["turns"], "raw_turns": episode["raw_turns"], **result["metrics"], "error": result["error"]}
        with (output / "summary.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"  -> {result['status']} after {episode['raw_turns']} calls ({episode['turns']} turns), "
              f"{result['metrics']['prompt_tokens']} prompt tokens, {result['metrics']['seconds']} s", flush=True)
    manifest = split_manifest_for(episode_paths)
    (output / "split_manifest.json").write_text(manifest.model_dump_json(indent=1), encoding="utf-8")
    print(f"{len(episode_paths)} episodes; split manifest written to {output / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
