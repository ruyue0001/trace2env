"""Command-line interface for reconstruction, inspection, simulation, and AgentWorldBench evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from trace2env.adapters import load_raw_traces, segment_transitions
from trace2env.agentworld import (
    aggregate_scores,
    export_episodes,
    format_score_report,
    judge_rows,
    load_rows,
    split_rows,
    trace2env_diagnostics,
)
from trace2env.agentworld_runner import AgentWorldRunner
from trace2env.compiler import EnvironmentCompiler
from trace2env.examples import build_ledger_demo
from trace2env.llm import (
    CachedAgentLLM,
    CachedChatLLM,
    CachedLLM,
    TracingLLM,
    TracingAgentLLM,
    OpenAIChatLLM,
    OpenAIChatStructuredLLM,
    OpenAIResponsesLLM,
    OpenAIToolLLM,
    StructuredAgentLLM,
)
from trace2env.models import (
    EnvironmentState,
    NormalizedAction,
    PackagePatch,
    ReconstructionArtifacts,
    ReconstructionConfig,
    ReplayCase,
    SplitManifest,
    StepRequest,
)
from trace2env.package import EnvironmentPackage, PackageInspector
from trace2env.patching import apply_package_patch
from trace2env.reconstruction import TraceReconstructor
from trace2env.replay import evaluate_package
from trace2env.runtime import RuntimeHarness
from trace2env.storage import read_json, read_jsonl, write_json, write_jsonl

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_JUDGE_MODEL = "gpt-5.2-2025-12-11"


def _json_argument(value: str) -> dict:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _text_argument(value: str) -> str:
    return Path(value[1:]).read_text(encoding="utf-8") if value.startswith("@") else value


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _api_key(args: argparse.Namespace, attribute: str = "api_key_env") -> str | None:
    """Keys are read from a named environment variable so they never appear on a command line."""
    name = getattr(args, attribute, None) or "OPENAI_API_KEY"
    return os.environ.get(name)


def _chat_reasoning(args: argparse.Namespace) -> str | None:
    """Reasoning effort for Chat Completions transports; ``off`` sends no reasoning parameter at all."""
    value = getattr(args, "chat_reasoning_effort", None)
    return None if value in (None, "", "off") else value


def _chat_reasoning_json(args: argparse.Namespace) -> dict | None:
    """The OpenRouter-style ``reasoning`` object for Chat Completions transports (``--chat-reasoning-json``);
    when set it replaces ``--chat-reasoning-effort`` in the request."""
    value = getattr(args, "chat_reasoning_json", None)
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SystemExit("--chat-reasoning-json must be a JSON object such as '{\"enabled\": false}'")
    return parsed


def _chat_provider_json(args: argparse.Namespace) -> dict | None:
    """OpenRouter provider routing for the backbone's Chat Completions calls (``--chat-provider-json``), e.g.
    '{"order": ["DeepSeek"], "allow_fallbacks": false}' pins the upstream provider; never applied to the judge."""
    value = getattr(args, "chat_provider_json", None)
    if not value:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise SystemExit("--chat-provider-json must be a JSON object such as '{\"order\": [\"DeepSeek\"], \"allow_fallbacks\": false}'")
    return parsed


def _features(args: argparse.Namespace) -> set[str] | str:
    """Parse ``--features``: the named sets pass through, ``none`` is empty, otherwise a comma list in which
    ``default`` / ``all`` expand to the named set (``default,evidence`` adds one feature to the defaults)."""
    value = getattr(args, "features", "default") or "default"
    if value in ("all", "default"):
        return value
    if value == "none":
        return set()
    from trace2env.runtime import DEFAULT_FEATURES, HARNESS_FEATURES

    selected: set[str] = set()
    for name in (item.strip() for item in value.split(",")):
        if name == "default":
            selected.update(DEFAULT_FEATURES)
        elif name == "all":
            selected.update(HARNESS_FEATURES)
        elif name and name != "none":
            selected.add(name)
    return selected


def _trace_corpus(args: argparse.Namespace):
    """Index the raw-trace directory named by ``--trace-corpus`` (under the session root, else a temp dir)."""
    source = getattr(args, "trace_corpus", None)
    if not source:
        return None
    from trace2env.trace_corpus import TraceCorpus

    root = getattr(args, "session_root", None)
    database = (Path(root) if root else Path(tempfile.mkdtemp(prefix="trace2env-corpus-"))) / "trace_corpus.sqlite"
    corpus = TraceCorpus.build(source, database)
    if corpus.turns == 0:
        raise ValueError(f"No action/observation turns were indexed from {source}")
    return corpus


def _resolved_features(args: argparse.Namespace) -> set[str]:
    """The concrete feature set for a harness constructed directly (``simulate``)."""
    from trace2env.runtime import DEFAULT_FEATURES, HARNESS_FEATURES

    selected = _features(args)
    if selected == "default":
        return set(DEFAULT_FEATURES)
    if selected == "all":
        return set(HARNESS_FEATURES)
    return set(selected)


def _llm(args: argparse.Namespace, *, max_tokens: int | None = None):
    base_url = getattr(args, "base_url", None)
    if getattr(args, "provider", "responses") == "chat":
        provider = OpenAIChatStructuredLLM(model=args.model, base_url=base_url, api_key=_api_key(args),
                                           temperature=getattr(args, "structured_temperature", None),
                                           reasoning_effort=_chat_reasoning(args), reasoning=_chat_reasoning_json(args),
                                           provider_routing=_chat_provider_json(args),
                                           max_tokens=max_tokens or getattr(args, "max_output_tokens", None) or 65536,
                                           schema_mode=getattr(args, "chat_schema_mode", "json_schema"))
    else:
        provider = OpenAIResponsesLLM(model=args.model, reasoning_effort=args.reasoning_effort, base_url=base_url,
                                      api_key=_api_key(args))
    cache_dir = getattr(args, "cache_dir", None)
    llm = CachedLLM(provider, cache_dir) if cache_dir else provider
    call_log = getattr(args, "call_log", None)
    return TracingLLM(llm, call_log) if call_log else llm


def _chat_llm(args: argparse.Namespace, *, model: str, base_url: str | None, key_attribute: str = "api_key_env"):
    provider = OpenAIChatLLM(model=model, base_url=base_url, api_key=_api_key(args, key_attribute),
                             temperature=args.temperature, max_tokens=args.max_tokens,
                             reasoning=_chat_reasoning_json(args) if key_attribute == "api_key_env" else None,
                             provider_routing=_chat_provider_json(args) if key_attribute == "api_key_env" else None)
    cache_dir = getattr(args, "cache_dir", None)
    return CachedChatLLM(provider, cache_dir) if cache_dir else provider


def _agent_llm(args: argparse.Namespace):
    """The world-model agent: native function calling, or typed turns over the structured provider."""
    cap = getattr(args, "agent_max_output_tokens", None)
    if getattr(args, "agent_transport", "structured") == "tools":
        provider = OpenAIToolLLM(model=args.model, base_url=getattr(args, "base_url", None), api_key=_api_key(args),
                                 temperature=getattr(args, "structured_temperature", None),
                                 reasoning_effort=_chat_reasoning(args), reasoning=_chat_reasoning_json(args),
                                 provider_routing=_chat_provider_json(args),
                                 **({"max_tokens": cap} if cap else {}))
        cache_dir = getattr(args, "cache_dir", None)
        agent = CachedAgentLLM(provider, cache_dir) if cache_dir else provider
        call_log = getattr(args, "call_log", None)
        return TracingAgentLLM(agent, call_log) if call_log else agent
    return StructuredAgentLLM(_llm(args, max_tokens=cap))


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default="medium", help="Responses API reasoning effort")
    parser.add_argument("--chat-reasoning-effort", default="medium",
                        help="Reasoning effort for Chat Completions transports (--provider chat, --agent-transport tools); "
                             "'medium' is the setting every reported run used, 'off' sends no reasoning parameter "
                             "(servers that reject it)")
    parser.add_argument("--chat-reasoning-json", default=None,
                        help="OpenRouter-style reasoning object for Chat Completions transports, e.g. '{\"enabled\": false}' "
                             "or '{\"effort\": \"low\"}'; sent as the request's `reasoning` field instead of "
                             "--chat-reasoning-effort (for models that ignore reasoning_effort, such as moonshotai/kimi-k2.6)")
    parser.add_argument("--chat-provider-json", default=None,
                        help="OpenRouter provider routing object for the backbone's Chat Completions calls, e.g. "
                             "'{\"order\": [\"DeepSeek\"], \"allow_fallbacks\": false}' to pin the upstream provider "
                             "(never applied to the judge); part of the call cache identity")
    parser.add_argument("--chat-schema-mode", choices=["json_schema", "json_object"], default="json_schema",
                        help="How structured chat calls (--provider chat) send the response schema: json_schema "
                             "(response_format json_schema, the reported runs) or json_object (response_format json_object "
                             "with the schema quoted in the system text, for Anthropic models via OpenRouter, whose "
                             "schema-constrained decoding rejects Any-typed fields and returns free-form maps empty)")
    parser.add_argument("--structured-temperature", type=float, default=None,
                        help="Sampling temperature for structured/tool chat calls; omitted by default (reasoning models reject it)")
    parser.add_argument("--max-output-tokens", type=int, default=65536,
                        help="Output token limit for structured chat calls (default 65536, sized for gpt-5.6-sol: induction "
                             "results with provenance and long tracked screens exceed smaller limits)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint (OpenRouter, vLLM, SGLang, ...)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Environment variable holding the API key (e.g. OPENROUTER_API_KEY); EMPTY for local servers")
    parser.add_argument("--provider", choices=["responses", "chat"], default="responses",
                        help="Structured-call transport: the OpenAI Responses API or Chat Completions JSON schema")
    parser.add_argument("--cache-dir")
    parser.add_argument("--call-log", help="Append every structured model call (prompt, payload, output, usage) to this JSONL file")


def _add_agent_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-transport", choices=["structured", "tools"], default="structured",
                        help="How the world-model agent runs: typed AgentTurn calls over --provider, or native "
                             "function calling over Chat Completions (--base-url servers such as vLLM)")
    parser.add_argument("--fast-path", choices=["confident", "always", "never"], default="confident",
                        help="When an applicable rule may short-circuit the agent: confident (templated, high "
                             "confidence), always, or never")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Workspace tool budget per simulated action")
    parser.add_argument("--trust-policy", choices=["rules_only", "schema_checked"], default=None,
                        help="schema_checked lets the agent propose effects on declared mutable paths")
    parser.add_argument("--features", default="default",
                        help="Online-harness features: 'default' (history,wait,compact,unknown,evidence — the measured best set), "
                             "'all' (adds the scaffold), 'none' (legacy harness), or a comma list of "
                             "history,wait,compact,scaffold,unknown")
    parser.add_argument("--history-turn-chars", type=int, default=6000,
                        help="Verbatim chars per recent turn in the agent's brief before head/tail clipping")
    parser.add_argument("--history-total-chars", type=int, default=60000, help="Byte budget for recent turns in the brief")
    parser.add_argument("--agent-max-output-tokens", type=int, default=16384,
                        help="Output token limit for world-model agent turns and single-shot predictions (default 16384: the "
                             "99.9th percentile of observed agent answers is ~6k tokens; the tracker keeps --max-output-tokens)")
    parser.add_argument("--prediction-mode", choices=["agent", "single_shot"], default="agent",
                        help="How a step without a confident rule is predicted: the tool-using agent loop (default), or "
                             "single_shot — one model call over the brief plus a fixed retrieval procedure, no tools "
                             "(the workspace_single_shot control)")
    parser.add_argument("--no-state-tracking", dest="state_tracking", action="store_false", default=True,
                        help="Disable persistent structured state tracking and the state tools (the trace2env_no_state "
                             "control): observed turns stay in memory, the agent relies on history and knowledge, effects are dropped")
    parser.add_argument("--no-package-knowledge", dest="package_knowledge", action="store_false", default=True,
                        help="Hide the package from the agent entirely (the harness_only control): no action specs, rules, "
                             "contracts, demonstrations, notes, evidence, or knowledge tools; only the environment prompt, "
                             "the episode's history, and the memory tools remain (combine with --no-state-tracking)")
    parser.add_argument("--no-knowledge-tools", dest="knowledge_tools", action="store_false", default=True,
                        help="Remove the knowledge tools (list_actions, inspect_action, search_knowledge, read_evidence) and "
                             "similar-turn retrieval while keeping schemas, state tracking, memory tools, and the loop "
                             "(the schema_only_hv3 control)")
    parser.add_argument("--trace-corpus", metavar="DIR",
                        help="Directory of raw construction trace files to index for the agent's search_traces / "
                             "read_trace_turn tools (the agentic_raw_traces control; pair it with a schema-only package)")


def _progress(label: str):
    def report(done: int, total: int) -> None:
        if done % 10 == 0 or done == total:
            print(f"[{label}] {done}/{total}", file=sys.stderr)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trace2env", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Normalize and segment raw traces without an API call")
    ingest.add_argument("inputs", nargs="+")
    ingest.add_argument("--output", required=True)

    reconstruct = subparsers.add_parser("reconstruct", help="Recover and compile an environment package")
    reconstruct.add_argument("inputs", nargs="+")
    reconstruct.add_argument("--work-dir", required=True)
    reconstruct.add_argument("--output", required=True)
    reconstruct.add_argument("--environment-id", required=True)
    reconstruct.add_argument("--name", required=True)
    reconstruct.add_argument("--description", default="", help="Text or @path to a file, e.g. a domain description")
    reconstruct.add_argument("--domain", action="append", default=[])
    reconstruct.add_argument("--tenant-scope")
    reconstruct.add_argument("--environment-version")
    reconstruct.add_argument("--max-prompt-bytes", type=int, default=600000,
                             help="Byte budget per induction call (default 600000, sized for a 1M-context model; lower it for smaller contexts)")
    reconstruct.add_argument("--induction-batch-size", type=int, default=100,
                             help="Evidence records per induction chunk (default 100, the setting of the reported packages)")
    reconstruct.add_argument("--split-manifest", help="SplitManifest JSON keyed by source content digests")
    _add_model_options(reconstruct)

    inspect = subparsers.add_parser("inspect", help="Validate and inspect a compiled package")
    inspect.add_argument("package")
    inspect.add_argument("--action")
    inspect.add_argument("--search")

    simulate = subparsers.add_parser("simulate", help="Execute one transition in a persistent session")
    simulate.add_argument("package")
    simulate.add_argument("--session", required=True)
    simulate.add_argument("--action", required=True, help='JSON or @path, e.g. {"type":"balance"}')
    simulate.add_argument("--agent", "--agentic-fallback", dest="agent", action="store_true",
                          help="Let the world-model agent simulate the action through the workspace tools")
    simulate.add_argument("--llm-verifier", action="store_true")
    simulate.add_argument("--llm-renderer", action="store_true")
    simulate.add_argument("--allow-unvalidated", action="store_true", help="Explicitly simulate an unvalidated candidate")
    simulate.add_argument("--initial-state", help="EnvironmentState JSON or @path for a new reconstructed session")
    _add_model_options(simulate)
    _add_agent_options(simulate)

    long_run = subparsers.add_parser("envscaler-long-horizon", help="Task-agent/world-model rollouts with oracle-verified pass@k")
    long_run.add_argument("package", help="Reconstructed package for one EnvScaler environment")
    long_run.add_argument("--env-defs", required=True, help="Directory with evaluation-only metadata and scenarios")
    long_run.add_argument("--env-id", required=True, help="Environment ID, e.g. env_160_rl")
    long_selection = long_run.add_mutually_exclusive_group(required=True)
    long_selection.add_argument("--task-id", action="append", help="Run this scenario ID; repeat for multiple tasks")
    long_selection.add_argument("--benchmark-dir", help="Held-out rollout directory; use only its task IDs (never its actions or observations)")
    long_run.add_argument("--output", required=True, help="Fresh output directory for attempt sessions and scores")
    long_run.add_argument("--attempts", type=int, default=1, help="Fresh attempts per task (pass@k up to this value)")
    long_run.add_argument("--max-steps", type=int, default=50, help="Maximum task-agent actions per attempt")
    long_run.add_argument("--task-model", help="Task-agent model (defaults to --model)")
    long_run.add_argument("--task-temperature", type=float, help="Task-agent sampling temperature for --provider chat")
    long_run.add_argument("--allow-unvalidated", action="store_true", help="Allow a candidate package")
    _add_model_options(long_run)
    long_run.add_argument("--agent-transport", choices=["structured", "tools"], default="structured")
    long_run.add_argument("--agent-max-output-tokens", type=int, default=16384)
    long_run.add_argument("--max-tool-calls", type=int, default=8)
    long_run.add_argument("--trust-policy", choices=["rules_only", "schema_checked"], default="rules_only")
    long_run.add_argument("--fast-path", choices=["confident", "always", "never"], default="confident")
    long_run.add_argument("--features", default="default",
                          help="Harness feature set: default, all, none, or comma-separated names")

    demo = subparsers.add_parser("init-demo", help="Build the deterministic ledger environment package")
    demo.add_argument("--output", required=True)

    replay = subparsers.add_parser("replay", help="Validate a candidate against supplied observations; no API calls")
    replay.add_argument("package")
    replay.add_argument("--cases", required=True, help="ReplayCase JSONL; provide independent validation trajectories")
    replay.add_argument("--output", required=True, help="Case-level replay report")
    replay.add_argument("--baseline")
    replay.add_argument("--promote-to", help="New immutable package destination; requires passing validation")

    patch = subparsers.add_parser("patch", help="Apply typed edits to a new candidate, then replay it")
    patch.add_argument("package")
    patch.add_argument("--patch", required=True, help="PackagePatch JSON, anchored to the base artifact digest")
    patch.add_argument("--output", required=True, help="New candidate package directory")
    patch.add_argument("--cases", required=True)
    patch.add_argument("--report", required=True)
    patch.add_argument("--promote-to")

    export = subparsers.add_parser("awb-export", help="Write AgentWorldBench trajectories as episodes for reconstruction")
    export.add_argument("inputs", nargs="+", help="AgentWorldBench JSONL files or a directory of *_test.jsonl")
    export.add_argument("--output", required=True, help="Directory receiving one episode JSON per trajectory")
    export.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    export.add_argument("--manifest", help="Where to write the SplitManifest (default: <output>/split_manifest.json)")

    atif = subparsers.add_parser("atif-export", help="Convert Harbor/Terminus-2 ATIF trajectories into episodes and benchmark rows")
    atif.add_argument("inputs", nargs="+", help="Harbor job directories (searched recursively for trajectory.json)")
    atif.add_argument("--output", required=True, help="Directory receiving one episode JSON per trial")
    atif.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    atif.add_argument("--manifest", help="Where to write the SplitManifest (default: <output>/split_manifest.json)")
    atif.add_argument("--rows", help="Also write AgentWorldBench-format records (JSONL) for the exported episodes")
    atif.add_argument("--turns-per-trajectory", type=int, help="Evenly spaced evaluated turns per trajectory in --rows (default: all)")
    atif.add_argument("--summary", help="Write per-trial summary JSON (turns, reward, tokens, cost)")

    mcp = subparsers.add_parser("mcp-export", help="Convert sampled MCPMark / Toolathlon trajectories into episodes and benchmark rows")
    mcp.add_argument("manifest", help="Sample manifest JSON with entries of {source, dir, split} (see scripts/mcp_sample.py)")
    mcp.add_argument("--output", required=True, help="Directory receiving one episode JSON per trajectory (+ tools/)")
    mcp.add_argument("--root", help="Base directory for relative entry dirs (default: the manifest's directory)")
    mcp.add_argument("--split-manifest", help="Where to write the SplitManifest (default: <output>/split_manifest.json)")
    mcp.add_argument("--rows", help="Also write AgentWorldBench-format records (JSONL) for the exported episodes")
    mcp.add_argument("--rows-split", choices=["train", "validation", "test", "all"], default="validation",
                     help="Which exported episodes get rows (default: validation)")
    mcp.add_argument("--turns-per-trajectory", type=int, help="Evenly spaced evaluated turns per trajectory in --rows (default: all)")
    mcp.add_argument("--summary", help="Write per-trajectory summary JSON (source, split, turns, outcome)")

    envscaler = subparsers.add_parser("envscaler-export", help="Convert EnvScaler rollouts into episodes, ground truth, and benchmark rows")
    envscaler.add_argument("inputs", nargs="+", help="Rollout JSON files or directories (searched recursively; other JSON is skipped)")
    envscaler.add_argument("--output", required=True, help="Directory receiving one episode JSON per task (+ tools/)")
    envscaler.add_argument("--env-defs", help="Directory with <env>_metadata.json / <env>_scenarios.json (tool definitions, task texts)")
    envscaler.add_argument("--assign-split", choices=["train", "validation", "test"],
                           help="Assign every exported rollout to this split (default: trajectory-group hash, 80/10/10)")
    envscaler.add_argument("--manifest", help="Where to write the SplitManifest (default: <output>/split_manifest.json)")
    envscaler.add_argument("--ground-truth", help="Also write per-episode ground truth (initial/final database, per-turn diff and "
                                                  "outcome label) to this directory; evaluation only, never a reconstruction input")
    envscaler.add_argument("--descriptions", help="Also draft one domain description per environment (<env>.md) in this directory, "
                                                  "from the train-split episodes; an existing file is kept, never overwritten")
    envscaler.add_argument("--rows", help="Also write AgentWorldBench-format records (JSONL) for the exported episodes")
    envscaler.add_argument("--rows-split", choices=["train", "validation", "test", "all"], default="all",
                           help="Which exported episodes get rows (default: all)")
    envscaler.add_argument("--turns-per-trajectory", type=int, help="Evenly spaced evaluated turns per trajectory in --rows (default: all)")
    envscaler.add_argument("--unique-turns", choices=["none", "episode", "environment", "answer"], default="none",
                           help="Evaluate only unique turns (histories stay complete). episode: drop calls to tools the environment "
                                "lacks and, when a call repeats in an episode, all but its last occurrence; environment: also keep one of "
                                "the same call with the same answer across episodes; answer: also keep one per tool and identical "
                                "answer text (a much stronger cut)")
    envscaler.add_argument("--repeated-calls", choices=["last", "same-answer"], default="last",
                           help="With --unique-turns, a call repeated within an episode keeps only its last occurrence (last), or "
                                "loses an occurrence only when a later one returns the same answer (same-answer: a rejection that "
                                "later succeeds, or a read before and after a write, both stay)")
    envscaler.add_argument("--filter-report", help="Write what --unique-turns dropped, and why, to this JSON file")
    envscaler.add_argument("--examples-from", help="Directory of exported construction episodes (split train): the first call of each "
                                                   "tool there fills the rows' few-shot section, as in the official mcp records")
    envscaler.add_argument("--few-shot-limit", type=_nonnegative_int,
                           help="Maximum construction examples in each environment prompt; selected evenly across tool-definition "
                                "order (default: all; 0 removes the few-shot section)")
    envscaler_state = envscaler.add_mutually_exclusive_group()
    envscaler_state.add_argument("--initial-state", dest="initial_state", action="store_true", default=False,
                                 help="Show the initial database before turn 1 (opt-in ablation; official mcp rows do not "
                                      "contain an environment-initialization section)")
    envscaler_state.add_argument("--no-initial-state", dest="initial_state", action="store_false",
                                 help="Keep the initial database latent (default; retained as an explicit compatibility flag)")
    envscaler.add_argument("--summary", help="Write per-episode summary JSON (split, turns, reward, outcome-label counts)")

    probe = subparsers.add_parser("envscaler-probe", help="Counterfactual benchmark rows for held-out EnvScaler rollouts, with ground "
                                                          "truth from the environment's own source (executes that source; evaluation only)")
    probe.add_argument("inputs", nargs="+", help="Held-out rollout JSON files or directories")
    probe.add_argument("--env-defs", required=True, help="Directory with <env>_metadata.json: the environment source is executed from it")
    probe.add_argument("--output", required=True, help="Probe rows (JSONL), same record format as envscaler-export --rows plus a probe id")
    probe.add_argument("--ground-truth", help="Directory receiving probes/<task>.json: operator, outcome label, state diff per probe")
    probe.add_argument("--per-trajectory", type=int, default=10, help="Probe rows per rollout (default 10)")
    probe.add_argument("--assign-split", choices=["train", "validation", "test"], default="test",
                       help="Split recorded on the rows (default: test; probes are an evaluation device)")
    probe.add_argument("--examples-from", help="Directory of exported construction episodes for the few-shot section (use the "
                                               "same one as for envscaler-export, so probe rows share the recorded rows' system prompt)")
    probe.add_argument("--few-shot-limit", type=_nonnegative_int,
                       help="Maximum construction examples in each environment prompt, selected as for envscaler-export")
    probe_state = probe.add_mutually_exclusive_group()
    probe_state.add_argument("--initial-state", dest="initial_state", action="store_true", default=False,
                             help="Show the initial database before turn 1 (opt-in ablation)")
    probe_state.add_argument("--no-initial-state", dest="initial_state", action="store_false",
                             help="Keep the initial database latent (default; retained as an explicit compatibility flag)")
    probe.add_argument("--seed", type=int, default=0, help="Seed of the ids the environment generates during probe execution")

    exact = subparsers.add_parser("envscaler-score", help="Exact (judge-free) accuracy of predicted EnvScaler observations")
    exact.add_argument("--predictions", required=True, help="Predictions JSONL from awb-run on EnvScaler rows (recorded and/or probe rows)")
    exact.add_argument("--summary", help="Optional JSON summary path")
    exact.add_argument("--scored", help="Optional JSONL with the per-row result added under 'exact'")

    run = subparsers.add_parser("awb-run", help="Predict AgentWorldBench observations (Trace2Env harness or prompting baseline)")
    run.add_argument("inputs", nargs="+", help="AgentWorldBench JSONL files or a directory of *_test.jsonl")
    run.add_argument("--output", required=True, help="Predictions JSONL: benchmark rows plus gen and trace2env fields")
    run.add_argument("--mode", choices=["agentic", "prompting", "prompting_rag", "envpack_prompting"], required=True,
                     help="agentic: the Trace2Env harness; prompting: the official baseline; prompting_rag: the official "
                          "baseline plus a fixed top-k of raw turns retrieved lexically from --trace-corpus; envpack_prompting: "
                          "the official baseline plus a fixed non-agentic view of --package (schemas, rules, contracts, notes, "
                          "demonstrations, top-k evidence retrieved lexically for the current action), one call per row")
    run.add_argument("--rag-top-k", type=int, default=5, help="prompting_rag: retrieved raw turns per prediction")
    run.add_argument("--rag-turn-chars", type=int, default=4000, help="prompting_rag: observation characters shown per retrieved turn")
    run.add_argument("--rag-style", choices=["terminal", "generic"], default="terminal",
                     help="prompting_rag: 'terminal' (default; every reported AgentWorldBench run) queries with the typed command "
                          "words, else the action type and argument words, under the terminal wording of the retrieved-examples "
                          "block; 'generic' (the EnvScaler runs) queries rows without keystrokes with the trace corpus's normalized "
                          "action query under environment-neutral wording")
    run.add_argument("--envpack-top-k", type=int, default=6, help="envpack_prompting: package evidence turns retrieved per prediction")
    run.add_argument("--envpack-evidence-chars", type=int, default=3000, help="envpack_prompting: observation characters shown per evidence turn")
    run.add_argument("--package", help="Environment package for --mode agentic")
    run.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    run.add_argument("--limit", type=int)
    run.add_argument("--effect-rejection", choices=["fail", "keep_observation"], default="keep_observation",
                     help="When a model-proposed transition's effects fail validation/verification: fail the step (the GPT-5.6-Sol "
                          "runs' behaviour) or keep the agent's observation with the effects dropped (default since 2026-09-21)")
    run.add_argument("--history-window", type=int, default=12, help="Observed turns shown in the world-model agent's brief")
    run.add_argument("--official-input", action="store_true",
                     help="Harness v5.1 information preservation: the benchmark's own model input for the turn (system prompt, "
                          "full history, current message, built by the prompting baseline's input code) is placed once, verbatim "
                          "and unclipped, in the world-model agent's input; loop, tools, tracking, and verification unchanged")
    run.add_argument("--knowledge-gate", action="store_true",
                     help="Harness v5.2 compatibility gate: every package item shown to the agent (brief and tools) carries an "
                          "applicability label, provenance and reason; foreign concrete values are masked in the model-facing view; "
                          "contradicted notes are dropped; the brief says when the gate abstains. Off reproduces v5.1")
    run.add_argument("--knowledge-gate-mode", choices=["full", "format_only"], default="full",
                     help="With --knowledge-gate: 'full' is v5.2; 'format_only' withholds supporting and uncertain evidence so the "
                          "agent sees only sanitized format-level knowledge (pre-analysis ablation)")
    run.add_argument("--knowledge-names", choices=["paths", "pages", "screens"], default="paths",
                     help="Harness v5.3.2: with --knowledge-gate, 'pages' makes page identity the supporting signal (an evidence "
                          "item about this episode's current page, or about the page a browser_navigate leads to, is shown intact) "
                          "for web environments whose sites are one shared instance; 'screens' (v5.3.3) does the same for Android "
                          "screens (same app and resource ids, or the same state id, or the same opened app); 'paths' is the v5.3.1 file-name model")
    run.add_argument("--knowledge-judge", action="store_true",
                     help="Harness v5.3: with --knowledge-gate, the structured provider also judges each candidate's applicability "
                          "(role knowledge_applicability); 'supporting' is honoured only when an anchor it names is verified in "
                          "the episode's own transcript or current action; failures fall back to the deterministic label")
    run.add_argument("--history-full", action="store_true",
                     help="Harness v4 information parity: every observed turn of the trajectory verbatim in the brief, ignoring "
                          "--history-window and the history character budgets (the same transcript the prompting layout sees)")
    run.add_argument("--state-window", type=int, default=8, help="Observed turns per state-tracking model call")
    run.add_argument("--observation-chars", type=int, default=4000, help="Screen chars per turn in the state tracker's payload")
    run.add_argument("--environment-prompt-chars", type=int, default=24000,
                     help="Bound on the record's system prompt as the environment description of the state tracker, the agent "
                          "(outside --official-input, which places the official system message whole) and the renderer; 24000 is "
                          "the setting of every reported AgentWorldBench run, 0 passes the prompt whole (the EnvScaler runs)")
    run.add_argument("--allow-unvalidated", action="store_true", help="Evaluate a candidate package")
    run.add_argument("--no-model", action="store_true", help="Agentic mode with rules only: no tracking, agent, or rendering model")
    run.add_argument("--session-root", help="Keep per-trajectory sessions (state, memory, audits) under this directory")
    run.add_argument("--temperature", type=float, default=0.6, help="Prompting-baseline sampling temperature (official default)")
    run.add_argument("--max-tokens", type=int, default=32768)
    _add_model_options(run)
    _add_agent_options(run)

    judge = subparsers.add_parser("awb-judge", help="Score predictions with the official AgentWorldBench LLM judge")
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--output", required=True, help="Judged JSONL with the official score fields")
    judge.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    judge.add_argument("--judge-base-url")
    judge.add_argument("--judge-api-key-env", default="OPENAI_API_KEY",
                       help="Environment variable holding the judge API key")
    judge.add_argument("--temperature", type=float, default=0.6)
    judge.add_argument("--max-tokens", type=int, default=32768)
    judge.add_argument("--max-retries", type=int, default=3)
    judge.add_argument("--cache-dir")
    judge.add_argument("--summary", help="Optional JSON summary path")

    score = subparsers.add_parser("awb-score", help="Aggregate judged predictions into the official per-domain report")
    score.add_argument("--predictions", required=True, help="Judged JSONL")
    score.add_argument("--summary", help="Optional JSON summary path")
    return parser


def _envscaler_examples(directory: str | None, env_id: str, tool_definitions: list | None,
                        limit: int | None = None) -> list[dict] | None:
    """Few-shot examples of one environment from a directory of exported construction episodes."""
    if not directory:
        return None
    from trace2env.envscaler import few_shot_examples, limit_few_shot_examples

    episodes = [read_json(path) for path in sorted(Path(directory).glob("*.json")) if not path.name.startswith(".")]
    episodes = [episode for episode in episodes if isinstance(episode, dict) and episode.get("env_id") == env_id and "events" in episode]
    return limit_few_shot_examples(few_shot_examples(episodes, tool_definitions), limit)


def _awb_rows(args: argparse.Namespace) -> list[dict]:
    rows = load_rows(args.inputs)
    split = getattr(args, "split", "all")
    return rows if split == "all" else split_rows(rows, split)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        episodes = [episode for path in args.inputs for episode in load_raw_traces(path)]
        transitions = [transition for episode in episodes for transition in segment_transitions(episode)]
        output = Path(args.output)
        write_jsonl(output / "episodes.jsonl", episodes)
        write_jsonl(output / "transitions.jsonl", transitions)
        print(json.dumps({"episodes": len(episodes), "transitions": len(transitions), "output": str(output)}))
        return 0

    if args.command == "reconstruct":
        config = ReconstructionConfig(
            environment_id=args.environment_id,
            name=args.name,
            description=_text_argument(args.description),
            domains=args.domain,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            tenant_scope=args.tenant_scope,
            environment_version=args.environment_version,
            max_prompt_bytes=args.max_prompt_bytes,
            induction_batch_size=args.induction_batch_size,
        )
        manifest = SplitManifest.model_validate(read_json(Path(args.split_manifest))) if args.split_manifest else None
        output = TraceReconstructor(_llm(args), config, args.work_dir).run(args.inputs, args.output, manifest)
        print(str(output.resolve()))
        return 0

    if args.command == "inspect":
        inspector = PackageInspector(EnvironmentPackage(args.package))
        if args.action:
            value = inspector.inspect_action(args.action)
        elif args.search:
            value = inspector.search(args.search)
        else:
            value = inspector.summary()
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
        return 0

    if args.command == "simulate":
        raw = _json_argument(args.action)
        request = (
            StepRequest.model_validate(raw)
            if "action" in raw
            else StepRequest(action=NormalizedAction.model_validate(raw))
        )
        single_shot = args.prediction_mode == "single_shot"
        llm = _llm(args) if any((args.llm_verifier, args.llm_renderer)) else None
        harness = RuntimeHarness(
            args.package,
            args.session,
            agent_llm=_agent_llm(args) if args.agent and not single_shot else None,
            single_shot_llm=_llm(args, max_tokens=args.agent_max_output_tokens) if args.agent and single_shot else None,
            prediction_mode=args.prediction_mode,
            trace_corpus=_trace_corpus(args),
            state_tracking=args.state_tracking,
            package_knowledge=args.package_knowledge,
            knowledge_tools=args.knowledge_tools,
            verifier_llm=llm if args.llm_verifier else None,
            renderer_llm=llm if args.llm_renderer else None,
            allow_unvalidated=args.allow_unvalidated,
            initial_state=EnvironmentState.model_validate(_json_argument(args.initial_state)) if args.initial_state else None,
            trust_policy=args.trust_policy or "rules_only",
            fast_path=args.fast_path,
            max_tool_calls=args.max_tool_calls,
            features=_resolved_features(args),
            history_turn_chars=args.history_turn_chars,
            history_total_chars=args.history_total_chars,
        )
        result = harness.step(request)
        print(result.model_dump_json(indent=2))
        return 0

    if args.command == "envscaler-long-horizon":
        from trace2env.envscaler_oracle import load_metadata
        from trace2env.long_horizon import LongHorizonRunner, load_scenarios

        metadata = load_metadata(args.env_defs, args.env_id)
        if args.benchmark_dir:
            from trace2env.envscaler import iter_rollouts

            selected_rollouts = [row for _, row in iter_rollouts([args.benchmark_dir])]
            if not selected_rollouts or any(row["env_id"] != args.env_id for row in selected_rollouts):
                raise ValueError("Benchmark directory is empty or contains another environment")
            task_ids = {row["task_id"] for row in selected_rollouts}
            if len(task_ids) != len(selected_rollouts):
                raise ValueError("Benchmark directory has duplicate task IDs")
        else:
            task_ids = set(args.task_id)
        scenarios = load_scenarios(Path(args.env_defs) / f"{args.env_id}_scenarios.json", args.env_id, task_ids)
        if args.benchmark_dir:
            scenario_by_id = {row["task_id"]: row for row in scenarios}
            if any(scenario_by_id[row["task_id"]]["init_config"] != row["init_state"] for row in selected_rollouts):
                raise ValueError("A scenario initial state differs from its held-out rollout")
        task_args = argparse.Namespace(**vars(args))
        task_args.model = args.task_model or args.model
        task_args.structured_temperature = args.task_temperature
        task_args.cache_dir = None  # repeated attempts must make fresh task-agent calls
        runner = LongHorizonRunner(
            package_dir=args.package, metadata=metadata, task_llm=_llm(task_args),
            world_agent_llm=_agent_llm(args), output_dir=args.output,
            attempts_per_task=args.attempts, max_steps=args.max_steps,
            allow_unvalidated=args.allow_unvalidated, trust_policy=args.trust_policy or "rules_only",
            fast_path=args.fast_path, max_tool_calls=args.max_tool_calls, features=_resolved_features(args),
        )
        summary = runner.run(scenarios)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "init-demo":
        print(str(build_ledger_demo(args.output).resolve()))
        return 0

    if args.command in {"replay", "patch"}:
        package = EnvironmentPackage(args.package)
        artifacts = ReconstructionArtifacts.model_validate(read_json(package.root / "construction" / "artifacts.json"))
        config = ReconstructionConfig.model_validate(read_json(package.root / "construction" / "config.json"))
        cases = [ReplayCase.model_validate(item) for item in read_jsonl(Path(args.cases))]
        baseline = getattr(args, "baseline", None)
        package_dir = package.root
        if args.command == "patch":
            patch = PackagePatch.model_validate(read_json(Path(args.patch)))
            if set(patch.failed_case_ids) - {case.id for case in cases}:
                raise ValueError("Patch cites failed cases that were not supplied for replay")
            artifacts = apply_package_patch(artifacts, patch)
            package_dir = EnvironmentCompiler(config).compile(artifacts, args.output)
            baseline = package.root
        report = evaluate_package(package_dir, cases, baseline)
        report_path = args.report if args.command == "patch" else args.output
        write_json(Path(report_path), report)
        if args.promote_to:
            if not report.promotion_eligible:
                print(report.model_dump_json(indent=2))
                return 1
            EnvironmentCompiler(config).compile(artifacts, args.promote_to, report)
        print(report.model_dump_json(indent=2))
        return 0 if all(case.passed for case in report.cases) and report.cases else 1

    if args.command == "awb-export":
        rows = load_rows(args.inputs)
        paths, manifest = export_episodes(rows, args.output, split=None if args.split == "all" else args.split)
        manifest_path = Path(args.manifest) if args.manifest else Path(args.output) / "split_manifest.json"
        write_json(manifest_path, manifest)
        print(json.dumps({"trajectories": len(paths), "records": len(rows), "output": str(Path(args.output).resolve()),
                          "manifest": str(manifest_path.resolve())}))
        return 0

    if args.command == "atif-export":
        from trace2env.atif import episode_rows, export_trajectories

        paths, manifest, summaries = export_trajectories(args.inputs, args.output,
                                                          split=None if args.split == "all" else args.split)
        manifest_path = Path(args.manifest) if args.manifest else Path(args.output) / "split_manifest.json"
        write_json(manifest_path, manifest)
        if args.summary:
            write_json(Path(args.summary), summaries)
        rows_written = 0
        if args.rows:
            rows = [row for path in paths for row in episode_rows(read_json(path), turns_per_trajectory=args.turns_per_trajectory)]
            write_jsonl(Path(args.rows), rows)
            rows_written = len(rows)
        print(json.dumps({"trials": len(paths), "skipped": sum(1 for item in summaries if item.get("skipped")),
                          "turns": sum(item.get("turns", 0) for item in summaries), "rows": rows_written,
                          "cost_usd": round(sum(float(item.get("cost_usd") or 0) for item in summaries), 4),
                          "output": str(Path(args.output).resolve()), "manifest": str(manifest_path.resolve())}))
        return 0

    if args.command == "mcp-export":
        from trace2env.mcp_traces import episode_rows as mcp_rows, export_manifest

        paths, manifest, summaries = export_manifest(args.manifest, args.output, root=args.root)
        manifest_path = Path(args.split_manifest) if args.split_manifest else Path(args.output) / "split_manifest.json"
        write_json(manifest_path, manifest)
        if args.summary:
            write_json(Path(args.summary), summaries)
        rows_written = 0
        if args.rows:
            rows = []
            for path in paths:
                episode = read_json(path)
                if args.rows_split != "all" and episode["split"] != args.rows_split:
                    continue
                definitions = read_json(Path(args.output) / episode["tool_definitions"]) if episode.get("tool_definitions") else None
                rows.extend(mcp_rows(episode, tool_definitions=definitions, turns_per_trajectory=args.turns_per_trajectory))
            write_jsonl(Path(args.rows), rows)
            rows_written = len(rows)
        by_split = Counter(item.get("split") for item in summaries if not item.get("skipped"))
        print(json.dumps({"episodes": len(paths), "skipped": sum(1 for item in summaries if item.get("skipped")),
                          "turns": sum(item.get("turns", 0) for item in summaries), "splits": dict(by_split), "rows": rows_written,
                          "output": str(Path(args.output).resolve()), "split_manifest": str(manifest_path.resolve())}))
        return 0

    if args.command == "envscaler-export":
        from trace2env.envscaler import domain_description, episode_rows as envscaler_rows, export_rollouts, load_environment, state_layout

        paths, manifest, summaries = export_rollouts(args.inputs, args.output, env_defs=args.env_defs, split=args.assign_split,
                                                     ground_truth_dir=args.ground_truth)
        manifest_path = Path(args.manifest) if args.manifest else Path(args.output) / "split_manifest.json"
        write_json(manifest_path, manifest)
        if args.summary:
            write_json(Path(args.summary), summaries)
        episodes = [read_json(path) for path in paths]
        rows = []
        examples_by_env: dict[str, list | None] = {}
        shapes_by_env: dict[str, list[str] | None] = {}
        kept_turns: dict[str, list[int]] = {}
        filter_reports = {}
        if args.rows and args.unique_turns != "none":
            from trace2env.envscaler import unique_turns

            evaluated = [episode for episode in episodes if args.rows_split == "all" or episode["split"] == args.rows_split]
            for env_id in sorted({episode["env_id"] for episode in evaluated}):
                kept, filter_reports[env_id] = unique_turns([episode for episode in evaluated if episode["env_id"] == env_id],
                                                            args.unique_turns, args.repeated_calls)
                kept_turns.update(kept)
            if args.filter_report:
                write_json(Path(args.filter_report), filter_reports)
        if args.rows:
            for episode in episodes:
                if args.rows_split != "all" and episode["split"] != args.rows_split:
                    continue
                definitions = read_json(Path(args.output) / episode["tool_definitions"]) if episode.get("tool_definitions") else None
                if episode["env_id"] not in examples_by_env:
                    examples_by_env[episode["env_id"]] = _envscaler_examples(
                        args.examples_from, episode["env_id"], definitions, args.few_shot_limit
                    )
                initial = read_json(Path(episode["source_path"])).get("init_state") if args.initial_state else None
                if episode["env_id"] not in shapes_by_env:
                    shapes_by_env[episode["env_id"]] = load_environment(args.env_defs, episode["env_id"])["generated_shapes"]
                rows.extend(envscaler_rows(episode, tool_definitions=definitions, initial_state=initial,
                                           turns_per_trajectory=args.turns_per_trajectory, examples=examples_by_env[episode["env_id"]],
                                           turns=kept_turns.get(episode["trajectory_id"]) if filter_reports else None,
                                           generated_shapes=shapes_by_env[episode["env_id"]]))
            write_jsonl(Path(args.rows), rows)
        descriptions, kept_descriptions = [], []
        if args.descriptions:
            # Construction inputs only: a held-out rollout must not shape what the reconstructor is told.
            train = [episode for episode in episodes if episode["split"] == "train"]
            for env_id in sorted({episode["env_id"] for episode in train}):
                target = Path(args.descriptions) / f"{env_id}.md"
                if target.exists():
                    kept_descriptions.append(str(target))  # a description is edited by hand; a re-export never overwrites it
                    continue
                members = [episode for episode in train if episode["env_id"] == env_id]
                layout = state_layout(read_json(Path(episode["source_path"])).get("init_state") for episode in members)
                text = domain_description(load_environment(args.env_defs, env_id), layout,
                                          [name for episode in members for name in episode["tool_names"]])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                descriptions.append(str(target))
        kept = [item for item in summaries if not item.get("skipped")]
        print(json.dumps({"episodes": len(paths), "skipped": len(summaries) - len(kept),
                          "turns": sum(item["turns"] for item in kept), "splits": dict(Counter(item["split"] for item in kept)),
                          "environments": dict(Counter(item["env_id"] for item in kept)),
                          "rows": len(rows), "rows_by_split": dict(Counter(row["split"] for row in rows)),
                          "few_shot_examples": {env: len(items or []) for env, items in examples_by_env.items()},
                          "unique_turns": {env: {"level": item["level"], "repeated_calls": item["repeated_calls"],
                                                 "turns": item["turns"], "kept": item["kept"],
                                                 "dropped_by_reason": item["dropped_by_reason"]} for env, item in filter_reports.items()},
                          "without_instruction": sum(1 for item in kept if not item["instruction"]),
                          "stale_raw_observations": sum(item["stale_raw_observations"] for item in kept),
                          "descriptions": descriptions, "descriptions_kept": kept_descriptions,
                          "output": str(Path(args.output).resolve()),
                          "split_manifest": str(manifest_path.resolve())}))
        return 0

    if args.command == "envscaler-probe":
        from trace2env.envscaler import iter_rollouts, load_environment, rollout_episode
        from trace2env.envscaler_oracle import (EnvironmentOracle, load_metadata, probe_rows, replay_check, reproduces,
                                                system_prompt_for)

        by_env: dict[str, list[tuple[Path, dict]]] = {}
        for path, rollout in iter_rollouts(args.inputs):
            by_env.setdefault(str(rollout["env_id"]), []).append((path, rollout))
        rows, report = [], {}
        for env_id, members in sorted(by_env.items()):
            oracle = EnvironmentOracle(load_metadata(args.env_defs, env_id), seed=args.seed)
            check = replay_check((rollout for _, rollout in members), oracle)
            if not reproduces(check):
                # The source on disk is not the environment these rollouts were recorded in: its answers are not ground truth.
                print(json.dumps({"env_id": env_id, "replay_check": check}), file=sys.stderr)
                raise ValueError(f"the environment source of {env_id} does not reproduce its recorded rollouts; no probes were generated")
            environment = load_environment(args.env_defs, env_id)
            examples = _envscaler_examples(args.examples_from, env_id, environment["tools"], args.few_shot_limit)
            coverage: Counter = Counter()
            env_rows, env_truth = [], []
            for path, rollout in members:
                episode = rollout_episode(path, split=args.assign_split, environment=environment)
                made, truth = probe_rows(rollout, episode, oracle, system_prompt=system_prompt_for(oracle, examples),
                                         initial_state=rollout.get("init_state") if args.initial_state else None,
                                         per_trajectory=args.per_trajectory, coverage=coverage,
                                         generated_shapes=environment["generated_shapes"])
                env_rows.extend(made)
                env_truth.extend(truth)
                if args.ground_truth:
                    write_json(Path(args.ground_truth) / "probes" / f"{episode['benchmark_task']}.json", truth)
            rows.extend(env_rows)
            report[env_id] = {"replay_check": check, "rollouts": len(members), "probes": len(env_rows),
                              "outcome_labels": dict(Counter(item["outcome_label"] for item in env_truth)),
                              "operators": dict(Counter(item["operator"] for item in env_truth)),
                              "distinct_signatures": len(coverage),
                              "distinct_rejections": sum(1 for signature in coverage if signature[1] == "error")}
        write_jsonl(Path(args.output), rows)
        print(json.dumps({"rows": len(rows), "environments": report, "output": str(Path(args.output).resolve())}))
        return 0

    if args.command == "envscaler-score":
        from trace2env.envscaler_scoring import aggregate, format_report, score_rows

        scored = score_rows(read_jsonl(Path(args.predictions)))
        summary = aggregate(scored)
        if args.summary:
            write_json(Path(args.summary), summary)
        if args.scored:
            write_jsonl(Path(args.scored), scored)
        print(format_report(summary))
        return 0

    if args.command == "awb-run":
        rows = _awb_rows(args)
        if args.mode == "agentic":
            if not args.package:
                raise ValueError("--mode agentic requires --package")
            runner = AgentWorldRunner(
                mode="agentic", package_dir=args.package, llm=None if args.no_model else _llm(args),
                agent_llm=None if args.no_model else _agent_llm(args),
                history_window=args.history_window, state_window=args.state_window,
                observation_chars=args.observation_chars, trust_policy=args.trust_policy or "schema_checked",
                fast_path=args.fast_path, max_tool_calls=args.max_tool_calls,
                allow_unvalidated=args.allow_unvalidated, session_root=args.session_root,
                features=_features(args),
                history_turn_chars=args.history_turn_chars, history_total_chars=args.history_total_chars,
                history_full=args.history_full, official_input=args.official_input, knowledge_gate=args.knowledge_gate,
                knowledge_gate_mode=args.knowledge_gate_mode, knowledge_judge=args.knowledge_judge, knowledge_names=args.knowledge_names,
                prediction_mode=args.prediction_mode, trace_corpus=_trace_corpus(args),
                single_shot_llm=None if args.no_model else _llm(args, max_tokens=args.agent_max_output_tokens),
                state_tracking=args.state_tracking, package_knowledge=args.package_knowledge, knowledge_tools=args.knowledge_tools,
                effect_rejection=args.effect_rejection,
                environment_prompt_chars=args.environment_prompt_chars if args.environment_prompt_chars > 0 else None,
            )
        else:
            if args.mode == "envpack_prompting" and not args.package:
                raise ValueError("--mode envpack_prompting requires --package")
            runner = AgentWorldRunner(mode=args.mode, chat_llm=_chat_llm(args, model=args.model, base_url=args.base_url),
                                      rag_corpus=_trace_corpus(args) if args.mode == "prompting_rag" else None,
                                      rag_top_k=args.rag_top_k, rag_turn_chars=args.rag_turn_chars, rag_style=args.rag_style,
                                      package_dir=args.package if args.mode == "envpack_prompting" else None,
                                      allow_unvalidated=args.allow_unvalidated,
                                      envpack_top_k=args.envpack_top_k, envpack_evidence_chars=args.envpack_evidence_chars)
        predictions = runner.run(rows, limit=args.limit, progress=_progress("awb-run"))
        write_jsonl(Path(args.output), predictions)
        diagnostics = trace2env_diagnostics(predictions)
        print(json.dumps({"predictions": len(predictions), "missing": sum(not row.get("gen") for row in predictions),
                          "output": str(Path(args.output).resolve()), **diagnostics}))
        return 0

    if args.command == "awb-judge":
        rows = read_jsonl(Path(args.predictions))
        llm = _chat_llm(args, model=args.judge_model, base_url=args.judge_base_url, key_attribute="judge_api_key_env")
        judged = judge_rows(rows, llm, max_retries=args.max_retries, progress=_progress("awb-judge"))
        write_jsonl(Path(args.output), judged)
        summary = aggregate_scores(judged)
        if args.summary:
            write_json(Path(args.summary), summary)
        print(format_score_report(summary, trace2env_diagnostics(judged)))
        return 0

    if args.command == "awb-score":
        rows = read_jsonl(Path(args.predictions))
        summary = aggregate_scores(rows)
        if args.summary:
            write_json(Path(args.summary), summary)
        print(format_score_report(summary, trace2env_diagnostics(rows)))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
