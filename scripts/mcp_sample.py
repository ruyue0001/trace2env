"""Select MCP construction traces that do not overlap the AgentWorldBench mcp split.

Two steps, both deterministic and re-runnable:

    python scripts/mcp_sample.py overlap --benchmark work/agentworldbench/mcp_test.jsonl \
        --mcpmark work/mcp/traces/mcpmark/mcpmark-v1-0905/claude-opus-4-1__* \
        --toolathlon-run work/mcp/traces/toolathlon/opus48/aws_anthropic_bedrock-claude-opus-4-8_1 \
        --output work/mcp/benchmark_overlap.json

    python scripts/mcp_sample.py sample --overlap work/mcp/benchmark_overlap.json \
        --mcpmark work/mcp/traces/mcpmark/mcpmark-v1-0905/claude-opus-4-1__* \
        --toolathlon-runs work/mcp/traces/toolathlon/opus48/aws_anthropic_bedrock-claude-opus-4-8_{1,2,3} \
        --output work/mcp/sample_manifest.json

``overlap`` matches every benchmark trajectory to a local task by the rare content tokens shared
between the benchmark's observations and the local trajectory's tool outputs (paths and ids differ
between runs, so exact argument matching fails). A confident match excludes its task; a weak match
(one-row trajectories carry little text) excludes both its best and second candidates.
``sample`` draws one trajectory per remaining task: per-service quotas for MCPMark, a fixed count
for Toolathlon preferring tasks whose servers the benchmark exercises and a passing run when one
exists, and marks a small validation hold-out whose trajectory groups are disjoint from training.
Rewards are recorded, never used as a filter: a failed attempt is still a valid environment trace.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import re
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{5,}")
SKIP_PREFIXES = ("backup_", "mcpmark", "toolu_", "call_", "tooluse_")
BENCHMARK_SERVERS = {"emails", "canvas", "excel", "pdf-tools", "filesystem", "github", "k8s", "snowflake", "google_sheet",
                     "google-cloud", "arxiv_local", "scholarly", "terminal", "memory", "google_calendar", "pptx", "yahoo-finance"}
STRONG_SCORE = 30.0
CONTEXT_TOOL_PREFIXES = ("local_search_overlong", "local_view_overlong", "local_check_context", "local_manage_context", "local_history")


def tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN.findall(text) if not t.lower().startswith(SKIP_PREFIXES)}


def expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(Path(m) for m in matches) if matches else paths.append(Path(pattern))
    return paths


# ─── Local trajectories ───────────────────────────────────────────────────────

def mcpmark_runs(service_dirs: list[Path]) -> dict[str, dict]:
    """``task -> {service, dir, solved, harness_error, turns, text}`` for one run per task."""
    runs: dict[str, dict] = {}
    for service_dir in service_dirs:
        for meta_path in sorted(service_dir.glob("run-*/*/meta.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            messages = json.loads((meta_path.parent / "messages.json").read_text(encoding="utf-8"))
            text = " ".join((m.get("output") or "") + " " + (m.get("arguments") or "") for m in messages
                            if m.get("type") in ("function_call", "function_call_output"))
            task = str(meta["task_name"])
            if task in runs:
                continue  # first run per task
            result = meta.get("execution_result") or {}
            runs[task] = {"service": meta["mcp"], "dir": str(meta_path.parent), "solved": bool(result.get("success")),
                          "harness_error": bool(result.get("error_message")), "turns": meta.get("turn_count"),
                          "text": text + " " + str(messages[0].get("content", "") if messages else "")}
    return runs


def toolathlon_run(run_dir: Path) -> dict[str, dict]:
    """``task -> {dir, solved, tool_calls, servers, text}`` for every task directory of one run."""
    tasks: dict[str, dict] = {}
    for log_path in sorted(run_dir.glob("finalpool/*/traj_log.json")):
        log = json.loads(log_path.read_text(encoding="utf-8"))
        verdict_path = log_path.parent / "eval_res.json"
        verdict = json.loads(verdict_path.read_text(encoding="utf-8")) if verdict_path.is_file() else {}
        calls = sum(len(m.get("tool_calls") or []) for m in log.get("messages") or [] if m.get("role") == "assistant"
                    if not all(str(c.get("function", {}).get("name", "")).startswith(CONTEXT_TOOL_PREFIXES) for c in m.get("tool_calls") or []))
        text = " ".join(json.dumps(m.get("content", "")) + json.dumps(m.get("tool_calls", "")) for m in log.get("messages") or [])
        tasks[log_path.parent.name] = {"dir": str(log_path.parent), "solved": verdict.get("pass"), "tool_calls": calls,
                                       "servers": list((log.get("config") or {}).get("needed_mcp_servers") or []),
                                       "status": log.get("status"), "text": text}
    return tasks


# ─── Benchmark trajectories ───────────────────────────────────────────────────

def benchmark_trajectories(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["id"])].append(row)
    out = {}
    for tid, group in grouped.items():
        names = set()
        for row in group:
            match = re.search(r'"name"\s*:\s*"([^"]+)"', row["current_prompt"])
            if match:
                names.add(match.group(1))
        toolathlon = any(("-" in n and not n.startswith("API-")) or n == "local-claim_done" for n in names)
        text = " ".join(r["response"][-1] for r in group) + " ".join(group[0]["prompt"]) + " ".join(r["current_prompt"] for r in group)
        backups = sorted({m for m in re.findall(r"backup_(?:filesystem|postgres|notion|github|playwright)_([a-z0-9_]+?)_\d+", text)})
        out[tid] = {"source": "toolathlon" if toolathlon else "mcpmark", "rows": len(group), "total_turns": group[0]["total_turns"],
                    "tools": sorted(names), "tokens": tokens(text), "backup_names": backups}
    return out


def match(query: set[str], corpus: dict[str, set[str]], *, max_df: int = 6) -> list[tuple[float, str, int]]:
    df: collections.Counter = collections.Counter()
    for keys in corpus.values():
        df.update(keys)
    n = len(corpus)
    scored = []
    for name, keys in corpus.items():
        shared = query & keys
        score = sum(math.log(n / df[t]) for t in shared if df[t] <= max_df)
        scored.append((round(score, 1), name, len(shared)))
    return sorted(scored, reverse=True)


def cmd_overlap(args: argparse.Namespace) -> None:
    bench = benchmark_trajectories(Path(args.benchmark))
    mcpmark = mcpmark_runs(expand(args.mcpmark))
    toolathlon = toolathlon_run(Path(args.toolathlon_run))
    corpora = {"mcpmark": {t: tokens(v["text"]) for t, v in mcpmark.items()},
               "toolathlon": {t: tokens(v["text"]) for t, v in toolathlon.items()}}
    backup_to_task = {}
    for task in mcpmark:
        family, _, rest = task.partition("__")
        backup_to_task[f"{family}_{rest}"] = task
    records, excluded = [], {"mcpmark": set(), "toolathlon": set()}
    for tid, info in bench.items():
        ranked = match(info["tokens"], corpora[info["source"]])
        best, second = ranked[0], ranked[1]
        confirmed = [backup_to_task[b] for b in info["backup_names"] if b in backup_to_task]
        strong = bool(confirmed) or best[0] >= STRONG_SCORE
        chosen = confirmed or [best[1]] if strong else [best[1], second[1]]
        excluded[info["source"]].update(chosen)
        records.append({"trajectory": tid, "source": info["source"], "rows": info["rows"], "total_turns": info["total_turns"],
                        "tools": info["tools"][:6], "best": best, "second": second, "confirmed_by_backup_name": confirmed,
                        "strong": strong, "excluded_tasks": chosen})
    summary = {"benchmark": str(args.benchmark), "trajectories": len(bench),
               "by_source": dict(collections.Counter(r["source"] for r in records)),
               "strong_matches": sum(r["strong"] for r in records), "strong_score": STRONG_SCORE,
               "excluded": {k: sorted(v) for k, v in excluded.items()}, "matches": records}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "matches"} | {"excluded": {k: len(v) for k, v in excluded.items()}}, indent=1))


# ─── Sampling ─────────────────────────────────────────────────────────────────

def parse_quota(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (item.split("=") for item in text.split(",") if item)}


def cmd_sample(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    overlap = json.loads(Path(args.overlap).read_text(encoding="utf-8"))
    excluded = {k: set(v) for k, v in overlap["excluded"].items()}
    manifest_dir = Path(args.output).resolve().parent

    def rel(path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(manifest_dir))
        except ValueError:
            return str(Path(path).resolve())

    # MCPMark: per-service quotas over the non-excluded tasks; validation = single-task families.
    runs = mcpmark_runs(expand(args.mcpmark))
    quota = parse_quota(args.quota)
    pool: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for task, info in runs.items():
        if task in excluded["mcpmark"]:
            continue
        service = info["service"].replace("playwright_webarena", "playwright")
        pool[service].append((task, info))
    entries = []
    for service, count in quota.items():
        candidates = sorted(pool[service])
        rng.shuffle(candidates)
        chosen = candidates[:count]
        families = collections.Counter(task.split("__")[0] for task, _ in chosen)
        singles = [task for task, _ in chosen if families[task.split("__")[0]] == 1]
        validation = set(rng.sample(sorted(singles), min(args.validation_per_service, len(singles))))
        for task, info in sorted(chosen):
            entries.append({"source": "mcpmark", "task": task, "service": info["service"], "dir": rel(info["dir"]),
                            "split": "validation" if task in validation else "train", "solved": info["solved"],
                            "harness_error": info["harness_error"], "turns": info["turns"]})
    # Toolathlon: one trajectory per task, preferring a passing run, then the lowest run id.
    per_task: dict[str, list[dict]] = collections.defaultdict(list)
    for run_dir in expand(args.toolathlon_runs):
        run = run_dir.name.rsplit("_", 1)[-1]
        for task, info in toolathlon_run(run_dir).items():
            if task in excluded["toolathlon"] or not info["tool_calls"]:
                continue
            per_task[task].append({**info, "run": run})
    candidates = []
    for task, infos in per_task.items():
        infos.sort(key=lambda i: (i["solved"] is not True, i["run"]))
        candidates.append((task, infos[0]))
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: -len(set(item[1]["servers"]) & BENCHMARK_SERVERS))
    chosen_ta = sorted(candidates[:args.toolathlon])
    validation_ta = set(rng.sample([task for task, _ in chosen_ta], min(args.validation_toolathlon, len(chosen_ta))))
    for task, info in chosen_ta:
        entries.append({"source": "toolathlon", "task": task, "service": ",".join(info["servers"]), "dir": rel(info["dir"]),
                        "split": "validation" if task in validation_ta else "train", "solved": info["solved"],
                        "run": info["run"], "tool_calls": info["tool_calls"]})
    manifest = {"seed": args.seed, "overlap": str(args.overlap), "quota": quota, "toolathlon": args.toolathlon,
                "protocol": "task-level exclusion of benchmark-matched tasks; one trajectory per task; validation groups disjoint "
                            "from training (MCPMark: single-task families); rewards recorded, not filtered",
                "pool": {"mcpmark": {s: len(v) for s, v in sorted(pool.items())}, "toolathlon": len(candidates)},
                "entries": entries}
    Path(args.output).write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = collections.Counter((e["source"], e["split"]) for e in entries)
    print(json.dumps({"entries": len(entries), "counts": {f"{s}/{sp}": n for (s, sp), n in sorted(counts.items())},
                      "solved": sum(1 for e in entries if e["solved"]), "pool": manifest["pool"], "output": str(args.output)}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    overlap = sub.add_parser("overlap", help="Match benchmark trajectories to local tasks and list exclusions")
    overlap.add_argument("--benchmark", required=True, help="AgentWorldBench mcp_test.jsonl")
    overlap.add_argument("--mcpmark", nargs="+", required=True, help="MCPMark <model>__<service> directories (globs allowed)")
    overlap.add_argument("--toolathlon-run", required=True, help="One Toolathlon run directory (finalpool/<task>/traj_log.json)")
    overlap.add_argument("--output", required=True)
    overlap.set_defaults(func=cmd_overlap)
    sample = sub.add_parser("sample", help="Draw the construction sample and write the manifest for `trace2env mcp-export`")
    sample.add_argument("--overlap", required=True)
    sample.add_argument("--mcpmark", nargs="+", required=True)
    sample.add_argument("--toolathlon-runs", nargs="+", required=True, help="Toolathlon run directories, in preference order")
    sample.add_argument("--quota", default="filesystem=18,notion=13,postgres=10,github=12,playwright=12")
    sample.add_argument("--toolathlon", type=int, default=35)
    sample.add_argument("--validation-per-service", type=int, default=1)
    sample.add_argument("--validation-toolathlon", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    sample.add_argument("--output", required=True)
    sample.set_defaults(func=cmd_sample)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
