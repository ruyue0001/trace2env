#!/usr/bin/env python3
"""Statistics and sanity checks of an EnvScaler rollout collection, per environment. No model, nothing executed.

    python work/exp-envscaler/data_stats.py --data work/envscaler --output work/exp-envscaler/data_stats.json

Reads the raw rollout files (`benchmark/<env>[__*]`, `reserve/<env>`), `env_defs/`, and `selection_manifest.json`
as they are, before any export. Per environment and split: rollouts and their success fields, tool calls, the
shape of every response (`data` / `message` / `error`), what a `data` response is made of (stored records copied
verbatim, nothing, or something computed), generated values, calls to the manifest's target tools, sizes
(including the initial database size, although evaluation now keeps it latent by default), and tool coverage
of the construction side. `problems` lists what a
study built on the collection has to know; the exit status is non-zero only for integrity errors (a checksum
mismatch, an unreadable or unsuccessful rollout), not for design gaps.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.envscaler import AGENT_SIDE_TOOLS, database, iter_rollouts, rollout_turns  # noqa: E402
from trace2env.envscaler_scoring import UNPARSED, kind_of, parse_observation  # noqa: E402

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CLOCK = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b1[5-9]\d{8}(?:\.\d+)?\b")
SPLITS = ("benchmark", "reserve")


def env_directories(data: Path) -> dict[str, dict[str, Path]]:
    found: dict[str, dict[str, Path]] = defaultdict(dict)
    for split in SPLITS:
        for directory in sorted((data / split).glob("*")) if (data / split).is_dir() else []:
            if directory.is_dir():
                found[directory.name.split("__")[0]][split] = directory
    return dict(found)


def records_of(state: dict[str, Any] | None) -> list[Any]:
    """Every stored record of a database: the values of its collections (dicts keyed by id, or lists)."""
    records: list[Any] = []
    for collection in (database(state) or {}).values():
        if isinstance(collection, dict):
            records.extend(collection.values())
        elif isinstance(collection, list):
            records.extend(collection)
    return records


def data_class(value: Any, records: list[Any]) -> str:
    """What a `data` response is made of: `copy` (stored records verbatim), `empty`, or `computed` (anything else)."""
    if value in ([], {}, None, ""):
        return "empty"
    items = value if isinstance(value, list) else [value]
    return "copy" if all(any(item == record for record in records) for item in items) else "computed"


def quantiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(ordered), "median": int(statistics.median(ordered)), "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max": ordered[-1]}


def source_scan(code: str) -> dict[str, Any]:
    tree = ast.parse(code)
    imports, calls = set(), Counter()
    sets = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(str(node.module))
        elif isinstance(node, (ast.Set, ast.SetComp)):
            sets += 1
        elif isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if re.search(r"uuid|\bnow\b|utcnow|today|time\.time|random|^set$|^frozenset$", name):
                calls[name] += 1
    return {"imports": sorted(imports), "generated_or_unordered_calls": dict(calls), "set_literals_or_comprehensions": sets,
            "rejection_returns": len(re.findall(r"[\"']success[\"']\s*:\s*False", code))}


def split_stats(directory: Path | None, tools: set[str], targets: set[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {"rollouts": 0}
    if directory is None:
        return stats
    kinds, labels, per_tool, classes, class_by_tool = Counter(), Counter(), Counter(), Counter(), defaultdict(Counter)
    target_calls, target_kinds, target_answers = Counter(), defaultdict(Counter), defaultdict(Counter)
    lengths, state_sizes, turn_counts, tasks, integrity, notes = [], [], [], [], [], []
    counts = Counter()
    for path, rollout in iter_rollouts([directory]):
        stats["rollouts"] += 1
        task = str(rollout.get("task_id"))
        tasks.append(task)
        turns = rollout_turns(rollout)
        trajectory = rollout.get("trajectory") or []
        turn_counts.append(len(turns))
        counts["agent_side_calls"] += sum(1 for step in trajectory if (step.get("action") or {}).get("name") in AGENT_SIDE_TOOLS)
        state_sizes.append(len(json.dumps(database(rollout.get("init_state")) or {}, ensure_ascii=False)))
        if not (rollout.get("reward") == 1.0 and rollout.get("label") == "success"):
            integrity.append(f"{path.name}: reward={rollout.get('reward')} label={rollout.get('label')}")
        if rollout.get("truncated") or not rollout.get("terminated"):
            # The step limit ended the rollout after the task was already solved: every recorded turn is still a real answer.
            notes.append(f"{path.name}: ended by the step limit ({len(trajectory)} steps; terminated={rollout.get('terminated')}, reward {rollout.get('reward')})")
        if rollout.get("checklist_pass") != rollout.get("checklist_total"):
            integrity.append(f"{path.name}: checklist {rollout.get('checklist_pass')}/{rollout.get('checklist_total')}")
        if rollout.get("steps") not in (len(trajectory), len(turns)):
            counts["steps_field_differs"] += 1
        steps = {step.get("step", index): step for index, step in enumerate(trajectory, start=1)}
        for turn in turns:
            counts["tool_calls"] += 1
            name, text = turn["name"], turn["output"]
            per_tool[name] += 1
            value = parse_observation(text)
            kind = kind_of(value)
            kinds[kind] += 1
            labels[str(turn["truth"]["outcome_label"])] += 1
            lengths.append(len(text))
            counts["state_changing"] += 1 if turn["truth"]["n_delta"] else 0
            counts["undefined_tool"] += 1 if name not in tools else 0
            counts["harness_error"] += 1 if turn["harness_error"] else 0
            counts["stale_observation_raw"] += 1 if turn["stale_raw"] else 0
            counts["unparsed_observation"] += 1 if value is UNPARSED else 0
            counts["with_uuid"] += 1 if UUID.search(text) else 0
            counts["with_clock_value"] += 1 if CLOCK.search(text) else 0
            if kind == "data":
                made_of = data_class(value.get("data"), records_of((steps.get(turn["step"]) or {}).get("state_before")))
                classes[made_of] += 1
                class_by_tool[name][made_of] += 1
            if name in targets:
                target_calls[name] += 1
                target_kinds[name][kind] += 1
                target_answers[name][text] += 1
    stats.update({
        "tasks": sorted(tasks), "tool_calls": counts["tool_calls"], "turns_per_rollout": quantiles(turn_counts),
        "response_kinds": dict(kinds), "outcome_labels": dict(labels),
        "rows_without_successful_reads": counts["tool_calls"] - kinds["data"],
        "data_responses_made_of": dict(classes), "data_responses_by_tool": {name: dict(made) for name, made in sorted(class_by_tool.items())},
        "state_changing_calls": counts["state_changing"], "calls_per_tool": dict(per_tool.most_common()),
        "tools_called": len(per_tool),
        # A target tool whose answer is always the same is answered by a constant, whatever it computes.
        "target_tool_calls": {name: {"calls": target_calls[name], "kinds": dict(target_kinds[name]), "distinct_answers": len(target_answers[name]),
                                     "most_common_answer_share": round(100 * max(target_answers[name].values()) / target_calls[name], 1)
                                     if target_calls[name] else None} for name in sorted(targets)},
        "observation_chars": quantiles(lengths), "initial_database_chars": quantiles(state_sizes),
        "flags": {key: counts[key] for key in ("agent_side_calls", "undefined_tool", "harness_error", "stale_observation_raw",
                                                "unparsed_observation", "with_uuid", "with_clock_value", "steps_field_differs")},
        "integrity": integrity, "notes": notes,
    })
    return stats


def manifest_check(data: Path) -> dict[str, Any]:
    path = data / "selection_manifest.json"
    if not path.is_file():
        return {"present": False}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    missing, mismatched, verified = Counter(), [], 0
    for entry in manifest.get("files") or []:
        target = data / entry["path"]
        if not target.is_file():
            missing["/".join(entry["path"].split("/")[:-1])] += 1
        elif hashlib.sha256(target.read_bytes()).hexdigest() != entry.get("sha256"):
            mismatched.append(entry["path"])
        else:
            verified += 1
    return {"present": True, "listed": len(manifest.get("files") or []), "verified": verified, "missing_by_directory": dict(missing),
            "checksum_mismatch": mismatched, "envs": manifest.get("envs") or {}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "work" / "envscaler")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "exp-envscaler" / "data_stats.json")
    parser.add_argument("--min-construction", type=int, default=10, help="fewer construction rollouts than this is reported")
    args = parser.parse_args()
    manifest = manifest_check(args.data)
    directories = env_directories(args.data)
    defined = sorted(path.name[:-len("_metadata.json")] for path in (args.data / "env_defs").glob("*_metadata.json"))
    report: dict[str, Any] = {"data": str(args.data), "manifest": {key: value for key, value in manifest.items() if key != "envs"},
                              "environments": {}, "problems": []}
    problems: list[str] = report["problems"]
    errors = 0
    if manifest.get("checksum_mismatch"):
        errors += 1
        problems.append(f"INTEGRITY: {len(manifest['checksum_mismatch'])} files differ from the manifest's checksum")
    for where, count in (manifest.get("missing_by_directory") or {}).items():
        problems.append(f"manifest lists {count} files under {where}/ that are not on disk")
    for env in sorted(set(defined) - set(directories)):
        problems.append(f"{env}: defined in env_defs/ but no rollouts on disk")
    for env, where in sorted(directories.items()):
        metadata_path = args.data / "env_defs" / f"{env}_metadata.json"
        if not metadata_path.is_file():
            errors += 1
            problems.append(f"INTEGRITY: {env} has rollouts but no env_defs metadata")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scenarios_path = args.data / "env_defs" / f"{env}_scenarios.json"
        scenarios = json.loads(scenarios_path.read_text(encoding="utf-8")) if scenarios_path.is_file() else []
        tools = {str((tool.get("function") or tool).get("name")) for tool in metadata.get("tools") or []}
        declared = (manifest.get("envs") or {}).get(env) or {}
        targets = set(declared.get("target_tools") or {})
        entry: dict[str, Any] = {"class": metadata.get("env_class_name"), "summary": metadata.get("environment_summary"),
                                 "tools_defined": len(tools), "scenario_tasks": len(scenarios),
                                 "manifest_status": declared.get("status"), "source": source_scan(metadata["env_class_code"])}
        for split in SPLITS:
            entry[split] = split_stats(where.get(split), tools, targets)
            errors += len(entry[split].get("integrity") or [])
            problems.extend(f"INTEGRITY: {env}/{split}: {item}" for item in entry[split].get("integrity") or [])
        benchmark, reserve = entry["benchmark"], entry["reserve"]
        shared = sorted(set(benchmark.get("tasks") or []) & set(reserve.get("tasks") or []))
        if shared:
            errors += 1
            problems.append(f"INTEGRITY: {env}: tasks in both benchmark and reserve: {shared}")
        uncovered = sorted(set(benchmark.get("calls_per_tool") or {}) - set(reserve.get("calls_per_tool") or {}))
        entry["benchmark_tools_never_called_in_reserve"] = uncovered
        if reserve["rollouts"] < args.min_construction:
            problems.append(f"{env}: {reserve['rollouts']} reserve rollouts ({reserve.get('tool_calls', 0)} tool calls) as construction set; "
                            f"{len(uncovered)} of the {benchmark.get('tools_called', 0)} tools the benchmark calls never occur there")
        for split in SPLITS:
            flags = entry[split].get("flags") or {}
            for key in ("undefined_tool", "harness_error", "unparsed_observation"):
                if flags.get(key):
                    problems.append(f"{env}/{split}: {flags[key]} calls flagged {key}")
        for name, calls in (benchmark.get("target_tool_calls") or {}).items():
            if not calls["calls"]:
                problems.append(f"{env}: target tool {name} is never called in the benchmark")
            elif calls["distinct_answers"] == 1 and calls["calls"] > 1:
                problems.append(f"{env}: target tool {name} gives the same answer in all {calls['calls']} benchmark calls")
        for split in SPLITS:
            problems.extend(f"note: {env}/{split}: {item}" for item in entry[split].get("notes") or [])
        kinds = benchmark.get("response_kinds") or {}
        if benchmark.get("tool_calls"):
            entry["benchmark_share_error"] = round(100 * kinds.get("error", 0) / benchmark["tool_calls"], 1)
        report["environments"][env] = entry
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    header = f"{'environment':<12}{'split':<10}{'rollouts':>9}{'calls':>7}{'data':>6}{'msg':>6}{'error':>6}{'copy':>6}{'empty':>6}{'comp.':>6}{'uuid':>6}{'obs p95':>9}{'db chars':>10}"
    print(header)
    print("-" * len(header))
    for env, entry in report["environments"].items():
        for split in SPLITS:
            stats = entry[split]
            if not stats["rollouts"]:
                print(f"{env:<12}{split:<10}{0:>9}")
                continue
            kinds, made = stats["response_kinds"], stats["data_responses_made_of"]
            print(f"{env:<12}{split:<10}{stats['rollouts']:>9}{stats['tool_calls']:>7}{kinds.get('data', 0):>6}{kinds.get('message', 0):>6}"
                  f"{kinds.get('error', 0):>6}{made.get('copy', 0):>6}{made.get('empty', 0):>6}{made.get('computed', 0):>6}"
                  f"{stats['flags']['with_uuid']:>6}{stats['observation_chars']['p95']:>9}{stats['initial_database_chars']['median']:>10}")
    print(f"\n{len(problems)} findings:")
    for problem in problems:
        print(" -", problem)
    print(f"-> {args.output}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
