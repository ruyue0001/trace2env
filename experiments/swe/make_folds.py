"""SWE cross-fit (k=5): trajectories, scaffold sub-sources, stratified folds, per-(scaffold, fold) construction corpora,
the trajectory -> package / corpus maps, shards and the offline path-overlap diagnostic. Rows are the benchmark's own
swe_test.jsonl (never modified); ids are unique in this split, so the derived full472_rows.jsonl carries every field
unchanged plus `benchmark_id`. Every trajectory is a developer session on its own repository with one coding-agent
scaffold (Claude Code tools or Gemini/Qwen CLI tools); a row is predicted with the package of its scaffold built from
the trajectories outside its fold.
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "work/exp-swe-cv")
from trace2env.agentworld import load_rows, normalize_action, trajectory_episode  # noqa: E402
from trace2env.models import SplitAssignment, SplitManifest  # noqa: E402
from trace2env.storage import write_json  # noqa: E402
from trajectory_identity import cluster_trajectories, unique_id  # noqa: E402

ROOT = Path("work/exp-swe-cv")
K = 5
SEED = 20260924
CLAUDE = {"Read", "Edit", "Bash", "Glob", "Grep", "TodoWrite", "Task", "Write", "WebSearch", "ExitPlanMode", "AskUserQuestion",
          "KillShell", "BashOutput", "WebFetch", "MultiEdit", "NotebookEdit", "Skill", "SlashCommand"}
GEMINI = {"run_shell_command", "read_file", "write_file", "list_directory", "glob", "todo_write", "read_many_files", "edit",
          "search_file_content", "replace", "web_fetch", "google_web_search", "save_memory", "exit_plan_mode"}
STYLES = ("claude_code", "gemini_cli")
PATH_TOKEN = re.compile(r"(?<![\w.])((?:[\w.-]+/){2,}[\w.-]+\.[A-Za-z0-9]{1,6})")


def scaffold_of(prompts: list[str]) -> str:
    names = {normalize_action("swe", p).type for p in prompts}
    if names & CLAUDE and not names & GEMINI:
        return "claude_code"
    if names & GEMINI and not names & CLAUDE:
        return "gemini_cli"
    return "claude_code" if len(names & CLAUDE) >= len(names & GEMINI) else "gemini_cli"


def rare_paths(text: str) -> set[str]:
    return {t for t in PATH_TOKEN.findall(text) if "node_modules" not in t and "site-packages" not in t}


def main() -> None:
    rows = load_rows(["work/agentworldbench/swe_test.jsonl"])
    clusters = cluster_trajectories(rows)
    uid_of_row: dict[int, str] = {}
    trajectories: dict[str, dict] = {}
    paths: dict[str, set[str]] = {}
    for (task, tid, ordinal), members in clusters.items():
        uid = unique_id(task, tid, ordinal)
        longest = max(members, key=lambda r: len(r["prompt"]))
        style = scaffold_of(longest["prompt"])
        paths[uid] = rare_paths("\n".join(longest["prompt"]) + "\n".join(longest["response"]))
        trajectories[uid] = {
            "benchmark_id": tid, "style": style, "rows": len(members), "turns": [int(r["turn_idx"]) for r in members],
            "visible_turns": len(longest["prompt"]), "total_turns": longest.get("total_turns"),
            "first_sha": hashlib.sha256(longest["prompt"][0].encode()).hexdigest()[:12],
            "tools": sorted({normalize_action("swe", p).type for p in longest["prompt"]}),
        }
        for row in members:
            uid_of_row[id(row)] = uid
    derived = [{**row, "id": uid_of_row[id(row)], "benchmark_id": str(row["id"])} for row in rows]
    assert len(derived) == len(rows) and len(set(r["id"] for r in derived)) == len(trajectories)
    with open(ROOT / "full472_rows.jsonl", "w", encoding="utf-8") as f:
        for row in derived:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # groups: identical first turn (exact duplicates); repositories shared by several sessions stay allowed across folds,
    # as the same app was on android — coverage is the point of the cross-fit
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for uid, t in trajectories.items():
        groups[f"first:{t['first_sha']}"].append(uid)
    rng = random.Random(SEED)
    fold_style: dict[tuple[int, str], int] = collections.Counter()
    fold_turns: dict[int, int] = collections.Counter()
    fold_of: dict[str, int] = {}
    ordered = sorted(groups.items(), key=lambda kv: (trajectories[kv[1][0]]["style"], -sum(trajectories[u]["visible_turns"] for u in kv[1]), kv[0]))
    for g, uids in ordered:
        style = trajectories[uids[0]]["style"]
        keys = [(fold_style[(f, style)], fold_turns[f], rng.random()) for f in range(K)]
        fold = min(range(K), key=lambda f: keys[f])
        for u in uids:
            fold_of[u] = fold
            fold_style[(fold, style)] += 1
            fold_turns[fold] += trajectories[u]["visible_turns"]
    for uid, t in trajectories.items():
        t["fold"] = fold_of[uid]
        t["package"] = f"ws/{t['style']}-f{t['fold']}/packages/{t['style']}-f{t['fold']}"
        t["corpus"] = f"corpora/{t['style']}-f{t['fold']}/episodes"

    longest_rows: dict[str, dict] = {}
    for row in derived:
        if row["id"] not in longest_rows or len(row["prompt"]) > len(longest_rows[row["id"]]["prompt"]):
            longest_rows[row["id"]] = row
    corpora = {}
    for style in STYLES:
        for fold in range(K):
            members = sorted(u for u, t in trajectories.items() if t["style"] == style and t["fold"] != fold)
            out = ROOT / "corpora" / f"{style}-f{fold}" / "episodes"
            out.mkdir(parents=True, exist_ok=True)
            assignments = {}
            for uid in members:
                episode = trajectory_episode(longest_rows[uid])
                episode["split"] = "train"
                path = out / f"swe_{uid}.json"
                write_json(path, episode)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assignments[f"src_{digest}"] = SplitAssignment(split="train", trajectory_group=episode["trajectory_group"])
            write_json(out.parent / "split_manifest.json", SplitManifest(assignments=assignments).model_dump(mode="json"))
            corpora[f"{style}-f{fold}"] = {"style": style, "fold": fold, "trajectories": members, "count": len(members),
                                           "transitions": sum(trajectories[u]["visible_turns"] for u in members)}
    shards = ROOT / "full472_shards"
    shards.mkdir(exist_ok=True)
    for old in shards.glob("traj_*.jsonl"):
        old.unlink()
    for uid in trajectories:
        with open(shards / f"traj_{uid}.jsonl", "w", encoding="utf-8") as f:
            for row in sorted((r for r in derived if r["id"] == uid), key=lambda r: int(r["turn_idx"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(ROOT / "full472_rows_ids.json", [{"id": r["id"], "turn_idx": r["turn_idx"]} for r in derived])
    write_json(ROOT / "trajectories.json", trajectories)
    write_json(ROOT / "corpora.json", corpora)
    write_json(ROOT / "package_map.json", {uid: t["package"] for uid, t in trajectories.items()})
    write_json(ROOT / "package_map_schema.json", {uid: t["package"] + "-schema_only" for uid, t in trajectories.items()})
    write_json(ROOT / "corpus_map.json", {uid: t["corpus"] for uid, t in trajectories.items()})

    # strata: repository shared with an out-of-fold trajectory of the same scaffold (≥ 5 rare paths in common)
    df = collections.Counter(t for s in paths.values() for t in s)
    strata = {}
    for row in derived:
        t = trajectories[row["id"]]
        pool = [u for u, x in trajectories.items() if x["style"] == t["style"] and x["fold"] != t["fold"]]
        shared = max((sum(1 for p in paths[row["id"]] & paths[u] if df[p] <= 3) for u in pool), default=0)
        strata[f"{row['id']}|{row['turn_idx']}"] = {"style": t["style"], "fold": t["fold"], "repo_shared": shared >= 5, "shared_paths": shared}
    write_json(ROOT / "row_strata.json", strata)
    print(f"rows {len(derived)} | trajectories {len(trajectories)} | groups {len(groups)}")
    for style in STYLES:
        ts = [t for t in trajectories.values() if t["style"] == style]
        print(f"  {style}: {len(ts)} trajectories, {sum(t['rows'] for t in ts)} rows, {sum(t['visible_turns'] for t in ts)} visible transitions | per fold: "
              + ", ".join(f"f{f}={sum(1 for t in ts if t['fold'] == f)}/{sum(t['rows'] for t in ts if t['fold'] == f)}r/{sum(t['visible_turns'] for t in ts if t['fold'] == f)}t" for f in range(K)))
    print("  rows with a shared repository out of fold:", sum(1 for s in strata.values() if s["repo_shared"]))


if __name__ == "__main__":
    main()
