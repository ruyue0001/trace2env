"""Android cross-fit (k=5): unique trajectory ids, task-grouped stratified folds, per-(sub-source, fold) construction
corpora, the trajectory -> package map, and the offline screen-identity diagnostic.

Rows are the benchmark's own android_test.jsonl (never modified): the derived full200_rows.jsonl carries every benchmark
field, `id` made unique per trajectory (`<id>~2`, `<id>~3` for the second/third trajectory sharing a benchmark id, in
file order) and the original in `benchmark_id`. A row is predicted with the package built from the trajectories of its
sub-source that are NOT in its fold; the construction corpus of (sub-source, fold f) is every longest visible record of
the sub-source outside fold f, exported as episodes with a train-only split manifest.
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
sys.path.insert(0, "work/exp-android-cv")
from trace2env.agentworld import case_from_row, load_rows, trajectory_episode  # noqa: E402
from trace2env.knowledge_gate import RESOURCE_ID, SYSTEM_PACKAGES, same_screen, screen_signature  # noqa: E402
from trace2env.models import SplitAssignment, SplitManifest  # noqa: E402
from trace2env.storage import write_json  # noqa: E402
from trajectory_identity import cluster_trajectories, unique_id  # noqa: E402

ROOT = Path("work/exp-android-cv")
K = 5
SEED = 20260924
TASK_TEXT = re.compile(r"\*\*Task Instruction:\*\*\s*\n(.*?)(?:\n\n|\Z)", re.S)
APP_LINE = re.compile(r"\*\*App:\*\*\s*(?:[^\n(]*\()?([a-zA-Z][\w.]*)\)?")
STYLES = ("json", "phone", "unparsed")


def style_of(text: str) -> str:
    if "**State ID:**" in text or "Accessibility Tree" in text:
        return "unparsed"
    if "**Current Phone State:**" in text:
        return "phone"
    return "json"


def state_section(prompt: str) -> str:
    return prompt.split("**Action:**")[0]


def primary_app(style: str, uid: str, rows: list[dict]) -> str:
    longest = max(rows, key=lambda r: len(r["prompt"]))
    if style == "unparsed":
        return uid.split("__", 1)[1] if "__" in uid else "unknown"
    texts = [state_section(p) for p in longest["prompt"]] + list(longest["response"])
    if style == "phone":
        for text in texts:
            m = APP_LINE.search(text)
            if m and m.group(1) not in SYSTEM_PACKAGES:
                return m.group(1)
    for prompt in longest["prompt"]:  # json: the first opened app, else the dominant non-system package on screen
        case_prompt = prompt
        m = re.search(r'"app_name"\s*:\s*"([^"]+)"', case_prompt.split("**Action:**")[-1])
        if m:
            return "app:" + m.group(1).strip().lower()
    counts = collections.Counter(i.split(":id/")[0] for text in texts for i in RESOURCE_ID.findall(text))
    for pkg, _ in counts.most_common():
        if pkg not in SYSTEM_PACKAGES:
            return pkg
    return "unknown"


def main() -> None:
    rows = load_rows(["work/agentworldbench/android_test.jsonl"])
    clusters = cluster_trajectories(rows)
    uid_of_row: dict[int, str] = {}
    trajectories: dict[str, dict] = {}
    for (task, tid, ordinal), members in clusters.items():
        uid = unique_id(task, tid, ordinal)
        longest = max(members, key=lambda r: len(r["prompt"]))
        style = style_of(case_from_row(longest).current_prompt)
        first = state_section(longest["prompt"][0])
        task_match = TASK_TEXT.search(longest["prompt"][0])
        task_text = " ".join(task_match.group(1).split()) if task_match else ""
        trajectories[uid] = {
            "benchmark_id": tid, "style": style, "rows": len(members), "turns": [int(r["turn_idx"]) for r in members],
            "visible_turns": len(longest["prompt"]), "total_turns": longest.get("total_turns"),
            "task_sha": hashlib.sha256(task_text.encode()).hexdigest()[:12] if task_text else "",  # the text itself stays out of the record
            "first_screen": hashlib.sha256(first.encode()).hexdigest()[:12], "app": primary_app(style, uid, members),
        }
        for row in members:
            uid_of_row[id(row)] = uid
    derived = [{**row, "id": uid_of_row[id(row)], "benchmark_id": str(row["id"])} for row in rows]  # file order
    assert len(derived) == len(rows) == 200 and len(set(r["id"] for r in derived)) == len(trajectories)
    with open(ROOT / "full200_rows.jsonl", "w") as f:
        for row in derived:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # task groups: identical task text, else identical first screen
    group_of: dict[str, str] = {}
    for uid, t in trajectories.items():
        group_of[uid] = f"task:{t['task_sha']}" if t["task_sha"] else f"screen:{t['first_screen']}"
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for uid, g in group_of.items():
        groups[g].append(uid)
    # stratified greedy assignment: balance sub-source counts per fold first, then app spread, then transitions
    rng = random.Random(SEED)
    fold_style: dict[tuple[int, str], int] = collections.Counter()
    fold_app: dict[tuple[int, str, str], int] = collections.Counter()
    fold_turns: dict[int, int] = collections.Counter()
    fold_of: dict[str, int] = {}
    ordered = sorted(groups.items(), key=lambda kv: (trajectories[kv[1][0]]["style"], -len(kv[1]), kv[0]))
    for g, uids in ordered:
        style = trajectories[uids[0]]["style"]
        apps = [trajectories[u]["app"] for u in uids]
        turns = sum(trajectories[u]["visible_turns"] for u in uids)
        keys = [(fold_style[(f, style)], sum(fold_app[(f, style, a)] for a in apps), fold_turns[f], rng.random()) for f in range(K)]
        fold = min(range(K), key=lambda f: keys[f])
        for u, a in zip(uids, apps):
            fold_of[u] = fold
            fold_style[(fold, style)] += 1
            fold_app[(fold, style, a)] += 1
            fold_turns[fold] += trajectories[u]["visible_turns"]
    for uid, t in trajectories.items():
        t["fold"] = fold_of[uid]
        t["group"] = group_of[uid]
        t["package"] = f"ws/{t['style']}-f{t['fold']}/packages/{t['style']}-f{t['fold']}"  # relative to work/exp-android-cv

    # construction corpora
    longest_rows = {}
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
                path = out / f"android_{uid}.json"
                write_json(path, episode)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assignments[f"src_{digest}"] = SplitAssignment(split="train", trajectory_group=episode["trajectory_group"])
            write_json(out.parent / "split_manifest.json", SplitManifest(assignments=assignments).model_dump(mode="json"))
            corpora[f"{style}-f{fold}"] = {"style": style, "fold": fold, "trajectories": members, "count": len(members),
                                           "transitions": sum(trajectories[u]["visible_turns"] for u in members)}
    shards = ROOT / "full200_shards"
    shards.mkdir(exist_ok=True)
    for old in shards.glob("traj_*.jsonl"):
        old.unlink()
    for uid in trajectories:
        with open(shards / f"traj_{uid}.jsonl", "w") as f:
            for row in sorted((r for r in derived if r["id"] == uid), key=lambda r: int(r["turn_idx"])):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(ROOT / "full200_rows_ids.json", [{"id": r["id"], "benchmark_id": r["benchmark_id"], "turn_idx": r["turn_idx"]} for r in derived])
    write_json(ROOT / "trajectories.json", trajectories)
    write_json(ROOT / "corpora.json", corpora)
    write_json(ROOT / "package_map.json", {uid: t["package"] for uid, t in trajectories.items()})

    # summary
    print(f"rows {len(derived)} | trajectories {len(trajectories)} | groups {len(groups)} (multi-trajectory: {sum(1 for v in groups.values() if len(v) > 1)})")
    for style in STYLES:
        ts = [t for t in trajectories.values() if t["style"] == style]
        print(f"  {style}: {len(ts)} trajectories, {sum(t['rows'] for t in ts)} rows, {sum(t['visible_turns'] for t in ts)} visible transitions, "
              f"{len(set(t['app'] for t in ts))} apps | per fold: " + ", ".join(f"f{f}={sum(1 for t in ts if t['fold'] == f)}/{sum(t['rows'] for t in ts if t['fold'] == f)}r" for f in range(K)))
    apps_per_fold = {f: collections.Counter(t["app"] for t in trajectories.values() if t["fold"] == f and t["style"] == "json") for f in range(K)}
    json_apps = collections.Counter(t["app"] for t in trajectories.values() if t["style"] == "json")
    covered = sum(1 for f in range(K) for t in trajectories.values() if t["style"] == "json" and t["fold"] == f and json_apps[t["app"]] - apps_per_fold[f][t["app"]] > 0)
    print(f"  json rows whose app appears in another fold (out-of-fold app coverage): {sum(t['rows'] for t in trajectories.values() if t['style']=='json' and json_apps[t['app']] - apps_per_fold[t['fold']][t['app']] > 0)}/{sum(t['rows'] for t in trajectories.values() if t['style']=='json')} rows, {covered}/{sum(1 for t in trajectories.values() if t['style']=='json')} trajectories")

    # screen-identity diagnostic: does the row's current screen appear (before an action) in its out-of-fold corpus?
    screens: dict[str, list] = collections.defaultdict(list)  # uid -> signatures of screens before each visible action
    for uid, row in longest_rows.items():
        for prompt in row["prompt"]:
            screens[uid].append(screen_signature(state_section(prompt)))
    stats = collections.defaultdict(lambda: collections.Counter())
    for row in derived:
        t = trajectories[row["id"]]
        current = screen_signature(state_section(row["prompt"][-1]))
        stats[t["style"]]["rows"] += 1
        if current is None:
            stats[t["style"]]["no_signature"] += 1
            continue
        pool = [u for u, x in trajectories.items() if x["style"] == t["style"] and x["fold"] != t["fold"]]
        if any(same_screen(s, current) for u in pool for s in screens[u]):
            stats[t["style"]]["out_of_fold_match"] += 1
        if any(same_screen(s, current) for u, x in trajectories.items() if x["style"] == t["style"] and u != row["id"] for s in screens[u]):
            stats[t["style"]]["any_other_trajectory_match"] += 1
        if any(same_screen(s, current) for s in screens[row["id"]][: int(row["turn_idx"]) - 1]):
            stats[t["style"]]["own_history_match"] += 1
    for style in STYLES:
        print(f"  screen identity {style}: {dict(stats[style])}")
    write_json(ROOT / "screen_diagnostic.json", {s: dict(stats[s]) for s in STYLES})


if __name__ == "__main__":
    main()
