"""Distinct benchmark trajectories: rows of one (task, id) whose prompts are consistent prefixes of one record.

AgentWorldBench ``id`` values are not unique within a split: several distinct trajectories (different tasks, even
different sub-sources) share an id. Two rows belong to the same trajectory when their history prompts are identical
and each row's current prompt equals the other record's history prompt at that turn plus the benchmark's per-turn
instruction tail (a prefix relation), with prefix-equal responses.
"""
from __future__ import annotations

import collections
from typing import Any


def same_trajectory(row: dict[str, Any], rep: dict[str, Any]) -> bool:
    """Is ``row`` a prefix view of the (longer or equal) record ``rep``?"""
    n = len(row["prompt"])
    if n > len(rep["prompt"]) or len(row["response"]) > len(rep["response"]):
        return False
    for k in range(n - 1):
        if row["prompt"][k] != rep["prompt"][k]:
            return False
    last_row, last_rep = row["prompt"][n - 1], rep["prompt"][n - 1]
    if not (last_row == last_rep or last_row.startswith(last_rep) or last_rep.startswith(last_row)):
        return False
    return row["response"][: len(row["response"])] == rep["response"][: len(row["response"])]


def cluster_trajectories(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    """(task, id, ordinal) -> rows; ordinal 0 is the first trajectory of the id in file order (by its earliest row)."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[(str(row["task"]), str(row["id"]))].append(row)
    result: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for (task, tid), members in groups.items():
        clusters: list[list[dict[str, Any]]] = []
        for row in sorted(members, key=lambda r: -len(r["prompt"])):
            for cluster in clusters:
                if same_trajectory(row, cluster[0]):
                    cluster.append(row)
                    break
            else:
                clusters.append([row])
        clusters.sort(key=lambda c: min(members.index(r) for r in c))
        for ordinal, cluster in enumerate(clusters):
            result[(task, tid, ordinal)] = sorted(cluster, key=lambda r: int(r["turn_idx"]))
    return result


def unique_id(task: str, tid: str, ordinal: int) -> str:
    return tid if ordinal == 0 else f"{tid}~{ordinal + 1}"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from trace2env.agentworld import load_rows
    for name in sys.argv[1:] or ("android", "web", "terminal"):
        rows = load_rows([f"work/agentworldbench/{name}_test.jsonl"])
        clusters = cluster_trajectories(rows)
        ids = collections.Counter((t, i) for t, i, _ in clusters)
        collided = {k: v for k, v in ids.items() if v > 1}
        affected = sum(len(v) for k, v in clusters.items() if ids[(k[0], k[1])] > 1)
        print(f"{name}: rows {len(rows)} | (task,id) {len(ids)} | distinct trajectories {len(clusters)} | ids with >1 trajectory {len(collided)} "
              f"(rows in collided ids: {affected}) | {sorted(collided.items())[:8]}")
