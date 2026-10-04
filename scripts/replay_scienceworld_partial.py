"""Replay only completed ScienceWorld world-model rollouts for an interim W2R."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from collect_scienceworld_measurement import WORK, configure_java, write_json
from run_scienceworld_measurement import replay_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Four-slice world-model run directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    selection = json.loads((WORK / "measurement-30-gold-v1/selection.json").read_text(encoding="utf-8"))
    rows = {row["item_id"]: row for row in selection["evaluation_rows"][:40]}
    sources = {}
    for path in args.source.glob("p*/attempts/sciworld_*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        item_id = record["item_id"]
        if item_id not in rows or item_id in sources:
            raise ValueError(f"Unexpected or duplicate completed task {item_id}")
        sources[item_id] = record
    if not sources:
        raise ValueError("No completed tasks yet")
    configure_java()
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv()
    replayed = []
    try:
        for item_id in selection["evaluation_ids"][:40]:
            if item_id not in sources:
                continue
            output = args.output / "attempts" / f"sciworld_{item_id}.json"
            if output.is_file():
                record = json.loads(output.read_text(encoding="utf-8"))
            else:
                record = replay_one(env, rows[item_id], sources[item_id], 50)
                write_json(output, record)
            replayed.append(record)
            print(f"replay {len(replayed)}/{len(sources)} item={item_id} "
                  f"success={record['success']} score={record['score']}", flush=True)
    finally:
        env.close()
    ids = [record["item_id"] for record in replayed]
    real = [json.loads((WORK / "real-gpt56sol-40/aggregate/attempts" /
                        f"sciworld_{item_id}.json").read_text(encoding="utf-8")) for item_id in ids]
    prompt_replay = [json.loads((WORK / "prompt-gpt56sol-40-replay/attempts" /
                                 f"sciworld_{item_id}.json").read_text(encoding="utf-8")) for item_id in ids]
    metrics = {"tasks": len(ids), "item_ids": ids,
               "trace2env_wm_successes": sum(sources[item_id]["wm_success"] for item_id in ids),
               "trace2env_w2r_successes": sum(record["success"] for record in replayed),
               "real_successes_on_subset": sum(record["success"] for record in real),
               "prompt_w2r_successes_on_subset": sum(record["success"] for record in prompt_replay),
               "errors": sum(record["status"] != "replay_complete" for record in replayed),
               "label": "interim completion-order subset; not the final 40-task result"}
    write_json(args.output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
