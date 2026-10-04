"""Record the post hoc 40-task total after replacing seven wait-repair rows.

This combines two Trace2Env package versions; it is not a full repaired-package
evaluation. No model or simulator calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

from collect_scienceworld_measurement import WORK, write_json


REPLACEMENTS = {
    1088: ("trace-batch-a", "replay-batch-a"),
    1017: ("trace-batch-a", "replay-1017"),
    1003: ("trace-batch-a", "replay-1003"),
    1029: ("trace-batch-b", "replay-batch-b"),
    1026: ("trace-1026", "replay-1026"),
    1023: ("trace-batch-b", "replay-1023"),
    1024: ("trace-1024", "replay-1024"),
}


def attempt(directory: Path, item_id: int) -> dict:
    path = directory / "attempts" / f"sciworld_{item_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = WORK / "wait-repair-7"
    ids = [int(row["item_id"].rsplit("_", 1)[1]) for row in
           json.loads((WORK / "sciworld_test_measurement_40.json").read_text(encoding="utf-8"))]
    if len(ids) != 40 or len(set(ids)) != 40 or not set(REPLACEMENTS) <= set(ids):
        raise ValueError("Frozen test IDs or replacement IDs changed")
    package_hashes = set()
    rows = []
    for item_id in ids:
        real = attempt(WORK / "real-gpt56sol-40/aggregate", item_id)
        prompt = attempt(WORK / "prompt-gpt56sol-40/aggregate", item_id)
        prompt_replay = attempt(WORK / "prompt-gpt56sol-40-replay", item_id)
        old_wm = attempt(WORK / "trace2env-gpt56sol-40-r2/aggregate", item_id)
        old_replay = attempt(WORK / "trace2env-gpt56sol-40-r2-replay", item_id)
        if item_id in REPLACEMENTS:
            trace_name, replay_name = REPLACEMENTS[item_id]
            new_wm = attempt(root / trace_name, item_id)
            new_replay = attempt(root / replay_name, item_id)
            selection = json.loads((root / trace_name / "selection.json").read_text(encoding="utf-8"))
            package_hashes.add(selection["package_manifest_sha256"])
            if (new_wm["error"] or new_replay["error"] or
                    new_wm["final_state_step"] != new_wm["steps"] or
                    new_wm["final_memory_entries"] != new_wm["steps"] + 1 or
                    new_replay["status"] != "replay_complete"):
                raise ValueError(f"Incomplete replacement record: {item_id}")
            wm, replay = new_wm, new_replay
            source = "wait_repair_v3"
        else:
            wm, replay = old_wm, old_replay
            source = "original_v2"
        if any(record.get("error") for record in (real, prompt, prompt_replay, old_wm, old_replay)):
            raise ValueError(f"Baseline record has an error: {item_id}")
        rows.append({
            "item_id": item_id,
            "trace_source": source,
            "real_success": bool(real["success"]),
            "prompt_wm_success": bool(prompt["wm_success"]),
            "prompt_w2r_success": bool(prompt_replay["success"]),
            "original_trace_wm_success": bool(old_wm["wm_success"]),
            "original_trace_w2r_success": bool(old_replay["success"]),
            "stitched_trace_wm_success": bool(wm["wm_success"]),
            "stitched_trace_w2r_success": bool(replay["success"]),
            "stitched_trace_replay_score": replay["score"],
        })
    if len(package_hashes) != 1:
        raise ValueError("Repaired reruns do not share a single package manifest")
    real = sum(row["real_success"] for row in rows)
    scores = {}
    for label, wm_key, w2r_key in (
        ("direct_prompting", "prompt_wm_success", "prompt_w2r_success"),
        ("original_trace2env", "original_trace_wm_success", "original_trace_w2r_success"),
        ("stitched_trace2env", "stitched_trace_wm_success", "stitched_trace_w2r_success"),
    ):
        wm = sum(row[wm_key] for row in rows)
        w2r = sum(row[w2r_key] for row in rows)
        scores[label] = {
            "real": real, "wm": wm, "w2r": w2r,
            "real_rate": real / 40, "wm_rate": wm / 40, "w2r_rate": w2r / 40,
            "cr": w2r / real,
        }
    write_json(root / "stitched-40-metrics.json", {
        "definition": "Post hoc hybrid: replace seven original Trace2Env rows with wait-repair-v3 reruns; retain 33 original Trace2Env rows and all direct-prompting/real rows.",
        "not_a_full_repaired_package_evaluation": True,
        "denominator": 40,
        "replacement_ids": list(REPLACEMENTS),
        "repaired_package_manifest_sha256": next(iter(package_hashes)),
        "scores": scores,
        "rows": rows,
    })
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
