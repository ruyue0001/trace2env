"""Per-row strata for the android cross-fit report: sub-source, fold, whether the row's app is covered by another fold
(json) and whether its current screen matches a screen recorded before an action in its out-of-fold corpus."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from trace2env.knowledge_gate import same_screen, screen_signature  # noqa: E402
from trace2env.storage import write_json  # noqa: E402

ROOT = Path("work/exp-android-cv")


def state_section(prompt: str) -> str:
    return prompt.split("**Action:**")[0]


def main() -> None:
    trajectories = json.load(open(ROOT / "trajectories.json"))
    rows = [json.loads(line) for line in open(ROOT / "full200_rows.jsonl", encoding="utf-8")]
    longest: dict[str, dict] = {}
    for row in rows:
        if row["id"] not in longest or len(row["prompt"]) > len(longest[row["id"]]["prompt"]):
            longest[row["id"]] = row
    screens = {uid: [screen_signature(state_section(p)) for p in row["prompt"]] for uid, row in longest.items()}
    apps = collections.Counter((t["style"], t["app"]) for t in trajectories.values())
    apps_in_fold = collections.Counter((t["style"], t["app"], t["fold"]) for t in trajectories.values())
    strata = {}
    for row in rows:
        t = trajectories[row["id"]]
        current = screen_signature(state_section(row["prompt"][-1]))
        pool = [u for u, x in trajectories.items() if x["style"] == t["style"] and x["fold"] != t["fold"]]
        strata[f"{row['id']}|{row['turn_idx']}"] = {
            "style": t["style"], "fold": t["fold"], "app": t["app"],
            "app_covered": apps[(t["style"], t["app"])] - apps_in_fold[(t["style"], t["app"], t["fold"])] > 0 and t["app"] != "unknown",
            "screen_match": current is not None and any(same_screen(s, current) for u in pool for s in screens[u]),
            "has_signature": current is not None,
        }
    write_json(ROOT / "row_strata.json", strata)
    print(f"rows {len(strata)} | app_covered {sum(s['app_covered'] for s in strata.values())} | screen_match {sum(s['screen_match'] for s in strata.values())} | "
          + ", ".join(f"{st} {sum(1 for s in strata.values() if s['style'] == st)}" for st in ("json", "phone", "unparsed")))


if __name__ == "__main__":
    main()
