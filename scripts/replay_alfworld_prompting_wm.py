"""Replay saved world-model task-agent actions in the real ALFWorld simulator.

This is evaluation-side only: simulator observations never enter the completed
world-model or task-agent rollouts being evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "Word2World"


def action_from_react(react: str) -> str:
    """Match AgentGym's ReAct extraction and AlfWorldEnvClient end-token trim."""
    if react.endswith("</s>"):
        react = react[:-5]
    parts = react.rsplit("Action:", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def replay_one(wrapper: Any, record: dict[str, Any]) -> dict[str, Any]:
    if record.get("status") == "api_or_harness_error":
        # The task agent stopped when its world model failed. Count that task
        # as an end-to-end failure without treating it as a simulator error.
        return {"item_id": record["item_id"], "wm_success": False,
                "real_success": False, "real_reward": 0.0,
                "status": "source_harness_error", "error": record.get("error"),
                "actions_replayed": 0, "turns": []}
    environment_id = None
    transcript = []
    status = "actions_exhausted"
    reward = 0.0
    error = None
    try:
        created = wrapper.create()
        if "error" in created:
            raise RuntimeError(created["error"])
        environment_id = created["id"]
        initial = wrapper.reset(environment_id, record["item_id"], "Text")
        if "error" in initial:
            raise RuntimeError(initial["error"])
        if "task_type" in record and "task_id" in record:
            expected_game = f"{record['task_type']}/{record['task_id']}"
            if initial.get("task_type") != expected_game:
                raise RuntimeError(f"Wrong ALFWorld game for {record['item_id']}: "
                                   f"{initial.get('task_type')} != {expected_game}")
        for turn in record["turns"]:
            action = action_from_react(turn["react"]) if "react" in turn else turn["action"]
            if action != turn["action"].removesuffix("</s>").strip():
                raise RuntimeError(f"Saved ReAct/action mismatch on turn {turn['turn']}")
            if not action:
                # The released W2R client does not call the simulator for an
                # unparseable ReAct message.
                observation = ("Invalid Action.\n\n" + wrapper.get_observation(environment_id) +
                               "\nAVAILABLE ACTIONS: " +
                               ",".join(wrapper.get_available_actions(environment_id)))
                result = {"observation": observation,
                          "reward": 0.0, "done": False}
            else:
                result = wrapper.step(environment_id, action)
            if "error" in result:
                raise RuntimeError(result["error"])
            reward = float(result["reward"])
            transcript.append({"turn": turn["turn"], "action": action,
                               "wm_observation": turn["observation"],
                               "real_observation": result["observation"],
                               "real_reward": reward, "real_done": bool(result["done"])})
            if result["done"]:
                status = "real_done"
                break
    except Exception as exc:
        status = "env_error"
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if environment_id is not None:
            # Match the proven cleanup in scripts/run_alfworld_real.py.
            with wrapper._lock:
                instance = wrapper.env_init.pop(environment_id, None)
                wrapper.env.pop(environment_id, None)
                wrapper.info.pop(environment_id, None)
                if environment_id in wrapper.ls:
                    wrapper.ls.remove(environment_id)
            if instance is not None:
                instance.close()
    return {"item_id": record["item_id"], "wm_success": record["wm_success"],
            "real_success": status == "real_done" and reward in (1.0, 100.0), "real_reward": reward,
            "status": status, "error": error, "actions_replayed": len(transcript),
            "turns": transcript}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path,
                        default=ROOT / "work" / "exp-alfworld-real" / "simulator" / "alfworld")
    args = parser.parse_args()
    selection = json.loads((args.input / "selection.json").read_text(encoding="utf-8"))
    ids = selection["task_ids"]
    os.environ["ALFWORLD_DATA"] = str(args.data_root.resolve())
    source = BASELINE / "AgentGym" / "agentenv-alfworld"
    sys.path.insert(0, str(source))
    from agentenv_alfworld.env_wrapper import ALFWorld_Wrapper

    wrapper = ALFWorld_Wrapper(data_path=str(args.data_root.resolve()),
                               config_path=str(source / "configs" / "base_config.yaml"))
    results = []
    for index, item_id in enumerate(ids, 1):
        source_path = args.input / "attempts" / f"alfworld_{item_id}.json"
        record = json.loads(source_path.read_text(encoding="utf-8"))
        output_path = args.output / "attempts" / f"alfworld_{item_id}.json"
        if output_path.exists():
            result = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            result = replay_one(wrapper, record)
            write_json(output_path, result)
        results.append(result)
        if index % 10 == 0 or index == len(ids):
            print(f"{index}/{len(ids)} replayed; real successes={sum(x['real_success'] for x in results)}", flush=True)
    errors = [x["item_id"] for x in results if x["status"] == "env_error"]
    source_errors = [x["item_id"] for x in results if x["status"] == "source_harness_error"]
    summary = {"tasks": len(ids), "wm_successes": sum(x["wm_success"] for x in results),
               "real_successes": sum(x["real_success"] for x in results),
               "wm_task_success_rate": sum(x["wm_success"] for x in results) / len(ids),
               "wm2real_success_rate": sum(x["real_success"] for x in results) / len(ids) if not errors else None,
               "env_errors": errors, "source_harness_errors": source_errors,
               "definition": "WM2Real: replay saved world-model task-agent actions on fresh real ALFWorld games; source harness errors count as failures"}
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
