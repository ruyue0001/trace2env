#!/usr/bin/env bash
# EnvScaler rollouts -> construction episodes, held-out benchmark rows (recorded turns; counterfactual probes only on
# request; 4 shards by trajectory), ground truth, a description draft per environment, and the overlap and difficulty
# reports. No model call. Roles: reserve/ = construction (train), benchmark/ = evaluation (test).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PYTHON:-python}"
# DATA / EXP can point elsewhere (a scratch copy for a look at new data); the tools stay in work/exp-envscaler.
DATA="${DATA:-work/envscaler}"; TOOLS="work/exp-envscaler"; EXP="${EXP:-$TOOLS}"
# Every environment that has held-out rollouts: benchmark/<env> or benchmark/<env>__<role>.
ENVS="${ENVS:-$(ls "$DATA/benchmark" | sed 's/__.*//' | sort -u | tr '\n' ' ')}"
# Probes are opt-in (PROBES_PER_TRAJECTORY=10 ...): the benchmark is the recorded held-out turns; probes are a separate
# diagnostic whose composition is set by a generator, not by agents.
PROBES_PER_TRAJECTORY="${PROBES_PER_TRAJECTORY:-0}"
# Which recorded turns are evaluated (histories always stay complete): environment = drop loader artifacts, keep the last
# occurrence of a call repeated within an episode, keep one of the same call with the same answer across episodes.
# none = every turn; episode = the first two rules only; answer = also one row per tool and identical answer text.
UNIQUE_TURNS="${UNIQUE_TURNS:-environment}"
# A call repeated within an episode: same-answer (the study's choice) = drop an occurrence only when a later one returns
# the same answer, so a rejection that later succeeds and a read before and after a write all keep their rows;
# last = keep only the last occurrence of a call, whatever it returned.
REPEATED_CALLS="${REPEATED_CALLS:-same-answer}"
# Some environments build lists from a set; a fixed hash seed makes the exported probe text byte-reproducible.
export PYTHONHASHSEED=0
for env in $ENVS; do
  E="$EXP/$env"
  benchmark=""
  for candidate in "$DATA/benchmark/$env" "$DATA/benchmark/${env}"__*; do
    if [ -d "$candidate" ]; then benchmark="$candidate"; break; fi
  done
  [ -n "$benchmark" ] || { echo "no benchmark directory for $env under $DATA/benchmark" >&2; exit 1; }
  # Episode directories are rebuilt from scratch so a removed rollout cannot linger as a construction input.
  rm -rf "$E/traces" "$E/ground_truth" "$E/awb/shards" "$E/unique_turns.json" "$E/construction_summary.json"
  mkdir -p "$E/awb"
  # An environment may have no reserve rollouts: then there is no construction set, and no few-shot examples either
  # (they come from construction episodes only).
  EXAMPLES=()
  if [ -n "$(ls "$DATA/reserve/$env"/*.json 2>/dev/null)" ]; then
    echo "[$(date +%FT%T)] $env construction <- $DATA/reserve/$env"
    "$PY" -m trace2env envscaler-export "$DATA/reserve/$env" --env-defs "$DATA/env_defs" --assign-split train \
      --output "$E/traces/reconstruction" --manifest "$E/traces/split_manifest.json" \
      --descriptions "$EXP/descriptions" --summary "$E/construction_summary.json"
    EXAMPLES=(--examples-from "$E/traces/reconstruction")
  else
    echo "[$(date +%FT%T)] $env has NO reserve rollouts: no construction episodes, no few-shot examples"
  fi
  echo "[$(date +%FT%T)] $env evaluation <- $benchmark"
  "$PY" -m trace2env envscaler-export "$benchmark" --env-defs "$DATA/env_defs" --assign-split test \
    --output "$E/traces/eval" --manifest "$E/traces/eval_split_manifest.json" --ground-truth "$E/ground_truth" \
    --rows "$E/awb/rows-recorded.jsonl" ${EXAMPLES[@]+"${EXAMPLES[@]}"} --summary "$E/evaluation_summary.json" \
    --unique-turns "$UNIQUE_TURNS" --repeated-calls "$REPEATED_CALLS" --filter-report "$E/unique_turns.json"
  # Probes: other actions at the recorded held-out states, answered by the environment's own source (which this
  # command executes, after checking that it reproduces every recorded step). Evaluation only.
  if [ "$PROBES_PER_TRAJECTORY" -gt 0 ]; then
    "$PY" -m trace2env envscaler-probe "$benchmark" --env-defs "$DATA/env_defs" --per-trajectory "$PROBES_PER_TRAJECTORY" \
      ${EXAMPLES[@]+"${EXAMPLES[@]}"} --output "$E/awb/rows-probe.jsonl" --ground-truth "$E/ground_truth" \
      > "$E/probe_summary.json"
  else
    : > "$E/awb/rows-probe.jsonl"; rm -f "$E/probe_summary.json"
  fi
  cat "$E/awb/rows-recorded.jsonl" "$E/awb/rows-probe.jsonl" > "$E/awb/rows.jsonl"
  # Shards are identical for every system: trajectories sorted by id, dealt round-robin; a trajectory's recorded
  # and probe rows stay together, so the runner rebuilds each prefix once.
  "$PY" - "$E/awb" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); rows = [json.loads(line) for line in (root / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
ids = sorted({row["id"] for row in rows}); shards = 4
(root / "shards").mkdir(exist_ok=True)
for index in range(shards):
    mine = set(ids[index::shards])
    lines = [json.dumps(row, ensure_ascii=False) for row in rows if row["id"] in mine]
    (root / "shards" / f"shard{index}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"rows": len(rows), "recorded": sum(1 for row in rows if not row.get("probe")),
                  "probes": sum(1 for row in rows if row.get("probe")), "trajectories": len(ids), "shards": shards}))
PYEOF
done
"$PY" "$TOOLS/data_stats.py" --data "$DATA" --output "$EXP/data_stats.json" > "$EXP/data_stats.txt" || echo "data_stats: integrity findings, see $EXP/data_stats.txt"
"$PY" "$TOOLS/check_overlap.py" --data "$DATA" --output "$EXP/overlap.json"
"$PY" "$TOOLS/analyze_difficulty.py" --data "$DATA" --experiment "$EXP" --output "$EXP/difficulty.json"
# The scorer must give the known verdict on every mutated ground truth of every exported row (exits non-zero otherwise).
"$PY" "$TOOLS/audit_scoring.py" --experiment "$EXP" --output "$EXP/scoring_audit.json"
# The rows themselves, checked against the raw rollout files by code that does not share the exporter's (selection
# recomputed, every history turn and target compared, ground truth re-derived by executing the environment source).
"$PY" "$TOOLS/verify_rows.py" --data "$DATA" --experiment "$EXP" --output "$EXP/rows_verification.json"
echo "[$(date +%FT%T)] EXPORT DONE"
