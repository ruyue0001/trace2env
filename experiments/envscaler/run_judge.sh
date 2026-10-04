#!/usr/bin/env bash
# Official AgentWorldBench judge (mcp judge prompt, verbatim) on prediction sets of one EnvScaler environment,
# 4 shards each, then official scoring. Response caching is off by default; set CACHE=1 to share a cache.
#
#   work/exp-envscaler/run_judge.sh env_174_rl prompting agentic-from-env_174_rl agentic-from-env_151_rl
#   CACHE=1 work/exp-envscaler/run_judge.sh env_174_rl prompting
#   DRY_RUN=1 work/exp-envscaler/run_judge.sh env_174_rl prompting
set -euo pipefail
cd "$(dirname "$0")/../.."
ROWS_ENV="${1:?usage: run_judge.sh ROWS_ENV LABEL [LABEL ...]}"; shift
[ "$#" -ge 1 ] || { echo "usage: run_judge.sh ROWS_ENV LABEL [LABEL ...]" >&2; exit 1; }
PY="${PYTHON:-python}"
A="${ROWS_DIR:-work/exp-envscaler/$ROWS_ENV/awb}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"; JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENROUTER_API_KEY}"
JUDGE=(--judge-model "$JUDGE_MODEL" --judge-base-url "$JUDGE_BASE_URL" --judge-api-key-env "$JUDGE_API_KEY_ENV")
if [ "${CACHE:-0}" = 1 ]; then JUDGE+=(--cache-dir "${JUDGE_CACHE_DIR:-$A/judge-cache}"); fi
if [ -z "${DRY_RUN:-}" ]; then : "${!JUDGE_API_KEY_ENV:?set $JUDGE_API_KEY_ENV first}"; fi
for name in "$@"; do
  # Judging reads the per-shard predictions, while deterministic scoring and reports commonly read the
  # merged file.  Refuse to score if a resumed run updated only one representation: otherwise missing or
  # stale shard generations can silently produce a different judge population from the reported baseline.
  "$PY" - "$A" "$name" <<'PYEOF'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
label = sys.argv[2]

def load(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"missing prediction file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

merged = load(root / f"pred-{label}.jsonl")
sharded = []
for shard in range(4):
    sharded.extend(load(root / f"pred-{label}-shard{shard}.jsonl"))

def signature(row: dict) -> tuple[str, int, str]:
    return str(row.get("id", "")), int(row.get("turn_idx", -1)), str(row.get("gen", ""))

if Counter(map(signature, merged)) != Counter(map(signature, sharded)):
    merged_empty = sum(not row.get("gen") for row in merged)
    shard_empty = sum(not row.get("gen") for row in sharded)
    raise SystemExit(
        f"merged/shard prediction mismatch for {label}: "
        f"merged={len(merged)} (empty gen={merged_empty}), "
        f"shards={len(sharded)} (empty gen={shard_empty})"
    )
PYEOF
  echo "[$(date +%FT%T)] judging $name on $ROWS_ENV"
  for i in 0 1 2 3; do
    ARGS=("$PY" -m trace2env awb-judge --predictions "$A/pred-$name-shard$i.jsonl" --output "$A/judged-$name-shard$i.jsonl" "${JUDGE[@]}")
    if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${ARGS[@]}"; echo; else "${ARGS[@]}" > "$A/judge-$name-shard$i.log" 2>&1 & fi
  done
  [ -z "${DRY_RUN:-}" ] || continue
  wait
  cat "$A"/judged-"$name"-shard{0,1,2,3}.jsonl > "$A/judged-$name.jsonl"
  "$PY" -m trace2env awb-score --predictions "$A/judged-$name.jsonl" --summary "$A/score-$name.json" | tail -9
done
echo "[$(date +%FT%T)] JUDGING DONE"
