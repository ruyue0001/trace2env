#!/usr/bin/env bash
# Knowledge-tier ablations under the v3 harness: examples_only (evidence + demonstrations) and structure_only
# (rules + contracts + notes), run in parallel (8 shards), judged, and compared with every other system.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
A="work/exp-v1_20_r=1/awb"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
V3=(--features default,evidence --agent-max-output-tokens 16384)
echo "[$(date -Is)] tiers chain start"
"$A/run_ablation.sh" examples_only "packages/v1_20_r=1-examples_only" "${V3[@]}" > "$A/examples_only.log" 2>&1 &
"$A/run_ablation.sh" structure_only "packages/v1_20_r=1-structure_only" "${V3[@]}" > "$A/structure_only.log" 2>&1 &
wait
echo "[$(date -Is)] predictions done: $(wc -l < "$A/pred-examples_only.jsonl") $(wc -l < "$A/pred-structure_only.jsonl")"
"$A/run_judge.sh" examples_only structure_only > "$A/tiers-judge.log" 2>&1
echo "[$(date -Is)] TIERS CHAIN DONE"
