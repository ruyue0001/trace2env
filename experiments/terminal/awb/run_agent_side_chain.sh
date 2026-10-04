#!/usr/bin/env bash
# Agent-side ablations under harness v3 on the full 354 rows, fresh sessions, same judge:
#   workspace_single_shot_hv3  full package + v3 tracking, fixed retrieval, one prediction call, no tools
#   trace2env_no_state_hv3     full package + v3 agent loop/retrieval/history, no persistent state tracking or state tools
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
A="work/exp-v1_20_r=1/awb"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
V3=(--features default,evidence --agent-max-output-tokens 16384)
echo "[$(date -Is)] agent-side chain start"
"$A/run_ablation.sh" single_shot_hv3 "packages/v1_20_r=1" --prediction-mode single_shot "${V3[@]}" > "$A/single_shot_hv3.log" 2>&1 &
"$A/run_ablation.sh" no_state_hv3 "packages/v1_20_r=1" --no-state-tracking "${V3[@]}" > "$A/no_state_hv3.log" 2>&1 &
wait
echo "[$(date -Is)] predictions done: $(wc -l < "$A/pred-single_shot_hv3.jsonl") $(wc -l < "$A/pred-no_state_hv3.jsonl")"
"$A/run_judge.sh" single_shot_hv3 no_state_hv3 > "$A/agent-side-judge.log" 2>&1
echo "[$(date -Is)] AGENT-SIDE CHAIN DONE"
