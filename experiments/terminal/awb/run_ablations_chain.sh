#!/usr/bin/env bash
# The three terminal baselines/ablations end to end: predictions (two cheap ones in parallel, then the
# raw-trace one), then the official judge for each, then the multi-system comparison.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
A="work/exp-v1_20_r=1/awb"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
echo "[$(date -Is)] chain start"
"$A/run_ablation.sh" schema_only "packages/v1_20_r=1-schema_only" > "$A/schema_only.log" 2>&1 &
"$A/run_ablation.sh" single_shot "packages/v1_20_r=1" --prediction-mode single_shot > "$A/single_shot.log" 2>&1 &
wait
"$A/run_ablation.sh" raw_traces "packages/v1_20_r=1-schema_only" --trace-corpus work/exp-v1_20_r=1/traces/reconstruction > "$A/raw_traces.log" 2>&1
echo "[$(date -Is)] predictions done: $(wc -l < "$A/pred-schema_only.jsonl") $(wc -l < "$A/pred-single_shot.jsonl") $(wc -l < "$A/pred-raw_traces.jsonl")"
"$A/run_judge.sh" schema_only single_shot raw_traces > "$A/ablations-judge.log" 2>&1
python "$A/compare_systems.py" --system prompting="$A/judged-prompting.jsonl" --system v2="$A/judged-v2.jsonl" \
  --system schema_only="$A/judged-schema_only.jsonl" --system raw_traces="$A/judged-raw_traces.jsonl" \
  --system single_shot="$A/judged-single_shot.jsonl" --baseline prompting --reference v2 --out "$A/report-ablations.json"
echo "[$(date -Is)] CHAIN DONE"
