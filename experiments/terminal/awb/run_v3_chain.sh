#!/usr/bin/env bash
# Harness v3 (evidence read path) on the full 354 rows: predictions, official judge, six-system comparison.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
A="work/exp-v1_20_r=1/awb"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
echo "[$(date -Is)] v3 chain start"
"$A/run_ablation.sh" v3 "packages/v1_20_r=1" --features default,evidence --agent-max-output-tokens 16384 > "$A/v3.log" 2>&1
echo "[$(date -Is)] predictions done: $(wc -l < "$A/pred-v3.jsonl")"
"$A/run_judge.sh" v3 > "$A/v3-judge.log" 2>&1
python "$A/compare_systems.py" --system prompting="$A/judged-prompting.jsonl" --system v2="$A/judged-v2.jsonl" \
  --system schema_only="$A/judged-schema_only.jsonl" --system raw_traces="$A/judged-raw_traces.jsonl" \
  --system single_shot="$A/judged-single_shot.jsonl" --system v3="$A/judged-v3.jsonl" \
  --baseline prompting --reference v2 --out "$A/report-v3.json"
echo "[$(date -Is)] V3 CHAIN DONE"
