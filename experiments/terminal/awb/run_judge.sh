#!/usr/bin/env bash
# Official judge (gpt-5.2 via OpenRouter) on both prediction sets, 4 shards each, then official scoring.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "$A/judge-cache")
for name in ${@:-prompting agentic}; do
  echo "[$(date -Is)] judging $name"
  for i in 0 1 2 3; do
    python -m trace2env awb-judge --predictions "$A/pred-$name-shard$i.jsonl" --output "$A/judged-$name-shard$i.jsonl" "${JUDGE[@]}" > "$A/judge-$name-shard$i.log" 2>&1 &
  done
  wait
  cat "$A"/judged-$name-shard{0,1,2,3}.jsonl > "$A/judged-$name.jsonl"
  python -m trace2env awb-score --predictions "$A/judged-$name.jsonl" --summary "$A/score-$name.json" | tail -9
done
echo "[$(date -Is)] JUDGING DONE"
