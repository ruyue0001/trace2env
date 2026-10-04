#!/usr/bin/env bash
# Official judge (gpt-5.2 via OpenRouter, shared content-addressed judge cache) on a backbone's prediction shards.
#   run_backbone_judge.sh SLUG LABEL...
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SLUG="$1"; shift
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"; B="$EXP/backbones/$SLUG"
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "$A/judge-cache")
for name in "$@"; do
  echo "[$(date -Is)] judging $SLUG/$name"
  for f in "$B"/pred-$name-shard*.jsonl; do
    i="${f##*-shard}"; i="${i%.jsonl}"
    python -m trace2env awb-judge --predictions "$f" --output "$B/judged-$name-shard$i.jsonl" "${JUDGE[@]}" > "$B/judge-$name-shard$i.log" 2>&1 &
  done
  wait
  cat "$B"/judged-$name-shard*.jsonl > "$B/judged-$name.jsonl"
  python -m trace2env awb-score --predictions "$B/judged-$name.jsonl" --summary "$B/score-$name.json" | tail -9
done
echo "[$(date -Is)] JUDGING DONE"
