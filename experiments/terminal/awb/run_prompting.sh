#!/usr/bin/env bash
# Official prompting world-model baseline on the full AgentWorldBench terminal set (4 shards in parallel).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
echo "[$(date -Is)] prompting baseline start"
for i in 0 1 2 3; do
  python -m trace2env awb-run "$A/shards/shard$i.jsonl" --mode prompting --split all "${OR[@]}" --cache-dir "$A/cache" \
    --output "$A/pred-prompting-shard$i.jsonl" > "$A/prompting-shard$i.log" 2>&1 &
done
wait
cat "$A"/pred-prompting-shard{0,1,2,3}.jsonl > "$A/pred-prompting.jsonl"
echo "[$(date -Is)] PROMPTING DONE rows=$(wc -l < "$A/pred-prompting.jsonl")"
