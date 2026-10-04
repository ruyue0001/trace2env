#!/usr/bin/env bash
# Full-benchmark agentic run with the improved harness on the same 4 shards as the reference runs, then judging.
#   run_agentic_v2.sh LABEL FEATURES
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LABEL="$1"; FEATURES="$2"
A="work/exp-v1_20_r=1/awb"; PKG="work/exp-v1_20_r=1/packages/v1_20_r=1"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "$A/judge-cache")
sha256sum "$PKG/manifest.json" | cut -c1-16 > "$A/package-manifest-$LABEL.sha256"
echo "[$(date -Is)] $LABEL features=$FEATURES start (package $(cat $A/package-manifest-$LABEL.sha256))"
for i in 0 1 2 3; do
  python -m trace2env awb-run "$A/shards/shard$i.jsonl" --mode agentic --package "$PKG" --allow-unvalidated --split all "${OR[@]}" \
    --features "$FEATURES" --cache-dir "$A/cache-$LABEL" --call-log "$A/calls-$LABEL-shard$i.jsonl" --session-root "$A/sessions-$LABEL-shard$i" \
    --output "$A/pred-$LABEL-shard$i.jsonl" > "$A/$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$A"/pred-$LABEL-shard{0,1,2,3}.jsonl > "$A/pred-$LABEL.jsonl"
echo "[$(date -Is)] $LABEL predictions done rows=$(wc -l < "$A/pred-$LABEL.jsonl")"
for i in 0 1 2 3; do
  python -m trace2env awb-judge --predictions "$A/pred-$LABEL-shard$i.jsonl" --output "$A/judged-$LABEL-shard$i.jsonl" "${JUDGE[@]}" > "$A/judge-$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$A"/judged-$LABEL-shard{0,1,2,3}.jsonl > "$A/judged-$LABEL.jsonl"
python -m trace2env awb-score --predictions "$A/judged-$LABEL.jsonl" --summary "$A/score-$LABEL.json" | tail -9
echo "[$(date -Is)] $LABEL DONE"
