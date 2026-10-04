#!/usr/bin/env bash
# Evaluate the scaling packages with harness v3 on the same 354 rows (4 shards each, fresh sessions), two packages at a
# time, then judge them with the official judge. Outputs under work/exp-scaling/<name>/awb/.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
SH="work/exp-v1_20_r=1/awb/shards"; JC="work/exp-v1_20_r=1/awb/judge-cache"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
V3=(--features default,evidence --agent-max-output-tokens 16384)
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "$JC")
predict() {
  local NAME="$1"; local EXP="work/exp-scaling/$NAME"; local PKG="$EXP/packages/$NAME"; local A="$EXP/awb"
  mkdir -p "$A"; sha256sum "$PKG/manifest.json" | cut -c1-16 > "$A/package-manifest.sha256"
  echo "[$(date -Is)] $NAME predictions start ($(cat "$A/package-manifest.sha256"))"
  for i in 0 1 2 3; do
    python -m trace2env awb-run "$SH/shard$i.jsonl" --mode agentic --package "$PKG" --allow-unvalidated --split all "${OR[@]}" "${V3[@]}" \
      --cache-dir "$A/cache" --call-log "$A/calls-shard$i.jsonl" --session-root "$A/sessions-shard$i" --output "$A/pred-shard$i.jsonl" > "$A/run-shard$i.log" 2>&1 &
  done
  wait
  cat "$A"/pred-shard{0,1,2,3}.jsonl > "$A/pred.jsonl"
  echo "[$(date -Is)] $NAME predictions done rows=$(wc -l < "$A/pred.jsonl")"
}
judge() {
  local NAME="$1"; local A="work/exp-scaling/$NAME/awb"
  echo "[$(date -Is)] $NAME judging"
  for i in 0 1 2 3; do
    python -m trace2env awb-judge --predictions "$A/pred-shard$i.jsonl" --output "$A/judged-shard$i.jsonl" "${JUDGE[@]}" > "$A/judge-shard$i.log" 2>&1 &
  done
  wait
  cat "$A"/judged-shard{0,1,2,3}.jsonl > "$A/judged.jsonl"
  python -m trace2env awb-score --predictions "$A/judged.jsonl" --summary "$A/score.json" | tail -3
}
# Usage: run_eval_chain.sh NAME [NAME ...]  — predicts the named packages two at a time, then judges each.
NAMES=("$@"); [ ${#NAMES[@]} -gt 0 ] || NAMES=(v1_1_r=1 v1_5_r=1 v1_10_r=1 v1_36_r=1)
for ((i = 0; i < ${#NAMES[@]}; i += 2)); do
  predict "${NAMES[i]}" & [ -n "${NAMES[i+1]:-}" ] && predict "${NAMES[i+1]}" & wait
done
for NAME in "${NAMES[@]}"; do judge "$NAME"; done
echo "[$(date -Is)] SCALING EVAL DONE: ${NAMES[*]}"
