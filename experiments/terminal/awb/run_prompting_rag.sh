#!/usr/bin/env bash
# prompting+rag baseline: the official prompting input (same model, temperature 0.6, max tokens 32768 as run_prompting.sh)
# plus a fixed top-5 of raw action->observation turns retrieved lexically (FTS5, BM25) from the 20 construction trace files
# of v1_20_r=1, each observation shown up to 4000 characters, appended to the official system prompt. Query = the command
# words of the current action's keystrokes. Same 4 shards; judged with the official judge; compared with every system.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"; LABEL="prompting+rag"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
RAG=(--mode prompting_rag --trace-corpus "$EXP/traces/reconstruction" --rag-top-k 5 --rag-turn-chars 4000)
echo "[$(date -Is)] $LABEL start: ${RAG[*]}"
for i in 0 1 2 3; do
  python -m trace2env awb-run "$A/shards/shard$i.jsonl" "${RAG[@]}" --split all "${OR[@]}" --cache-dir "$A/cache" \
    --session-root "$A/sessions-$LABEL-shard$i" --output "$A/pred-$LABEL-shard$i.jsonl" > "$A/$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$A"/pred-$LABEL-shard{0,1,2,3}.jsonl > "$A/pred-$LABEL.jsonl"
echo "[$(date -Is)] $LABEL PREDICTIONS DONE rows=$(wc -l < "$A/pred-$LABEL.jsonl")"
"$A/run_judge.sh" "$LABEL" > "$A/$LABEL-judge.log" 2>&1
echo "[$(date -Is)] $LABEL JUDGED"
