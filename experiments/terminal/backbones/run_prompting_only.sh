#!/usr/bin/env bash
# Official prompting baseline only, for one OpenRouter backbone, on the 16 trajectory shards, then judged and scored.
#   BACKBONE_MODEL=deepseek/deepseek-v4-pro REASONING_ARGS="--chat-reasoning-effort low" run_prompting_only.sh
# EXTRA_ARGS (e.g. "--max-tokens 65536") is appended to awb-run; BACKBONE_SLUG overrides the output directory name.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"; : "${BACKBONE_MODEL:?}"
export REASONING_ARGS="${REASONING_ARGS:---chat-reasoning-effort low}"
EXP="work/exp-v1_20_r=1"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"; B="$EXP/backbones/$SLUG"; mkdir -p "$B"
read -r -a REASONING <<< "$REASONING_ARGS"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model "$BACKBONE_MODEL" "${REASONING[@]}" --max-output-tokens 65536)
echo "[$(date -Is)] $SLUG/prompting start model=$BACKBONE_MODEL reasoning='${REASONING[*]}'"
for i in $(seq 0 15); do
  python -m trace2env awb-run "$EXP/backbones/shards16/shard$i.jsonl" --mode prompting --split all "${OR[@]}" ${EXTRA_ARGS:-} --cache-dir "$B/cache" \
    --output "$B/pred-prompting-shard$i.jsonl" > "$B/prompting-shard$i.log" 2>&1 &
done
wait
cat "$B"/pred-prompting-shard*.jsonl > "$B/pred-prompting.jsonl"
echo "[$(date -Is)] $SLUG/prompting PREDICTIONS DONE rows=$(wc -l < "$B/pred-prompting.jsonl")"
"$EXP/backbones/run_backbone_judge.sh" "$SLUG" prompting > "$B/judge.log" 2>&1
echo "[$(date -Is)] $SLUG/prompting chain done"
