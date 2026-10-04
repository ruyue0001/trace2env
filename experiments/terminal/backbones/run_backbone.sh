#!/usr/bin/env bash
# Backbone-transfer runs for any OpenRouter model with the GPT-5.6-Sol settings (structured transport, reasoning effort
# medium, same caps, shards, package and judge). Outputs go to work/exp-v1_20_r=1/backbones/<slug>/.
#   BACKBONE_MODEL=deepseek/deepseek-v4-pro run_backbone.sh LABEL prompting|prompting_rag|agentic [PACKAGE_REL] [extra args]
# REASONING_ARGS (default "--chat-reasoning-effort medium") is the only per-model transport knob; SMOKE_ROWS=<rows.jsonl>
# runs one process on a small row file. Judging: run_backbone_judge.sh <slug> LABEL...
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LABEL="$1"; MODE="$2"; shift 2
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"; : "${BACKBONE_MODEL:?set BACKBONE_MODEL, e.g. deepseek/deepseek-v4-pro}"
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"; B="$EXP/backbones/$SLUG"; mkdir -p "$B"
read -r -a REASONING <<< "${REASONING_ARGS:---chat-reasoning-effort medium}"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model "$BACKBONE_MODEL" "${REASONING[@]}" --max-output-tokens 65536)
CACHE="${BACKBONE_CACHE_DIR:-$B/cache}"
case "$MODE" in
  prompting)     ARGS=(--mode prompting --split all "${OR[@]}" --cache-dir "$CACHE") ;;
  prompting_rag) ARGS=(--mode prompting_rag --trace-corpus "$EXP/traces/reconstruction" --rag-top-k 5 --rag-turn-chars 4000 --split all "${OR[@]}" --cache-dir "$CACHE") ;;
  agentic)       PKG_REL="$1"; shift; PKG="$EXP/$PKG_REL"
                 sha256sum "$PKG/manifest.json" | cut -c1-16 > "$B/package-manifest-$LABEL.sha256"
                 ARGS=(--mode agentic --package "$PKG" --allow-unvalidated --split all "${OR[@]}" --features default "$@" --cache-dir "$CACHE") ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
echo "[$(date -Is)] $SLUG/$LABEL mode=$MODE model=$BACKBONE_MODEL reasoning='${REASONING[*]}' extra: $* start"
if [ -n "${SMOKE_ROWS:-}" ]; then
  python -m trace2env awb-run "$SMOKE_ROWS" "${ARGS[@]}" --call-log "$B/calls-$LABEL.jsonl" --session-root "$B/sessions-$LABEL" --output "$B/pred-$LABEL.jsonl" > "$B/$LABEL.run.log" 2>&1
  echo "[$(date -Is)] $SLUG/$LABEL SMOKE DONE rows=$(wc -l < "$B/pred-$LABEL.jsonl")"; exit 0
fi
for i in 0 1 2 3; do
  python -m trace2env awb-run "$A/shards/shard$i.jsonl" "${ARGS[@]}" --call-log "$B/calls-$LABEL-shard$i.jsonl" --session-root "$B/sessions-$LABEL-shard$i" --output "$B/pred-$LABEL-shard$i.jsonl" > "$B/$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$B"/pred-$LABEL-shard{0,1,2,3}.jsonl > "$B/pred-$LABEL.jsonl"
echo "[$(date -Is)] $SLUG/$LABEL PREDICTIONS DONE rows=$(wc -l < "$B/pred-$LABEL.jsonl")"
