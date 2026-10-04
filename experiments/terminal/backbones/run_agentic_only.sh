#!/usr/bin/env bash
# One agentic system for one backbone on the 16 trajectory shards, then judged: used to re-run Trace2Env v3 after a
# transport fix so that unchanged calls replay from the cache and only the changed turns are new.
#   BACKBONE_MODEL=deepseek/deepseek-v4-pro REASONING_ARGS="--chat-reasoning-effort low" AGENT_MAX_OUTPUT_TOKENS=65536 \
#     run_agentic_only.sh LABEL PACKAGE_REL [extra awb-run args]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LABEL="$1"; PKG_REL="$2"; shift 2
: "${OPENROUTER_API_KEY:?}"; : "${BACKBONE_MODEL:?}"
export REASONING_ARGS="${REASONING_ARGS:---chat-reasoning-effort low}"
EXP="work/exp-v1_20_r=1"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"; B="$EXP/backbones/$SLUG"; PKG="$EXP/$PKG_REL"
read -r -a REASONING <<< "$REASONING_ARGS"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model "$BACKBONE_MODEL" "${REASONING[@]}" --max-output-tokens 65536)
V3=(--features default,evidence --agent-max-output-tokens "${AGENT_MAX_OUTPUT_TOKENS:-16384}")
sha256sum "$PKG/manifest.json" | cut -c1-16 > "$B/package-manifest-$LABEL.sha256"
echo "[$(date -Is)] $SLUG/$LABEL agentic pkg=$PKG_REL reasoning='${REASONING[*]}' agent_cap=${AGENT_MAX_OUTPUT_TOKENS:-16384} extra: $* start"
for i in $(seq 0 15); do
  python -m trace2env awb-run "$EXP/backbones/shards16/shard$i.jsonl" --mode agentic --package "$PKG" --allow-unvalidated --split all "${OR[@]}" \
    --features default "${V3[@]}" "$@" --cache-dir "$B/cache" --call-log "$B/calls-$LABEL-shard$i.jsonl" --session-root "$B/sessions-$LABEL-shard$i" \
    --output "$B/pred-$LABEL-shard$i.jsonl" > "$B/$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$B"/pred-$LABEL-shard*.jsonl > "$B/pred-$LABEL.jsonl"
echo "[$(date -Is)] $SLUG/$LABEL PREDICTIONS DONE rows=$(wc -l < "$B/pred-$LABEL.jsonl")"
"$EXP/backbones/run_backbone_judge.sh" "$SLUG" "$LABEL" > "$B/judge-$LABEL.log" 2>&1
echo "[$(date -Is)] $SLUG/$LABEL chain done"
