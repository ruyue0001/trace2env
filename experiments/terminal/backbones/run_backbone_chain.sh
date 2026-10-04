#!/usr/bin/env bash
# Full backbone-transfer chain for one OpenRouter model: the four terminal systems on 16 trajectory-level shards, then the unchanged gpt-5.2 judge. Settings = the GPT-5.6-Sol runs'
# flags; REASONING_ARGS (default "--chat-reasoning-effort low" here, set for deepseek-v4.1-flash) is the
# one transport knob, recorded in every log line; AGENT_MAX_OUTPUT_TOKENS (default 16384, the GPT runs' cap) sizes the
# world-model agent's output cap, which must include the backbone's reasoning tokens on Chat Completions.
#   BACKBONE_MODEL=deepseek/deepseek-v4.1-flash REASONING_ARGS="--chat-reasoning-effort low" run_backbone_chain.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"; : "${BACKBONE_MODEL:?}"
export REASONING_ARGS="${REASONING_ARGS:---chat-reasoning-effort low}"
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"; B="$EXP/backbones/$SLUG"; mkdir -p "$B"
read -r -a REASONING <<< "$REASONING_ARGS"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model "$BACKBONE_MODEL" "${REASONING[@]}" --max-output-tokens 65536)
V3=(--features default,evidence --agent-max-output-tokens "${AGENT_MAX_OUTPUT_TOKENS:-16384}")
run_agentic() {  # LABEL PKG_REL [extra]
  local label="$1" pkg="$EXP/$2"; shift 2
  sha256sum "$pkg/manifest.json" | cut -c1-16 > "$B/package-manifest-$label.sha256"
  echo "[$(date -Is)] $SLUG/$label agentic pkg=$pkg reasoning='${REASONING[*]}' extra: $* start"
  for i in $(seq 0 15); do
    python -m trace2env awb-run "$EXP/backbones/shards16/shard$i.jsonl" --mode agentic --package "$pkg" --allow-unvalidated --split all "${OR[@]}" \
      --features default "$@" --cache-dir "$B/cache" --call-log "$B/calls-$label-shard$i.jsonl" --session-root "$B/sessions-$label-shard$i" \
      --output "$B/pred-$label-shard$i.jsonl" > "$B/$label-shard$i.log" 2>&1 &
  done
}
run_prompting() {  # LABEL MODE-ARGS...   (16 trajectory shards too: one ~2-minute reasoning call per row)
  local label="$1"; shift
  echo "[$(date -Is)] $SLUG/$label prompting-family reasoning='${REASONING[*]}' start"
  for i in $(seq 0 15); do
    python -m trace2env awb-run "$EXP/backbones/shards16/shard$i.jsonl" "$@" --split all "${OR[@]}" --cache-dir "$B/cache" \
      --output "$B/pred-$label-shard$i.jsonl" > "$B/$label-shard$i.log" 2>&1 &
  done
}
echo "[$(date -Is)] $SLUG chain start model=$BACKBONE_MODEL reasoning='${REASONING[*]}' agent_cap=${AGENT_MAX_OUTPUT_TOKENS:-16384} effect_rejection=keep_observation(default)"
run_agentic v3 packages/v1_20_r=1 "${V3[@]}"
run_agentic harness_only_hv3 packages/v1_20_r=1-schema_only --no-state-tracking --no-package-knowledge "${V3[@]}"
run_prompting prompting --mode prompting
run_prompting prompting+rag --mode prompting_rag --trace-corpus "$EXP/traces/reconstruction" --rag-top-k 5 --rag-turn-chars 4000
wait
for label in v3 harness_only_hv3 prompting prompting+rag; do cat "$B"/pred-$label-shard*.jsonl > "$B/pred-$label.jsonl"; echo "[$(date -Is)] $SLUG/$label PREDICTIONS DONE rows=$(wc -l < "$B/pred-$label.jsonl")"; done
"$EXP/backbones/run_backbone_judge.sh" "$SLUG" prompting prompting+rag harness_only_hv3 v3 > "$B/judge.log" 2>&1
echo "[$(date -Is)] $SLUG chain done"
