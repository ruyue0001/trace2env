#!/usr/bin/env bash
# Baselines / ablations on the same 4 AgentWorldBench terminal shards as the v2 run, with the v2 harness settings.
#   run_ablation.sh LABEL PACKAGE [extra awb-run args...]
# Examples (see docs/EXPERIMENTS.md, "Baselines and ablations"):
#   run_ablation.sh schema_only  packages/v1_20_r=1-schema_only
#   run_ablation.sh raw_traces   packages/v1_20_r=1-schema_only --trace-corpus work/exp-v1_20_r=1/traces/reconstruction
#   run_ablation.sh single_shot  packages/v1_20_r=1 --prediction-mode single_shot
# The model cache is shared with the v2 run (identical state-tracking calls replay for free); agent and
# single-shot prompts differ from v2, so their calls are new. Judging is a separate step (run_judge.sh LABEL).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
LABEL="$1"; PKG_REL="$2"; shift 2
EXP="work/exp-v1_20_r=1"; A="$EXP/awb"; PKG="$EXP/$PKG_REL"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
sha256sum "$PKG/manifest.json" | cut -c1-16 > "$A/package-manifest-$LABEL.sha256"
echo "[$(date -Is)] $LABEL package=$PKG_REL ($(cat "$A/package-manifest-$LABEL.sha256")) extra: $* start"
for i in 0 1 2 3; do
  python -m trace2env awb-run "$A/shards/shard$i.jsonl" --mode agentic --package "$PKG" --allow-unvalidated --split all "${OR[@]}" \
    --features default "$@" --cache-dir "$A/cache-v2" --call-log "$A/calls-$LABEL-shard$i.jsonl" --session-root "$A/sessions-$LABEL-shard$i" \
    --output "$A/pred-$LABEL-shard$i.jsonl" > "$A/$LABEL-shard$i.log" 2>&1 &
done
wait
cat "$A"/pred-$LABEL-shard{0,1,2,3}.jsonl > "$A/pred-$LABEL.jsonl"
echo "[$(date -Is)] $LABEL PREDICTIONS DONE rows=$(wc -l < "$A/pred-$LABEL.jsonl")"
