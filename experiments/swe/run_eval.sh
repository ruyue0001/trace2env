#!/usr/bin/env bash
# SWE cross-fit evaluation on all 472 rows (99 trajectories): every trajectory's shard runs with the package of its
# scaffold built without its fold (PKG_MAP = package_map.json; schema-only twins via PKG_MAP_SCHEMA; the raw-trace corpus
# of prompting_rag via TRACE_CORPUS_MAP). Same launcher and settings as the other environments; judge gpt-5.2 via OpenRouter.
# Usage: run_eval.sh gpt|deepseek   (SYSTEMS overrides the system list; default: prompting trace2env_v531)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"
EXP="work/exp-swe-cv"
BACKBONE="${1:?gpt|deepseek}"
export EXP_ROOT="$EXP" PKG_MAP=package_map.json PKG_MAP_SCHEMA=package_map_schema.json TRACE_CORPUS_MAP=corpus_map.json \
  PKG_FULL=ws/claude_code-f0/packages/claude_code-f0 PKG_SCHEMA=ws/claude_code-f0/packages/claude_code-f0-schema_only \
  TRACE_CORPUS=work/exp-swe-cv/corpora/claude_code-f0/episodes \
  RUN_NAME=full472 ROWS_FILE=./full472_rows.jsonl SHARDS_DIR=./full472_shards SYSTEMS="${SYSTEMS:-prompting trace2env_v531}"
case "$BACKBONE" in
  gpt)
    BACKBONE_MODEL=openai/gpt-5.6-sol BACKBONE_SLUG=gpt-5.6-sol REASONING_ARGS="--chat-reasoning-effort medium" AGENT_MAX_OUTPUT_TOKENS=16384 MAX_PARALLEL=16 \
      bash work/exp-v1_20_r=1/backbones/run_heldout_systems.sh ;;
  deepseek)
    : "${DEEPSEEK_API_KEY:?}"
    BACKBONE_MODEL=deepseek-flash BACKBONE_SLUG=deepseek-flash-official BASE_URL=https://api.deepseek.com API_KEY_ENV=DEEPSEEK_API_KEY \
      REASONING_ARGS="--chat-reasoning-effort low" AGENT_MAX_OUTPUT_TOKENS=65536 SCHEMA_ARGS="--chat-schema-mode json_object" AGENT_ARGS="--agent-transport tools" MAX_PARALLEL=8 \
      bash work/exp-v1_20_r=1/backbones/run_heldout_systems.sh ;;
  *) echo "unknown backbone $BACKBONE"; exit 2 ;;
esac
