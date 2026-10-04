#!/usr/bin/env bash
# Detached collection run for one batch of WebArena task ids (the sites must be up under the benchmark hostnames).
#   scripts/webarena/run_collection.sh admin 486,548,694,699,4,62,95,198,291,344,474,543,705
# Writes work/webarena/traces/{raw,episodes,summary.jsonl} and work/webarena/traces/run-<batch>.log; resumable (existing
# episodes are skipped), so a final invocation with the whole selection rebuilds split_manifest.json over all episodes.
set -euo pipefail
BATCH=$1; IDS=$2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-$(cat ~/.config/trace2env/openrouter_key)}   # Node 18+ must be on PATH
exec python scripts/collect_webarena.py --task-list work/webarena/test.raw.json --ids "$IDS" --output work/webarena/traces \
  --node "$(command -v node)" --mcp-cli scripts/webarena/node_modules/@playwright/mcp/cli.js \
  --base-url https://openrouter.ai/api/v1 --model openai/gpt-5.6-sol --api-key-env OPENROUTER_API_KEY \
  --reasoning-effort medium --max-output-tokens 8192 --max-steps 30 --keep-results 6 --run r1 \
  > "work/webarena/traces/run-$BATCH.log" 2>&1
