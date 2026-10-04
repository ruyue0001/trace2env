#!/usr/bin/env bash
# Stage-by-stage reconstruction of the web-v1_50 package from the 50 WebArena traces in work/webarena/traces
# (all versions, calls, and code snapshots kept under stages/). Same model settings as v1_20_r=1 and mcp-v1_20:
# openai/gpt-5.6-sol via OpenRouter, chat provider, reasoning medium, 600000-byte prompt budget, 65536 output
# tokens for induction and 16384 for extraction, induction batch 100. Requires OPENROUTER_API_KEY. Pass STAGES to
# override the stage list (default: everything from ingest).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="work/exp-web-v1_50"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
OR=(--description "@$EXP/description.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model openai/gpt-5.6-sol
    --provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --chat-reasoning-effort medium --induction-batch-size 100)
STAGES="${STAGES:-ingest extract schema rules renderers notes compile}"
for stage in $STAGES; do
  echo "[$(date -Is)] $stage"
  case "$stage" in
    ingest)
      python scripts/pilot_stages.py "$EXP" ingest --episodes work/webarena/traces/episodes --split-manifest work/webarena/traces/split_manifest.json \
        --environment-id awb-web --name "AgentWorldBench web (WebArena sites through Playwright MCP)" --domain web "${OR[@]}" \
        --note "web-v1_50: 50 own WebArena traces (GPT-5.6-Sol task agent, Playwright MCP 0.0.68), the benchmark's own task suite" ;;
    extract)
      python scripts/pilot_stages.py "$EXP" extract --workers 6 "${OR[@]}" --max-output-tokens 16384 --note "web-v1_50 extract (16k output cap)" ;;
    compile)
      python scripts/pilot_stages.py "$EXP" compile --package-label "web-v1_50" "${OR[@]}" --note "web-v1_50 compile" ;;
    *)
      python scripts/pilot_stages.py "$EXP" "$stage" "${OR[@]}" --note "web-v1_50 $stage" ;;
  esac
done
echo "[$(date -Is)] RECONSTRUCTION DONE"
