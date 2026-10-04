#!/usr/bin/env bash
# Stage-by-stage reconstruction of the v1_20_r=1 package (all versions, calls, and code snapshots kept under stages/).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP="work/exp-v1_20_r=1"
OR=(--description "@$EXP/description.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model openai/gpt-5.6-sol
    --provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --chat-reasoning-effort medium --induction-batch-size 100)
echo "[$(date -Is)] ingest"
python scripts/pilot_stages.py "$EXP" ingest --episodes "$EXP/traces/reconstruction" --split-manifest "$EXP/traces/split_manifest.json" \
  --environment-id tb2-terminal --name "Terminal-Bench 2.0 terminal (Terminus-2 / tmux shell)" --domain terminal "${OR[@]}" \
  --note "v1_20_r=1: 20 solved traces sampled with seed 42 after excluding 25 AgentWorldBench-overlapping tasks"
for stage in extract schema rules renderers notes; do
  echo "[$(date -Is)] $stage"
  extra=(); [ "$stage" = extract ] && extra=(--workers 6)
  python scripts/pilot_stages.py "$EXP" "$stage" "${extra[@]}" "${OR[@]}" --note "v1_20_r=1 $stage (prompts as of the pilot's final versions)"
done
echo "[$(date -Is)] compile"
python scripts/pilot_stages.py "$EXP" compile --package-label "v1_20_r=1" "${OR[@]}" --note "v1_20_r=1 compile"
echo "[$(date -Is)] RECONSTRUCTION DONE"
