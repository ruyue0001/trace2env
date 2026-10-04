#!/usr/bin/env bash
# Stage-by-stage reconstruction of one scaling package with exactly the v1_20_r=1 settings (model, prompts, options).
#   run_reconstruction.sh NAME      (NAME in v1_1_r=1 v1_5_r=1 v1_10_r=1 v1_36_r=1; traces prepared by prepare_traces.py)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
NAME="$1"; EXP="work/exp-scaling/$NAME"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
OR=(--description "@work/exp-v1_20_r=1/description.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model openai/gpt-5.6-sol
    --provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --chat-reasoning-effort medium --induction-batch-size 100)
echo "[$(date -Is)] $NAME ingest"
python scripts/pilot_stages.py "$EXP" ingest --episodes "$EXP/traces/reconstruction" --split-manifest "$EXP/traces/split_manifest.json" \
  --environment-id tb2-terminal --name "Terminal-Bench 2.0 terminal (Terminus-2 / tmux shell)" --domain terminal "${OR[@]}" \
  --note "$NAME: trace-scaling construction set (work/exp-scaling/manifest.json)"
for stage in extract schema rules renderers notes; do
  echo "[$(date -Is)] $NAME $stage"
  extra=(); [ "$stage" = extract ] && extra=(--workers 6)
  python scripts/pilot_stages.py "$EXP" "$stage" "${extra[@]}" "${OR[@]}" --note "$NAME $stage (same prompts/settings as v1_20_r=1)"
done
echo "[$(date -Is)] $NAME compile"
python scripts/pilot_stages.py "$EXP" compile --package-label "$NAME" "${OR[@]}" --note "$NAME compile"
sha256sum "$EXP/packages/$NAME/manifest.json" | cut -c1-16 > "$EXP/package-manifest.sha256"
echo "[$(date -Is)] $NAME RECONSTRUCTION DONE ($(cat "$EXP/package-manifest.sha256"))"
