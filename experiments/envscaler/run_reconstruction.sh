#!/usr/bin/env bash
# Stage-by-stage reconstruction of one EnvScaler environment's package from its construction episodes
# (all versions, calls, and code snapshots kept under <env>/stages/). Run run_export.sh first and read
# descriptions/<env>.md: it is a draft, and it is the one hand-editable input of the build.
#
#   work/exp-envscaler/run_reconstruction.sh env_151_rl
#   STAGES="rules renderers notes compile" work/exp-envscaler/run_reconstruction.sh env_151_rl   # re-run from a stage
#   DRY_RUN=1 work/exp-envscaler/run_reconstruction.sh env_151_rl                                # print the commands only
#
# Model settings default to those of every reported build (docs/EXPERIMENTS.md); override with MODEL, BASE_URL,
# API_KEY_ENV, PROVIDER. The key is read from the environment variable named by API_KEY_ENV.
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_ID="${1:?usage: run_reconstruction.sh ENV_ID   (e.g. env_151_rl)}"
PY="${PYTHON:-python}"
EXP="work/exp-envscaler"; W="$EXP/$ENV_ID"; LABEL="${LABEL:-envscaler-$ENV_ID-v1}"
MODEL="${MODEL:-openai/gpt-5.6-sol}"; BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"; PROVIDER="${PROVIDER:-chat}"
run() { if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "$@"; echo; else "$@"; fi; }
if [ -z "${DRY_RUN:-}" ]; then : "${!API_KEY_ENV:?set $API_KEY_ENV first}"; fi
[ -d "$W/traces/reconstruction" ] || { echo "no construction episodes in $W/traces/reconstruction; run $EXP/run_export.sh" >&2; exit 1; }
[ -f "$EXP/descriptions/$ENV_ID.md" ] || { echo "no description at $EXP/descriptions/$ENV_ID.md; run $EXP/run_export.sh" >&2; exit 1; }
OR=(--description "@$EXP/descriptions/$ENV_ID.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model "$MODEL"
    --provider "$PROVIDER" --base-url "$BASE_URL" --api-key-env "$API_KEY_ENV" --chat-reasoning-effort medium --induction-batch-size 100)
STAGES="${STAGES:-ingest extract schema rules renderers notes compile}"
for stage in $STAGES; do
  echo "[$(date +%FT%T)] $ENV_ID $stage"
  case "$stage" in
    ingest)
      run "$PY" scripts/pilot_stages.py "$W" ingest --episodes "$W/traces/reconstruction" --split-manifest "$W/traces/split_manifest.json" \
        --environment-id "envscaler-$ENV_ID" --name "EnvScaler $ENV_ID" --domain tools "${OR[@]}" \
        --note "$LABEL: construction episodes from work/envscaler/reserve/$ENV_ID (GPT-5.4 success rollouts; benchmark/ tasks held out)" ;;
    extract)
      # Extraction answers are small here (one tool call, one short dict); the 16k cap bounds a runaway
      # generation, as in the mcp build, while induction keeps the 65k budget.
      run "$PY" scripts/pilot_stages.py "$W" extract --workers 6 "${OR[@]}" --max-output-tokens 16384 --note "$LABEL extract (16k output cap)" ;;
    compile)
      run "$PY" scripts/pilot_stages.py "$W" compile --package-label "$LABEL" "${OR[@]}" --note "$LABEL compile"
      # Freeze: the manifest hash is recorded before any evaluation; the package is never edited afterwards.
      if [ -z "${DRY_RUN:-}" ]; then
        "$PY" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$W/packages/$LABEL/manifest.json" \
          > "$W/package-manifest-$LABEL.sha256"
        echo "package $LABEL manifest sha256 $(cat "$W/package-manifest-$LABEL.sha256")"
      fi ;;
    *)
      run "$PY" scripts/pilot_stages.py "$W" "$stage" "${OR[@]}" --note "$LABEL $stage" ;;
  esac
done
echo "[$(date +%FT%T)] $ENV_ID RECONSTRUCTION DONE"
