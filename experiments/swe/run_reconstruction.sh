#!/usr/bin/env bash
# Stage-by-stage reconstruction of one swe cross-fit package (CORPUS = <style>-f<fold>, e.g. claude_code-f0) from the
# corpus exported by make_folds.py (the sub-source's trajectories outside the fold). Same model settings as web-v1_50:
# openai/gpt-5.6-sol via OpenRouter, chat provider, reasoning medium, 600000-byte prompt budget, 65536 output tokens
# for induction and 16384 for extraction, induction batch 100. The cache is shared by every corpus of the experiment
# (content-addressed on the prompt, role, provider and model): a transition is extracted once for the whole cross-fit
# as long as corpora of the same sub-source run one after another. Requires OPENROUTER_API_KEY.
# Usage: run_reconstruction.sh CORPUS [STAGES="ingest extract schema rules renderers notes compile"]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
CORPUS="${1:?corpus name, e.g. claude_code-f0}"
STYLE="${CORPUS%%-f*}"
EXP="work/exp-swe-cv"
WS="$EXP/ws/$CORPUS"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY first}"
OR=(--description "@$EXP/descriptions/$STYLE.md" --max-prompt-bytes 600000 --max-output-tokens 65536 --model openai/gpt-5.6-sol
    --provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --chat-reasoning-effort medium
    --induction-batch-size 100 --cache-dir "$EXP/cache")
STAGES="${STAGES:-ingest extract schema rules renderers notes compile}"
for stage in $STAGES; do
  echo "[$(date -Is)] $CORPUS $stage"
  case "$stage" in
    ingest)
      python scripts/pilot_stages.py "$WS" ingest --episodes "$EXP/corpora/$CORPUS/episodes" --split-manifest "$EXP/corpora/$CORPUS/split_manifest.json" \
        --environment-id "awb-swe-$STYLE" --name "AgentWorldBench swe ($STYLE sub-source), cross-fit $CORPUS" --domain swe "${OR[@]}" \
        --note "swe cross-fit $CORPUS: the $STYLE trajectories of the swe split outside fold ${CORPUS##*-f} (longest visible prefixes)" ;;
    extract)
      python scripts/pilot_stages.py "$WS" extract --workers 12 "${OR[@]}" --max-output-tokens 16384 --note "swe cross-fit $CORPUS extract (16k output cap)" ;;
    compile)
      python scripts/pilot_stages.py "$WS" compile --package-label "$CORPUS" "${OR[@]}" --note "swe cross-fit $CORPUS compile" ;;
    *)
      python scripts/pilot_stages.py "$WS" "$stage" "${OR[@]}" --note "swe cross-fit $CORPUS $stage" ;;
  esac
done
echo "[$(date -Is)] $CORPUS RECONSTRUCTION DONE"
