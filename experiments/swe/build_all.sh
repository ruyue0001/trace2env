#!/usr/bin/env bash
# Build the 10 cross-fit packages: per sub-source, ingest+extract the five folds one after another (the shared cache
# makes every transition a paid call exactly once), then induce and compile the five folds in parallel.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"
EXP="work/exp-swe-cv"
for style in "${@:-claude_code gemini_cli}"; do
  (
    for f in 0 1 2 3 4; do STAGES="ingest extract" bash "$EXP/run_reconstruction.sh" "$style-f$f" || echo "[$(date -Is)] $style-f$f EXTRACT FAILED"; done
    for f in 0 1 2 3 4; do
      STAGES="schema rules renderers notes compile" bash "$EXP/run_reconstruction.sh" "$style-f$f" > "$EXP/build-$style-f$f.log" 2>&1 || echo "[$(date -Is)] $style-f$f INDUCTION FAILED" &
    done
    wait
    echo "[$(date -Is)] $style ALL FOLDS DONE"
  ) > "$EXP/build-$style.log" 2>&1 &
done
wait
echo "[$(date -Is)] BUILD_ALL DONE"
