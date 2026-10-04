#!/usr/bin/env bash
# Collect one Terminus-2 trajectory per Terminal-Bench 2.0 task with the official Harbor harness.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   scripts/collect_tb2.sh [JOB_NAME] [N_CONCURRENT] [EXTRA HARBOR ARGS...]
#
# Task images are pulled from Docker Hub (~300-500 MB each). Because Harbor keeps images after a
# trial (it only deletes containers), a janitor loop removes Terminal-Bench images that no longer
# have a container, so the collection needs disk for only ~N_CONCURRENT images at a time.
# Results: work/tb2/jobs/<JOB_NAME>/<task>__<id>/{agent/trajectory.json,result.json,...}
# Convert afterwards with: trace2env atif-export work/tb2/jobs/<JOB_NAME> --output work/tb2/episodes --rows ...
set -euo pipefail

JOB_NAME="${1:-tb2-opus46}"
N_CONCURRENT="${2:-4}"
shift $(( $# >= 2 ? 2 : $# )) || true
MODEL="${TB2_MODEL:-openrouter/anthropic/claude-opus-4.6}"
JOBS_DIR="${TB2_JOBS_DIR:-work/tb2/jobs}"
IMAGE_PREFIX="${TB2_IMAGE_PREFIX:-alexgshaw/}"

: "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY first}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$JOBS_DIR"
LOG="$JOBS_DIR/$JOB_NAME.log"

janitor() {
  # Remove finished tasks' images while the job runs; never touch images that still have a container.
  while kill -0 "$1" 2>/dev/null; do
    sleep 120
    for image in $(docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${IMAGE_PREFIX}" || true); do
      if [ -z "$(docker ps -a -q --filter "ancestor=$image")" ]; then
        docker image rm "$image" >/dev/null 2>&1 || true
      fi
    done
  done
}

echo "[$(date -Is)] job=$JOB_NAME model=$MODEL concurrency=$N_CONCURRENT extra=$*" | tee -a "$LOG"
harbor run -d terminal-bench@2.0 -a terminus-2 -m "$MODEL" -o "$JOBS_DIR" --job-name "$JOB_NAME" \
  -n "$N_CONCURRENT" -y -q "$@" >>"$LOG" 2>&1 &
HARBOR_PID=$!
janitor "$HARBOR_PID" &
JANITOR_PID=$!
wait "$HARBOR_PID"
STATUS=$?
kill "$JANITOR_PID" 2>/dev/null || true
echo "[$(date -Is)] harbor exit=$STATUS" | tee -a "$LOG"
ls "$JOBS_DIR/$JOB_NAME"/*/result.json 2>/dev/null | wc -l | xargs -I{} echo "trials with result.json: {}" | tee -a "$LOG"
exit "$STATUS"
