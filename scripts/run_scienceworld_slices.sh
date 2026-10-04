#!/usr/bin/env bash
# Four disjoint Word2World Measurement slices, each with its own ScienceWorld JVM.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:?mode: real, prompt, or trace2env}"
OUTPUT="${2:?output directory}"
PACKAGE="${3:-}"
MAX_STEPS="${MAX_STEPS:-50}"
TOTAL="${TOTAL:-53}"
PY="$PWD/work/exp-scienceworld/.venv/bin/python"

if [[ "$MODE" != real && "$MODE" != prompt && "$MODE" != trace2env ]]; then
    echo "Unsupported mode: $MODE" >&2
    exit 2
fi
if [[ "$MODE" == trace2env && -z "$PACKAGE" ]]; then
    echo "trace2env requires a package directory" >&2
    exit 2
fi

pids=()
if [[ "$TOTAL" == 40 ]]; then
    offsets=(0 10 20 30)
    lengths=(10 10 10 10)
elif [[ "$TOTAL" == 53 ]]; then
    offsets=(0 14 28 41)
    lengths=(14 14 13 12)
else
    echo "TOTAL must be 40 or 53" >&2
    exit 2
fi
for i in 0 1 2 3; do
    args=("$PY" scripts/run_scienceworld_measurement.py "$MODE"
          --output "$OUTPUT/p$i" --start-index "${offsets[$i]}"
          --limit "${lengths[$i]}" --max-steps "$MAX_STEPS")
    if [[ "$MODE" == trace2env ]]; then
        args+=(--package "$PACKAGE")
    fi
    "${args[@]}" >"$OUTPUT-p$i.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
"$PY" scripts/merge_scienceworld_measurement.py --mode "$MODE" --parts \
    "$OUTPUT/p0" "$OUTPUT/p1" "$OUTPUT/p2" "$OUTPUT/p3" --output "$OUTPUT/aggregate" --limit "$TOTAL"
exit "$failed"
