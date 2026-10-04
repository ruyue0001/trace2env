#!/usr/bin/env bash
# Build a candidate ALFWorld package only after 40 distinct real wins are exported.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE=work/exp-alfworld-real/trace2env-40-success-v1
BUILD="$SOURCE/build"
LABEL="${PACKAGE_LABEL:-alfworld-40-success-v1}"
PY="${PYTHON:-python3}"
MODEL="${MODEL:-openai/gpt-5.6-sol}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"
STAGES="${STAGES:-ingest extract schema rules renderers notes compile}"

"$PY" -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
summary = json.loads((p / "collection_summary.json").read_text())
manifest = json.loads((p / "trace_manifest.json").read_text())
traces = manifest["traces"]
if not summary["complete"] or len(traces) != 40 or len({tuple(row["signature"]) for row in traces}) != 40:
    raise SystemExit("Need 40 distinct simulator-rewarded wins before reconstruction")
if any(not (p / row["trace"]).is_file() for row in traces):
    raise SystemExit("A selected trace file is missing")' "$SOURCE"

OPTIONS=(--provider chat --base-url "$BASE_URL" --api-key-env "$API_KEY_ENV"
         --model "$MODEL" --chat-reasoning-effort medium
         --max-output-tokens 65536 --max-prompt-bytes 600000 --induction-batch-size 100)
for stage in $STAGES; do
    echo "[$(date +%FT%T)] $stage"
    case "$stage" in
        ingest)
            "$PY" scripts/pilot_stages.py "$BUILD" ingest \
                --episodes "$SOURCE/traces" --split-manifest "$SOURCE/split_manifest.json" \
                --environment-id alfworld-word2world --name "ALFWorld Word2World" \
                --description "@$SOURCE/domain_description.md" --domain text_game \
                "${OPTIONS[@]}" --note "$LABEL: 40 selected real wins; evaluation rows 0-99 held out" ;;
        extract)
            "$PY" scripts/pilot_stages.py "$BUILD" extract --workers 4 \
                "${OPTIONS[@]}" --max-output-tokens 16384 --note "$LABEL extraction" ;;
        compile)
            "$PY" scripts/pilot_stages.py "$BUILD" compile --package-label "$LABEL" \
                "${OPTIONS[@]}" --note "$LABEL candidate compilation" ;;
        schema|rules|renderers|notes)
            "$PY" scripts/pilot_stages.py "$BUILD" "$stage" \
                "${OPTIONS[@]}" --note "$LABEL $stage" ;;
        *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
done
