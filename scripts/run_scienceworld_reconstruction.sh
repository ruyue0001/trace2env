#!/usr/bin/env bash
# Build from 30 simulator-verified Measurement trajectories.
set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE=work/exp-scienceworld/measurement-30-gold-v1
BUILD="$SOURCE/build"
PY="$PWD/work/exp-scienceworld/.venv/bin/python"
MODEL="${MODEL:-openai/gpt-5.6-sol}"
API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
STAGES="${STAGES:-extract schema rules renderers notes compile}"
LABEL="${PACKAGE_LABEL:-sciworld-measurement-30-v1}"

"$PY" -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); m = json.loads((p / "trace_manifest.json").read_text())
rows = m["traces"]
if not m["complete"] or len(rows) != 30 or len({r["item_id"] for r in rows}) != 30:
    raise SystemExit("Need 30 distinct simulator-verified wins")
if any(r["score"] != 100 or not (p / r["trace"]).is_file() for r in rows):
    raise SystemExit("A win lacks score 100 or its episode file")' "$SOURCE"

OPTIONS=(--provider chat --base-url "$BASE_URL" --api-key-env "$API_KEY_ENV"
         --model "$MODEL" --chat-reasoning-effort medium
         --max-output-tokens 65536 --max-prompt-bytes 600000 --induction-batch-size 100)
for stage in $STAGES; do
    echo "[$(date +%FT%T)] $stage"
    case "$stage" in
        extract)
            "$PY" scripts/pilot_stages.py "$BUILD" extract --workers 16 \
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
