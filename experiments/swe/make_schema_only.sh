#!/usr/bin/env bash
# Schema-only twins of the 10 cross-fit packages (for harness_only_v51 later) and the matching PKG_MAP_SCHEMA file.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
EXP=work/exp-swe-cv
for m in "$EXP"/ws/*/packages/*/manifest.json; do
  pkg="$(dirname "$m")"; dst="$pkg-schema_only"
  [ -d "$dst" ] || python scripts/ablate_package.py "$pkg" --output "$dst" --keep schemas
done
python3 - <<'PY'
import json
m = json.load(open("work/exp-swe-cv/package_map.json"))
json.dump({k: v + "-schema_only" for k, v in m.items()}, open("work/exp-swe-cv/package_map_schema.json", "w"), indent=1)
print("package_map_schema.json written")
PY
