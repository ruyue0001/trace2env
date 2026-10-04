#!/usr/bin/env bash
# Judge one frozen backbone's six prediction sets with the pinned official AgentWorldBench eval.py.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=work/exp-envscaler/baselines-20260924
JUDGE_TARGET_BACKBONE="${JUDGE_TARGET_BACKBONE:-deepseek-v41-flash}"
case "$JUDGE_TARGET_BACKBONE" in deepseek-v41-flash|gpt56sol) ;; *) echo "unknown backbone: $JUDGE_TARGET_BACKBONE" >&2; exit 1 ;; esac
PYTHON="$PWD/.venv/bin/python"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENROUTER_API_KEY}"
export PYTHON JUDGE_MODEL JUDGE_BASE_URL JUDGE_API_KEY_ENV
"$PYTHON" "$ROOT/validate.py"
"$PYTHON" - "$ROOT" "$JUDGE_TARGET_BACKBONE" <<'PYEOF'
import json, sys
from collections import Counter
from pathlib import Path
root = Path(sys.argv[1])
backbone = sys.argv[2]
plan = json.loads((root / "plan.json").read_text())
for env in plan["environment_order"]:
    directory = root / "rows" / env
    for label in (f"envpack-prompting-{backbone}", f"harness-only-v51-{backbone}"):
        def read(path):
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        merged = read(directory / f"pred-{label}.jsonl")
        shards = [row for i in range(4) for row in read(directory / f"pred-{label}-shard{i}.jsonl")]
        signature = lambda row: (row["id"], row["turn_idx"], row["gen"])
        if len(merged) != plan["environments"][env]["rows"] or Counter(map(signature, merged)) != Counter(map(signature, shards)):
            raise SystemExit(f"merged/shard mismatch: {env}/{label}")
        if any(not row.get("gen") or (row.get("trace2env") or {}).get("error") for row in merged):
            raise SystemExit(f"empty or failed prediction: {env}/{label}")
print("Six prediction sets verified for judging")
PYEOF
if [ -z "${DRY_RUN:-}" ]; then
  : "${!JUDGE_API_KEY_ENV:?set $JUDGE_API_KEY_ENV first}"
  "$PYTHON" - <<'PYEOF'
import os, sys
from openai import OpenAI
client = OpenAI(api_key=os.environ[os.environ["JUDGE_API_KEY_ENV"]], base_url=os.environ["JUDGE_BASE_URL"], max_retries=0)
try:
    client.chat.completions.create(model=os.environ["JUDGE_MODEL"], messages=[{"role": "user", "content": "Reply OK."}], max_tokens=16, temperature=0.6)
except Exception as error:
    raise SystemExit(f"Judge preflight failed: {type(error).__name__}: {error}")
print("Judge model preflight passed")
PYEOF
fi
for env in env_160_rl env_172_rl env_174_rl; do
  export ROWS_DIR="$ROOT/rows/$env"
  for label in "envpack-prompting-$JUDGE_TARGET_BACKBONE" "harness-only-v51-$JUDGE_TARGET_BACKBONE"; do
    if [ -z "${DRY_RUN:-}" ] && [ -f "$ROWS_DIR/official-judge-run-$label.json" ]; then
      "$PYTHON" - "$ROWS_DIR" "$label" <<'PYEOF'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in (root / f"official-judged-{label}.jsonl").read_text().splitlines()]
source = [json.loads(line) for line in (root / f"pred-{label}.jsonl").read_text().splitlines()]
record = json.loads((root / f"official-judge-run-{label}.json").read_text())
if len(rows) != len(source) or record["rows"] != len(source) or any(row.get("failed") != 0.0 for row in rows):
    raise SystemExit(f"Existing judge result is incomplete: {root}/{label}")
PYEOF
      echo "skip completed judge $env/$label"
      continue
    fi
    echo "JUDGE $env $label"
    work/exp-envscaler/run_judge_official.sh "$env" "$label"
    if [ -z "${DRY_RUN:-}" ]; then
      "$PYTHON" - "$ROWS_DIR" "$label" <<'PYEOF'
import json, sys
from pathlib import Path
root, label = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in (root / f"official-judged-{label}.jsonl").read_text().splitlines()]
source = [json.loads(line) for line in (root / f"pred-{label}.jsonl").read_text().splitlines()]
failed = sum(row.get("failed") != 0.0 for row in rows)
if len(rows) != len(source) or failed:
    raise SystemExit(f"Judge incomplete: {root}/{label}; judged={len(rows)} expected={len(source)} failed={failed}")
PYEOF
    fi
  done
done
if [ -z "${DRY_RUN:-}" ]; then "$PYTHON" "$ROOT/summarize_judge.py" "$JUDGE_TARGET_BACKBONE"; fi
