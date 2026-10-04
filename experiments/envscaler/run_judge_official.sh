#!/usr/bin/env bash
# Judge and score predictions with the ORIGINAL AgentWorldBench evaluation code, unmodified: QwenLM/Qwen-AgentWorld
# eval/eval.py (`judge`, then `score`), fetched at a pinned commit into the git-ignored work/agentworldbench/.
# The repo's own `awb-judge` / `awb-score` are a port of the same code (same prompts, parser, aggregation); this
# script exists so a reported number can be said to come from the original. The port re-aggregates the original's
# judged file at the end and the two totals are compared.
#
#   work/exp-envscaler/run_judge_official.sh env_151_rl prompting [more labels]
#   DRY_RUN=1 work/exp-envscaler/run_judge_official.sh env_151_rl prompting
#
# The original reads the key from OPENAI_API_KEY; it is set for the judge processes only, from the variable named
# by JUDGE_API_KEY_ENV, and never appears on a command line. The original has no cache: a re-run is billed again.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROWS_ENV="${1:?usage: run_judge_official.sh ROWS_ENV LABEL [LABEL ...]}"; shift
[ "$#" -ge 1 ] || { echo "usage: run_judge_official.sh ROWS_ENV LABEL [LABEL ...]" >&2; exit 1; }
PY="$(command -v "${PYTHON:-python}")"
ROOT="$PWD"; A="${ROWS_DIR:-$ROOT/work/exp-envscaler/$ROWS_ENV/awb}"
case "$A" in /*) ;; *) A="$ROOT/$A" ;; esac
COMMIT="${AWB_COMMIT:-cd0aa83dc7a9c733695eb9c4652e0a68b6e6ecde}"   # QwenLM/Qwen-AgentWorld main on 2026-09-21
UP="$ROOT/work/agentworldbench/Qwen-AgentWorld-$COMMIT"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-5.2}"; JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://openrouter.ai/api/v1}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-OPENROUTER_API_KEY}"
if [ ! -f "$UP/eval/eval.py" ]; then
  echo "fetching QwenLM/Qwen-AgentWorld@$COMMIT (eval code and prompts) into $UP"
  files="eval/eval.py eval/lwm_eval_utils/__init__.py eval/lwm_eval_utils/task_configs.py eval/lwm_eval_utils/judge_parser.py eval/lwm_eval_utils/output_parser.py LICENSE"
  for domain in terminal swe search mcp android web os; do files="$files prompts/$domain/judge_system_prompt.txt prompts/$domain/system_prompt.txt"; done
  for file in $files; do
    mkdir -p "$UP/$(dirname "$file")"
    curl -fsSL --retry 3 -o "$UP/$file" "https://raw.githubusercontent.com/QwenLM/Qwen-AgentWorld/$COMMIT/$file"
  done
fi
if [ -z "${DRY_RUN:-}" ]; then : "${!JUDGE_API_KEY_ENV:?set $JUDGE_API_KEY_ENV first}"; fi
for name in "$@"; do
  STARTED="$(date +%FT%T)"
  echo "[$STARTED] original eval.py judge: $name on $ROWS_ENV ($JUDGE_MODEL)"
  for i in 0 1 2 3; do
    ARGS=("$PY" eval.py judge --predictions "$A/pred-$name-shard$i.jsonl" --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL"
          --output-dir "$A/official-judged-$name-shard$i")
    if [ -n "${DRY_RUN:-}" ]; then printf '(cd %q/eval && OPENAI_API_KEY=$%s ' "$UP" "$JUDGE_API_KEY_ENV"; printf '%q ' "${ARGS[@]}"; echo ")"; continue; fi
    (cd "$UP/eval" && OPENAI_API_KEY="${!JUDGE_API_KEY_ENV}" "${ARGS[@]}" > "$A/official-judge-$name-shard$i.out" 2>&1) &
  done
  [ -z "${DRY_RUN:-}" ] || continue
  wait
  cat "$A"/official-judged-"$name"-shard{0,1,2,3}/judged.jsonl > "$A/official-judged-$name.jsonl"
  expected=$(wc -l < "$A/pred-$name.jsonl"); got=$(wc -l < "$A/official-judged-$name.jsonl")
  [ "$got" -eq "$expected" ] || { echo "judged $got of $expected rows: check $A/official-judge-$name-shard*.out" >&2; exit 1; }
  (cd "$UP/eval" && "$PY" eval.py score --predictions "$A/official-judged-$name.jsonl") 2>&1 | tee "$A/official-score-$name.txt" | tail -12
  # The port on the original's judged file: the same aggregate, as JSON.
  "$PY" -m trace2env awb-score --predictions "$A/official-judged-$name.jsonl" --summary "$A/official-score-$name.json" > /dev/null
  "$PY" - "$A/official-score-$name.txt" "$A/official-score-$name.json" <<'PYEOF'
import json, re, sys
text, summary = open(sys.argv[1]).read(), json.load(open(sys.argv[2]))
original = float(re.search(r"Overall:\s*([0-9.]+)", text).group(1))
print(f"overall: original eval.py {original:.2f} | port on the same judged file {summary['overall']:.2f}")
sys.exit(0 if abs(original - summary["overall"]) < 0.01 else "the port and the original aggregate differently")
PYEOF
  # The judging's record (never the key): which code and which judge produced official-judged-$name.jsonl.
  printf '{\n  "label": "%s",\n  "rows_env": "%s",\n  "code": "QwenLM/Qwen-AgentWorld eval/eval.py judge + score, unmodified",\n  "commit": "%s",\n  "judge_model": "%s",\n  "judge_base_url": "%s",\n  "settings": "eval.py defaults: temperature 0.6, max_tokens 32768, max_retries 3",\n  "rows": %s,\n  "started": "%s",\n  "finished": "%s"\n}\n' \
    "$name" "$ROWS_ENV" "$COMMIT" "$JUDGE_MODEL" "$JUDGE_BASE_URL" "$((got))" "$STARTED" "$(date +%FT%T)" > "$A/official-judge-run-$name.json"
done
echo "[$(date +%FT%T)] ORIGINAL JUDGING DONE"
