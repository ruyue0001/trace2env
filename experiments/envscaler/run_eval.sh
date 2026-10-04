#!/usr/bin/env bash
# Predict the held-out rows of one EnvScaler environment (recorded turns and probes, 4 shards in parallel), then
# report exact accuracy (envscaler-score, no judge). The official judge is a separate step: run_judge_official.sh
# (the original AgentWorldBench eval.py) or run_judge.sh (the repo's port, cached).
#
#   work/exp-envscaler/run_eval.sh prompting env_174_rl               # official prompting baseline, same model
#   work/exp-envscaler/run_eval.sh tracerag  env_174_rl               # prompting + top-5 construction-trace retrieval
#   work/exp-envscaler/run_eval.sh envpack_prompting env_174_rl       # one prompt with a fixed view of the package
#   work/exp-envscaler/run_eval.sh harness_only env_174_rl            # agent loop + episodic memory, no package knowledge/state
#   work/exp-envscaler/run_eval.sh agentic   env_174_rl               # Trace2Env with env_174_rl's own package (in-domain)
#   work/exp-envscaler/run_eval.sh agentic   env_174_rl env_151_rl    # Trace2Env with the package built from env_151_rl (transfer)
#   ROWS_DIR=work/exp-envscaler/env_174_rl/awb/custom work/exp-envscaler/run_eval.sh tracerag env_174_rl
#   CACHE=1 work/exp-envscaler/run_eval.sh agentic env_174_rl         # opt in to response caching
#   DRY_RUN=1 work/exp-envscaler/run_eval.sh agentic env_174_rl env_151_rl   # print the commands only
#
# Every system reads the same shards. Predictions land in <rows env>/awb/pred-<label>.jsonl with
# label = prompting | agentic-from-<package env>. Extra awb-run flags go in EXTRA (e.g. EXTRA="--features none").
set -euo pipefail
cd "$(dirname "$0")/../.."
SYSTEM="${1:?usage: run_eval.sh prompting|tracerag|envpack_prompting|harness_only|agentic ROWS_ENV [PACKAGE_ENV]}"
ROWS_ENV="${2:?usage: run_eval.sh prompting|tracerag|envpack_prompting|harness_only|agentic ROWS_ENV [PACKAGE_ENV]}"
PACKAGE_ENV="${3:-$ROWS_ENV}"
PY="${PYTHON:-python}"
EXP="work/exp-envscaler"; A="${ROWS_DIR:-$EXP/$ROWS_ENV/awb}"
MODEL="${MODEL:-openai/gpt-5.6-sol}"; BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"
API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"; PROVIDER="${PROVIDER:-chat}"
OR=(--provider "$PROVIDER" --base-url "$BASE_URL" --api-key-env "$API_KEY_ENV" --model "$MODEL" --chat-reasoning-effort medium --max-output-tokens 65536)
read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
if [ -z "${DRY_RUN:-}" ]; then : "${!API_KEY_ENV:?set $API_KEY_ENV first}"; fi
[ -f "$A/shards/shard0.jsonl" ] || { echo "no shards in $A/shards; run $EXP/run_export.sh" >&2; exit 1; }
case "$SYSTEM" in
  prompting)
    LABEL="${LABEL:-prompting}"
    # The baseline is the official inference: awb-run's free-text defaults (temperature 0.6, 32768 tokens, no reasoning
    # parameter) are upstream eval.py's; the structured-role flags in OR are not used by this mode.
    SETTINGS="official inference defaults: temperature 0.6, max_tokens 32768, no reasoning parameter"
    MODE=(--mode prompting) ;;
  tracerag)
    LABEL="${LABEL:-tracerag}"
    CORPUS="$EXP/$ROWS_ENV/traces/reconstruction"
    [ -d "$CORPUS" ] || { echo "no construction traces at $CORPUS; run $EXP/run_export.sh" >&2; exit 1; }
    SETTINGS="official inference + prompting_rag: top_k=5, turn_chars=4000, construction traces only"
    MODE=(--mode prompting_rag --trace-corpus "$CORPUS" --rag-top-k 5 --rag-turn-chars 4000) ;;
  envpack_prompting)
    LABEL="${LABEL:-envpack-prompting-from-$PACKAGE_ENV}"
    PKG="${PKG:-$EXP/$PACKAGE_ENV/packages/envscaler-$PACKAGE_ENV-v1}"
    [ -n "${DRY_RUN:-}" ] || [ -f "$PKG/manifest.json" ] || { echo "no package at $PKG; run $EXP/run_reconstruction.sh $PACKAGE_ENV" >&2; exit 1; }
    SETTINGS="official inference defaults + fixed package view: top_k=6, evidence_chars=3000, one chat call per row"
    MODE=(--mode envpack_prompting --package "$PKG" --allow-unvalidated --envpack-top-k 6 --envpack-evidence-chars 3000) ;;
  harness_only)
    LABEL="${LABEL:-harness-only}"
    PKG="${PKG:-$EXP/$PACKAGE_ENV/packages/envscaler-$PACKAGE_ENV-v1-schema-only}"
    [ -n "${DRY_RUN:-}" ] || [ -f "$PKG/manifest.json" ] || { echo "no schema-only package at $PKG; derive it with scripts/ablate_package.py" >&2; exit 1; }
    SETTINGS="--chat-reasoning-effort medium --max-output-tokens 65536 --agent-max-output-tokens 16384 --features default --no-package-knowledge --no-state-tracking"
    MODE=(--mode agentic --package "$PKG" --allow-unvalidated --features default
          --no-package-knowledge --no-state-tracking) ;;
  agentic)
    LABEL="${LABEL:-agentic-from-$PACKAGE_ENV}"
    PKG="${PKG:-$EXP/$PACKAGE_ENV/packages/envscaler-$PACKAGE_ENV-v1}"
    [ -n "${DRY_RUN:-}" ] || [ -f "$PKG/manifest.json" ] || { echo "no package at $PKG; run $EXP/run_reconstruction.sh $PACKAGE_ENV" >&2; exit 1; }
    # Keep long recorded tool observations unclipped for the state tracker (the default clip is 4000).
    SETTINGS="--chat-reasoning-effort medium --max-output-tokens 65536 --features default --observation-chars 12000"
    MODE=(--mode agentic --package "$PKG" --allow-unvalidated --features default --observation-chars 12000) ;;
  *) echo "unknown system $SYSTEM (prompting | tracerag | envpack_prompting | harness_only | agentic)" >&2; exit 1 ;;
esac
# Caches create one hash-named JSON file per distinct model request. They are useful for resuming expensive runs but
# are not part of the result, so EnvScaler experiments create them only when explicitly requested.
if [ "${CACHE:-0}" = 1 ]; then MODE+=(--cache-dir "${CACHE_DIR:-$A/cache-$LABEL}"); fi
STARTED="$(date +%FT%T)"
echo "[$STARTED] $LABEL on $ROWS_ENV rows start"
for i in 0 1 2 3; do
  ARGS=("$PY" -m trace2env awb-run "$A/shards/shard$i.jsonl" "${MODE[@]}" --split all "${OR[@]}" --output "$A/pred-$LABEL-shard$i.jsonl")
  if [ "$SYSTEM" = agentic ] || [ "$SYSTEM" = harness_only ]; then
    ARGS+=(--call-log "$A/calls-$LABEL-shard$i.jsonl" --session-root "$A/sessions-$LABEL-shard$i")
  fi
  ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${ARGS[@]}"; echo; else "${ARGS[@]}" > "$A/$LABEL-shard$i.log" 2>&1 & fi
done
[ -z "${DRY_RUN:-}" ] || exit 0
wait
cat "$A"/pred-"$LABEL"-shard{0,1,2,3}.jsonl > "$A/pred-$LABEL.jsonl"
# A shard that crashed is invisible to `wait`: the merged row count must equal the exported row count before judging.
expected=$(wc -l < "$A/rows.jsonl"); got=$(wc -l < "$A/pred-$LABEL.jsonl"); FINISHED="$(date +%FT%T)"
echo "[$FINISHED] $LABEL PREDICTIONS DONE rows=$got expected=$expected"
[ "$got" -eq "$expected" ] || { echo "row count mismatch: check $A/$LABEL-shard*.log before judging" >&2; exit 1; }
# The run's record: what produced pred-$LABEL.jsonl (never the key) and the usage the rows report, cached rows included.
RUN_SYSTEM="$SYSTEM" RUN_LABEL="$LABEL" RUN_ROWS_ENV="$ROWS_ENV" RUN_MODEL="$MODEL" RUN_BASE_URL="$BASE_URL" RUN_PROVIDER="$PROVIDER" \
RUN_PACKAGE="${PKG:-}" RUN_SETTINGS="$SETTINGS" RUN_EXTRA="${EXTRA:-}" RUN_STARTED="$STARTED" RUN_FINISHED="$FINISHED" "$PY" - "$A/pred-$LABEL.jsonl" "$A/run-$LABEL.json" <<'PYEOF'
import json, os, sys
rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
usage: dict = {}
for row in rows:
    for key, value in ((row.get("trace2env") or {}).get("usage") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            usage[key] = round(usage.get(key, 0) + value, 6)
names = ("system", "label", "rows_env", "model", "base_url", "provider", "package", "settings", "extra", "started", "finished")
record = {name: os.environ[f"RUN_{name.upper()}"] for name in names if os.environ.get(f"RUN_{name.upper()}")}
record.update(rows=len(rows), usage_reported_by_rows=usage)
json.dump(record, open(sys.argv[2], "w", encoding="utf-8"), indent=2)
PYEOF
# Exact accuracy needs no judge: recorded turns and probes are reported separately (set:recorded / set:probe).
"$PY" -m trace2env envscaler-score --predictions "$A/pred-$LABEL.jsonl" --summary "$A/exact-$LABEL.json" --scored "$A/exact-$LABEL.jsonl"
