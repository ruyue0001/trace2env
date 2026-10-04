#!/usr/bin/env bash
# Harness v5.x on the fixed 30-row evaluation set (eval30_rows.jsonl, seed 20260922, 25 trajectories), resumable:
#   harness_only_v51  schema-only package, no state tracking, no package knowledge, agent loop + memory tools, --official-input
#   trace2env_v51     full package v1_20_r=1, state tracking, evidence read path, all tools, --official-input
#   trace2env_v52     trace2env_v51 plus --knowledge-gate (v5.2 compatibility-gated package knowledge)
#   state_only_v52    v5.2 runtime with state tracking but no package-derived knowledge (--no-package-knowledge); pre-analysis ablation
#   format_only_v52   v5.2 gate in format_only mode: supporting/uncertain evidence withheld; pre-analysis ablation
#   trace2env_v53     v5.2 gate with the corrected tokenizer plus the applicability judge (--knowledge-judge)
#   trace2env_v531    the deterministic v5.3.1 gate alone (--knowledge-gate at 57d284c or later, no judge); run it with a
#                     fresh BACKBONE_CACHE_DIR so that rows whose input equals trace2env_v52's are fresh samples (noise)
# One process per trajectory (the runner tracks state per trajectory); a system whose 30 predictions already exist is
# not re-run. Then the unchanged gpt-5.2 judge and the model-boundary audit of information preservation.
#   BACKBONE_MODEL=openai/gpt-5.6-sol REASONING_ARGS="--chat-reasoning-effort medium" AGENT_MAX_OUTPUT_TOKENS=16384 \
#     BACKBONE_CACHE_DIR=work/exp-v1_20_r=1/awb/cache-v2 [SYSTEMS="harness_only_v51 trace2env_v51"] run_v51_systems.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"; : "${BACKBONE_MODEL:?}"
EXP="work/exp-v1_20_r=1"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"; B="$EXP/backbones/$SLUG"; mkdir -p "$B"
read -r -a REASONING <<< "${REASONING_ARGS:---chat-reasoning-effort medium}"
CACHE="${BACKBONE_CACHE_DIR:-$B/cache}"; CAP="${AGENT_MAX_OUTPUT_TOKENS:-16384}"; SYSTEMS="${SYSTEMS:-harness_only_v51 trace2env_v51}"
OR=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --model "$BACKBONE_MODEL" "${REASONING[@]}" --max-output-tokens 65536)
COMMON=(--mode agentic --allow-unvalidated --split all "${OR[@]}" --features default,evidence --agent-max-output-tokens "$CAP" --official-input --cache-dir "$CACHE")
git rev-parse --short HEAD > "$B/code-commit-v51.txt"
run_system() {  # LABEL PKG_REL [extra]
  local label="$1" pkg_rel="$2" pkg="$EXP/$2"; shift 2
  if [ -f "$B/pred-$label.jsonl" ] && [ "$(wc -l < "$B/pred-$label.jsonl")" -ge 30 ]; then
    echo "[$(date -Is)] $SLUG/$label predictions already present; not re-run"; return
  fi
  sha256sum "$pkg/manifest.json" | cut -c1-16 > "$B/package-manifest-$label.sha256"
  echo "[$(date -Is)] $SLUG/$label eval30 pkg=$pkg_rel reasoning='${REASONING[*]}' cap=$CAP extra: $* start"
  for f in "$EXP"/backbones/eval30_shards/traj_*.jsonl; do
    t="${f##*/traj_}"; t="${t%.jsonl}"
    python -m trace2env awb-run "$f" --package "$pkg" "${COMMON[@]}" "$@" --call-log "$B/calls-$label-$t.jsonl" \
      --session-root "$B/sessions-$label-$t" --output "$B/pred-$label-$t.jsonl" > "$B/$label-$t.log" 2>&1 &
  done
  wait
  cat "$B"/pred-$label-[0-9]*.jsonl > "$B/pred-$label.jsonl"
  echo "[$(date -Is)] $SLUG/$label PREDICTIONS DONE rows=$(wc -l < "$B/pred-$label.jsonl")"
}
for label in $SYSTEMS; do
  case "$label" in
    harness_only_v51) run_system harness_only_v51 packages/v1_20_r=1-schema_only --no-state-tracking --no-package-knowledge ;;
    trace2env_v51) run_system trace2env_v51 packages/v1_20_r=1 ;;
    trace2env_v52) run_system trace2env_v52 packages/v1_20_r=1 --knowledge-gate ;;
    state_only_v52) run_system state_only_v52 packages/v1_20_r=1 --no-package-knowledge ;;
    format_only_v52) run_system format_only_v52 packages/v1_20_r=1 --knowledge-gate --knowledge-gate-mode format_only ;;
    trace2env_v53) run_system trace2env_v53 packages/v1_20_r=1 --knowledge-gate --knowledge-judge ;;
    trace2env_v531) run_system trace2env_v531 packages/v1_20_r=1 --knowledge-gate ;;
    *) echo "unknown system $label"; exit 2 ;;
  esac
done
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "$EXP/awb/judge-cache")
for label in $SYSTEMS; do
  python -m trace2env awb-judge --predictions "$B/pred-$label.jsonl" --output "$B/judged-$label.jsonl" "${JUDGE[@]}" > "$B/judge-$label.log" 2>&1 &
done
wait
for label in $SYSTEMS; do
  python -m trace2env awb-score --predictions "$B/judged-$label.jsonl" --summary "$B/score-$label.json" | tail -3
  python "$EXP/backbones/official_input_audit.py" --rows "$EXP/backbones/eval30_rows.jsonl" --calls "$B/calls-$label-*.jsonl" --out "$B/audit-$label.json" | head -12
  if [ "$label" = trace2env_v52 ] || [ "$label" = format_only_v52 ] || [ "$label" = trace2env_v53 ] || [ "$label" = trace2env_v531 ]; then
    python "$EXP/backbones/knowledge_gate_audit.py" --rows "$EXP/backbones/eval30_rows.jsonl" --calls "$B/calls-$label-*.jsonl" \
      --judged "$B/judged-$label.jsonl" --out "$B/gate-audit-$label.json" | head -30
  fi
done
echo "[$(date -Is)] $SLUG v5.1 EVAL30 DONE"
