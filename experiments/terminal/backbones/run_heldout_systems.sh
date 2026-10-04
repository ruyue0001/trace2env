#!/usr/bin/env bash
# Held-out validation (rows outside the development trajectories: heldout_rows.jsonl, shards heldout_shards/), resumable.
# Same protocol as run_v51_systems.sh: one awb-run process per trajectory (at most MAX_PARALLEL at a time), the unchanged
# gpt-5.2 judge (JUDGE_SHARDS parallel shards), official-input audit and knowledge-gate audit. Outputs under
# backbones/<slug>/heldout/. Systems: harness_only_v51, trace2env_v531 (the v5.3.1 gate, no judge).
#   BACKBONE_MODEL=openai/gpt-5.6-sol REASONING_ARGS="--chat-reasoning-effort medium" AGENT_MAX_OUTPUT_TOKENS=16384 \
#     BACKBONE_CACHE_DIR=work/exp-v1_20_r=1/awb/cache-v2 SYSTEMS="harness_only_v51 trace2env_v531" run_heldout_systems.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${OPENROUTER_API_KEY:?}"; : "${BACKBONE_MODEL:?}"
# Backbone transport overrides (the judge always uses OpenRouter): BASE_URL, API_KEY_ENV, SCHEMA_ARGS, e.g. the official
# DeepSeek API: BASE_URL=https://api.deepseek.com API_KEY_ENV=DEEPSEEK_API_KEY SCHEMA_ARGS="--chat-schema-mode json_object"
BASE_URL="${BASE_URL:-https://openrouter.ai/api/v1}"; API_KEY_ENV="${API_KEY_ENV:-OPENROUTER_API_KEY}"; : "${!API_KEY_ENV:?}"
read -r -a SCHEMA <<< "${SCHEMA_ARGS:-}"
# AGENT_ARGS, e.g. "--agent-transport tools": native function calling for the world-model agent (the official DeepSeek
# API's JSON mode returns blank output on long briefs; tracking and single-shot stay on JSON mode)
read -r -a AGENT <<< "${AGENT_ARGS:-}"
# EXP_ROOT selects the experiment (data, packages, outputs); the scripts stay in the terminal experiment's backbones/ dir.
# PKG_FULL / PKG_SCHEMA name the packages relative to EXP_ROOT (defaults: the terminal packages).
EXP="${EXP_ROOT:-work/exp-v1_20_r=1}"; SCRIPTS="work/exp-v1_20_r=1/backbones"; SLUG="${BACKBONE_SLUG:-$(echo "$BACKBONE_MODEL" | sed 's#.*/##; s/[^A-Za-z0-9.]/-/g')}"
PKG_FULL="${PKG_FULL:-packages/v1_20_r=1}"; PKG_SCHEMA="${PKG_SCHEMA:-packages/v1_20_r=1-schema_only}"
# TRACE_CORPUS: the raw construction traces for prompting_rag (a directory of trace files; the terminal default).
TRACE_CORPUS="${TRACE_CORPUS:-work/exp-v1_20_r=1/traces/reconstruction}"
# RUN_NAME / ROWS_FILE / SHARDS_DIR select the row set: the default is the 100-row held-out set; the final evaluation
# uses RUN_NAME=full354 ROWS_FILE=full354_rows.jsonl SHARDS_DIR=full354_shards (all 354 rows, 76 trajectories).
RUN_NAME="${RUN_NAME:-heldout}"; B="$EXP/backbones/$SLUG/$RUN_NAME"; mkdir -p "$B"
# ROWS_FILE / SHARDS_DIR: a bare name lives under $EXP/backbones/ (the terminal sets); a path with a slash is relative to $EXP.
ROWS_FILE="${ROWS_FILE:-heldout_rows.jsonl}"; SHARDS_DIR="${SHARDS_DIR:-heldout_shards}"
case "$ROWS_FILE" in */*) ROWS="$EXP/$ROWS_FILE" ;; *) ROWS="$EXP/backbones/$ROWS_FILE" ;; esac
case "$SHARDS_DIR" in */*) SHARDS="$EXP/$SHARDS_DIR" ;; *) SHARDS="$EXP/backbones/$SHARDS_DIR" ;; esac
N_ROWS="$(wc -l < "$ROWS")"
read -r -a REASONING <<< "${REASONING_ARGS:---chat-reasoning-effort medium}"
# PROVIDER_JSON pins OpenRouter's upstream provider for every backbone call (not the judge), e.g.
#   PROVIDER_JSON='{"order": ["DeepSeek"], "allow_fallbacks": false}'; use BACKBONE_SLUG to keep outputs apart.
PROVIDER=(); if [ -n "${PROVIDER_JSON:-}" ]; then PROVIDER=(--chat-provider-json "$PROVIDER_JSON"); fi
CACHE="${BACKBONE_CACHE_DIR:-$EXP/backbones/$SLUG/cache}"; CAP="${AGENT_MAX_OUTPUT_TOKENS:-16384}"; SYSTEMS="${SYSTEMS:-harness_only_v51 trace2env_v531}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"; JUDGE_SHARDS="${JUDGE_SHARDS:-8}"
OR=(--provider chat --base-url "$BASE_URL" --api-key-env "$API_KEY_ENV" --model "$BACKBONE_MODEL" "${REASONING[@]}" "${PROVIDER[@]}" "${SCHEMA[@]}" --max-output-tokens 65536)
COMMON=(--mode agentic --allow-unvalidated --split all "${OR[@]}" --features default,evidence --agent-max-output-tokens "$CAP" --official-input --cache-dir "$CACHE" "${AGENT[@]}")
git rev-parse --short HEAD > "$B/code-commit.txt"
# TRACE_CORPUS_MAP (optional, relative to EXP_ROOT): {trajectory id: raw-trace directory relative to EXP_ROOT}, replacing
# TRACE_CORPUS per shard (prompting_rag under the cross-fit).
# PKG_MAP (optional, relative to EXP_ROOT): a JSON object {trajectory id: package dir relative to EXP_ROOT} for
# cross-fitted evaluations (the android cross-fit): each trajectory's shard runs with its own out-of-fold package
# instead of PKG_FULL; PKG_MAP_SCHEMA does the same for the systems that take PKG_SCHEMA. The mapped packages'
# manifest digests are recorded together.
shard_pkg() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }
run_system() {  # LABEL PKG_REL [extra]
  local label="$1" pkg_rel="$2" pkg="$EXP/$2" map=""; shift 2
  if [ -f "$B/pred-$label.jsonl" ] && [ "$(wc -l < "$B/pred-$label.jsonl")" -ge "$N_ROWS" ]; then
    echo "[$(date -Is)] $SLUG/$label $RUN_NAME predictions already present; not re-run"; return
  fi
  if [ -n "${PKG_MAP:-}" ]; then
    if [ "$pkg_rel" = "$PKG_SCHEMA" ]; then map="$EXP/${PKG_MAP_SCHEMA:?PKG_MAP_SCHEMA is required with PKG_MAP for schema-only systems}"; else map="$EXP/$PKG_MAP"; fi
    python3 -c "import json, sys; print(sorted(set(json.load(open(sys.argv[1])).values())))" "$map" > "$B/package-map-$label.txt"
    for p in $(python3 -c "import json, sys; print(' '.join(sorted(set(json.load(open(sys.argv[1])).values()))))" "$map"); do
      printf '%s %s\n' "$p" "$(sha256sum "$EXP/$p/manifest.json" | cut -c1-16)"; done > "$B/package-manifest-$label.sha256"
  else
    sha256sum "$pkg/manifest.json" | cut -c1-16 > "$B/package-manifest-$label.sha256"
  fi
  echo "[$(date -Is)] $SLUG/$label $RUN_NAME ($N_ROWS rows) pkg=$pkg_rel base=$BASE_URL model=$BACKBONE_MODEL reasoning='${REASONING[*]}' schema='${SCHEMA_ARGS:-}' agent='${AGENT_ARGS:-}' provider='${PROVIDER_JSON:-}' cap=$CAP parallel=$MAX_PARALLEL extra: $* start"
  local running=0
  for f in "$SHARDS"/traj_*.jsonl; do
    t="${f##*/traj_}"; t="${t%.jsonl}"
    if [ -f "$B/pred-$label-$t.jsonl" ] && [ "$(wc -l < "$B/pred-$label-$t.jsonl")" -ge "$(wc -l < "$f")" ]; then continue; fi
    if [ -n "$map" ]; then pkg="$EXP/$(shard_pkg "$map" "$t")"; fi
    local -a extra=("$@")
    if [ -n "${TRACE_CORPUS_MAP:-}" ]; then  # cross-fit: the raw-trace corpus is per trajectory too (its out-of-fold corpus)
      local corpus="$EXP/$(shard_pkg "$EXP/$TRACE_CORPUS_MAP" "$t")"
      for i in "${!extra[@]}"; do if [ "${extra[$i]}" = "--trace-corpus" ]; then extra[$((i+1))]="$corpus"; fi; done
    fi
    python -m trace2env awb-run "$f" --package "$pkg" "${COMMON[@]}" "${extra[@]}" --call-log "$B/calls-$label-$t.jsonl" \
      --session-root "$B/sessions-$label-$t" --output "$B/pred-$label-$t.jsonl" > "$B/$label-$t.log" 2>&1 &
    running=$((running + 1))
    if [ "$running" -ge "$MAX_PARALLEL" ]; then wait -n || true; running=$((running - 1)); fi  # a failed shard must not abort the chain
  done
  wait || true
  cat "$B"/pred-$label-[0-9]*.jsonl > "$B/pred-$label.jsonl"
  echo "[$(date -Is)] $SLUG/$label $RUN_NAME PREDICTIONS DONE rows=$(wc -l < "$B/pred-$label.jsonl")"
}
for label in $SYSTEMS; do
  case "$label" in
    harness_only_v51) run_system harness_only_v51 "$PKG_SCHEMA" --no-state-tracking --no-package-knowledge ;;
    trace2env_v531) run_system trace2env_v531 "$PKG_FULL" --knowledge-gate ;;
    trace2env_v532) run_system trace2env_v532 "$PKG_FULL" --knowledge-gate --knowledge-names pages ;;  # v5.3.2: page identity (web)
    trace2env_v533) run_system trace2env_v533 "$PKG_FULL" --knowledge-gate --knowledge-names screens ;;  # v5.3.3: screen identity (Android)
    prompting) run_system prompting "$PKG_FULL" --mode prompting ;;
    envpack_prompting) run_system envpack_prompting "$PKG_FULL" --mode envpack_prompting ;;  # one free-text call over official input + package block
    prompting_rag) run_system prompting_rag "$PKG_FULL" --mode prompting_rag --trace-corpus "$TRACE_CORPUS" --rag-top-k 5 --rag-turn-chars 4000 ;;  # official input + top-5 raw turns
    *) echo "unknown system $label"; exit 2 ;;
  esac
done
JUDGE=(--judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir "${JUDGE_CACHE:-work/exp-v1_20_r=1/awb/judge-cache}")
for label in $SYSTEMS; do
  if [ -f "$B/judged-$label.jsonl" ] && [ "$(wc -l < "$B/judged-$label.jsonl")" -ge "$N_ROWS" ]; then echo "$label already judged"; continue; fi
  rm -f "$B/judge-shard-$label-"*.jsonl
  split -n "l/$JUDGE_SHARDS" -d --additional-suffix=.jsonl "$B/pred-$label.jsonl" "$B/judge-shard-$label-"
  for shard in "$B/judge-shard-$label-"*.jsonl; do
    [[ "$shard" == *-judged.jsonl ]] && continue
    python -m trace2env awb-judge --predictions "$shard" --output "${shard%.jsonl}-judged.jsonl" "${JUDGE[@]}" > "${shard%.jsonl}.log" 2>&1 &
  done
  wait
  cat "$B/judge-shard-$label-"*-judged.jsonl > "$B/judged-$label.jsonl"
  echo "[$(date -Is)] $SLUG/$label JUDGED rows=$(wc -l < "$B/judged-$label.jsonl")"
done
for label in $SYSTEMS; do
  python -m trace2env awb-score --predictions "$B/judged-$label.jsonl" --summary "$B/score-$label.json" | tail -3
  if [ "$label" != prompting ]; then
    python "$SCRIPTS/official_input_audit.py" --rows "$ROWS" --calls "$B/calls-$label-*.jsonl" --out "$B/audit-$label.json" | head -12
  fi
  if [ "$label" = trace2env_v531 ] && [ -z "${PKG_MAP:-}" ]; then  # cross-fits audit per package afterwards (gate_audit_crossfit.py)
    python "$SCRIPTS/knowledge_gate_audit.py" --rows "$ROWS" --calls "$B/calls-$label-*.jsonl" --judged "$B/judged-$label.jsonl" --package "$EXP/$PKG_FULL" --out "$B/gate-audit-$label.json" | head -30
  fi
done
echo "[$(date -Is)] $SLUG $RUN_NAME DONE"
