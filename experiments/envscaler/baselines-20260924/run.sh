#!/usr/bin/env bash
# Frozen 3-environment x 2-backbone x 2-baseline matrix.
# Order: env 160, 172, 174; within each, GPT then DeepSeek; within each pair, EnvPack then harness-only v5.1.
# Requires the key for each selected backbone. DRY_RUN=1 prints every awb-run command.
# ONLY_BACKBONE can select gpt56sol or deepseek-v41-flash.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=work/exp-envscaler/baselines-20260924
PYTHON=.venv/bin/python
export PYTHON
"$PYTHON" "$ROOT/validate.py"

for env in env_160_rl env_172_rl env_174_rl; do
  ROWS_DIR="$ROOT/rows/$env"
  full_package="work/exp-envscaler/$env/packages/envscaler-$env-v1"
  schema_package="$ROOT/packages/$env-schema-only"
  for backbone in gpt56sol deepseek-v41-flash; do
    if [ -n "${ONLY_BACKBONE:-}" ] && [ "$backbone" != "$ONLY_BACKBONE" ]; then continue; fi
    case "$backbone" in
      gpt56sol)
        model=openai/gpt-5.6-sol
        base_url=https://openrouter.ai/api/v1
        api_key_env=OPENROUTER_API_KEY
        agent_extra='--official-input --features default,evidence --agent-max-output-tokens 16384 --effect-rejection fail'
        ;;
      deepseek-v41-flash)
        model=deepseek-flash
        base_url=https://api.deepseek.com
        api_key_env=DEEPSEEK_API_KEY
        agent_extra='--official-input --features default,evidence --agent-max-output-tokens 65536 --effect-rejection fail --chat-reasoning-effort low --chat-schema-mode json_object --agent-transport tools'
        ;;
    esac
    for baseline in envpack_prompting harness_only_v51; do
      case "$baseline" in
        envpack_prompting)
          system=envpack_prompting
          package="$full_package"
          extra=''
          label="envpack-prompting-$backbone"
          ;;
        harness_only_v51)
          system=harness_only
          package="$schema_package"
          extra="$agent_extra"
          label="harness-only-v51-$backbone"
          ;;
      esac
      if [ -z "${DRY_RUN:-}" ] && [ -f "$ROWS_DIR/run-$label.json" ]; then
        echo "skip completed $env/$label"
        continue
      fi
      echo "RUN $env $backbone $baseline"
      ROWS_DIR="$ROWS_DIR" MODEL="$model" BASE_URL="$base_url" API_KEY_ENV="$api_key_env" PROVIDER=chat \
        PKG="$package" LABEL="$label" EXTRA="$extra" DRY_RUN="${DRY_RUN:-}" \
        work/exp-envscaler/run_eval.sh "$system" "$env"
      if [ -z "${DRY_RUN:-}" ]; then
        "$PYTHON" - "$ROWS_DIR/pred-$label.jsonl" <<'PYEOF'
import json, sys
errors = []
with open(sys.argv[1], encoding="utf-8") as source:
    for line in source:
        row = json.loads(line)
        error = (row.get("trace2env") or {}).get("error")
        if error:
            errors.append((row.get("id"), error))
if errors:
    print(f"{len(errors)} prediction errors in {sys.argv[1]}; first: {errors[0]}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
      fi
    done
  done
done

if [ -z "${DRY_RUN:-}" ]; then
  "$PYTHON" "$ROOT/summarize.py"
fi
