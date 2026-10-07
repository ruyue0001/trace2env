# Reproducing the experiments

This guide maps the paper's experiments to the code, the scripts and the data. Every reported number is a single run per
system with the settings below. The scripts and reports use the lab's run labels, which map to the paper's system names
as follows; the code's "package" is the paper's worldbook.

| Run label | Paper name |
|---|---|
| `prompting` | Direct Prompting |
| `prompting_rag` | Trace RAG Prompting |
| `envpack_prompting` | Worldbook Prompting |
| `harness_only_v51` | Harness only |
| `trace2env_v531` (terminal, swe), `trace2env_v532` (web), `trace2env_v533` (android) | Trace2Env |

The three Trace2Env labels differ only in the Applicability Gate's identity signal for evidence about the current episode
(`--knowledge-names paths|pages|screens`: file paths, web pages, Android screens); everything else is identical.

## What you need

| Item | Where it comes from |
|---|---|
| AgentWorldBench test files (`terminal`, `swe`, `android`, `web`) | the [Hugging Face dataset](https://huggingface.co/datasets/Qwen/AgentWorldBench), placed in `work/agentworldbench/<split>_test.jsonl` (never modified) |
| Terminal-Bench 2.0 traces (20 construction + 16 scaling trajectories, own collection) | on request (see below); or re-collect with Harbor and Terminus-2 ([docs/TRACE_COLLECTION.md](docs/TRACE_COLLECTION.md)) |
| WebArena traces (50 own trajectories) | on request; or self-host WebArena and re-collect through Playwright MCP (`scripts/webarena/`, `scripts/collect_webarena.py`) |
| Android and SWE construction data | the benchmark's own trajectories under a task-grouped k=5 cross-fit (`experiments/android/make_folds.py`, `experiments/swe/make_folds.py`); nothing to download beyond the test files |
| EnvScaler rollouts and environment definitions | on request (unpack into `work/envscaler/`), converted with `trace2env envscaler-export` (`experiments/envscaler/run_export.sh`) |
| ALFWorld and SciWorld | the `baseline/Word2World` submodule (task lists, ALFWorld grammar, AgentGym adapters) plus the simulators themselves ([docs/ALFWORLD_REAL.md](docs/ALFWORLD_REAL.md), [docs/ALFWORLD_TRACE2ENV.md](docs/ALFWORLD_TRACE2ENV.md)) |
| Frozen worldbooks, predictions, judge outputs, call logs of the reported runs | the Terminal worldbook is published with the [live demo](https://huggingface.co/spaces/Quanyu001/trace2env_demo/tree/main/worldbook); the others are available on request and unpack into `work/` with the layout the scripts expect |
| Model access | an OpenAI-compatible endpoint: the reported runs used `openai/gpt-5.6-sol` through OpenRouter (`OPENROUTER_API_KEY`) and `deepseek-v4.1-flash` through DeepSeek's API (`DEEPSEEK_API_KEY`); the judge is `openai/gpt-5.2` through OpenRouter |

Every worldbook's `manifest.json` lists the SHA-256 of each of its files, and the launchers record the manifest hash of
every worldbook they ran, so a reproduction can check that it used the same one. Keys are read from environment
variables (`--api-key-env`); never put them on a command line.

## Worldbook construction

All worldbooks were built with `gpt-5.6-sol` through the stage-by-stage driver, with the library defaults:

```bash
export OPENROUTER_API_KEY=...
MODEL=(--provider chat --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
       --model openai/gpt-5.6-sol --chat-reasoning-effort medium --max-output-tokens 65536)
python scripts/pilot_stages.py WORK ingest --episodes EPISODES --split-manifest MANIFEST \
  --environment-id ID --name NAME --description @DESCRIPTION.md --domain terminal "${MODEL[@]}"
python scripts/pilot_stages.py WORK extract --workers 6 "${MODEL[@]}"
for stage in schema rules renderers notes compile; do python scripts/pilot_stages.py WORK $stage "${MODEL[@]}"; done
```

The per-environment scripts hold the exact invocations: `experiments/terminal/run_reconstruction.sh`,
`experiments/web/run_reconstruction.sh`, `experiments/android/build_all.sh` and `experiments/swe/build_all.sh` (one
worldbook per sub-source and fold, shared extraction cache), `experiments/scaling/run_reconstruction.sh` (nested
subsets of 1 to 36 traces), `experiments/envscaler/run_reconstruction.sh`, `scripts/run_alfworld_reconstruction.sh` and
`scripts/run_scienceworld_reconstruction.sh`. Domain descriptions are under `experiments/*/description*.md` and
`experiments/*/descriptions/`. The schema-only twin of a worldbook that Harness only runs on is derived with
`scripts/ablate_package.py` (`experiments/android/make_schema_only.sh` shows the call).

## Single-turn next-state prediction (Table 1 and Appendix B)

One `awb-run` call per system and shard, the same rows for every system, then the official judge and scorer. The
launcher `experiments/terminal/backbones/run_heldout_systems.sh` runs all five systems and the judge for one backbone;
`experiments/android/run_eval.sh` and `experiments/swe/run_eval.sh` wrap it with the per-fold worldbook maps
(`PKG_MAP`, `PKG_MAP_SCHEMA`, `TRACE_CORPUS_MAP`). The flags that define each system:

| Paper name | `awb-run` flags |
|---|---|
| Direct Prompting | `--mode prompting` (official layout; temperature 0.6, 32,768 output tokens, no reasoning parameter) |
| Trace RAG Prompting | `--mode prompting_rag --trace-corpus TRACES --rag-top-k 5 --rag-turn-chars 4000` (`--rag-style terminal`, the default) |
| Worldbook Prompting | `--mode envpack_prompting --package WORLDBOOK` (defaults: 6 evidence turns at 3,000 characters, 4 demonstrations) |
| Harness only | `--mode agentic --package SCHEMA_ONLY_TWIN --no-state-tracking --no-package-knowledge` + the common agentic flags |
| Trace2Env | `--mode agentic --package WORLDBOOK --knowledge-gate` + the common agentic flags; `--knowledge-names pages` on web, `--knowledge-names screens` on android, `paths` (default) on terminal and swe |

Common agentic flags: `--allow-unvalidated --official-input --features default,evidence --effect-rejection
keep_observation --agent-max-output-tokens 16384 --environment-prompt-chars 24000` (both defaults) with the backbone's
model options. GPT: `--provider chat --base-url https://openrouter.ai/api/v1 --model openai/gpt-5.6-sol
--chat-reasoning-effort medium --max-output-tokens 65536`. DeepSeek: `--base-url https://api.deepseek.com --model
deepseek-v4.1-flash --chat-reasoning-effort low --agent-transport tools --agent-max-output-tokens 65536
--chat-schema-mode json_object`. Judge and score:

```bash
python -m trace2env awb-judge --predictions pred.jsonl --output judged.jsonl \
  --judge-model openai/gpt-5.2 --judge-base-url https://openrouter.ai/api/v1 --judge-api-key-env OPENROUTER_API_KEY --cache-dir JUDGE_CACHE
python -m trace2env awb-score --predictions judged.jsonl --summary score.json
```

Paired statistics (mean row difference, row-level SE, trajectory-clustered bootstrap CIs with 10,000 resamples,
win/tie/loss) and the per-environment tables come from `experiments/terminal/backbones/full354_report.py`,
`experiments/web/full200_report.py`, `experiments/android/full200_report.py` and `experiments/swe/full472_report.py`;
the unseen-task strata of Appendix B from `experiments/terminal/awb/construction_task_split.py`, the web
`construction_overlap` analysis in `full200_report.py`, and the overlap strata of the cross-fits from `experiments/android/row_strata.py` and `experiments/swe/make_folds.py` (`row_strata.json`). The two
integrity audits are `experiments/terminal/backbones/official_input_audit.py` and `knowledge_gate_audit.py`
(`experiments/android/gate_audit_v533.py`, `experiments/swe/gate_audit_crossfit.py` for the cross-fits).

## Ablations and trace scaling (Section 4 and Appendix B)

Terminal, `gpt-5.6-sol`: `experiments/terminal/awb/run_tiers_chain.sh` (schema-only, structure, example tiers),
`run_agent_side_chain.sh` (no adaptive loop: `--prediction-mode single_shot`; no state tracking:
`--no-state-tracking`), `run_ablations_chain.sh`, and `experiments/scaling/` (`select_traces.py`,
`run_reconstruction.sh`, `run_eval_chain.sh`, `analyze_scaling.py`) for the 1 to 36 trace curve.

## Multi-turn interaction (Table 2)

ALFWorld: `scripts/run_alfworld_real.py` (Real), `scripts/run_alfworld_prompting_wm.py` and
`scripts/replay_alfworld_prompting_wm.py` (prompted world model, WM and W2R), `scripts/run_alfworld_trace2env.py`
(Trace2Env, with the real-environment replay), after `scripts/collect_alfworld_trace2env.py` and
`scripts/run_alfworld_reconstruction.sh` built the worldbook from 40 successful trajectories. SciWorld:
`scripts/collect_scienceworld_measurement.py`, `scripts/run_scienceworld_reconstruction.sh`,
`scripts/run_scienceworld_measurement.py` (all three world models and the replay), `scripts/run_scienceworld_slices.sh`.
The consistency ratio is W2R / Real. EnvScaler closed-loop runs use `trace2env envscaler-long-horizon`
([docs/LONG_HORIZON.md](docs/LONG_HORIZON.md)).

## Costs

At list prices the worldbooks cost about $27 (terminal), $19 (web), $154 (android, 15 worldbooks) and $340 (swe,
10 worldbooks) to build; an AgentWorldBench evaluation of one system on one environment costs between a few dollars
(prompting) and a few tens of dollars (Trace2Env), plus the judge. Pass `--cache-dir` everywhere: calls are
content-addressed, so an interrupted build or evaluation resumes without re-sampling.
