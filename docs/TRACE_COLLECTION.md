# Collecting and converting traces

Trace2Env builds a worldbook from recorded interactions of the environment: for each one a sequence of actions and the
observations the real environment returned. Any source works as long as the actions and observations are recorded
verbatim; nothing in this repository rewrites observations. The converters below produce the episode layout that
`reconstruct` and `scripts/pilot_stages.py` ingest (one `user` instruction event, then alternating `action` /
`observation` events) together with a `SplitManifest` that assigns every trajectory family to `train`, `validation`
or `test`. Construction, validation and evaluation families must stay disjoint.

| Source | Converter | Notes |
|---|---|---|
| Any JSON / JSONL / role-prefixed text trace | `trace2env ingest FILES --output DIR` (`reconstruct` runs it too) | deterministic parse; episodes that lack clear action/observation kinds get one annotation model call that marks spans without rewriting text |
| Terminal-Bench 2.0 runs with Harbor and the Terminus-2 agent (ATIF trajectories) | `scripts/collect_tb2.sh`, then `trace2env atif-export` | the terminal worldbook of the paper: 20 finished, reward-1 trajectories on tasks whose benchmark trajectories do not match a trace |
| WebArena through Playwright MCP | `scripts/webarena/host_sites.sh up`, `scripts/collect_webarena.py` | self-hosted WebArena under the benchmark hostnames; `@playwright/mcp` pinned to the version whose result text matches the benchmark's; `browser_snapshot` calls are folded into the preceding action's observation |
| AgentWorldBench trajectories (Android, SWE cross-fits) | `trace2env awb-export`, `experiments/android/make_folds.py`, `experiments/swe/make_folds.py` | the longest visible record of each trajectory becomes one episode; task-grouped k=5 folds, one worldbook per sub-source and fold, every record predicted out of fold |
| EnvScaler rollouts | `trace2env envscaler-export` (`experiments/envscaler/run_export.sh`) | the recorded `observation` string is the evidence; database snapshots, diffs, outcome labels and the environment source never enter an episode and are exported separately for evaluation only |
| ALFWorld, SciWorld (real simulators) | `scripts/collect_alfworld_trace2env.py`, `scripts/collect_scienceworld_measurement.py` | successful real-simulator trajectories on tasks disjoint from the evaluation tasks; see `ALFWORLD_TRACE2ENV.md` |
| MCPMark / Toolathlon trajectories | `scripts/mcp_sample.py`, `trace2env mcp-export` | tooling only; Toolathlon's verified trajectories are gated and must not be used as training data or as a retrieval corpus that answers a benchmark |

## Terminal-Bench 2.0 with Harbor

```bash
scripts/collect_tb2.sh                      # Harbor + Terminus-2; set the model and key variables inside (OpenRouter by default)
python -m trace2env atif-export RUN_DIR --output work/tb2/episodes --manifest work/tb2/split_manifest.json
```

Exclude tasks whose benchmark trajectories match a trace strongly (shared task files or workspace paths) before
selecting construction trajectories; `experiments/terminal/awb/construction_task_split.py` reports the overlap of the
selected traces with the benchmark.

## WebArena through Playwright MCP

Node 18+ and a hosted WebArena are required (about 140 GB of Docker images; `scripts/webarena/private_daemon.sh` runs
a separate docker daemon with its data root in `WA_DOCKER_ROOT`).

```bash
scripts/webarena/host_sites.sh pull && scripts/webarena/host_sites.sh up    # four sites + :80 proxy under the benchmark hostnames
export OPENROUTER_API_KEY=...
python scripts/collect_webarena.py --task-list work/webarena/test.raw.json --selection work/webarena/collection_tasks.json \
  --output work/webarena/traces --node "$(command -v node)" --mcp-cli scripts/webarena/node_modules/@playwright/mcp/cli.js \
  --base-url https://openrouter.ai/api/v1 --model openai/gpt-5.6-sol --api-key-env OPENROUTER_API_KEY
python -m trace2env ingest work/webarena/traces/episodes/*.json --output work/webarena/ingest-preview     # parser check, no API
```

`--scripted FILE` replays fixed tool calls without a model (used by the tests). Choose tasks for page-kind coverage of
the shared sites rather than to match evaluation tasks: a worldbook covers the environment, not the task.

## EnvScaler

```bash
python -m trace2env envscaler-export ROLLOUTS/reserve/ENV --env-defs ROLLOUTS/env_defs --assign-split train \
  --output work/envscaler/ENV/traces --descriptions work/envscaler/ENV/descriptions
python -m trace2env envscaler-export ROLLOUTS/benchmark/ENV --env-defs ROLLOUTS/env_defs --assign-split test \
  --output work/envscaler/ENV/eval --ground-truth work/envscaler/ENV/ground_truth --rows work/envscaler/ENV/rows.jsonl
python -m trace2env envscaler-score --predictions PRED.jsonl --summary exact.json      # judge-free exact accuracy
```

`envscaler-probe` generates counterfactual evaluation rows by executing the environment's own source at recorded states
(evaluation only, after a replay check that the source reproduces every recorded step).

## Writing the domain description

Every worldbook build takes a short description of the domain: what the environment is, how actions are normalized, and
a suggested state vocabulary (`world.*` paths). Write it from what the construction traces show, never from the
evaluation data or the environment's source; the descriptions used for the paper are under `experiments/*/descriptions/`
and `experiments/*/description.md`.
