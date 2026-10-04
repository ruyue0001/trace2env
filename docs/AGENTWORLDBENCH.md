# Evaluating on AgentWorldBench

[AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench) (Qwen-AgentWorld) evaluates a language world
model one step at a time: a record gives the real prefix of an agent trajectory and the current action, and the world
model predicts the environment's next observation. An LLM judge scores Format, Factuality, Consistency, Realism and
Quality from 1 to 5; the official scorer rescales each to 0–100 and averages the five. `trace2env` ports the official
input layout, judge prompts (verbatim, under `src/trace2env/agentworld_prompts/`) and aggregation, and adds the harness
as a world model.

## Data

Download the test files of the dataset into `work/agentworldbench/` (`terminal_test.jsonl`, `swe_test.jsonl`,
`android_test.jsonl`, `web_test.jsonl`, ...). The files are never modified. A trajectory contributes several records,
one per evaluated turn, so statistics must treat the trajectory as the unit of dependence. The Android file reuses some
identifiers for distinct trajectories; the cross-fit tooling under `experiments/android/` derives unique ones.

## Commands

```bash
# Benchmark trajectories as construction episodes (smoke tests and cross-fits; a SplitManifest is written)
python -m trace2env awb-export work/agentworldbench/terminal_test.jsonl --output work/awb/train

# Predict every record with one of the four systems (see the table), then judge and score
python -m trace2env awb-run work/agentworldbench/terminal_test.jsonl --mode agentic --package WORLDBOOK \
  --official-input --knowledge-gate --features default,evidence --allow-unvalidated --output work/awb/pred.jsonl \
  --provider chat --base-url URL --api-key-env KEY_VAR --model MODEL --cache-dir work/awb/cache
python -m trace2env awb-judge --predictions work/awb/pred.jsonl --output work/awb/judged.jsonl \
  --judge-model gpt-5.2 --judge-api-key-env KEY_VAR --cache-dir work/awb/judge-cache
python -m trace2env awb-score --predictions work/awb/judged.jsonl --summary work/awb/score.json
```

| `--mode` | What is predicted from | Use |
|---|---|---|
| `prompting` | The record's own system prompt and full history in one chat call, exactly as the official inference code builds it | The official baseline (Direct Prompting) |
| `prompting_rag` | The official input plus `--rag-top-k` raw turns retrieved lexically from `--trace-corpus` (a directory of construction traces) | Trace RAG Prompting |
| `envpack_prompting` | The official input plus a fixed, non-agentic view of `--package` (schemas, rules, contracts, notes, demonstrations, retrieved evidence) | Worldbook Prompting |
| `agentic` | The harness: the prefix is reconstructed into session state and episodic memory, and the world-model agent simulates the turn with its tools; `--no-state-tracking --no-package-knowledge` on a schema-only twin of the worldbook gives the package-free control (Harness only) | Trace2Env |

Output rows keep every benchmark field plus `gen` (the `<predicted_observation>`-wrapped prediction) and a
`trace2env` field with the route (`rule`, `planner`, `fallback_single_shot`, `fallback`, `error`), tool calls, usage and
gate decisions, so the official `eval.py judge` and `score` also accept them. Records the judge cannot score are excluded,
as in the official scorer. Pass `--cache-dir` to both commands: calls are content-addressed and an interrupted run resumes.

## Run-time state reconstruction

A record has no explicit state, so before the harness predicts a turn the `StateTracker` reads the observed prefix: a
rule whose rendered observation matches the real one is applied deterministically, the remaining turns are explained in
windows of `--state-window` turns by bounded, schema-checked `track_state` model calls, and an eligible rule never
overrides an observed output. With `--official-input` the benchmark's own model input is placed once and verbatim in the
agent's input; with `--knowledge-gate` every worldbook item shown carries an applicability label and foreign values are
masked (`--knowledge-names paths|pages|screens` chooses the identity signal: file paths, web pages, Android screens).
`--environment-prompt-chars` bounds the record's system prompt as the environment description (24,000 by default,
0 for the whole prompt).

## Reading the numbers

One record is one teacher-forced step; there is no rollout. Scores from trajectories that share a task with the
construction traces are not transfer and should be reported separately (`experiments/terminal/awb/construction_task_split.py`,
`experiments/android/row_strata.py`, `experiments/swe/make_folds.py`). Numbers from hash-split or cross-fit trajectories are not leaderboard-comparable.
`REPRODUCING.md` lists the exact flags of the systems reported in the paper.
