# Trace2Env on ALFWorld

> `work/` is this repository's gitignored scratch directory; the `work/...` paths below are the layout the scripts assume, not files tracked here. See [REPRODUCING.md](../REPRODUCING.md) for the data and the settings of the reported runs.

This experiment separates construction and evaluation by Word2World test-index
row. The task agent and world-model agent are both GPT-5.6-Sol. The real
ALFWorld simulator supplies construction trajectories and later scores action
replay; it does not supply observations during Trace2Env interaction.

## Construction selection and package

Rows 0–99 (IDs 2420–2519) are evaluation only. Rows 100–199 (IDs 2520–2619)
form the construction pool. [`collect_alfworld_trace2env.py`](../scripts/collect_alfworld_trace2env.py)
reads task metadata and game-file hashes before running an agent. It excludes
any construction task whose goal-family/object/destination signature appears
in the evaluation set, then excludes repeated signatures and exact game-file
duplicates within the pool. It orders the remaining tasks to spread calls
across goal families, objects, and destinations. The collector runs the
released AgentGym ReAct task prompt against fresh **real** ALFWorld games and
stops at 40 simulator-rewarded wins. Failed attempts are retained for audit but
are not reconstruction inputs.

Each selected win is exported as a separate episode with the initial visible
observation and every action/real observation pair. The action is normalized
to a stable typed interface while retaining the original text in `raw`.
The source files contain no simulator database snapshots, PDDL state, or
reward labels. `split_manifest.json` binds their content hashes to the train
split; `trace_manifest.json` records the selected signatures and the held-out
evaluation IDs. The generic ALFWorld command interface supplied during
reconstruction is documented in `domain_description.md` in the work directory.

```bash
python3 scripts/collect_alfworld_trace2env.py select \
  --output work/exp-alfworld-real/trace2env-40-success-v1

arch -x86_64 work/exp-alfworld-real/.venv-x86/bin/python \
  scripts/collect_alfworld_trace2env.py collect \
  --output work/exp-alfworld-real/trace2env-40-success-v1

python3 scripts/collect_alfworld_trace2env.py export \
  --output work/exp-alfworld-real/trace2env-40-success-v1

uv venv work/exp-alfworld-real/.venv-build --python 3.12
uv pip install --python work/exp-alfworld-real/.venv-build/bin/python -e .
PYTHON=work/exp-alfworld-real/.venv-build/bin/python \
  bash scripts/run_alfworld_reconstruction.sh
```

The collection command uses `OPENROUTER_API_KEY` from the process environment.
It never writes the key to an artifact. The reconstruction script requires the
40-win export and writes a candidate package under
`trace2env-40-success-v1/build/packages/alfworld-40-success-v1/`. See the
work-directory `REPORT.md` for the completed selection, package, and
evaluation results.

## Current-turn interaction

[`run_alfworld_trace2env.py`](../scripts/run_alfworld_trace2env.py) starts a
fresh persistent session for each evaluation task. It initializes the session
once with the same task-specific Word2World initial description used by the
prompting-world-model comparison and the initial task-agent observation. That
privileged description comes from the simulator files **only at episode
initialization**; no later real simulator observation enters the rollout.

At each turn, the task agent sends one action to `RuntimeHarness.step`. The
request carries the current normalized command and a fixed environment
protocol; it does **not** carry earlier actions or observations. Trace2Env
reloads structured state and episodic memory from the session's SQLite file.
The agent brief includes recent entries and exposes `recall` and `read_turn`
for older observations. Every committed prediction is recorded after the
state commit. The runner checks that the memory-entry count equals the
committed step count plus the turn-0 seed before and after every step. It also
checks the seed, the latest turn number and kind, and that the saved
observation matches the one returned to the task agent. This detects a missing
or mismatched memory write instead of silently continuing with an incomplete
prefix. The task agent's own chat history is separate and remains full, as in
Word2World.

The world-model success marker is a model claim. Replay its saved task-agent
actions in fresh real ALFWorld games with
[`replay_alfworld_prompting_wm.py`](../scripts/replay_alfworld_prompting_wm.py)
to measure grounded W2R success.
The runner accepts `--start-index` and `--limit` for disjoint held-out slices
or fresh retries of an API-error task without reusing a partial session.
After all slices complete, `scripts/merge_alfworld_trace2env.py --inputs
SLICE_DIR... --output AGGREGATE_DIR` checks that their IDs cover all 100 tasks,
their package and protocol settings match, and each completed task has a full
memory prefix. It accepts `--retry DIR` for a fresh one-task replacement of an
API-error record. With `--allow-errors`, unresolved harness errors remain in
the 100-task aggregate and count as task failures. The replay script records
them as `source_harness_error` without running their incomplete action
sequence in the simulator; it reports simulator errors separately. Keep the
untouched first-attempt aggregate alongside any retry-completed aggregate so
the single-run and retry-completed rates are distinguishable. The aggregate
directory is suitable as input to the real simulator replay script.
