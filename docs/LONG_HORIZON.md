# Long-horizon task-agent evaluation (EnvScaler)

> `work/` is this repository's gitignored scratch directory; the `work/...` paths below are the layout the scripts assume, not files tracked here. See [REPRODUCING.md](../REPRODUCING.md) for the data and the settings of the reported runs.

`envscaler-long-horizon` follows the task-agent/world-model interaction in
Word2World's `baseline/Word2World/scripts/interact_with_world_model/run.py`:
the task agent chooses an action, `RuntimeHarness.step` returns a simulated
observation, and the task agent continues or finishes. Unlike turn-wise `awb-run`,
all actions in an attempt use the same persistent session. Every new attempt gets
a fresh session with an explicitly empty *known* state (`EnvironmentState()`), not
the hidden scenario database. Candidate packages require `--allow-unvalidated`;
`rules_only` remains the default trust policy.

The task agent sees task text, sanitized tool definitions, its actions, and only
simulated observations. The world model sees the sanitized tool interface, package,
and persistent session. After the attempt, the chosen actions are replayed on a
fresh EnvScaler oracle from the real initial state and the final state is checked
against every official checklist function. Oracle state, observations, and
checklist code/results never enter either model's prompt. A world-model success
claim alone does not count. Recorded `chat_with_user` completion messages are not
environment tools; the live task agent instead returns `finish`.
Invalid tool calls or arguments return an error observation and leave the session
state unchanged; the task agent may recover on the next turn. Oracle replay likewise
keeps the state unchanged for a failed call before evaluating the final checklist.

```bash
PYTHONPATH=src python -m trace2env envscaler-long-horizon \
  work/exp-envscaler/env_160_rl/packages/envscaler-env_160_rl-v1 \
  --env-defs work/envscaler/env_defs --env-id env_160_rl \
  --benchmark-dir work/envscaler/benchmark/env_160_rl \
  --output work/my-long-horizon-env160 --attempts 3 --max-steps 50 \
  --allow-unvalidated --provider chat --model YOUR_MODEL --agent-transport tools
```

Use a fresh output directory and a compatible model endpoint. The task agent uses
`--model` unless `--task-model` is set; `--task-temperature` applies to chat models
that accept it. Its disk cache is disabled, so attempts make fresh calls, though
deterministic settings can still give duplicate trajectories. Repeat `--task-id`
instead of `--benchmark-dir` for a smaller smoke test; that does not establish a
held-out split. The benchmark selector reads only rollout task IDs and checks
their initial states against the scenarios; recorded actions and observations are
not prompts.

Outputs include one trace and persistent session per attempt and a `summary.json`.
`pass@k` is the fraction of tasks with at least one **oracle-verified,
full-checklist** success among their first k fresh attempts. This is empirical
best-of-k, not the combinatorial code-generation estimator. The report also gives
per-attempt success rate. No live long-horizon model result is claimed here.
