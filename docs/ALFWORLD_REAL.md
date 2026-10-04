# Real ALFWorld task-agent evaluation

> `work/` is this repository's gitignored scratch directory; the `work/...` paths below are the layout the scripts assume, not files tracked here. See [REPRODUCING.md](../REPRODUCING.md) for the data and the settings of the reported runs.

Word2World reports **Real Task Success Rate** for task-agent interaction with
ALFWorld, distinct from **WM Task Success Rate** (interaction with a learned world
model) and **WM2Real Success Rate** (replay of world-model actions in ALFWorld).
For one fresh attempt per task, Real Task Success Rate is pass@1. For `k > 1`,
the additional empirical pass@k is the fraction of tasks solved in at least one
of the first k independent attempts. A simulator reward of 1 (or 100 in the
AgentGym convention) counts as success; a model or environment error makes the
reported metric incomplete until the failed attempt is rerun.

[`scripts/run_alfworld_real.py`](../scripts/run_alfworld_real.py) follows the
original test ordering and ReAct protocol: it selects the first 100 rows of
`eval/alfworld_test.json` (`--num_examples 100` in Word2World), uses the bundled
`mappings_test.json` IDs and reads the initial ReAct prompt from AgentGym's
`AlfWorldAdapter`. Each attempt creates and resets a real ALFWorld TextWorld game.
The 50-round cap matches Word2World's default. GPT-5-family chat requests omit
temperature, top-p, and legacy `max_tokens`, matching the baseline's Azure
branch and avoiding unsupported-parameter retry loops. The key comes from an
environment variable, not a command-line argument.

The Word2World ALFWorld archive is required: its grammar changes `put` to
`move` and adds `help`. Do not run the bundled full-data downloader if you want
to preserve an existing `~/.cache/alfworld`; it deletes that directory. Use a
task-specific extracted data root containing `logic/` and
`json_2.1.1/valid_train/`. The runner validates all selected game files before
model calls.

On Apple Silicon, the tested local simulator uses a Python 3.9 x86_64 virtual
environment under Rosetta; TextWorld and its PDDL library must both be x86_64.
The `work/exp-alfworld-real/.venv-x86` environment was smoke-tested with a real
reset and `look` action on item 2420.

```bash
arch -x86_64 work/exp-alfworld-real/.venv-x86/bin/python scripts/run_alfworld_real.py \
  --test-file work/exp-alfworld-real/data/alfworld_test.json \
  --data-root work/exp-alfworld-real/simulator/alfworld \
  --output work/exp-alfworld-real/real-gpt56sol-k1 \
  --model gpt-5.6-sol --limit 100 --attempts 1 --max-rounds 50 \
  --api-key-env OPENAI_API_KEY
```

If the model is served by an OpenAI-compatible provider other than the default,
add `--base-url "$OPENAI_BASE_URL"` with that variable set to its endpoint.

For OpenRouter, set `--model openai/gpt-5.6-sol`, `--base-url https://openrouter.ai/api/v1`,
and `--api-key-env OPENROUTER_API_KEY`. Keep `--max-workers 1`: TextWorld's shared
in-process state is not thread-safe. Separate processes can evaluate disjoint
test slices with `--start-index` and `--limit`.

Use `--dry-run` to write and inspect `selection.json` without invoking a model.
Set `--attempts 3` to report pass@1, pass@2, and pass@3 on the same 100 tasks.
`metrics.json` is updated as attempts finish; individual transcripts live under
`attempts/`. The runner resumes completed attempts only when the selection and
configuration manifest match. A partial run is marked `complete: false` and is
not a final pass@k result.
