# Prompt-only ALFWorld world-model evaluation

> `work/` is this repository's gitignored scratch directory; the `work/...` paths below are the layout the scripts assume, not files tracked here. See [REPRODUCING.md](../REPRODUCING.md) for the data and the settings of the reported runs.

[`scripts/run_alfworld_prompting_wm.py`](../scripts/run_alfworld_prompting_wm.py)
uses the first 100 Word2World ALFWorld tasks selected by the prior real-agent
run. Both the task agent and prompt-only world model use GPT-5.6-Sol. The world
model starts with Word2World's task-specific initial-state description from the
game files and the task agent's initial observation. No later real observation
enters either model. This is the same privileged initialization as the released
Word2World world-model setup; it is not an observation-only simulator.

The runner includes the simulator's exact command forms in the world-model prompt and checks them
against `logic/alfred.twl2` before making model calls, so that the prompted world model does not
accept commands the simulator rejects (`put ... on ...`, `toggle ...`).

At turn `t`, the task agent chooses an action from its own conversation. The
world-model request contains its initial context, every earlier action and
simulated observation, and the current action. Its response is appended before
the next turn. The runner stores the complete histories, all turns, request
message counts, and request-prefix hashes under `attempts/`. It applies
Word2World's ` [SUCCESS]` marker rule and 50-turn cap. A success marker is a
model claim; use real-environment replay for grounded task success.

The run requires the prior real-agent artifacts and Word2World ALFWorld data
under `work/exp-alfworld-real/`. Use a Python environment with `openai` and
`tqdm`. Make `OPENROUTER_API_KEY` available as an environment variable without
putting it in a command argument.

```bash
python3 scripts/run_alfworld_prompting_wm.py \
  --output work/exp-alfworld-real/prompting-wm-gpt56sol-grammar-v2-100 \
  --limit 100 --max-steps 50 --max-workers 16 --dry-run

python3 scripts/run_alfworld_prompting_wm.py \
  --output work/exp-alfworld-real/prompting-wm-gpt56sol-grammar-v2-100 \
  --limit 100 --max-steps 50 --max-workers 16
```

The run resumes completed task files and retries only `api_error` files when
called again with the same selection and prompt. For turn `t` in a task JSON,
reconstruct the exact world-model request with
`wm_history[:turns[t-1]["wm_request_message_count"]]`.

Replay the saved action sequences after the model run. This command requires
the local x86_64 ALFWorld/TextWorld environment on Apple Silicon and makes no
model API calls:

```bash
arch -x86_64 work/exp-alfworld-real/.venv-x86/bin/python \
  scripts/replay_alfworld_prompting_wm.py \
  --input work/exp-alfworld-real/prompting-wm-gpt56sol-grammar-v2-100 \
  --output work/exp-alfworld-real/prompting-wm-gpt56sol-grammar-v2-100-replay
```

The 100-task report (`work/exp-alfworld-real/prompting-wm-gpt56sol-100/REPORT.md`)
gives the model-reported and real-replay success rates, along with a per-turn
case study. The prior [real-agent evaluation](ALFWORLD_REAL.md) documents the
same 100-task selection.
