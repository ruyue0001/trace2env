# Trace2Env

**Recover an environment from its interaction traces and operate it as an agentic language world model, without
rebuilding the original executable system.**

Many environments that agents need to practice in are unavailable: the system is gone, private, or impractical to
reproduce, but recorded interactions remain. Trace2Env turns those traces into a running simulator in two steps.

1. **Offline reconstruction** organizes the traces into a reusable **environment worldbook**: action and state schemas,
   transition rules, invariants, observation contracts, grounded evidence turns, demonstrations and notes, every item
   tied to the traces that support it. Nothing is learned into model weights and the environment's source is never read.
2. **Agentic simulation** lets a **world-model agent** serve as the environment for a task agent. For each action it
   consults the worldbook, the persistent episode state and episodic memory, proposes the next observation together
   with the state changes that should persist, and a shared **runtime harness** verifies and commits the transition so
   that its consequences carry into later turns. One worldbook serves many episodes and any OpenAI-compatible backbone.

In the code the worldbook is called a *package* and one episode's state and audit a *session*.

## Install and try it

Python 3.11 or newer. The demo below needs no model and no API key.

```bash
git clone --recurse-submodules https://github.com/ruyue0001/trace2env.git
cd trace2env
python -m pip install -e '.[dev]'
python -m pytest                                   # 284 tests, no network

python -m trace2env init-demo --output .trace2env/demo-authored
python -m trace2env inspect .trace2env/demo-authored --action withdraw
python -m trace2env replay .trace2env/demo-authored \
  --cases examples/replay/ledger_cases.jsonl --output work/demo-replay.json \
  --promote-to .trace2env/demo-validated
python -m trace2env simulate .trace2env/demo-validated --session .trace2env/demo-session \
  --action '{"type":"withdraw","arguments":{"amount":25}}'     # -> "Withdrew 25. Balance: 75.0"
```

The demo worldbook is hand-authored to show the machinery: replay validation against reference cases, promotion to a
validated package, and a persistent session. [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) follows it file by file.
Package destinations (`--output`, `--promote-to`) must not already exist.

## Build a worldbook from your traces

```bash
python -m trace2env ingest TRACES --output work/parsed            # deterministic parse and segmentation, no model

export OPENAI_API_KEY=...                                         # or --base-url URL --api-key-env VAR for any OpenAI-compatible server
python -m trace2env reconstruct TRACES --environment-id my.env --name "My environment" \
  --description @description.md --domain terminal \
  --work-dir work/build --cache-dir work/cache --output work/my-env-candidate-v1
python -m trace2env inspect work/my-env-candidate-v1 --action some_action
python -m trace2env simulate work/my-env-candidate-v1 --session work/session-1 --allow-unvalidated \
  --initial-state @initial_state.json --agent --action '{"type":"some_action","arguments":{...}}'
```

[docs/TRACE_COLLECTION.md](docs/TRACE_COLLECTION.md) covers the trace formats and the converters for Terminal-Bench,
WebArena, AgentWorldBench, EnvScaler, ALFWorld and SciWorld traces; [docs/OFFLINE_VALIDATION.md](docs/OFFLINE_VALIDATION.md)
how to replay, promote and repair a worldbook; [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) what each generated file is
and who consumes it. For research runs, `scripts/pilot_stages.py` runs the reconstruction stage by stage and keeps an
inspectable version of every stage (inputs, every model call, outputs).

## Use it as a world model

[docs/RUNTIME_HARNESS.md](docs/RUNTIME_HARNESS.md) explains how one action is simulated: rule selection, the agent's
brief and tools, verification, trust policy, rendering, commit. Two evaluation protocols are built in:

- **Single-turn next-state prediction** on [AgentWorldBench](https://huggingface.co/datasets/Qwen/AgentWorldBench):
  `awb-run` predicts held-out turns with the harness or with the official prompting layout and its augmented variants,
  `awb-judge` and `awb-score` port the official judge and aggregation ([docs/AGENTWORLDBENCH.md](docs/AGENTWORLDBENCH.md)).
- **Multi-turn interaction**, where a task agent acts inside the simulated environment and its action sequence is
  replayed in the real one: `envscaler-long-horizon` for EnvScaler environments ([docs/LONG_HORIZON.md](docs/LONG_HORIZON.md)),
  and the ALFWorld and SciWorld scripts ([docs/ALFWORLD_TRACE2ENV.md](docs/ALFWORLD_TRACE2ENV.md)).

## Reproducing the paper

[REPRODUCING.md](REPRODUCING.md) maps every experiment of the paper to its data, scripts and flags. The scripts of each
environment are under [`experiments/`](experiments/); the frozen worldbooks and collected traces are distributed as a
release archive.

## Repository layout

| Path | Contents |
|---|---|
| `src/trace2env/` | Library and CLI: trace adapters, reconstruction, compiler, runtime harness, world-model agent, state tracking, benchmark adapters, long-horizon runner |
| `tests/` | 284 tests with scripted models; no network |
| `scripts/` | Stage-by-stage reconstruction driver, trace collection (Terminal-Bench 2.0 with Harbor, WebArena through Playwright MCP), ALFWorld and SciWorld runs, worldbook ablation |
| `experiments/` | Launchers, cross-fit tooling, audits and report scripts of the paper's experiments, per environment |
| `docs/` | Architecture, runtime harness, offline validation, prompt contracts, trace collection, benchmark protocol, long-horizon evaluation |
| `examples/` | The authored ledger demo and a synthetic benchmark file for smoke tests |
| `baseline/Word2World` | Submodule used by the ALFWorld and SciWorld scripts (task lists, grammar, simulator adapters) |

[DEVELOPMENT.md](DEVELOPMENT.md) has the commands, the code map and the design constraints to preserve.

## Data and licenses

No benchmark data is redistributed. AgentWorldBench, Terminal-Bench 2.0, WebArena, EnvScaler, ALFWorld and SciWorld are
obtained from their own distributions as described in the guides. The judge prompts under
`src/trace2env/agentworld_prompts/` are copied verbatim from Qwen-AgentWorld (Apache-2.0, see the NOTICE there).
Toolathlon's verified trajectories are gated and must not become training data or a retrieval corpus that answers a
benchmark. The code is released under Apache-2.0 ([LICENSE](LICENSE)); the Word2World submodule keeps its own terms.
