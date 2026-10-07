<div align="center">
  <img src="./assets/logo.png" alt="Trace2Env logo" width="250">
</div>

<h1 align="center">
  <b>From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation</b>
</h1>

<div align="center">
  <p>
    <a href="https://arxiv.org/abs/2610.06100"><img src="https://img.shields.io/badge/arXiv-2610.06100-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
    <a href="https://huggingface.co/papers/2610.06100"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Paper-page-ffd21e?style=for-the-badge" alt="Hugging Face paper page"></a>
    <a href="https://huggingface.co/spaces/Quanyu001/trace2env_demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Space-0f8a5f?style=for-the-badge" alt="Live demo"></a>
    <a href="https://quanyulong.net/trace2env/"><img src="https://img.shields.io/badge/Project-page-2457c5?style=for-the-badge&logo=githubpages&logoColor=white" alt="Project page"></a>
    <a href="https://github.com/ruyue0001/trace2env/stargazers"><img src="https://img.shields.io/github/stars/ruyue0001/trace2env?color=f1e05a&style=for-the-badge&logo=star&logoColor=white" alt="Stars"></a>
    <a href="https://github.com/ruyue0001/trace2env/forks"><img src="https://img.shields.io/github/forks/ruyue0001/trace2env?color=2ea44f&style=for-the-badge&logo=git&logoColor=white" alt="Forks"></a>
    <a href="https://github.com/ruyue0001/trace2env/issues"><img src="https://img.shields.io/github/issues/ruyue0001/trace2env?color=d73a49&style=for-the-badge&logo=github&logoColor=white" alt="Issues"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-Apache--2.0-2EA44F?style=for-the-badge" alt="License"></a>
  </p>
</div>


## 🧩 Overview

**Trace2Env** is a training-free framework for building **agentic language world models**. It is not tied to a particular environment domain: as long as past action–observation trajectories are available, Trace2Env can simulate an environment from the behavior they expose.

The [paper](https://arxiv.org/abs/2610.06100) evaluates Trace2Env on terminals, software repositories, Android and web apps, tool-based enterprise services, and text games. The same machinery applies to any system that interacts in text: internal enterprise systems and ticketing workflows, cloud and DevOps consoles, API and MCP tool backends, legacy systems that no longer run, and interactive fiction.

Trace2Env does not recover the original executable system. Instead, a model of the environment dynamics is reconstructed automatically from observed traces. 🕹️ Try our simulated Terminal environment in the **[live demo](https://huggingface.co/spaces/Quanyu001/trace2env_demo)**!

Specifically, Trace2Env works in two phases:

1. **Offline reconstruction.** Past trajectories are processed and organized into a **worldbook** containing environment schemas, transition rules, constraints, invariants, observation contracts, grounded evidence, and demonstrations. Check out what a worldbook looks like [here](https://huggingface.co/spaces/Quanyu001/trace2env_demo/tree/main/worldbook).

2. **Agentic simulation.** A **world model agent** actively calls tools to inspect the worldbook, together with the maintained episode state and episodic memory, and infers the next observation together with the state changes that should persist. A shared runtime harness verifies the proposal against the schemas, rules, and invariants and commits the accepted changes, so the consequences of a task agent's action carry into later turns.

<div align="center">
  <img src="./assets/overview.png" width="1000" alt="Trace2Env framework overview">
</div>


### ✨ Highlights

- **Build from tens of traces.** Worldbook construction needs only recorded interactions, not access to the original executable environment, its source code, dependencies, or infrastructure.

- **Agentic world modeling.** The world model agent actively consults the knowledge needed for each action instead of relying on a single flattened environment prompt.

- **Stateful, long-horizon simulation.** Verified state changes persist across turns, allowing the simulated environment to preserve the consequences of earlier actions throughout an interaction.

- **Evaluated as a world model.** Trace2Env supports both next-observation prediction ([Qwen-AgentWorld](https://github.com/QwenLM/Qwen-AgentWorld)) and successive interaction with the world model to evaluate its long-horizon consistency.

## 📰 News

- 🚀 **[2026-10]** Trace2Env is released. The paper, *From Traces to Agentic Worlds: Agentic Language World Models for
  Interactive Environment Simulation*, is on [arXiv](https://arxiv.org/abs/2610.06100) and
  [Hugging Face Papers](https://huggingface.co/papers/2610.06100); the [project page](https://quanyulong.net/trace2env/)
  summarizes the method and results, and the [live demo](https://huggingface.co/spaces/Quanyu001/trace2env_demo) lets you
  play against a reconstructed terminal.

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
validated package, and a persistent session. In the code, a worldbook is called a *package* and one episode's state and
audit a *session*. [docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) follows it file by file.
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
environment are under [`experiments/`](experiments/). The frozen Terminal worldbook is published with the
[live demo](https://huggingface.co/spaces/Quanyu001/trace2env_demo/tree/main/worldbook); the other frozen worldbooks and the collected traces are available on request.

## Repository layout

| Path | Contents                                                                                                                                                                 |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/trace2env/` | Library and CLI: trace adapters, reconstruction, compiler, runtime harness, world model agent, state tracking, benchmark adapters, long-horizon runner                   |
| `tests/` | 284 tests with scripted models; no network                                                                                                                               |
| `scripts/` | Stage-by-stage reconstruction driver, trace collection (Terminal-Bench 2.0 with Harbor, WebArena through Playwright MCP), ALFWorld and SciWorld runs, worldbook ablation |
| `experiments/` | Launchers, cross-fit tooling, audits and report scripts of the paper's experiments, per environment                                                                      |
| `docs/` | Architecture, runtime harness, offline validation, prompt contracts, trace collection, benchmark protocol, long-horizon evaluation                                       |
| `examples/` | The authored ledger demo and a synthetic benchmark file for smoke tests                                                                                                  |
| `baseline/Word2World` | Submodule used by the ALFWorld and SciWorld scripts (task lists, grammar, simulator adapters)                                                                            |

[DEVELOPMENT.md](DEVELOPMENT.md) has the commands, the code map and the design constraints to preserve.


## 📚 Citation

If you find Trace2Env useful in your research, please consider citing our paper:

```bibtex
@article{long2026agenticworlds,
  title={From Traces to Agentic Worlds: Agentic Language World Models for Interactive Environment Simulation},
  author={Long, Quanyu and Chen, Xiao and Chen, Jianda and Zhang, Haozhen and Hu, Qisheng and Bao, Jianzhu and Wang, Wenya},
  journal={arXiv preprint arXiv:2610.06100},
  year={2026}
}
```
