# Development guide

Commands, a map of the code, and the design constraints that every change must preserve. The paper's worldbook is the
code's **package**; the per-episode state and audit live in a **session**; the shared runtime is the **harness**.

## Commands

Python 3.11+; `pydantic` v2 and `openai` are the only runtime dependencies (`pyproject.toml`, hatchling). Run everything
from the repository root.

```bash
git submodule update --init baseline/Word2World     # three ALFWorld tests read its files
python -m pip install -e '.[dev]'
python -m pytest                                    # 284 tests (unittest-style classes; `python -m unittest discover -s tests` also works)
python -m pytest tests/test_offline_safety.py -k conflicting_equal_priority
```

API-free CLI smoke after runtime, compiler, or validation changes (package destinations must not exist):

```bash
python -m trace2env init-demo --output .trace2env/demo-authored
python -m trace2env inspect .trace2env/demo-authored --action withdraw
python -m trace2env replay .trace2env/demo-authored --cases examples/replay/ledger_cases.jsonl \
  --output work/demo-replay.json --promote-to .trace2env/demo-validated
python -m trace2env simulate .trace2env/demo-validated --session .trace2env/demo-session \
  --action '{"type":"withdraw","arguments":{"amount":25}}'      # -> "Withdrew 25. Balance: 75.0"
```

API-free AgentWorldBench smoke after adapter, tracker, or runner changes:

```bash
python -m trace2env awb-export examples/agentworldbench/synthetic_terminal.jsonl --output work/awb-smoke/train
python -m trace2env awb-run examples/agentworldbench/synthetic_terminal.jsonl --mode agentic \
  --package <any package> --no-model --output work/awb-smoke/predictions.jsonl
python -m trace2env awb-score --predictions work/awb-smoke/predictions.jsonl
```

Commands that call a model (`reconstruct`, `awb-run` without `--no-model`, `awb-judge`, `simulate --agent`, the
`envscaler-long-horizon` loop) read the key from the environment variable named by `--api-key-env` (default
`OPENAI_API_KEY`) and accept `--base-url` for any OpenAI-compatible server. Structured roles use the Responses API or,
with `--provider chat`, Chat Completions with a JSON schema (`--chat-schema-mode json_object` quotes the schema in the
system text for servers that reject schemas; `--chat-reasoning-effort`, `--max-output-tokens`, `--chat-provider-json`
tune the call). Pass `--cache-dir` so interrupted builds and evaluations resume: calls are content-addressed. Live
models are never needed to exercise the pipeline; tests use `ScriptedLLM`, `ScriptedChatLLM` and `ScriptedToolLLM`.

Stage-by-stage reconstruction with inspectable versions:

```bash
python scripts/pilot_stages.py WORK ingest --episodes DIR --split-manifest FILE --environment-id ID --name NAME \
  --description @FILE --domain terminal [model options]
python scripts/pilot_stages.py WORK extract --workers 6 [model options]
python scripts/pilot_stages.py WORK schema|rules|renderers|notes|compile [--note "why"] [model options]
```

Each stage writes `WORK/stages/<stage>/v<N>/` with `inputs.json`, `calls.jsonl` (every model call), the result, and a
`code/` snapshot with a diff against the previous version; `state.json` points each stage at its current version.

## Architecture

**Contracts (`models.py`).** Every artifact, model response, and runtime object is a Pydantic model with
`extra="forbid"`. State paths start with `world`, `session`, `surface`, or `epistemic`; dynamic values use
`{"$action_arg": ...}` or `{action.arguments.name}` tokens (`engine.resolve_*`, checked by `validation.py`). Mutation
ops: `set`/`create`, `delete`, `increment`/`decrement`, `append`, `schedule`, `merge`, `remove`.
Maps keyed by file paths or URLs (`world.files`) are edited with `merge`/`remove`, or addressed by name: a path
segment that starts with `/` or `~/` or contains `://` absorbs the segments after it, so `world.files./app/report.txt`
is the entry `/app/report.txt` (existing keys match longest-first; an object value writes the whole entry, a scalar
names the last segment as an attribute), and bracket notation such as `world.files["/app/report.txt"]` is
normalized to that form (`engine._step_key`, `models.normalize_state_path`).

**Offline construction (`reconstruction.py`).** `adapters.load_raw_traces` → episodes with content-derived ids →
`segment_transitions` (action/result pairing by call id, else sequential; ambiguous cases flagged, never guessed) →
`extract_transition` once per slice (bounded views; refusals and invalid answers become zero-confidence records) →
`induce_schema` (+ alias repair) → `induce_rules` with matched success/failure contrasts and a `falsify_rules` review →
`induce_renderers` → `induce_notes` → deterministic demonstration selection → `reconcile_artifacts` /
`reconcile_provenance` (repair selectors, quarantine schema-violating items, drop unanchored citations, downgrade
unsupported claims) → `EnvironmentCompiler.compile`. `BoundedInduction` enforces a prompt byte budget and record cap
with map/reduce roles. Prompt text lives only in `prompts.py`; `docs/PROMPT_SKETCHES.md` explains each stage's contract.

**Package (`compiler.py`, `package.py`).** Validates artifacts, filters executable rules through `eligibility.py`
(complete reviewed set stays in `candidates/`), builds the FTS knowledge catalog in `index.sqlite`, hashes every file
into `manifest.json`, publishes atomically. `validation_status` is `candidate`, `validated` (passing replay), or
`authored`. Hashes are re-verified on load; edit through `patch` into a new directory, never by hand.

**Runtime (`runtime.py`, `engine.py`, `workspace.py`, `agent.py`, `memory.py`).** `RuntimeHarness.step`: load state,
apply due events, canonicalize and type-check the action, select over all eligible rules (a confident templated rule
takes the recorded fast path), otherwise brief the world-model agent and run the tool loop (`inspect_action`,
`search_knowledge`, `read_state`, `recall`, `read_turn`, `apply_rule`, `dry_run`, `submit_transition`), validate and
apply the proposed mutations, verify (invariants, rule support, renderer fields, citations), render, commit state and
audit atomically, record episodic memory. Trust policy `rules_only` (default) or `schema_checked`; effect rejection
`fail` or `keep_observation`. `--official-input` places the benchmark's own input verbatim; `--knowledge-gate` labels
every package item shown with an applicability decision (`knowledge_gate.py`; `--knowledge-names paths|pages|screens`).
`metadata["environment_prompt"]` is bounded to 24,000 characters unless `environment_prompt_chars=None`.

**State tracking (`state_tracking.py`).** Turns an observed prefix into state: a rule whose rendered observation matches
the real one is applied deterministically, the rest is explained by bounded `track_state` model calls whose mutations
are schema-checked. A rule never overrides an observed output.

**Benchmarks (`agentworld.py`, `agentworld_runner.py`, `envpack.py`, `envscaler*.py`, `long_horizon.py`).**
`agentworld.py` mirrors the official Qwen-AgentWorld evaluation code (judge prompts verbatim under
`agentworld_prompts/`, never edited). `AgentWorldRunner` modes: `prompting`, `prompting_rag` (`--rag-style`),
`envpack_prompting`, `agentic`. `envscaler.py` converts EnvScaler rollouts to episodes and rows, `envscaler_oracle.py`
executes an environment's own source for evaluation only, `long_horizon.py` runs the closed task-agent/world-model loop.

**Replay and repair (`replay.py`, `patching.py`).** Replay runs cases in throwaway sessions with the supplied initial
state and compares observation, outcome, and state projections; promotion requires passing validation-split cases from
disjoint trajectory families, full rule coverage, and no regressions. Patches are typed edits against a base digest.

**Model boundary and test doubles (`llm.py`).** `StructuredLLM.complete(system, user, response_model, role)` is the
protocol; role names are the contract between stages and tests:

| Stage | Role | Response model |
|---|---|---|
| Trace annotation | `normalize_trace` | `TraceAnnotations` |
| Evidence extraction | `extract_transition` | `LocalTransitionEvidence` |
| Schema / rules / review / renderers / notes | `induce_schema`, `induce_rules`, `falsify_rules`, `induce_renderers`, `induce_notes` (+ `_chunk` / `_merge`) | the corresponding `*InductionResult` |
| World-model agent | `runtime_agent_turn` | `AgentTurn` (structured) or `AgentReply` (native tools) |
| Verifier / renderer | `runtime_verify`, `runtime_render` | `VerificationResult`, `RenderedObservation` |
| State tracking | `track_state` | `StateTrackingResult` |
| Applicability judge | `knowledge_applicability` | `ApplicabilityJudgement` |
| Benchmark baseline / judge (`ChatLLM`) | `agentworld_infer`, `agentworld_judge` | free text |

## Where to change what

| Task | Start with | Tests |
|---|---|---|
| Trace formats, ids, action/result matching | `adapters.py`, `models.py` | `AdapterSafetyTests`, `AdapterTests` |
| Evidence, induction, contrasts, demonstrations, prompts | `reconstruction.py`, `contrasts.py`, `induction.py`, `prompts.py` | `ReconstructionTests`, `InductionSafetyTests` |
| Artifact validity, eligibility, package layout | `validation.py`, `eligibility.py`, `compiler.py`, `package.py` | `PackageSafetyTests` |
| Step orchestration, fast path, trust policy, verification, rendering | `runtime.py`, `engine.py`, `session.py` | `RuntimeTests`, `HarnessRoutingTests` |
| Agent tools, tool loop, episodic memory | `workspace.py`, `agent.py`, `memory.py` | `test_agent_harness.py` |
| Applicability gate | `knowledge_gate.py` | `test_online_features.py` |
| Replay, splits, promotion, repair, CLI | `replay.py`, `patching.py`, `cli.py` | `PackageSafetyTests` |
| State reconstruction from a prefix | `state_tracking.py` | `StateTrackerTests` |
| AgentWorldBench data, judge port, runner | `agentworld.py`, `agentworld_runner.py` | `test_agentworld.py`, `test_prompting_rag.py` |
| Trajectory converters (Harbor, MCP, WebArena, EnvScaler) | `atif.py`, `mcp_traces.py`, `webarena_traces.py`, `envscaler.py` | `test_atif.py`, `test_mcp_traces.py`, `test_webarena_traces.py`, `test_envscaler.py` |
| EnvScaler oracle, probes, exact scoring; long-horizon loop | `envscaler_oracle.py`, `envscaler_scoring.py`, `long_horizon.py` | `test_envscaler_oracle.py`, `test_long_horizon.py` |
| Model transport, tracing, retries | `llm.py` | `OpenAIAdapterTests`, `test_pilot_tooling.py` |

## Design constraints to preserve

- **Two agents.** The task agent supplies actions; the world-model agent plays the environment. Internal inspections
  never choose the next environment action, and the harness rejects a plan whose action differs from the request.
- **Traces are evidence, not instructions.** Normalizers annotate original spans; nothing rewrites recorded
  observations. Source identity is content-derived; provenance and ambiguity flags are kept.
- **Local observation is not a rule.** Only `supported` rules above the confidence threshold and in scope are
  executable; withheld candidates stay in `candidates/`.
- **Package is not session.** Unknown episode state is never filled from schema defaults for reconstructed packages or
  replay; compile and promote only into fresh directories; SQLite is authoritative, JSON files are mirrors.
- **Fail closed.** Same-priority conflicts reject the step; effects must match cited rules exactly under `rules_only`;
  `schema_checked` accepts only declared, type-valid, invariant-preserving effects and flags them; rejected effects are
  never applied; the fast path is a recorded policy, never a hidden bypass.
- **Data hygiene.** Construction, validation, and test trajectory families stay disjoint; generated expectations cannot
  certify their own candidate; the benchmark's own system prompt enters the harness only through the documented input
  paths; gated datasets never become training data or an answer-retrieval corpus.
- **Reported settings stay reproducible.** The defaults of `awb-run` are the settings of the reported AgentWorldBench
  runs (bounded environment prompt, `--rag-style terminal`, `--effect-rejection keep_observation`); new behaviour is
  added behind flags, and `docs/` is updated with any change to defaults, schemas, or gates.
