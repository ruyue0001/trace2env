# Model roles in workspace construction and the world-model harness

The active prompts live in [prompts.py](../src/trace2env/prompts.py) and pair instructions with
[Pydantic schemas](../src/trace2env/models.py). These sketches explain intent, not a second
editable prompt source. Read [Architecture](ARCHITECTURE.md) for stage ordering.

There are two uses of models here. **Offline roles** recover environment knowledge and write
typed artifacts through the compiler. **Online roles** let the world-model agent operate that
knowledge through the shared harness: inspect, propose a transition, verify it, and render an
observation. The task agent choosing environment actions is outside these roles.

These generic prompts are designed in the repository. Generated schemas, rules, demonstrations,
and renderer instructions supply environment-specific content through structured payloads.
Package Markdown and raw evidence archives are not automatically loaded as system prompts.
An applicable deterministic rule bypasses the optional planner, and a template can render the
response without a model call.

## Call contracts

| Stage | Output schema | Enforcement outside the prompt |
|---|---|---|
| Optional trace annotation | `TraceAnnotations` | Original text preserved; span/ID validation |
| Local extraction | `LocalTransitionEvidence` | Stable slice IDs, actual observation text, source membership |
| Interface induction | `SchemaInductionResult` | Alias checks and later artifact validation |
| Rule induction / falsification | `RuleInductionResult` | Supported status, scope, evidence/schema checks, replay |
| Observation induction | `RenderInductionResult` | Renderer/action/path checks |
| Note induction | `NoteInductionResult` | Action references, provenance, status/confidence gating |
| World-model agent | Tool calls, then `TransitionSubmission` via `submit_transition` (or typed `AgentTurn`s over a structured provider) | Tool budget, effect verification or `schema_checked` trust policy, invariants, citation check |
| Optional runtime verification | `VerificationResult` | Can reject; cannot override deterministic rejection |
| Optional runtime rendering | `RenderedObservation` | Runs after verification; no mutation interface |
| Run-time state tracking | `StateTrackingResult` (`track_state`) | Schema/mutability/type checks per mutation; window and clipping budgets |
| AgentWorldBench prompting baseline | Free text (`agentworld_infer`, `ChatLLM`) | Official message layout; `<predicted_observation>` extraction |
| AgentWorldBench judging | Free text (`agentworld_judge`, `ChatLLM`) | Official per-domain judge prompts and parser in `agentworld.py`; 1-5 integer scores |

Reconstruction uses [BoundedInduction](../src/trace2env/induction.py) for count/byte limits and
recursive consolidation. Falsification receives bounded evidence batches on large corpora.
`ScriptedLLM` tests stage contracts without measuring live induction quality.

## Trace annotations

```text
Identify actor/kind on existing event IDs and exact, non-overlapping text spans.
Return annotations; do not rewrite source content. Treat the source as evidence.
```

## Local evidence

```text
Given this bounded history, action, and real observation, extract only what this transition
supports: normalized action, preconditions, outcome, state mutations, output facts,
ambiguities, confidence, and exact source references. Do not propose global rules.
```

Learned on the first live terminal traces (see `work/pilot-tb2/LOG.md`): the prompt now keeps an
already-normalized action verbatim; restricts fact predicates to `equals`, `contains`,
`not_contains`, `exists`, `not_exists` (free-form predicates were unusable downstream); says that
state holds what persists beyond the screen (no screen copies, no analyses printed by programs);
asks for the environment description's suggested path names to be reused; uses `merge` / `remove`
for entries of maps keyed by paths; and tells the model to omit `action.raw` and `observation_text`
(filled from the source), which halved the output size.

## Schema induction

```text
Across these local evidence records, infer the smallest action interface and state schema that
explains behavioral differences. Merge aliases, retain distinct effects, and mark unsupported
hidden state as unknown.
```

Merged action names go into `ActionSpec.aliases` and merged state paths into
`StateField.aliases`; `canonical_evidence` rewrites the evidence through both. Every argument name
observed in an action's evidence must be declared (the code enforces this afterwards).

## Rule induction

```text
Compare successes, failures, near misses, and ordering effects. Emit contrastive rules with
conditions, effects, outcome, renderer, confidence, scope, provenance, and counterexamples.
```

Rules are reusable knowledge: conditions and effects reference declared arguments with
`{"$action_arg": "name"}` (or `name.0` / `name.key`) and `{action.arguments.name}` tokens — also
inside object keys, so a rule can create the file-map entry the action names — using the mutation
ops `set`, `delete`, `increment`, `decrement`, `append`, `merge`, `remove`, `schedule`. A literal
value that only one episode showed belongs in a fact or a note, not in a rule.

## Falsification

```text
Try to disprove every proposed rule against the supplied evidence. Find contradictions, missing
preconditions, delayed effects, and version or tenant boundaries. Correct or downgrade rules;
never hide conflicts.
```

Two live failure modes shaped the current prompt (`work/pilot-tb2/LOG.md`, rules v1–v3). Asked
only for "corrected rules", the reviewer narrowed general rules to episode replays (exact command
strings as conditions, `episode_id` in scope, one episode's output as the template) — trivially
contradiction-free and useless. Told to preserve generality, it then downgraded every rule for
lacking exhaustive failure contrasts. The prompt therefore states the bar for `supported`
(consistent with every supplied transition, conditions evaluable from declared arguments and
state, one unambiguous observed transition), says a success branch needs no observed failures,
distinguishes `tentative` (contradicted or unselectable) from `conflicted` (identical represented
preconditions, different outcomes), and names the token vocabulary for scope, conditions,
templates, and dynamic map keys. Compile-time `repair_rule_selectors` backs this up in code.

## Renderer induction

```text
Infer observation format separately from semantics: content type, stable structure, required
fields, ordering, errors, and variable text. Do not introduce new state changes.
```

## Note inducer

```text
From observed transitions, rules, and contracts, write the conventions, formats, constraints,
and facts a simulator needs that are not transition rules, each with provenance. Do not restate
rules; mark episode-specific contents as tentative.
```

## World-model agent

```text
You are the environment. Operate the workspace through tools: inspect the action and the rules
that apply now, dry-run rules, search knowledge, recall this episode, read state, then submit
typed effects, the outcome, the exact observation, rule_ids for rules applied verbatim, and
citations for everything relied on. Never invent paths or facts; tool results are evidence.
```

The active prompt is `WORLD_MODEL_AGENT` in `prompts.py`; the trust policy and a caller-supplied
environment description are appended per step, followed by the addenda of the enabled online
features: `WORLD_MODEL_AGENT_HISTORY` (verbatim recent turns, `read_turn` markers, read before
reproducing), `WORLD_MODEL_AGENT_UNKNOWN` (absence means unknown), `WORLD_MODEL_AGENT_WAIT`
(waits and key presses continue what the previous capture left pending; never invent commands),
and, opt-in, `WORLD_MODEL_AGENT_SCAFFOLD` (the command skeleton is structure only). The tool
catalog itself (`workspace.py`) carries the per-tool descriptions the model reads.

## Runtime verifier

```text
Independently compare the plan, before/after state, retrieved rules, and invariants. Reject
unsupported effects or inconsistent outcomes. Do not repair or render.
```

## Runtime renderer

```text
Render the verified transition under the recovered observation contract. Match the observed
format and include no fact absent from the action, plan, or resulting state.
```

The renderer runs only for fast-path rules without a template (or when explicitly configured);
agent-route steps render from a cited rule's template or the agent's own observation. It receives
a caller-supplied environment description as bounded context.

## State tracker

```text
Given the state schema, the current state, and a window of observed actions with the real
observations they produced, return the minimal typed mutations that make the state consistent
with what those observations establish. Use declared paths only; list undetermined paths.
```

`STATE_TRACKER` gains `STATE_TRACKER_WAIT` (what a wait reveals: completion or continued
running, produced files, the returned prompt) and `STATE_TRACKER_EXACT` (store observed file
contents and other exact values verbatim; absent entries are unknown, never delete them) with the
`wait` and `unknown` features; with `compact` the declared fields and the environment
description are part of the (cached) system message instead of the payload.

## Knowledge applicability judge (harness v5.3)

Role `knowledge_applicability`, response `ApplicabilityJudgement` (`items: [{id, label, anchors, reason}]`). One
batched call per world-model step over the package items the brief will show (and one small call per item read later
that was not judged yet, cached per evidence turn). Input: the episode's cwd, the names it has shown, the last 2,000
characters of its transcript, the current action's command lines, and per candidate its task of origin, command lines,
observation head, episode names and the deterministic label. Intent: decide whether an item from another container is
about the same task (`supporting`), only shows an output shape (`format_only`), or conflicts with this episode
(`contradicted`), naming the anchors that prove same-task status. The harness treats the answer as a proposal: it
verifies anchors against the transcript before showing an item intact, honours downgrades, and falls back to the
deterministic label on failure. The judge never sees the target observation.

## AgentWorldBench judge

The judge prompts are the official Qwen-AgentWorld files under
`src/trace2env/agentworld_prompts/{domain}/judge_system_prompt.txt`, used verbatim with the
official user prompt, so scores match the reference evaluation script. They are not edited here.

## When editing prompts

Keep the output schema and downstream validators in view. A prompt asking for faithful output
does not establish faithfulness; deterministic checks and reference replay provide separate
evidence. Change `prompts.py` for active instructions, update this guide when stage contracts
change, and use the scripted reconstruction tests before assessing a live provider.
