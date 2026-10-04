# The runtime harness

The harness turns a worldbook (the code's *package*) into a running environment. The **task agent** supplies actions;
the **world-model agent** plays the environment and may only inspect knowledge and propose a transition; the harness
verifies the proposal, commits it to the **session**, and returns the observation. Three memory tiers back the agent:
semantic (the package), working (the session state), episodic (the turns of this episode, in `session.sqlite`).

## One action through `RuntimeHarness.step`

1. **Load** the committed state, apply due scheduled events to a working copy, canonicalize the action through the
   package's aliases and type-check its arguments.
2. **Select a rule** over all eligible package rules (a retrieval limit can never hide an applicable branch). Rules whose
   effects need absent arguments are inapplicable; two same-priority rules with different plans reject the step.
3. **Route.** A confident, templated rule takes the deterministic **fast path** (`--fast-path confident|always|never`,
   recorded in the audit). Otherwise the agent gets a **brief** (the action, a state summary, the recent turns verbatim
   within a budget, the retrieved action contract, rules, demonstrations and notes, and the tool budget) and runs a tool
   loop: `list_actions`, `inspect_action`, `search_knowledge` (full-text search over rules, notes, demonstrations,
   evidence, actions), `read_state`, `state_schema`, `recall`, `recent_turns`, `read_turn`, `apply_rule` and `dry_run`
   (dry runs), then `submit_transition` with typed effects, the outcome, the observation, the rules applied, and
   citations. The budget is `--max-tool-calls` (8); exhaustion forces the submission.
   `--prediction-mode single_shot` replaces the loop with one call over the same brief and a fixed retrieval procedure.
4. **Validate** the proposed mutations against the state schema (declared paths, mutability, types) and apply them to the
   working copy.
5. **Verify.** Invariants (only violations introduced by this step reject), rule support (cited rules eligible and
   applicable, outcome and effects exactly as the rules say), renderer required fields, and citations (an item that was
   never retrieved this step is flagged). An optional LLM verifier can only add rejections.
   `--trust-policy rules_only` (default) rejects any effect no cited rule supports; `schema_checked` accepts
   agent-proposed effects that passed step 4 and the invariants, and flags them in the audit.
   `--effect-rejection fail` fails the step when the effects are rejected; `keep_observation` keeps the agent's
   observation and drops the effects (state unchanged, warning recorded).
6. **Render** from the cited rule's template, else the renderer contract, else the agent's observation.
7. **Commit** state and audit atomically (optimistic revision check) and record the turn in episodic memory. Any
   exception records a failed audit and leaves the state untouched. An environment-level failure outcome (for example
   insufficient funds) is a committed step; a harness rejection is not.

## Inputs the caller controls

- `StepRequest.metadata["environment_prompt"]`: an environment description (for example a benchmark's world-model
  system prompt), appended to the agent's and renderer's system text, bounded to `ENVIRONMENT_PROMPT_CHARS` (24,000)
  unless the harness is built with `environment_prompt_chars=None`.
- `metadata["official_input"]` (`--official-input` in `awb-run`): a benchmark's own model input, placed once and verbatim
  in the agent's input instead of the recent-memory view.
- `--knowledge-gate`: every package item shown to the agent, in the brief and in tool results, carries an applicability
  decision (`supporting`, `uncertain`, `format_only`; notes `consistent`, `untested`, `contradicted`), foreign concrete
  values are masked, contradicted notes are dropped, and the brief says when the gate abstains.
  `--knowledge-names paths|pages|screens` sets the identity signal for evidence about the current episode
  (file paths; the same web page or navigation target; the same Android screen). `--knowledge-judge` adds a model
  judgement whose `supporting` label is honoured only when its anchor is found in the episode's own transcript.
- `--features default|all|none|<list>`: `history` (budgeted verbatim recent turns and `read_turn` paging), `wait`
  (effect-free, template-free rules never explain a turn; empty observations are valid predictions; a pending-work
  hint for waits and key presses), `compact` (slim briefs and a compact tracker schema), `unknown` (an absent state key
  means "not observed yet"; `known_files`; exact observed values kept), `evidence` (retrieved evidence turns in the
  brief), `scaffold` (opt-in command skeleton for terminal prompts).
- `--trace-corpus DIR`: raw construction traces behind `search_traces` / `read_trace_turn` tools.
- `--history-window`, `--history-full`, `--agent-transport structured|tools`, `--agent-max-output-tokens`.

## Reconstructing state from an observed prefix

`StateTracker.advance(state, turns)` turns observed action/observation pairs into state before the first prediction: a
rule whose rendered observation matches the real one is applied deterministically; the other turns are queued and
explained, in windows of `--state-window` turns with `--observation-chars` per observation, by one `track_state` model
call whose mutations are individually schema-checked. A rule that contradicts the observation is recorded and not
applied; invariant violations are diagnostics only; without a model the queued turns are skipped and the state is left
unchanged. The harness never lets a rule override an observed output.

## Sessions

`session.sqlite` is authoritative (current state and audit table); `state.json` and `audit.jsonl` are regenerated
mirrors. Sessions live outside packages and are never written into them. Startup gates: a `candidate` package needs
`--allow-unvalidated`; a new session on a reconstructed package needs an explicit `--initial-state` (schema defaults are
used only for authored packages); reopening a session ignores `--initial-state`.
