# Offline validation and package promotion

Trace2Env's core design is an environment agent workspace operated by an agentic world-model
harness. This guide covers how its environment knowledge is constructed, checked, and released
as a package. Start with [README](../README.md) for that motivation,
[Architecture](ARCHITECTURE.md) for the generated-file map, and
[Runtime harness](RUNTIME_HARNESS.md) for how those files become simulated behavior.

The commands below implement the offline side of that design. They produce package data and
validation records; the shared inspection/execution harness remains repository code, and each
active episode gets its own session state. [Walkthrough](WALKTHROUGH.md) connects both sides.

## Commands at a glance

| Command | Input | Output / next step |
|---|---|---|
| `ingest` | Raw trace files | Deterministic parsing preview; inspect episodes/slices |
| `reconstruct` | Training traces, scope, optional frozen split manifest | Statically checked candidate; supply replay cases next |
| `inspect` | Any package version | Manifest/action/rule views; no state transition |
| `replay` | Package, reference cases, optional baseline | Case report and optional new promoted version |
| `patch` | Base package, typed patch, reference cases | New artifacts, baseline comparison, optional promotion |
| `simulate` | Runnable package, session, action | Observation, plan, verification, committed session state |
| `awb-export` | AgentWorldBench records | Episode files plus a split manifest ready for `reconstruct` ([AgentWorldBench](AGENTWORLDBENCH.md)) |

Only reconstruction needs model calls by default. Simulation can enable optional model roles.
The `trace2env` commands below also work as `python -m trace2env` from an installed checkout.
`reconstruct --description` accepts `@path` to read a domain description from a file.

## What the safety checks establish

The implementation protects evidence identity, action/result attribution, executable rule eligibility, state mutation, and package publication. Replay evaluates predictions against supplied observations in temporary local sessions. It never invokes traced commands, contacts the original environment, or creates its own expected answers with an LLM.

These checks establish software behavior and agreement with the provided cases. They do not prove causal recovery, identify all hidden state, or establish benchmark improvement. The correctness and independence of the supplied reference data remain essential.

## Workflow

### 1. Construct an inspectable candidate

```powershell
trace2env reconstruct examples/raw/ledger_trace.txt `
  --environment-id internal.ledger --name "Internal Ledger" `
  --work-dir work/ledger-build --cache-dir work/ledger-cache `
  --output environments/ledger-candidate-v1
```

Construction uses the configured LLM as before. New reconstructed packages have `validation_status: candidate`. They can be inspected immediately. Normal simulation requires promotion; diagnostic simulation requires the explicit `--allow-unvalidated` option. That option does not bypass reference, rule-status, scope, or mutation checks.

Source snapshots are identified by SHA-256 of their original bytes. The raw UTF-8 text, original identifiers, normalized episodes, transition links, full candidate rules, and construction artifacts are retained. Normalizer responses contain role/type annotations and exact text spans instead of replacement observations. Overlapping calls remain marked as causally ambiguous even when their results are correctly matched.

### 2. Supply replay cases from authoritative observations

A replay file is JSONL, with one `ReplayCase` per trajectory. Example structure:

```json
{
  "id": "withdraw-heldout-1",
  "trajectory_group": "independent-account-run-1",
  "source_ids": ["src_<sha256-of-the-reference-trace>"],
  "split": "validation",
  "initial_state": {"world": {"balance": 100}},
  "steps": [{
    "action": {"type": "withdraw", "arguments": {"amount": 25}},
    "expected_observation": "Withdrew 25. Balance: 75",
    "observation_match": "exact",
    "expected_outcome": "success",
    "expected_state": {"world.balance": 75}
  }]
}
```

Replace the source placeholder with the content digest of the actual reference trace. Use independent trajectory-family identifiers, including copies and overlapping prefixes in the same family. Expected observations and state projections must come from the reference environment or a separately authored test oracle, not from the candidate package or its extracted hypotheses.

- Initial state is supplied explicitly; replay never fills gaps with schema defaults.
- `initial_state_complete` defaults to false. Missing fields remain unobserved. A potentially applicable rule that needs them makes the replay case fail with an uncertainty diagnostic.
- `known_absent_paths` can explicitly establish absence, such as `["world.cache"]`. Use `initial_state_complete: true` only for a complete authoritative state snapshot.
- `expected_state` is a flat projection of known state paths. Unobserved fields need not be invented.
- `observation_match: json` compares parsed JSON; `exact` compares text without normalization.
- A correctly predicted environment failure is a passing replay step.
- Steps run consecutively against predicted state. An execution exception stops that trajectory, and unattempted steps remain in the denominator. Other trajectories still run.

### 3. Evaluate and promote to a new version

```powershell
trace2env replay environments/ledger-candidate-v1 `
  --cases eval/ledger-replay.jsonl --output eval/ledger-report.json `
  --baseline environments/ledger-v0 `
  --promote-to environments/ledger-v1
```

Omit `--baseline` for the first package. Omit `--promote-to` to produce a report only. Promotion requires:

1. All supplied cases pass.
2. At least one validation trajectory is independent of the construction sources and trajectory groups.
3. No validation case overlaps construction data, and no test case is used for promotion.
4. Every executable rule is exercised by a passing case.
5. There are no passing-to-failing regressions against the optional baseline.
6. The report matches the exact construction artifacts and configuration.

This initial gate is deliberately strict. Source-only replay, incomplete branch coverage, and partial-state ambiguity produce a useful report without promotion. Passing all tests establishes coverage of those cases, not universal correctness.

Include supporting observations and counterexamples as replay cases when their initial state
is known, alongside independent validation. The current gate checks executable-rule coverage;
it does not automatically convert or require every source transition as a separate replay case.

For report-only runs, exit code 0 means every supplied case passed and at least one case ran.
It does not by itself mean promotion is allowed: inspect `promotion_eligible` and `reasons`.
When `--promote-to` is supplied, an ineligible report returns exit code 1. Invalid inputs or
package-integrity errors can raise before a replay report is written.

Package destinations must be new. Compilation writes a fresh sibling staging directory and renames it into place only after successful completion. Existing packages are never rebuilt in place. Failed staging directories remain available for diagnosis. Session state remains separate.

### 4. Start a reconstructed session from observed state

```powershell
trace2env simulate environments/ledger-v1 --session work/ledger-session `
  --initial-state '{"world":{"balance":100}}' `
  --action '{"type":"withdraw","arguments":{"amount":25}}'
```

The initial state is required only for a new reconstructed session. Later calls use the existing session. Hand-authored packages such as `init-demo` retain their explicit defaults. An existing session is not reset by an initial-state argument.

## Freeze data splits before reconstruction

Use `--split-manifest splits.json` on `reconstruct` to bind source digests to declared splits and trajectory families:

```json
{
  "assignments": {
    "src_<training-source-sha256>": {"split": "train", "trajectory_group": "family-a"},
    "src_<validation-source-sha256>": {"split": "validation", "trajectory_group": "family-b"},
    "src_<test-source-sha256>": {"split": "test", "trajectory_group": "family-c"}
  }
}
```

With a manifest, every reconstruction input must be assigned to train. Replay source IDs, split labels, and groups must match the retained manifest. The same family cannot cross splits. Without a manifest, supplied reconstruction inputs are treated as training data, and replay still checks source/group overlap. Declared held-out episodes are rejected during ingestion.

## Typed repairs

`trace2env patch` accepts explicit edits to rules, invariants, renderers, demonstrations, actions, or state fields. A `PackagePatch` contains the base artifact digest, rationale, failed case IDs, and an edit list:

```json
{
  "base_artifact_digest": "<digest from manifest metadata>",
  "rationale": "Correct the observed equality boundary.",
  "failed_case_ids": ["withdraw-equality"],
  "edits": [{
    "collection": "rules",
    "key": "withdraw_success",
    "operation": "replace",
    "value": {"id": "withdraw_success", "action_type": "withdraw", "description": "Supply the complete corrected rule here"}
  }]
}
```

The example shows the edit envelope; supply a complete corrected rule with its actual conditions, effects, renderer, and provenance. Edits are applied together, then validated as a whole. Conflicting edits, missing targets, invalid references, and stale base digests are rejected.

```powershell
trace2env patch environments/ledger-v1 --patch repairs/boundary.json `
  --output environments/ledger-candidate-v2 --cases eval/ledger-replay.jsonl `
  --report eval/ledger-v2-report.json --promote-to environments/ledger-v2
```

The previous package is the replay baseline. Failed candidates remain inspectable; promotion leaves the previous version intact. Patch proposal is explicit in this release. No autonomous loop repeatedly tunes against the test set.

## Compile-time reconciliation

Induction stages are separate model calls and can disagree: rules may cite renderer ids the
renderer stage never defined, reference undeclared arguments or state paths, or attach event
citations to the wrong episode. Instead of failing a long build, `reconstruct` reconciles
before compiling and records every decision in `exceptions/unresolved.json`:

- A renderer id that rules cite but no contract defines gets a template-free contract
  synthesized from the rule (flagged as synthesized).
- Rules, invariants, and contracts that still violate the schema are quarantined into
  `exceptions/rejected.json` with their reasons; notes with unknown action types become global.
- Citations are re-anchored to the episode their event ids belong to; unresolvable citations are
  dropped, and a `supported` rule or note that loses its observed support is downgraded to
  `tentative` (so it stays in the candidate archive but never executes).
- Rule selectors are kept evaluable (`repair_rule_selectors`): scope keys that are neither
  tenant/version/environment selectors nor state paths (an episode id, a command string, a
  free-text qualifier) are dropped with a note — they would otherwise make the rule silently
  ineligible forever — as are scope entries whose value spells an unset selector ("None");
  observation templates (and renderer templates) containing placeholder prose such as
  `<shell prompt>` are removed, since they are descriptions rather than literal output.

Structured model calls also repair their own output: validation errors are fed back for up to
two corrections, and an answer cut off by the output limit fails fast with an actionable
message. Rejections are diagnostic, not discarded evidence: the first pilot terminal package
showed rules built on derived arguments (`command`, `duration`) that the induced action schema
did not declare. Since then `declare_observed_arguments` adds every argument name the evidence
records for an action to its `ActionSpec` (reported in `unresolved`), so that failure class is
closed at the schema stage rather than at compile time.

## Induction and demonstration changes

- Action aliases and state-path aliases (`StateField.aliases`) are canonicalized before rule
  induction (`canonical_evidence`, which returns rewritten copies; the extracted records are kept
  as written).
- Matched contrasts share action, exact arguments, and declared scope, and come from different trajectory groups. Shared/differing observed preconditions are recorded. These are observational comparisons, not causal labels.
- Demonstration selection prioritizes outcome-branch coverage and episode diversity, retaining evidence-linked observed state projections and rule links.
- Note induction (`induce_notes`) records conventions, formats, constraints, and facts that rules and contracts do not express, with provenance; notes below the confidence threshold become tentative and are excluded from the runtime knowledge catalog.
- Every reconstruction LLM call has a byte budget covering its system text, JSON input, and response schema. `--max-prompt-bytes` defaults to 600000 and `--induction-batch-size` to 100 — the settings of the reported package builds, sized for a 1M-context model; lower the byte budget for smaller contexts.
- Map/reduce consolidation recurses under that budget. Oversized indivisible evidence, schemas, or rule sets raise a clear error instead of silently dropping or truncating evidence. This is a byte bound, not a provider-specific token estimate.
- Induction calls remain sequential; extraction can run concurrently (`workers`). Extraction and induction payloads are clipped views (`extraction_history_chars`, `extraction_event_chars`, `induction_value_chars`, `induction_observation_chars`); recorded observation text and evidence files are never clipped. Citations outside a slice are dropped and recorded; a provider content-policy refusal yields a zero-confidence evidence record instead of a failed build.
- Automatic exploratory probes, calibrated confidence learning, and semantic-equivalence benchmarking require further empirical work.

## Compatibility and verification

Existing schema-1.0 packages remain readable. The authored ledger demo and persistent sessions retain their behavior. Newly reconstructed packages use schema 1.1 and the candidate/promotion workflow. Newly qualified event/episode IDs differ from prior imports; original IDs remain in metadata. Custom normalizers must now return `TraceAnnotations`.

All rule execution now requires supported status and matching scope. Equal-priority rules with incompatible behavior reject the transition, including conflicts beyond the retrieval limit. Verification checks the requested action, complete effect order/multiplicity, rendering choice, and state types. This intentionally rejects ambiguous behavior that older versions could execute.

The test suite covers source identity, correlation, normalization, provenance/schema defects, withheld rules, scope, rollback, interrupted builds, replay coverage, leakage, partial state, promotion fingerprints, typed edits, missing predictions, and scripted reconstruction through promotion. Real-corpus fidelity and live-provider behavior have not been measured by this change.

### Verification record (2026-09-16)

- All 91 unit and integration tests passed (`tests/test_trace2env.py`, `tests/test_offline_safety.py`,
  `tests/test_agentworld.py`, `tests/test_agent_harness.py`, `tests/test_atif.py`), including scripted-model coverage of
  the world-model agent loop (native tool calling and structured transports), workspace tools,
  episodic memory, fast-path and trust routing, citation checks, note induction and indexing,
  run-time state tracking, AgentWorldBench export, prediction routing, and the official
  judge/score port.
- A smoke run on the real AgentWorldBench terminal file (354 records, 76 trajectories) normalized
  all 1,827 turn actions without error, exported the 59 training trajectories with a split
  manifest, ingested them deterministically into 1,343 transitions, and ran `awb-run --no-model`
  over 40 records with prefixes up to 151 turns and `awb-score` on the result without failures.
  62% of terminal turns normalize to compound `shell` input.
- The smoke run made no model calls; no benchmark score has been measured.

### Verification record (2026-09-15)

- All 54 unit and integration tests passed, including the adapted original suite.
- A separate CLI smoke run built the authored demo, replayed four consecutive steps covering all four executable rules, promoted a new package, and simulated a withdrawal successfully.
- A deliberately incorrect typed patch produced a regression report and failed promotion. The original package's file hashes remained unchanged.
- The smoke run made no external API calls. Its oracle was the synthetic fixture in `examples/replay/ledger_reference.txt`; the matching JSONL is `examples/replay/ledger_cases.jsonl`.
- Software checks passed. Real-trace prediction improvement, hidden-state recovery, and live-model induction quality remain unmeasured.
