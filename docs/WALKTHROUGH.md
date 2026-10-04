# Walkthrough: build a ledger workspace and operate it as a world model

The complete story is **recorded environment interactions → environment knowledge files →
agent harness operating a persistent workspace → observations returned to the task agent**.
This guide follows that story with a small ledger.

The induction discussion is illustrative. Runnable commands use the hand-authored
[ledger demo](../src/trace2env/examples.py), so you can inspect the files and run the harness
without model calls. The same harness also supports optional agentic model roles; this demo's
applicable rules take the deterministic route.

## 1. Start with recorded behavior

Suppose independent episodes contain these observations in the same environment version:

| Known balance before | Action | Observed result | Known balance after |
|---|---|---|---|
| 100 | Withdraw 25 | Success | 75 |
| 20 | Withdraw 25 | Insufficient funds | 20 |
| 25 | Withdraw 25 | Success | 0 |

These rows explain three different needs: state changes, expected failure, and the equality
boundary. The first row alone cannot distinguish `balance > amount` from `balance >= amount`.
An unknown balance remains unknown; the builder must not invent it from the answer it wants.

## 2. Preserve the trace and align its events

Run the deterministic parser on the repository's sample trace:

```powershell
python -m trace2env ingest examples/raw/ledger_trace.txt --output work/walkthrough-parsed
```

Inspect `episodes.jsonl` and `transitions.jsonl` in that directory. An episode contains ordered
events. A transition links action event IDs to observation event IDs and bounded history.
Source IDs are content hashes; original event/episode IDs remain in metadata.

The full `reconstruct` pipeline also archives original UTF-8 sources and can annotate exact
text spans when roles are unclear. For overlapping tool calls, matching a result by call ID
does not establish the ordering of effects; concurrency remains explicit.

## 3. Turn a local observation into a candidate rule

Local extraction records the normalized `withdraw` action, its `amount`, the observed outcome,
known pre/post facts, possible mutations, and source references. Then interface induction
introduces a numeric `world.balance` field and an action argument named `amount`.

Comparing the episodes motivates two branches. The typed core of the success branch looks
like this (an illustrative excerpt, not a complete provenance-bearing rule):

```json
{
  "id": "withdraw_success",
  "action_type": "withdraw",
  "conditions": [{
    "left": {"path": "world.balance"},
    "op": "gte",
    "right": {"action_arg": "amount"}
  }],
  "effects": [{
    "op": "decrement",
    "path": "world.balance",
    "value": {"$action_arg": "amount"}
  }],
  "outcome": "success",
  "renderer": "ledger",
  "observation_template": "Withdrew {action.arguments.amount}. Balance: {state_after.world.balance}"
}
```

The failure branch uses `balance < amount`, has outcome `failure`, and has no balance effects.
The renderer describes the response text; it cannot itself change balance. A complete induced
rule additionally needs its status, confidence, scope, and valid evidence support. Review and
static checks precede package compilation, and replay precedes normal use of a reconstruction.

## 4. See the environment encoded as files

After `python -m pip install -e .`, run from the repository root in PowerShell. Use fresh
package paths when repeating the example.

```powershell
python -m trace2env init-demo --output .trace2env/walkthrough-authored
python -m trace2env inspect .trace2env/walkthrough-authored --action withdraw
```

The action inspection returns `withdraw_success` and `withdraw_insufficient`, their renderer,
the numeric argument contract, and the nonnegative-balance invariant. The package also contains
`deposit_success` and `balance_read`. Read `construction/artifacts.json` for the complete object
and `manifest.json` for hashes and package status.

The authored demo has no reconstructed trace evidence. Its schema default of `100.0` is an
explicit fixture choice. In a real reconstruction, episode initial values are supplied separately.

The package turns the abstract ledger into the following workspace knowledge:

| File | What the ledger example stores there |
|---|---|
| `action_schema.json` | `deposit`, `withdraw`, and `balance`; numeric `amount` where required |
| `state_schema.json` | Numeric `world.balance`, with the authored initial default |
| `rules/index.jsonl` | Deposit, successful withdrawal, insufficient-funds withdrawal, balance-read rules |
| `invariants.json` | The balance must be nonnegative |
| `renderer/contracts.json` | Ledger response instructions, required balance field, example outputs |
| `manifest.json` | Environment identity, hashes, and authored/validated status |

The concrete response templates are attached to the rules in this demo. The renderer contract
defines their shared context. The compiler also creates the catalog, action shards, and
construction records. A reconstructed package adds source evidence and selected demonstrations;
the authored demo leaves those evidence/example collections empty.

The generic executor and verifier come from the repository's harness code. They are not
generated into this package. A session database is created only when the harness starts a run.

## 5. Replay an independent authored reference

[ledger_reference.txt](../examples/replay/ledger_reference.txt) is the synthetic oracle;
[ledger_cases.jsonl](../examples/replay/ledger_cases.jsonl) encodes its state, actions, expected
observations, outcomes, and state projections. Its source ID hashes the reference file.

```powershell
python -m trace2env replay .trace2env/walkthrough-authored `
  --cases examples/replay/ledger_cases.jsonl --output work/walkthrough-replay.json `
  --promote-to .trace2env/walkthrough-validated
```

The case runs consecutively from balance `100.0`:

| Step | Action | Balance after | Expected outcome |
|---|---|---|---|
| 1 | Read balance | 100.0 | success |
| 2 | Deposit 5 | 105.0 | success |
| 3 | Withdraw 25 | 80.0 | success |
| 4 | Withdraw 1000 | 80.0 | failure |

The report should have `total_cases: 1`, `passed_cases: 1`, `total_steps: 4`,
`uncovered_rule_ids: []`, and `promotion_eligible: true`. Replay uses temporary sessions;
it does not change the package or seed the next user session with `80.0`.

This confirms four authored branches on one synthetic reference. It does not test every amount,
establish independent real-world generalization, or test whether an LLM would induce the rules.

## 6. Let the harness play the environment

Here the CLI stands in for the task agent. It sends actions and receives simulated environment
observations; the world-model harness owns the ledger's evolving state.

```powershell
python -m trace2env simulate .trace2env/walkthrough-validated `
  --session .trace2env/walkthrough-session `
  --action '{"type":"withdraw","arguments":{"amount":25}}'
python -m trace2env simulate .trace2env/walkthrough-validated `
  --session .trace2env/walkthrough-session `
  --action '{"type":"withdraw","arguments":{"amount":1000}}'
```

The first call selects `withdraw_success`, decrements `100.0` to `75.0` on a copy, verifies the
effects/invariant, renders the response, and commits revision 1. The second call selects the
failure branch and preserves balance `75.0`. It still commits a valid transition and advances
the revision. An invariant violation or unsupported plan instead records a failed audit without
committing new state. An environment-level failure and a harness rejection are different events.

For the first withdrawal, the file-to-behavior path is:

| Harness step | File or object used | Concrete result |
|---|---|---|
| Load the environment | `manifest.json` and package loader | Verify this package and read its scope/status |
| Load the episode | `session.sqlite` | Obtain `world.balance = 100.0` |
| Understand the action | `action_schema.json` | Validate `withdraw(amount=25)` |
| Inspect and select | `rules/index.jsonl` | `withdraw_success` applies because `100.0 >= 25` |
| Plan and execute | Typed rule effect + `state_schema.json` | Resolve the amount and decrement balance on a copy |
| Verify | `invariants.json` and cited rule | Balance is nonnegative; the effect matches the supported plan |
| Render | Rule template + `renderer/contracts.json` | Return `Withdrew 25. Balance: 75.0` |
| Persist | `session.sqlite`, then JSON mirrors | Commit the new state and audit for the next interaction |

When a configured planner is needed, the harness exposes selected package views through typed
inspection tools and receives a typed plan in return. That plan goes through the same executor,
verifier, renderer, and commit boundary. The current ledger trace does not invoke that optional
loop, because a rule already applies.

Read `state.json` and `audit.jsonl` in the session directory for human-readable inspection.
`session.sqlite` is authoritative. With a reconstructed package, supply `--initial-state` on
the first call; subsequent calls reuse the existing session.

## 7. Find and repair an error

If an induced success rule used `gt`, the observed equality case would fail replay. Inspect its
rule, provenance, and report; create a `PackagePatch` replacing the complete rule with `gte`;
then run `patch` with that case and regression trajectories. The previous package is the baseline.
Successful promotion creates a new version; failed candidates remain available for diagnosis.

The CLI does not propose this edit automatically. Supply actual reference cases and retain a
separate final test split. Use [Offline validation](OFFLINE_VALIDATION.md) for the patch envelope,
split manifest, command options, and acceptance requirements.
