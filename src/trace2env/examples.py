"""Small deterministic package used for smoke tests and first-run exploration."""

from __future__ import annotations

from pathlib import Path

from trace2env.compiler import EnvironmentCompiler
from trace2env.models import (
    ActionSchema,
    ActionSpec,
    ArgumentSpec,
    Condition,
    EnvironmentNote,
    Invariant,
    Operand,
    Outcome,
    ReconstructionArtifacts,
    ReconstructionConfig,
    RenderContract,
    StateField,
    StateMutation,
    StateSchema,
    TransitionRule,
)


def build_ledger_demo(output_dir: str | Path) -> Path:
    config = ReconstructionConfig(
        environment_id="demo.ledger",
        name="Trace2Env Ledger Demo",
        description="A tiny balance environment demonstrating contrastive success/failure rules.",
        domains=["tool", "transaction"],
        construction_kind="authored",
    )
    actions = ActionSchema(
        actions=[
            ActionSpec(
                name="deposit",
                description="Add funds to the current balance.",
                arguments={"amount": ArgumentSpec(type="number", required=True)},
            ),
            ActionSpec(
                name="withdraw",
                description="Remove funds when the balance is sufficient.",
                arguments={"amount": ArgumentSpec(type="number", required=True)},
            ),
            ActionSpec(name="balance", description="Read the current balance."),
        ]
    )
    state = StateSchema(
        fields=[
            StateField(
                path="world.balance",
                type="number",
                description="Available account balance.",
                default=100.0,
            )
        ]
    )
    rules = [
        TransitionRule(
            id="deposit_success",
            action_type="deposit",
            description="Deposits increase the available balance.",
            priority=10,
            effects=[
                StateMutation(op="increment", path="world.balance", value={"$action_arg": "amount"})
            ],
            outcome=Outcome.SUCCESS,
            renderer="ledger",
            observation_template="Deposited {action.arguments.amount}. Balance: {state_after.world.balance}",
        ),
        TransitionRule(
            id="withdraw_success",
            action_type="withdraw",
            description="A covered withdrawal decreases the balance.",
            priority=10,
            conditions=[
                Condition(
                    left=Operand(path="world.balance"),
                    op="gte",
                    right=Operand(action_arg="amount"),
                )
            ],
            effects=[
                StateMutation(op="decrement", path="world.balance", value={"$action_arg": "amount"})
            ],
            outcome=Outcome.SUCCESS,
            renderer="ledger",
            observation_template="Withdrew {action.arguments.amount}. Balance: {state_after.world.balance}",
        ),
        TransitionRule(
            id="withdraw_insufficient",
            action_type="withdraw",
            description="An uncovered withdrawal fails without changing state.",
            priority=10,
            conditions=[
                Condition(
                    left=Operand(path="world.balance"),
                    op="lt",
                    right=Operand(action_arg="amount"),
                )
            ],
            outcome=Outcome.FAILURE,
            renderer="ledger",
            observation_template="Insufficient funds. Balance: {state_after.world.balance}",
        ),
        TransitionRule(
            id="balance_read",
            action_type="balance",
            description="Reading the balance has no state effect.",
            priority=10,
            outcome=Outcome.SUCCESS,
            renderer="ledger",
            observation_template="Balance: {state_after.world.balance}",
        ),
    ]
    invariant = Invariant(
        id="nonnegative_balance",
        description="Available balance must never be negative.",
        condition=Condition(
            left=Operand(path="world.balance"), op="gte", right=Operand(literal=0)
        ),
    )
    renderers = [
        RenderContract(
            id="ledger",
            action_types=["deposit", "withdraw", "balance"],
            content_type="text/plain",
            required_fields=["state_after.world.balance"],
            instructions="Return one concise status line and preserve numeric values.",
            examples=["Balance: 100.0", "Insufficient funds. Balance: 20.0"],
        )
    ]
    notes = [
        EnvironmentNote(
            id="balance_format",
            kind="format",
            statement="Balances are rendered as decimal numbers with one fractional digit, e.g. 75.0.",
            action_types=["deposit", "withdraw", "balance"],
        ),
        EnvironmentNote(
            id="single_account",
            kind="concept",
            statement="The environment holds exactly one account; every action refers to its balance.",
        ),
    ]
    artifacts = ReconstructionArtifacts(
        episodes=[],
        transitions=[],
        evidence=[],
        action_schema=actions,
        state_schema=state,
        rules=rules,
        invariants=[invariant],
        renderers=renderers,
        notes=notes,
    )
    return EnvironmentCompiler(config).compile(artifacts, output_dir)
