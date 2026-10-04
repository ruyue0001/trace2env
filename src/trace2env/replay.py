"""Offline replay against caller-supplied observations; no generated oracle or LLM calls."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from trace2env.engine import MISSING, get_path, process_due_events, resolve_operand, resolve_mutation, evaluate_condition
from trace2env.eligibility import rule_eligible
from trace2env.models import ReconstructionArtifacts, ReconstructionConfig, ReplayCase, ReplayCaseResult, ReplayReport
from trace2env.package import EnvironmentPackage
from trace2env.runtime import RuntimeHarness
from trace2env.storage import read_json
from trace2env.validation import artifact_digest, config_digest, unique


def _check_partial_state(harness, action, known_absent):
    before = harness.session.load()
    state, _ = process_due_events(before, before.step + 1)
    aliases = {alias: spec.name for spec in harness.package.action_schema.actions for alias in [spec.name, *spec.aliases]}
    action = action.model_copy(update={"type": aliases.get(action.type, action.type)})
    for rule in harness.package.rules:
        if rule.action_type != action.type or not rule_eligible(rule, harness.package.scope, state):
            continue
        unknown, false = False, False
        for condition in rule.conditions:
            missing = [operand for operand in [condition.left, condition.right] if operand is not None
                       and operand.path is not None and resolve_operand(operand, state, action) is MISSING
                       and operand.path not in known_absent]
            if missing:
                unknown = True
            elif not evaluate_condition(condition, state, action):
                false = True
        if unknown and not false:
            raise ValueError(f"Rule {rule.id} requires unobserved state; replay cannot establish applicability")
        if not false:
            for effect in rule.effects:
                effect = resolve_mutation(effect, action)
                if (effect.op in {"increment", "decrement", "append"}
                        and get_path(state.model_dump(mode="python"), effect.path, MISSING) is MISSING
                        and effect.path not in known_absent):
                    raise ValueError(f"Rule {rule.id} updates unobserved state at {effect.path}")


def _run(package_dir: str | Path, cases: list[ReplayCase]) -> list[ReplayCaseResult]:
    results = []
    from trace2env.models import StepRequest
    for case in cases:
        errors = []
        attempted = 0
        rule_ids = set()
        # Each trajectory starts in a fresh session. No schema defaults supplement known state.
        with tempfile.TemporaryDirectory(prefix="trace2env-replay-") as directory:
            try:
                harness = RuntimeHarness(package_dir, directory, allow_unvalidated=True,
                                         initial_state=case.initial_state)
                known_absent = set(case.known_absent_paths)
                for index, expected in enumerate(case.steps):
                    attempted += 1
                    if not case.initial_state_complete:
                        _check_partial_state(harness, expected.action, known_absent)
                    actual = harness.step(StepRequest(action=expected.action))
                    rule_ids.update(actual.plan.rule_ids)
                    for effect in actual.plan.effects:
                        if effect.op == "delete":
                            known_absent.add(effect.path)
                        elif effect.op != "schedule":
                            known_absent.discard(effect.path)
                    if expected.observation_match == "json":
                        matches = json.loads(actual.observation) == json.loads(expected.expected_observation)
                    else:
                        matches = actual.observation == expected.expected_observation
                    if not matches:
                        errors.append(f"step {index + 1}: observation mismatch")
                    if expected.expected_outcome is not None and actual.outcome != expected.expected_outcome:
                        errors.append(f"step {index + 1}: outcome mismatch")
                    for path, value in expected.expected_state.items():
                        observed = get_path(actual.state.model_dump(mode="python"), path, MISSING)
                        if observed is MISSING or observed != value:
                            errors.append(f"step {index + 1}: state mismatch at {path}")
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
        results.append(ReplayCaseResult(case_id=case.id, trajectory_group=case.trajectory_group,
            passed=not errors, attempted_steps=attempted, expected_steps=len(case.steps), errors=errors,
            rule_ids=sorted(rule_ids)))
    return results


def evaluate_package(package_dir: str | Path, cases: list[ReplayCase],
                     baseline_dir: str | Path | None = None) -> ReplayReport:
    unique([case.id for case in cases], "replay case IDs")
    groups = {}
    source_splits = {}
    for case in cases:
        if case.trajectory_group in groups and groups[case.trajectory_group] != case.split:
            raise ValueError("Trajectory group crosses replay splits")
        groups[case.trajectory_group] = case.split
        for source in case.source_ids:
            if source in source_splits and source_splits[source] != case.split:
                raise ValueError("Source snapshot crosses replay splits")
            source_splits[source] = case.split
    package = EnvironmentPackage(package_dir)
    artifacts_path = package.root / "construction" / "artifacts.json"
    if not artifacts_path.is_file():
        raise ValueError("Replay promotion requires a package with construction artifacts")
    artifacts = ReconstructionArtifacts.model_validate(read_json(artifacts_path))
    config = ReconstructionConfig.model_validate(read_json(package.root / "construction" / "config.json"))
    if artifacts.split_assignments:
        for case in cases:
            for source in case.source_ids:
                assignment = artifacts.split_assignments.get(source)
                if assignment is None or assignment.split != case.split or assignment.trajectory_group != case.trajectory_group:
                    raise ValueError("Replay case violates the construction split manifest")
    digest = artifact_digest(artifacts)
    if digest != package.manifest.metadata.get("artifact_digest"):
        raise ValueError("Construction artifact fingerprint does not match the package")
    if config_digest(config) != package.manifest.metadata.get("config_digest"):
        raise ValueError("Construction configuration fingerprint does not match the package")
    results = _run(package_dir, cases)
    baseline = _run(baseline_dir, cases) if baseline_dir is not None else None
    regressions = [result.case_id for old, result in zip(baseline or [], results) if old.passed and not result.passed]
    training_sources = {ep.source_id for ep in artifacts.episodes}
    training_groups = {str(ep.metadata.get("trajectory_group", ep.metadata.get("original_episode_id") or ep.id))
                       for ep in artifacts.episodes}
    independent = [case for case in cases if case.split == "validation"
                   and not (training_sources & set(case.source_ids)) and case.trajectory_group not in training_groups]
    reasons = []
    covered = {rule for result in results if result.passed for rule in result.rule_ids}
    uncovered = sorted({rule.id for rule in package.rules} - covered)
    if uncovered:
        reasons.append("Executable rules remain untested by passing replay cases")
    if not cases:
        reasons.append("No replay cases supplied")
    if any(case.split == "test" for case in cases):
        reasons.append("Test cases are reporting-only and cannot be used for promotion")
    if not independent:
        reasons.append("No independent validation trajectory supplied")
    if any(case.split == "validation" and (training_sources & set(case.source_ids) or case.trajectory_group in training_groups) for case in cases):
        reasons.append("Validation data overlaps construction evidence")
    if not all(item.passed for item in results):
        reasons.append("Candidate failed one or more replay cases")
    if regressions:
        reasons.append("Candidate regressed against the baseline")
    return ReplayReport(artifact_digest=digest, config_digest=config_digest(config),
        cases=results, total_cases=len(results), uncovered_rule_ids=uncovered,
        passed_cases=sum(result.passed for result in results),
        total_steps=sum(result.expected_steps for result in results),
        attempted_steps=sum(result.attempted_steps for result in results),
        baseline_passed_cases=sum(result.passed for result in baseline) if baseline is not None else None,
        regressions=regressions, promotion_eligible=not reasons, reasons=reasons)


def promote_package(package_dir: str | Path, cases: list[ReplayCase], output_dir: str | Path,
                    baseline_dir: str | Path | None = None) -> tuple[Path, ReplayReport]:
    from trace2env.compiler import EnvironmentCompiler
    report = evaluate_package(package_dir, cases, baseline_dir)
    if not report.promotion_eligible:
        raise ValueError("Promotion rejected: " + "; ".join(report.reasons))
    root = Path(package_dir)
    artifacts = ReconstructionArtifacts.model_validate(read_json(root / "construction" / "artifacts.json"))
    config = ReconstructionConfig.model_validate(read_json(root / "construction" / "config.json"))
    return EnvironmentCompiler(config).compile(artifacts, output_dir, report), report
