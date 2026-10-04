#!/usr/bin/env python3
"""Run offline reconstruction one stage at a time, keeping every version of every stage inspectable.

    python scripts/pilot_stages.py WORKSPACE ingest --episodes DIR --split-manifest FILE \
        --environment-id ID --name NAME --description @FILE --domain terminal
    python scripts/pilot_stages.py WORKSPACE extract [--tasks a,b] [--limit N] MODEL-OPTIONS
    python scripts/pilot_stages.py WORKSPACE schema | rules | renderers | notes  MODEL-OPTIONS
    python scripts/pilot_stages.py WORKSPACE compile --package-label v1

Layout: WORKSPACE/stages/<stage>/v<N>/ holds inputs.json (upstream versions, effective config, model,
options, summary), calls.jsonl (every model call: prompt, payload, validated output, usage, timing),
the stage result, work/ (the reconstructor's own files) and code/ (snapshots of the prompt and pipeline
sources plus CHANGES.diff against the previous version). WORKSPACE/state.json points each stage at its
current version; --use STAGE=vN picks another upstream version for one run, --note records why a version
exists. Model calls are cached under WORKSPACE/cache unless --cache-dir says otherwise.
"""

from __future__ import annotations

import argparse
import difflib
import json
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.cli import _add_model_options, _llm, _text_argument  # noqa: E402
from trace2env.compiler import EnvironmentCompiler  # noqa: E402
from trace2env.models import (  # noqa: E402
    Episode,
    LocalTransitionEvidence,
    NoteInductionResult,
    ReconstructionArtifacts,
    ReconstructionConfig,
    RenderInductionResult,
    RuleInductionResult,
    SchemaInductionResult,
    SplitAssignment,
    SplitManifest,
    TransitionSlice,
)
from trace2env.prompts import LOCAL_EVIDENCE_EXTRACTOR  # noqa: E402
from trace2env.reconstruction import (TraceReconstructor, canonical_evidence,
    declare_observed_actions,
    declare_observed_arguments, reconcile_artifacts,  # noqa: E402
                                      reconcile_provenance, refused_count, invalid_count)
from trace2env.storage import load_models, read_json, write_json, write_jsonl  # noqa: E402

STAGES = ["ingest", "extract", "schema", "rules", "renderers", "notes", "compile"]
UPSTREAM = {
    "ingest": [],
    "extract": ["ingest"],
    "schema": ["extract"],
    "rules": ["extract", "schema"],
    "renderers": ["extract", "schema", "rules"],
    "notes": ["extract", "schema", "rules", "renderers"],
    "compile": ["ingest", "extract", "schema", "rules", "renderers", "notes"],
}
CODE_FILES = ["prompts.py", "reconstruction.py", "induction.py", "validation.py", "contrasts.py", "compiler.py",
              "llm.py", "models.py"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state(workspace: Path) -> dict:
    path = workspace / "state.json"
    return read_json(path) if path.exists() else {"current": {}, "history": []}


def next_version(workspace: Path, stage: str) -> str:
    stage_dir = workspace / "stages" / stage
    numbers = [int(p.name[1:]) for p in stage_dir.glob("v*") if p.name[1:].isdigit()] if stage_dir.exists() else []
    return f"v{max(numbers, default=0) + 1}"


def snapshot_code(dest: Path, previous: Path | None) -> None:
    code = dest / "code"
    code.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        shutil.copy2(ROOT / "src" / "trace2env" / name, code / name)
    if previous is None or not (previous / "code").exists():
        return
    diff: list[str] = []
    for name in CODE_FILES:
        old_path = previous / "code" / name
        old = old_path.read_text(encoding="utf-8").splitlines(keepends=True) if old_path.exists() else []
        new = (code / name).read_text(encoding="utf-8").splitlines(keepends=True)
        diff.extend(difflib.unified_diff(old, new, fromfile=f"{previous.name}/{name}", tofile=f"{dest.name}/{name}"))
    (code / "CHANGES.diff").write_text("".join(diff) or f"(no code changes since {previous.name})\n", encoding="utf-8")


def effective_config(workspace: Path, args: argparse.Namespace) -> ReconstructionConfig:
    base = read_json(workspace / "config.json") if (workspace / "config.json").exists() else {}
    overrides = {}
    for key in ("environment_id", "name", "max_prompt_bytes", "induction_batch_size", "min_rule_confidence",
                "max_history_events"):
        if getattr(args, key, None) is not None:
            overrides[key] = getattr(args, key)
    if getattr(args, "description", None) is not None:
        overrides["description"] = _text_argument(args.description)
    if getattr(args, "domain", None):
        overrides["domains"] = list(args.domain)
    overrides["model"] = args.model
    if getattr(args, "provider", "responses") == "chat" and getattr(args, "chat_reasoning_effort", None):
        overrides["reasoning_effort"] = args.chat_reasoning_effort
    elif getattr(args, "reasoning_effort", None):
        overrides["reasoning_effort"] = args.reasoning_effort
    if "environment_id" not in {**base, **overrides}:
        sys.exit("--environment-id and --name are required the first time a workspace is used")
    config = ReconstructionConfig(**{**base, **overrides})
    write_json(workspace / "config.json", config)
    return config


def restore_ingest_meta(recon: TraceReconstructor, ingest_dir: Path) -> None:
    meta = read_json(ingest_dir / "ingest_meta.json")
    recon.source_snapshots = meta["source_snapshots"]
    recon.episode_groups = meta["episode_groups"]
    recon.episode_scopes = meta["episode_scopes"]
    recon.split_assignments = {key: SplitAssignment(**value) for key, value in meta["split_assignments"].items()}


def usage_summary(calls_path: Path) -> dict:
    totals: dict[str, float] = {"calls": 0, "cache_hits": 0, "errors": 0, "elapsed_s": 0.0}
    if not calls_path.exists():
        return totals
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        totals["calls"] += 1
        totals["cache_hits"] += 1 if record.get("cache_hit") else 0
        totals["errors"] += 1 if record.get("error") else 0
        totals["elapsed_s"] += record.get("elapsed_s") or 0.0
        for key, value in (record.get("usage") or {}).items():
            # Usage may also carry categorical provenance such as the OpenRouter
            # upstream provider.  The stage summary is an additive numeric total;
            # keep the full categorical value in calls.jsonl and do not add it.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue  # labels such as the answering `provider` are recorded per call, not summed
            totals[key] = totals.get(key, 0) + value
    return totals


# ─── Stages ────────────────────────────────────────────────────────────────────


def stage_ingest(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    episodes_arg = Path(args.episodes)
    # macOS sidecar files (._name.json) are AppleDouble metadata, not traces.
    paths = sorted(p for p in episodes_arg.glob("*.json") if not p.name.startswith("._")) if episodes_arg.is_dir() else [episodes_arg]
    manifest = SplitManifest.model_validate(read_json(Path(args.split_manifest))) if args.split_manifest else None
    episodes, transitions = recon.ingest(paths, manifest)
    write_jsonl(dest / "episodes.jsonl", episodes)
    write_jsonl(dest / "transitions.jsonl", transitions)
    write_json(dest / "ingest_meta.json", {
        "source_snapshots": recon.source_snapshots,
        "episode_groups": recon.episode_groups,
        "episode_scopes": recon.episode_scopes,
        "split_assignments": {k: v.model_dump(mode="json") for k, v in recon.split_assignments.items()},
        "paths": [str(p) for p in paths],
    })
    by_episode = {episode.id: episode for episode in episodes}
    sizes = [recon.bounded.size(LOCAL_EVIDENCE_EXTRACTOR, recon.extraction_payload(by_episode[t.episode_id], t),
                                LocalTransitionEvidence) for t in transitions]
    per_task = {}
    for transition in transitions:
        task = by_episode[transition.episode_id].metadata.get("benchmark_task", transition.episode_id)
        per_task[task] = per_task.get(task, 0) + 1
    return {
        "episodes": len(episodes), "transitions": len(transitions), "transitions_per_task": per_task,
        "ambiguous_transitions": sum(1 for t in transitions if t.alignment == "ambiguous"),
        "extraction_payload_bytes": {"mean": int(statistics.mean(sizes)) if sizes else 0, "max": max(sizes, default=0),
                                     "over_budget": sum(1 for s in sizes if s > recon.config.max_prompt_bytes)},
    }


def stage_extract(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, upstream["ingest"])
    episodes = load_models(upstream["ingest"] / "episodes.jsonl", Episode)
    transitions = load_models(upstream["ingest"] / "transitions.jsonl", TransitionSlice)
    if args.tasks:
        wanted = set(args.tasks.split(","))
        keep = {ep.id for ep in episodes if ep.metadata.get("benchmark_task") in wanted}
        transitions = [t for t in transitions if t.episode_id in keep]
    if args.limit:
        transitions = transitions[: args.limit]
    evidence = recon.extract_evidence(episodes, transitions)
    write_jsonl(dest / "evidence.jsonl", evidence)
    outcomes: dict[str, int] = {}
    for item in evidence:
        outcomes[item.outcome.value] = outcomes.get(item.outcome.value, 0) + 1
    return {
        "transitions": len(transitions), "evidence": len(evidence), "outcomes": outcomes,
        "refused_by_provider": refused_count(evidence),
        "invalid_output": invalid_count(evidence),
        "action_types": len({item.action.type for item in evidence}),
        "with_mutations": sum(1 for item in evidence if item.mutations),
        "with_ambiguities": sum(1 for item in evidence if item.ambiguities),
        "mean_confidence": round(statistics.mean(item.confidence.value for item in evidence), 3) if evidence else None,
    }


def stage_schema(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, _ingest_dir_for(workspace, upstream["extract"]))
    evidence = load_models(upstream["extract"] / "evidence.jsonl", LocalTransitionEvidence)
    schema = recon.induce_schema(evidence)
    write_json(dest / "schema.json", schema)
    return {"actions": len(schema.action_schema.actions), "state_fields": len(schema.state_schema.fields),
            "unresolved": len(schema.unresolved), "evidence": len(evidence)}


def _ingest_dir_for(workspace: Path, extract_dir: Path) -> Path:
    return workspace / "stages" / "ingest" / read_json(extract_dir / "inputs.json")["upstream"]["ingest"]


def stage_rules(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, _ingest_dir_for(workspace, upstream["extract"]))
    schema = SchemaInductionResult.model_validate(read_json(upstream["schema"] / "schema.json"))
    evidence = canonical_evidence(load_models(upstream["extract"] / "evidence.jsonl", LocalTransitionEvidence), schema)
    rules = recon.induce_rules(evidence, schema)
    write_json(dest / "rules.json", rules)
    statuses: dict[str, int] = {}
    for rule in rules.rules:
        statuses[rule.status] = statuses.get(rule.status, 0) + 1
    return {"rules": len(rules.rules), "by_status": statuses, "invariants": len(rules.invariants),
            "unresolved": len(rules.unresolved), "evidence": len(evidence)}


def stage_renderers(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, _ingest_dir_for(workspace, upstream["extract"]))
    schema = SchemaInductionResult.model_validate(read_json(upstream["schema"] / "schema.json"))
    raw_evidence = load_models(upstream["extract"] / "evidence.jsonl", LocalTransitionEvidence)
    # A schema induced before declare_observed_actions existed may lack rare evidence action types; the
    # deterministic repair runs here too so such a workspace compiles (a no-op when nothing is missing).
    schema.unresolved.extend(declare_observed_actions(schema, raw_evidence))
    schema.unresolved.extend(declare_observed_arguments(schema, raw_evidence))
    evidence = canonical_evidence(raw_evidence, schema)
    rules = RuleInductionResult.model_validate(read_json(upstream["rules"] / "rules.json"))
    renderers = recon.induce_renderers(evidence, rules)
    write_json(dest / "renderers.json", renderers)
    return {"renderers": len(renderers.renderers), "with_template": sum(1 for r in renderers.renderers if r.template),
            "unresolved": len(renderers.unresolved)}


def stage_notes(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, _ingest_dir_for(workspace, upstream["extract"]))
    schema = SchemaInductionResult.model_validate(read_json(upstream["schema"] / "schema.json"))
    evidence = canonical_evidence(load_models(upstream["extract"] / "evidence.jsonl", LocalTransitionEvidence), schema)
    rules = RuleInductionResult.model_validate(read_json(upstream["rules"] / "rules.json"))
    renderers = RenderInductionResult.model_validate(read_json(upstream["renderers"] / "renderers.json"))
    notes = recon.induce_notes(evidence, rules, renderers)
    write_json(dest / "notes.json", notes)
    kinds: dict[str, int] = {}
    for note in notes.notes:
        kinds[note.kind] = kinds.get(note.kind, 0) + 1
    return {"notes": len(notes.notes), "by_kind": kinds, "unresolved": len(notes.unresolved)}


def stage_compile(args, workspace: Path, recon: TraceReconstructor, dest: Path, upstream: dict[str, Path]) -> dict:
    restore_ingest_meta(recon, upstream["ingest"])
    episodes = load_models(upstream["ingest"] / "episodes.jsonl", Episode)
    transitions = load_models(upstream["ingest"] / "transitions.jsonl", TransitionSlice)
    schema = SchemaInductionResult.model_validate(read_json(upstream["schema"] / "schema.json"))
    raw_evidence = load_models(upstream["extract"] / "evidence.jsonl", LocalTransitionEvidence)
    # A schema induced before declare_observed_actions existed may lack rare evidence action types; the
    # deterministic repair runs at compile time too so such a workspace compiles (a no-op when nothing is missing).
    schema.unresolved.extend(declare_observed_actions(schema, raw_evidence))
    schema.unresolved.extend(declare_observed_arguments(schema, raw_evidence))
    evidence = canonical_evidence(raw_evidence, schema)
    rules = RuleInductionResult.model_validate(read_json(upstream["rules"] / "rules.json"))
    renderers = RenderInductionResult.model_validate(read_json(upstream["renderers"] / "renderers.json"))
    notes = NoteInductionResult.model_validate(read_json(upstream["notes"] / "notes.json"))
    kept_rules, kept_invariants, kept_renderers, kept_notes, reconciled, rejected = reconcile_artifacts(
        schema, rules.rules, rules.invariants, renderers.renderers, notes.notes)
    reconciled += reconcile_provenance(schema, kept_rules, kept_invariants, kept_renderers, kept_notes,
                                       episodes, evidence, transitions)
    unresolved = list(dict.fromkeys(schema.unresolved + rules.unresolved + renderers.unresolved + notes.unresolved + reconciled))
    artifacts = ReconstructionArtifacts(
        episodes=episodes, transitions=transitions, evidence=evidence,
        action_schema=schema.action_schema, state_schema=schema.state_schema,
        rules=kept_rules, invariants=kept_invariants, renderers=kept_renderers,
        demonstrations=recon.select_demonstrations(evidence, rules=kept_rules), notes=kept_notes,
        unresolved=unresolved, rejected=rejected,
        source_snapshots=recon.source_snapshots, split_assignments=recon.split_assignments,
    )
    write_json(dest / "artifacts.json", artifacts)
    write_json(dest / "rejected.json", rejected)
    write_json(dest / "unresolved.json", unresolved)
    package_dir = workspace / "packages" / args.package_label
    if package_dir.exists():
        sys.exit(f"package destination {package_dir} already exists; pick another --package-label")
    package = EnvironmentCompiler(recon.config).compile(artifacts, package_dir)
    return {"package": str(package), "rules_kept": len(kept_rules), "rules_rejected": sum(1 for r in rejected if r["kind"] == "rule"),
            "invariants": len(kept_invariants), "renderers": len(kept_renderers), "notes": len(kept_notes),
            "demonstrations": len(artifacts.demonstrations), "unresolved": len(unresolved)}


STAGE_FUNCTIONS = {"ingest": stage_ingest, "extract": stage_extract, "schema": stage_schema, "rules": stage_rules,
                   "renderers": stage_renderers, "notes": stage_notes, "compile": stage_compile}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workspace")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--note", default="", help="Why this version exists (recorded in inputs.json)")
    parser.add_argument("--use", action="append", default=[], metavar="STAGE=vN", help="Upstream version override")
    parser.add_argument("--version", help="Version name for this run (default: next v<N>)")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent extract_transition calls")
    # ingest options
    parser.add_argument("--episodes", help="Episode JSON directory or file (ingest)")
    parser.add_argument("--split-manifest")
    # extract options
    parser.add_argument("--tasks", help="Comma-separated benchmark task names to extract (subset run)")
    parser.add_argument("--limit", type=int, help="Only the first N transitions (subset run)")
    # compile options
    parser.add_argument("--package-label", default=None)
    # config overrides
    parser.add_argument("--environment-id")
    parser.add_argument("--name")
    parser.add_argument("--description", help="Text or @path")
    parser.add_argument("--domain", action="append")
    parser.add_argument("--max-prompt-bytes", type=int)
    parser.add_argument("--induction-batch-size", type=int)
    parser.add_argument("--min-rule-confidence", type=float)
    parser.add_argument("--max-history-events", type=int)
    _add_model_options(parser)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    state = load_state(workspace)
    overrides = dict(item.split("=", 1) for item in args.use)
    upstream: dict[str, Path] = {}
    for up in UPSTREAM[args.stage]:
        version = overrides.get(up) or state["current"].get(up)
        if not version:
            sys.exit(f"stage {args.stage} needs a completed {up} stage; none recorded in state.json (pass --use {up}=vN)")
        upstream[up] = workspace / "stages" / up / version
        if not upstream[up].exists():
            sys.exit(f"upstream version {upstream[up]} does not exist")
    version = args.version or next_version(workspace, args.stage)
    dest = workspace / "stages" / args.stage / version
    if dest.exists():
        sys.exit(f"{dest} already exists")
    dest.mkdir(parents=True)
    previous = state["current"].get(args.stage)
    snapshot_code(dest, workspace / "stages" / args.stage / previous if previous else None)

    config = effective_config(workspace, args)
    args.cache_dir = args.cache_dir or str(workspace / "cache")
    args.call_log = str(dest / "calls.jsonl")
    llm = _llm(args)
    recon = TraceReconstructor(llm, config, dest / "work", workers=args.workers)
    if args.stage == "compile" and not args.package_label:
        args.package_label = version
    inputs = {
        "stage": args.stage, "version": version, "note": args.note, "started": utc_now(),
        "upstream": {k: v.name for k, v in upstream.items()},
        "config": config.model_dump(mode="json"),
        "model": {"model": args.model, "provider": args.provider, "base_url": args.base_url,
                  "reasoning_effort": args.reasoning_effort, "chat_reasoning_effort": args.chat_reasoning_effort,
                  "max_output_tokens": args.max_output_tokens, "cache_dir": args.cache_dir},
        "options": {"tasks": args.tasks, "limit": args.limit, "workers": args.workers, "episodes": args.episodes,
                    "split_manifest": args.split_manifest, "package_label": args.package_label},
    }
    write_json(dest / "inputs.json", inputs)
    status, summary = "failed", None
    try:
        summary = STAGE_FUNCTIONS[args.stage](args, workspace, recon, dest, upstream)
        status = "completed"
    except Exception as exc:  # keep the partial version for inspection, but do not point state at it
        summary = {"error": f"{type(exc).__name__}: {exc}"}
        raise
    finally:
        inputs.update(finished=utc_now(), status=status, summary=summary, usage=usage_summary(dest / "calls.jsonl"))
        write_json(dest / "inputs.json", inputs)
        if status == "completed":
            state["current"][args.stage] = version
            state.setdefault("history", []).append({"stage": args.stage, "version": version, "finished": inputs["finished"],
                                                    "note": args.note, "upstream": inputs["upstream"]})
            write_json(workspace / "state.json", state)
        print(json.dumps({"stage": args.stage, "version": version, "status": inputs["status"], "summary": summary,
                          "usage": inputs["usage"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
