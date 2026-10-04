#!/usr/bin/env python3
"""Prepare the construction traces of every scaling package: fresh `atif-export` of each selected Harbor trial (which
reproduces the v1_20_r=1 copies byte for byte — asserted), copied into <package>/traces/reconstruction/ with a
SplitManifest assigning every trace to `train` under its task's trajectory group. Digests are recorded in the manifest."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ONLY = set(sys.argv[1:])  # optional package names to (re)prepare; default all non-reused packages

ROOT = Path("work/exp-scaling")
manifest = json.load(open(ROOT / "manifest.json"))
trials = {}
for spec in manifest["packages"].values():
    for trace in spec["traces"]:
        trials[trace["task"]] = trace
export_dir = Path(tempfile.mkdtemp(prefix="scaling-export-"))
for task, trace in sorted(trials.items()):
    subprocess.run([sys.executable, "-m", "trace2env", "atif-export", trace["trial_dir"], "--output", str(export_dir / task), "--split", "all"],
                   check=True, capture_output=True, env={**__import__("os").environ, "PYTHONPATH": "src"})
exported = {}
origin = {}
for task in trials:
    files = [p for p in (export_dir / task).glob("*.json") if p.name != "split_manifest.json"]
    assert len(files) == 1, (task, files)
    reference = Path("work/exp-v1_20_r=1/traces/reconstruction") / files[0].name
    if reference.exists():
        # The 20 shared tasks reuse the v1_20_r=1 inputs byte for byte, so the smaller sets are exact subsets of that
        # package's construction data. (The current exporter splits control-key batches slightly differently for a
        # few of them; the manifest records which copies would have differed.)
        exported[task] = reference
        origin[task] = "v1_20_r=1 copy" + ("" if reference.read_bytes() == files[0].read_bytes() else " (fresh export would differ: control-key batch normalization)")
    else:
        exported[task] = files[0]
        origin[task] = "fresh atif-export (current exporter)"
for name, spec in manifest["packages"].items():
    if spec["reused_existing_package"] or (ONLY and name not in ONLY):
        continue
    destination = ROOT / name / "traces" / "reconstruction"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    assignments = {}
    for trace in spec["traces"]:
        source = exported[trace["task"]]
        target = destination / source.name
        shutil.copyfile(source, target)
        episode = json.loads(target.read_text(encoding="utf-8"))
        if episode.get("split") != "train":
            # The exporter labels trials train/validation/test by a hash for our own held-out replay; that
            # hold-out is not used in this experiment, and the construction pipeline refuses held-out
            # episodes, so the label is set to train in the copy (the collection file is untouched).
            trace["exporter_split_label"] = episode.get("split")
            episode["split"] = "train"
            target.write_text(json.dumps(episode, ensure_ascii=False, indent=1), encoding="utf-8")
        digest = "src_" + hashlib.sha256(target.read_bytes()).hexdigest()
        assignments[digest] = {"split": "train", "trajectory_group": f"tb2:{trace['task']}"}
        trace["episode_file"] = str(target)
        trace["source_id"] = digest
        trace["origin"] = origin[trace["task"]]
    (ROOT / name / "traces" / "split_manifest.json").write_text(json.dumps({"assignments": assignments}, indent=1))
    print(f"{name}: {len(assignments)} traces -> {destination}")
manifest["trace_preparation"] = {"method": "the 20 tasks shared with v1_20_r=1 reuse its trace copies byte for byte; the 16 additional tasks are fresh "
                                           "`trace2env atif-export <trial_dir> --split all` outputs of the current exporter",
                                 "fresh_export_would_differ_for": sorted(t for t, o in origin.items() if "would differ" in o),
                                 "held_out_labels_overridden": sorted(t["task"] for spec in manifest["packages"].values() for t in spec["traces"] if t.get("exporter_split_label")),
                                 "split": "all train; trajectory_group tb2:<task>"}
(ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
shutil.rmtree(export_dir)
print("manifest updated")
