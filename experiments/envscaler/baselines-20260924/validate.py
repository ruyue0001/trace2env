"""Check that rows and packages still match the frozen experiment plan."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
plan = json.loads((root / "plan.json").read_text())
for env in plan["environment_order"]:
    record = plan["environments"][env]
    rows_dir = root / "rows" / env
    for name, expected in record["row_files_sha256"].items():
        path = rows_dir / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"row file changed: {path}")
    full = repo / "work" / "exp-envscaler" / env / "packages" / f"envscaler-{env}-v1" / "manifest.json"
    schema = root / "packages" / f"{env}-schema-only" / "manifest.json"
    for path, key in ((full, "full_package_manifest_sha256"), (schema, "schema_only_manifest_sha256")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != record[key]:
            raise ValueError(f"package manifest changed: {path}")
    rows = [json.loads(line) for line in (rows_dir / "rows.jsonl").read_text().splitlines()]
    if len(rows) != record["rows"]:
        raise ValueError(f"row count changed: {env}")
    print(f"{env}: {len(rows)} rows and two package manifests frozen")
