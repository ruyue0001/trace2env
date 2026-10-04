#!/usr/bin/env bash
# Seven-system comparison (prompting, v2, schema_only, raw_traces, single_shot, v3, examples_only, structure_only)
# and the structure-vs-mimicry slices, once every judged file exists.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
A="work/exp-v1_20_r=1/awb"
SYS=(--system prompting="$A/judged-prompting.jsonl" --system v2="$A/judged-v2.jsonl" --system schema_only="$A/judged-schema_only.jsonl"
     --system raw_traces="$A/judged-raw_traces.jsonl" --system single_shot="$A/judged-single_shot.jsonl" --system v3="$A/judged-v3.jsonl"
     --system examples_only="$A/judged-examples_only.jsonl" --system structure_only="$A/judged-structure_only.jsonl"
     --system single_shot_hv3="$A/judged-single_shot_hv3.jsonl" --system no_state_hv3="$A/judged-no_state_hv3.jsonl")
python "$A/compare_systems.py" "${SYS[@]}" --baseline prompting --reference v2 --out "$A/report-all-systems.json"
python "$A/analyze_structure.py" "${SYS[@]}" --reference v3 --corpus-programs "$A/corpus_programs.json" --out "$A/report-structure.json"
