#!/usr/bin/env python3
"""Audit of the exact scorer on the exported rows: mutate every ground truth in ways whose verdict is known.

    python work/exp-envscaler/audit_scoring.py --experiment work/exp-envscaler --output work/exp-envscaler/scoring_audit.json

No model and no environment code is involved: each "prediction" is the row's own ground truth, changed by one
rule, and the scorer must return the verdict that rule implies. A rule applies to a row only when the row has
what it needs (a generated id, a list of records, ...). Exit status is non-zero when any verdict is wrong.

    rule                        what changes                                                   match  strict  text  outcome
    identity                    nothing                                                         yes    yes    yes    yes
    fenced                      the answer wrapped in a code fence                              yes    yes    yes    yes
    json                        rendered as JSON instead of a Python literal                    yes    yes    no     yes
    reversed_keys               dict keys in reverse order                                      yes    yes    no     yes
    pretty_printed              line breaks and indentation                                     yes    yes    -      yes
    fresh_generated_id          every generated uuid replaced by another new uuid               yes    yes    yes    yes
    shifted_clock               every generated clock value replaced by another one             yes    yes    yes    yes
    changed_computed_clock      a date the model never saw, in an environment that reads no     no     no     no     yes
                                clock (so it is computed, not generated), replaced by another
    reversed_scalars            a list of scalars reversed                                      yes    no     no     yes
    reordered_group_in_string   a [...] or {...} group inside a string reversed                 yes    no     no     yes
    changed_string              one string value altered                                        no     no     no     yes
    changed_number              one number altered                                              no     no     no     yes
    changed_known_id            a uuid the context holds replaced by a new one                  no     no     no     yes
    reused_context_id           a generated uuid replaced by a uuid the context already holds   no     no     no     yes
    merged_generated_ids        two generated uuids replaced by one                             no     no     no     yes
    split_generated_id          a generated uuid that occurs twice given two values             no     no     no     yes
    reversed_records            a list of records reversed                                      no     no     no     yes
    dropped_key / extra_key     a key removed from / added to the response                      no     no     no     yes
    flipped_success             the success flag negated                                        no     no     no     no
    bool_as_int                 the success flag as 1 / 0                                       no     no     no     no
    prose                       a sentence in front of the answer (not parseable)               no     no     no     no
    empty                       no prediction at all                                            no     no     no     no
"""

from __future__ import annotations

import argparse
import copy
import json
import pprint
import random
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trace2env.agentworld import clean_response_marker, wrap_prediction  # noqa: E402
from trace2env.envscaler_scoring import (UNPARSED, _GROUPS, generated_tokens, parse_observation, row_context,  # noqa: E402
                                         score_row)

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
YES, NO, ANY = True, False, None
EXPECT = {  # rule -> (match, match_strict, text_match, outcome_match)
    "identity": (YES, YES, YES, YES), "fenced": (YES, YES, YES, YES), "json": (YES, YES, NO, YES),
    "reversed_keys": (YES, YES, NO, YES), "pretty_printed": (YES, YES, ANY, YES),
    "fresh_generated_id": (YES, YES, YES, YES), "shifted_clock": (YES, YES, YES, YES), "changed_computed_clock": (NO, NO, NO, YES),
    "reversed_scalars": (YES, NO, NO, YES), "reordered_group_in_string": (YES, NO, NO, YES),
    "changed_string": (NO, NO, NO, YES), "changed_number": (NO, NO, NO, YES), "changed_known_id": (NO, NO, NO, YES),
    "reused_context_id": (NO, NO, NO, YES), "merged_generated_ids": (NO, NO, NO, YES), "split_generated_id": (NO, NO, NO, YES),
    "reversed_records": (NO, NO, NO, YES), "dropped_key": (NO, NO, NO, YES), "extra_key": (NO, NO, NO, YES),
    "flipped_success": (NO, NO, NO, NO), "bool_as_int": (NO, NO, NO, NO), "prose": (NO, NO, NO, NO), "empty": (NO, NO, NO, NO),
}


def paths(value, prefix=()):
    """Every (path, leaf-or-container) of a parsed observation."""
    yield prefix, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from paths(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from paths(item, prefix + (index,))


def replaced(value, path, new):
    value = copy.deepcopy(value)
    if not path:
        return new
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = new
    return value


def reversed_keys(value):
    if isinstance(value, dict):
        return {key: reversed_keys(value[key]) for key in reversed(list(value))}
    return [reversed_keys(item) for item in value] if isinstance(value, list) else value


def masked(text: str, tokens) -> str:
    for token in sorted({match.group(0) for match in tokens}, key=len, reverse=True):
        text = text.replace(token, "<generated>")
    return text


def mutations(row: dict, rng: random.Random) -> dict[str, str | None]:
    """rule -> predicted text for this row (rules that do not apply are absent; ``None`` means no prediction)."""
    truth_text = clean_response_marker(str(row["response"][int(row["turn_idx"]) - 1]))
    context = row_context(row)
    truth = parse_observation(truth_text)
    out: dict[str, str | None] = {"identity": truth_text, "fenced": f"```python\n{truth_text}\n```",
                                  "prose": "The environment returns: " + truth_text, "empty": None}
    if truth is UNPARSED:
        return out
    # What the environment can generate (the row's ``generated_shapes``); a token of any other shape that the model never
    # saw is computed, and changing it must cost the match.
    tokens = generated_tokens(truth_text, context, row.get("generated_shapes"))
    computed = [match.group(0) for match in generated_tokens(truth_text, context)
                if match.group(0) not in {token.group(0) for token in tokens}]

    def shifted(token: str) -> str:
        return re.sub(r"\d", lambda digit: str((int(digit.group(0)) + 1) % 10) if digit.start() >= len(token) - 2 else digit.group(0), token)

    def fresh() -> str:
        while True:
            candidate = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            if candidate not in context and candidate not in truth_text:
                return candidate

    out["json"] = json.dumps(truth, ensure_ascii=False)
    out["pretty_printed"] = pprint.pformat(truth, width=30, sort_dicts=False)
    if any(isinstance(item, dict) and len(item) >= 2 for _, item in paths(truth)):
        out["reversed_keys"] = repr(reversed_keys(truth))
    generated_ids = list(dict.fromkeys(match.group(0) for match in tokens if match.lastgroup == "uuid"))
    clocks = list(dict.fromkeys(match.group(0) for match in tokens if match.lastgroup != "uuid"))
    if generated_ids:
        text = truth_text
        for token in generated_ids:
            text = text.replace(token, fresh())
        out["fresh_generated_id"] = text
        known = [item for item in dict.fromkeys(UUID.findall(context)) if item not in truth_text]
        if known:
            out["reused_context_id"] = truth_text.replace(generated_ids[0], known[0])
        if len(generated_ids) >= 2:
            same = fresh()
            out["merged_generated_ids"] = truth_text.replace(generated_ids[0], same).replace(generated_ids[1], same)
        if truth_text.count(generated_ids[0]) >= 2:
            out["split_generated_id"] = truth_text.replace(generated_ids[0], fresh(), 1).replace(generated_ids[0], fresh())
    if clocks:
        text = truth_text
        for token in clocks:
            text = text.replace(token, shifted(token))
        if text != truth_text:
            out["shifted_clock"] = text
    if computed and computed[0] not in UUID.findall(truth_text):
        out["changed_computed_clock"] = truth_text.replace(computed[0], shifted(computed[0]))
    known_ids = [item for item in dict.fromkeys(UUID.findall(truth_text)) if item in context]
    if known_ids:
        out["changed_known_id"] = truth_text.replace(known_ids[0], fresh())
    for path, item in paths(truth):
        if isinstance(item, str) and "changed_string" not in out:
            out["changed_string"] = repr(replaced(truth, path, item + "~"))
        if isinstance(item, (int, float)) and not isinstance(item, bool) and "changed_number" not in out and repr(item) in context + truth_text:
            if not any(match.group(0) == repr(item) for match in tokens):
                out["changed_number"] = repr(replaced(truth, path, item + 1))
        if isinstance(item, list) and len(item) >= 2:
            scalars = all(not isinstance(element, (dict, list)) for element in item)
            flipped = list(reversed(item))
            if scalars and flipped != item and "reversed_scalars" not in out:
                out["reversed_scalars"] = repr(replaced(truth, path, flipped))
            # Two records that differ only in generated values are interchangeable: reversing them is not an error.
            if not scalars and "reversed_records" not in out and masked(repr(flipped), tokens) != masked(repr(item), tokens):
                out["reversed_records"] = repr(replaced(truth, path, flipped))
        if isinstance(item, str) and "reordered_group_in_string" not in out:
            group = next((m for m in _GROUPS.finditer(item) if len({part.strip() for part in (m.group(1) or m.group(2) or "").split(",")}) >= 2), None)
            if group:
                body = group.group(1) if group.group(1) is not None else group.group(2)
                parts = [part.strip() for part in body.split(",")]
                rebuilt = item[:group.start() + 1] + ", ".join(reversed(parts)) + item[group.end() - 1:]
                if rebuilt != item:
                    out["reordered_group_in_string"] = repr(replaced(truth, path, rebuilt))
    if isinstance(truth, dict):
        if isinstance(truth.get("success"), bool):
            out["flipped_success"] = repr({**truth, "success": not truth["success"]})
            out["bool_as_int"] = repr({**truth, "success": int(truth["success"])})
        others = [key for key in truth if key != "success"]
        if others:
            out["dropped_key"] = repr({key: item for key, item in truth.items() if key != others[-1]})
        out["extra_key"] = repr({**truth, "unexpected_key": 1})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="work/exp-envscaler")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rng = random.Random(20260920)
    applied: Counter[str] = Counter()
    wrong: list[dict] = []
    rows = 0
    for path in sorted(Path(args.experiment).glob("*/awb/rows.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows += 1
            for rule, text in mutations(row, rng).items():
                applied[rule] += 1
                score = score_row({**row, "gen": wrap_prediction(text) if text is not None else ""})
                got = (score["match"], score["match_strict"], score["text_match"], score["outcome_match"])
                if any(want is not ANY and want != have for want, have in zip(EXPECT[rule], got)):
                    wrong.append({"rule": rule, "env_id": row.get("env_id"), "id": row["id"], "turn_idx": row["turn_idx"],
                                  "probe": row.get("probe"), "expected": EXPECT[rule], "got": got,
                                  "truth": clean_response_marker(str(row["response"][int(row["turn_idx"]) - 1]))[:300],
                                  "prediction": (text or "")[:300]})
    report = {"rows": rows, "checks": sum(applied.values()), "wrong": len(wrong),
              "rules": {rule: {"applied": applied[rule], "wrong": sum(1 for item in wrong if item["rule"] == rule)} for rule in EXPECT},
              "wrong_verdicts": wrong[:50]}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'rule':<28}{'applied':>9}{'wrong':>7}")
    for rule, item in report["rules"].items():
        print(f"{rule:<28}{item['applied']:>9}{item['wrong']:>7}")
    print(f"rows {rows} | checks {report['checks']} | wrong verdicts {report['wrong']}")
    for item in wrong[:8]:
        print("WRONG", item["rule"], item["id"], item["turn_idx"], item["probe"], "expected", item["expected"], "got", item["got"])
        print("   truth:", item["truth"][:160]); print("   pred :", item["prediction"][:160])
    if wrong:
        sys.exit(1)


if __name__ == "__main__":
    main()
