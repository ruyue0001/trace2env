"""Exact scoring of predicted EnvScaler observations against the recorded (or oracle) text. No judge, no model.

An EnvScaler environment is a program: its observation is a deterministic function of the database and the
call, except for what the program draws from outside (fresh ids, the clock, the interpreter's hash seed). So a
prediction is checked, not graded. ``match`` is the accuracy; the rules, each verified on the collection by
re-executing the environments' source (``work/exp-envscaler/audit_scoring.py``):

* **Values, not text.** Both texts are parsed (Python literal, else JSON), so ``{'success': True}`` and
  ``{"success": true}`` are the same answer and key order does not matter; ``text_match`` separately says
  whether the recorded surface form was reproduced (whitespace aside). Types are strict (``True`` is not ``1``),
  strings are exact (case, punctuation: ``'... not found'`` and ``'... not found.'`` are different branches of
  one environment), extra or missing keys fail, and records in a list are compared in order: re-executing
  all 2394 recorded and probe steps under different hash seeds never reorders a list in an observation.
* **Generated values match by shape.** A uuid, clock text, or epoch number in the truth that occurs nowhere
  in what the model was shown (the row's prompts up to the evaluated turn and the earlier responses) cannot
  be predicted; any text of the same shape is accepted there while everything around it stays exact. The
  acceptance is a *binding*: one generated value keeps one predicted value throughout the response, two
  generated values get two different ones, and a predicted id must itself be new (an id the context
  already holds belongs to another record). A value the context does hold must be reproduced exactly.
  Only what the environment can generate is accepted this way: a row's ``generated_shapes`` (read off the
  environment source by the exporter) names the shapes, so a *computed* date the model has not seen, such as
  a registration deadline derived from a session's start in an environment that never reads the clock, is
  compared exactly (``work/exp-envscaler/verify_rows.py`` checks every accepted token by re-execution).
* **Hash-seed order is not a fact about the environment.** Some tools build a list from a ``set`` or print
  a ``set`` inside a message (``"Allowed: {'pending', 'active'}"``); that order changes from one interpreter
  run to the next. ``match`` compares scalar lists, and ``[...]`` / ``{...}`` groups inside strings, without
  order; ``match_strict`` keeps every order and is a lower bound.

``outcome_match`` is the coarse question only (the ``success`` flag). Row kinds name the *shape* of the true
response (``data`` / ``message`` / ``error`` / ``other``), which is all a row holds: every ``data`` response of
the collection left the database unchanged and every ``error`` was a rejection, but 81 ``message`` responses
changed nothing either, so ``message`` does not mean "state changed".

The report includes ``reads:excluded``: the rows whose true response is not ``data``. Under the former
state-visible protocol, successful reads often copied records from the initial database or earlier observations.
The MCP-aligned default keeps that database latent, so the same slice now separates non-read behavior from
otherwise unobservable task-specific records. A rejected lookup is an ``error`` and stays in.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from typing import Any, Iterable

from trace2env.agentworld import RESPONSE_TAG, clean_response_marker, parse_model_output

UNPARSED = object()
_GENERATED = re.compile(
    r"(?P<uuid>[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"|(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)"
    r"|(?P<date>\d{4}-\d{2}-\d{2})"
    r"|(?P<epoch>\b1[6-9]\d{8}(?:\.\d+)?\b)"
)
_SHAPES = {"uuid": r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
           "stamp": r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?", "date": r"\d{4}-\d{2}-\d{2}",
           "epoch": r"\d{9,11}(?:\.\d+)?"}
_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n(.*?)\n?```$", re.S)
_GROUPS = re.compile(r"\[([^\[\]{}]*)\]|\{([^\[\]{}]*)\}")
_WHITESPACE = re.compile(r"\s+")
METRICS = ("predicted", "parsed", "outcome_match", "match", "match_strict", "text_match")


def parse_observation(text: str) -> Any:
    """The value an observation text denotes (Python literal, else JSON); ``UNPARSED`` when it is neither."""
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        pass
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        return UNPARSED


class Bindings:
    """Which predicted value stands for which generated value of the truth, within one response."""

    def __init__(self, context: str, shapes: Iterable[str] | None = None):
        self.context = context
        self.shapes = None if shapes is None else frozenset(shapes)  # the shapes the environment can generate; None = any
        self.forward: dict[str, str] = {}
        self.backward: dict[str, str] = {}

    def copy(self) -> "Bindings":
        other = Bindings(self.context, self.shapes)
        other.forward, other.backward = dict(self.forward), dict(self.backward)
        return other

    def adopt(self, other: "Bindings") -> None:
        self.forward, self.backward = other.forward, other.backward

    def bind(self, truth: str, predicted: str, shape: str) -> bool:
        if shape == "uuid" and predicted != truth and predicted in self.context:
            return False  # a generated id is new; one the context already holds names another record
        if self.forward.get(truth, predicted) != predicted or self.backward.get(predicted, truth) != truth:
            return False  # one generated value, one predicted value, and the other way round
        self.forward[truth], self.backward[predicted] = predicted, truth
        return True


def generated_tokens(truth: str, context: str, shapes: Iterable[str] | None = None) -> list[re.Match[str]]:
    """The uuid / clock / epoch tokens of ``truth`` that occur nowhere in ``context``.

    ``shapes`` names what the environment can generate (a row's ``generated_shapes``): a token of another shape is a
    computed value, however unfamiliar, and stays exact. ``None`` keeps every shape.
    """
    return [match for match in _GENERATED.finditer(truth)
            if match.group(0) not in context and (shapes is None or match.lastgroup in shapes)]


def _unordered_groups(text: str) -> str:
    """``[...]`` and ``{...}`` groups inside a string with their comma-separated parts sorted."""
    def ordered(match: re.Match[str]) -> str:
        opening, closing = ("[", "]") if match.group(1) is not None else ("{", "}")
        body = match.group(1) if match.group(1) is not None else match.group(2)
        return opening + ", ".join(sorted(part.strip() for part in body.split(","))) + closing
    return _GROUPS.sub(ordered, text)


def _text_equal(truth: str, predicted: str, bindings: Bindings, strict: bool) -> bool:
    candidates = [(truth, predicted)] if strict else [(truth, predicted), (_unordered_groups(truth), _unordered_groups(predicted))]
    for expected, actual in candidates:
        tokens = generated_tokens(expected, bindings.context, bindings.shapes)
        if not tokens:
            if expected == actual:
                return True
            continue
        parts, cursor = [], 0
        for index, token in enumerate(tokens):
            parts += [re.escape(expected[cursor:token.start()]), f"(?P<g{index}>{_SHAPES[token.lastgroup or 'uuid']})"]
            cursor = token.end()
        found = re.compile("".join(parts) + re.escape(expected[cursor:]), re.S).fullmatch(actual)
        if not found:
            continue
        trial = bindings.copy()
        if all(trial.bind(token.group(0), found.group(f"g{index}"), token.lastgroup or "uuid") for index, token in enumerate(tokens)):
            bindings.adopt(trial)
            return True
    return False


def _equal(truth: Any, predicted: Any, bindings: Bindings, strict: bool) -> bool:
    if isinstance(truth, bool) or isinstance(predicted, bool) or truth is None or predicted is None:
        return truth is predicted
    if isinstance(truth, dict):
        if not isinstance(predicted, dict) or len(truth) != len(predicted):
            return False
        trial = bindings.copy()
        remaining = {str(key): value for key, value in predicted.items()}  # JSON keys are text; compare keys as text
        if len(remaining) != len(predicted):
            return False
        for key, value in truth.items():
            # The same key, or (a record filed under a generated id) any key of that shape whose value also agrees.
            for other in ([str(key)] if str(key) in remaining else list(remaining)):
                attempt = trial.copy()
                if _text_equal(str(key), other, attempt, strict) and _equal(value, remaining[other], attempt, strict):
                    trial.adopt(attempt)
                    del remaining[other]
                    break
            else:
                return False
        bindings.adopt(trial)
        return True
    if isinstance(truth, (list, tuple)):
        if not isinstance(predicted, (list, tuple)) or len(truth) != len(predicted):
            return False
        trial = bindings.copy()
        if strict or any(isinstance(item, (dict, list, tuple)) for item in truth):
            if not all(_equal(a, b, trial, strict) for a, b in zip(truth, predicted)):
                return False
        else:
            remaining_items = list(predicted)
            for item in truth:
                for index, other in enumerate(remaining_items):
                    attempt = trial.copy()
                    if _equal(item, other, attempt, strict):
                        trial.adopt(attempt)
                        remaining_items.pop(index)
                        break
                else:
                    return False
        bindings.adopt(trial)
        return True
    if isinstance(truth, str):
        return isinstance(predicted, str) and _text_equal(truth, predicted, bindings, strict)
    if isinstance(truth, (int, float)):
        if not isinstance(predicted, (int, float)):
            return False
        generated = _GENERATED.fullmatch(repr(truth)) is not None and repr(truth) not in bindings.context
        return generated or truth == predicted
    return truth == predicted


def values_equal(truth: Any, predicted: Any, context: str = "", *, strict: bool = False,
                 shapes: Iterable[str] | None = None) -> bool:
    """Equality of two parsed observations under the rules in the module docstring."""
    return _equal(truth, predicted, Bindings(context, shapes), strict)


def kind_of(value: Any) -> str:
    """The shape of a response: ``error`` (``success`` false), ``data``, ``message`` (any other dict), ``other``."""
    if not isinstance(value, dict):
        return "other"
    return "error" if value.get("success") is False else "data" if "data" in value else "message"


def row_context(row: dict[str, Any]) -> str:
    """What the model was shown for the evaluated turn: every prompt through it and the earlier responses."""
    turn = int(row["turn_idx"])
    return "\n".join([*(str(item) for item in row["prompt"][:turn]), *(str(item) for item in row["response"][:turn - 1])])


def score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Compare a prediction row (``gen``) with the row's own ground truth for the evaluated turn."""
    truth_text = clean_response_marker(str(row["response"][int(row["turn_idx"]) - 1]))
    context = row_context(row)
    truth = parse_observation(truth_text)
    shapes = row.get("generated_shapes")  # absent in rows exported before the field existed: every shape
    score: dict[str, Any] = {"kind": kind_of(truth), "generated": bool(generated_tokens(truth_text, context, shapes)),
                             **{metric: False for metric in METRICS}}
    generation = row.get("gen") or ""
    if not generation:
        return score
    predicted_text = parse_model_output(generation, RESPONSE_TAG).strip()
    fenced = _FENCE.match(predicted_text)
    if fenced:
        predicted_text = fenced.group(1).strip()
    predicted = parse_observation(predicted_text)
    score["predicted"] = True
    score["parsed"] = predicted is not UNPARSED
    expected_text, actual_text = (_WHITESPACE.sub(" ", text).strip() for text in (truth_text, predicted_text))
    score["text_match"] = _text_equal(expected_text, actual_text, Bindings(context, shapes), strict=True)
    if truth is UNPARSED or predicted is UNPARSED:
        # Not a value on one side: only the texts can agree.
        score["match"] = score["match_strict"] = score["outcome_match"] = score["text_match"]
        return score
    score["match_strict"] = values_equal(truth, predicted, context, strict=True, shapes=shapes)
    score["match"] = score["match_strict"] or values_equal(truth, predicted, context, strict=False, shapes=shapes)
    if isinstance(truth, dict) and isinstance(predicted, dict):
        score["outcome_match"] = truth.get("success") is predicted.get("success")
    else:
        # No accept/reject flag on one side (a call the environment does not have returns ``None``).
        score["outcome_match"] = score["match"] and not isinstance(truth, dict) and not isinstance(predicted, dict)
    return score


def score_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows with an ``exact`` entry added; the input rows are not modified."""
    return [{**row, "exact": score_row(row)} for row in rows]


def aggregate(scored: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rates per metric: overall, per environment, per response kind, recorded turns vs probes, and their crossings.

    ``reads:excluded`` (the reported accuracy) leaves out the rows whose true response is ``data``.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        score = row["exact"]
        env = str(row.get("env_id") or "unknown")
        source = "probe" if row.get("probe") else "recorded"
        names = ["all", f"env:{env}", f"kind:{score['kind']}", f"set:{source}", f"env:{env}|set:{source}",
                 f"set:{source}|kind:{score['kind']}"]
        if score["kind"] != "data":
            names += ["reads:excluded", f"env:{env}|reads:excluded", f"set:{source}|reads:excluded"]
        for name in names:
            groups[name].append(score)
    return {name: {"rows": len(items), "generated": sum(1 for item in items if item["generated"]),
                   **{metric: round(100 * sum(1 for item in items if item[metric]) / len(items), 2) for metric in METRICS}}
            for name, items in sorted(groups.items())}


def format_report(summary: dict[str, Any]) -> str:
    header = f"{'group':<44}{'rows':>6}" + "".join(f"{metric:>15}" for metric in METRICS)
    lines = ["EnvScaler exact scoring (percent of rows; match is the accuracy)", header, "-" * len(header)]
    for name, entry in summary.items():
        lines.append(f"{name:<44}{entry['rows']:>6}" + "".join(f"{entry[metric]:>15.1f}" for metric in METRICS))
    return "\n".join(lines)
