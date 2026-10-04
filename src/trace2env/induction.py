"""Bounded map/reduce calls. Oversized individual records fail instead of being truncated."""
from __future__ import annotations

import json
from typing import Any, Callable

from trace2env.llm import ProviderRefusal


class PromptBudgetExceeded(ValueError):
    pass


class BoundedInduction:
    """Batches records under a byte and count budget, consolidates partial results recursively, and
    isolates records the model provider refuses to analyze (content policy) instead of failing the
    whole stage: suspect records (``suspect(record)`` — e.g. from episodes already refused at
    extraction) are dropped first, then the chunk is bisected; every excluded record lands in
    ``refused`` for the caller to report."""

    def __init__(self, llm, max_bytes: int, batch_size: int, suspect: Callable[[dict], bool] | None = None):
        self.llm, self.max_bytes, self.batch_size = llm, max_bytes, batch_size
        self.suspect = suspect or (lambda record: False)
        self.refused: list[Any] = []

    def size(self, system, payload, model):
        return len((system + json.dumps(payload, ensure_ascii=False, default=str)
                    + json.dumps(model.model_json_schema())).encode("utf-8"))

    def call(self, system, payload, model, role):
        if self.size(system, payload, model) > self.max_bytes:
            raise PromptBudgetExceeded(f"{role} exceeds the configured prompt byte budget")
        return self.llm.complete(system=system, user=json.dumps(payload, ensure_ascii=False, default=str),
                                 response_model=model, role=role)

    def groups(self, values, system, payload_for, model):
        groups, current = [], []
        for value in values:
            if current and (len(current) >= self.batch_size or self.size(system, payload_for(current + [value]), model) > self.max_bytes):
                groups.append(current)
                current = []
            if self.size(system, payload_for([value]), model) > self.max_bytes:
                raise PromptBudgetExceeded("An individual evidence/artifact record exceeds the prompt byte budget")
            current.append(value)
        if current:
            groups.append(current)
        return groups

    def _results(self, group, system, payload_for, model, role, *, refused_already=False, use_suspect=True):
        """Partial results for one chunk, working around provider refusals by exclusion and bisection."""
        if not refused_already:
            try:
                return [self.call(system, payload_for(group), model, role).model_dump(mode="json")]
            except ProviderRefusal:
                pass
        if use_suspect:
            kept = [record for record in group if not self.suspect(record)]
            dropped = [record for record in group if self.suspect(record)]
            if dropped:
                self.refused.extend(dropped)
                if not kept:
                    return []
                try:
                    return [self.call(system, payload_for(kept), model, role).model_dump(mode="json")]
                except ProviderRefusal:
                    group = kept
        if len(group) == 1:
            self.refused.append(group[0])
            return []
        middle = len(group) // 2
        return (self._results(group[:middle], system, payload_for, model, role, use_suspect=False)
                + self._results(group[middle:], system, payload_for, model, role, use_suspect=False))

    def induce(self, values, system, model, role, base, field, merged_field):
        payload_for = lambda group: {**base, field: group}
        groups = self.groups(values, system, payload_for, model)
        self.refused = []
        if len(groups) <= 1:
            try:
                return self.call(system, payload_for(values), model, role)
            except ProviderRefusal:
                partials = self._results(list(values), system, payload_for, model, role + "_chunk", refused_already=True)
        else:
            partials = [partial for group in groups
                        for partial in self._results(group, system, payload_for, model, role + "_chunk")]
        if not partials:
            raise ProviderRefusal(f"The model provider refused every record supplied to {role}")
        if len(partials) == 1:
            return model.model_validate(partials[0])
        return self.reduce(partials, system, model, role, base, merged_field)

    def reduce(self, partials, system, model, role, base, field):
        payload_for = lambda group: {**base, field: group,
            "instruction": "Consolidate without losing evidence anchors, scoped branches, or unresolved conflicts."}
        for _ in range(32):
            if len(partials) == 1:
                return model.model_validate(partials[0])
            groups = self.groups(partials, system, payload_for, model)
            if len(groups) <= 1:
                try:
                    return self.call(system, payload_for(partials), model, role)
                except ProviderRefusal:
                    merged = self._results(list(partials), system, payload_for, model, role + "_merge",
                                           refused_already=True, use_suspect=False)
                    if not merged:
                        raise
                    if len(merged) == 1:
                        return model.model_validate(merged[0])
                    partials = merged
                    continue
            if len(groups) >= len(partials):
                raise PromptBudgetExceeded(
                    f"Consolidation cannot make progress within the byte budget ({self.max_bytes} bytes): "
                    f"{len(partials)} partial results cannot be paired; raise --max-prompt-bytes for a "
                    "long-context model or reduce the corpus"
                )
            partials = [partial for group in groups
                        for partial in self._results(group, system, payload_for, model, role + "_merge", use_suspect=False)]
            if not partials:
                raise ProviderRefusal(f"The model provider refused every consolidation input for {role}")
        raise PromptBudgetExceeded("Consolidation depth exceeded")
