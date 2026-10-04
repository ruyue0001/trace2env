"""Loss-minimizing adapters from common raw trace shapes into Trace2Env events."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from trace2env.models import Actor, Episode, EventKind, RawEvent, TraceAnnotations, TransitionSlice

_TIMESTAMP = TypeAdapter(datetime | None)


def _parse_timestamp(value: Any, metadata: dict[str, Any]) -> datetime | None:
    """Keep unparsable timestamps as metadata instead of rejecting the whole event."""
    try:
        return _TIMESTAMP.validate_python(value)
    except ValidationError:
        metadata["raw_timestamp"] = value
        return None


ROLE_MAP = {
    "human": Actor.USER,
    "user": Actor.USER,
    "assistant": Actor.AGENT,
    "agent": Actor.AGENT,
    "environment": Actor.ENVIRONMENT,
    "observation": Actor.ENVIRONMENT,
    "tool": Actor.TOOL,
    "system": Actor.SYSTEM,
}


def stable_id(prefix: str, *parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(raw).hexdigest()[:16]}"


def _event_from_mapping(item: dict[str, Any], episode_id: str, index: int) -> list[RawEvent]:
    role = str(item.get("role", item.get("actor", item.get("source", "unknown")))).lower()
    actor = ROLE_MAP.get(role, Actor.UNKNOWN)
    explicit_kind = str(item.get("kind", item.get("type", ""))).lower()
    content = item.get("content", item.get("text", item.get("value", item)))
    event_id = stable_id("evt", episode_id, index, item.get("id"), item)
    metadata = dict(item.get("metadata", {}))
    metadata["original_event_id"] = item.get("id")
    for key in ("tool_call_id", "call_id", "name"):
        if key in item:
            metadata[key] = item[key]
    timestamp = _parse_timestamp(item.get("timestamp"), metadata)

    if explicit_kind in EventKind._value2member_map_:
        kind = EventKind(explicit_kind)
    elif explicit_kind in {"tool_call", "function_call", "command", "input", "request"}:
        kind = EventKind.ACTION
    elif explicit_kind in {"tool_result", "function_result", "output", "response", "result"}:
        kind = EventKind.OBSERVATION
    elif actor in (Actor.TOOL, Actor.ENVIRONMENT):
        kind = EventKind.OBSERVATION
    elif item.get("tool_calls") or item.get("action") is not None:
        kind = EventKind.ACTION
        content = item.get("action", {"tool_calls": item.get("tool_calls"), "content": content})
    elif actor == Actor.AGENT and isinstance(content, dict) and any(
        key in content for key in ("action", "tool", "command", "arguments")
    ):
        kind = EventKind.ACTION
    elif actor == Actor.AGENT and isinstance(content, str) and re.match(
        r"^\s*(action|tool|command)\s*:", content, re.IGNORECASE
    ):
        kind = EventKind.ACTION
    else:
        kind = EventKind.MESSAGE

    events = [
        RawEvent(
            id=event_id,
            actor=actor,
            kind=kind,
            content=content,
            timestamp=timestamp,
            metadata=metadata,
            raw_payload=item,
        )
    ]

    # Some APIs embed multiple calls in an assistant message. Preserve each call as an action.
    if item.get("tool_calls"):
        events = []
        for call_index, call in enumerate(item["tool_calls"]):
            events.append(
                RawEvent(
                    id=f"{event_id}_call_{call_index}",
                    actor=Actor.AGENT,
                    kind=EventKind.ACTION,
                    content=call,
                    timestamp=timestamp,
                    metadata={**metadata, "parent_event_id": event_id, "call_id": call.get("id"),
                              "batch_size": len(item["tool_calls"])},
                    raw_payload=call,
                )
            )
    return events


def _episode_from_object(obj: Any, source_id: str, ordinal: int = 0) -> Episode:
    if isinstance(obj, list):
        items, metadata = obj, {}
    elif isinstance(obj, dict):
        items = obj.get("events", obj.get("messages", obj.get("trajectory", obj.get("trace"))))
        if items is None:
            observation_key = next(
                (key for key in ("observation", "tool_result", "environment_response") if key in obj),
                None,
            )
            if "action" in obj and observation_key:
                items = [
                    {"actor": "agent", "kind": "action", "content": obj["action"]},
                    {"actor": "environment", "kind": "observation", "content": obj[observation_key]},
                ]
            else:
                items = [obj]
        metadata = {k: v for k, v in obj.items() if k not in {"events", "messages", "trajectory", "trace"}}
    else:
        items, metadata = [obj], {}

    original_id = metadata.get("episode_id", metadata.get("id"))
    episode_id = stable_id("ep", source_id, ordinal, original_id)
    metadata["original_episode_id"] = original_id
    events: list[RawEvent] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            observation_key = next(
                (key for key in ("observation", "tool_result", "environment_response") if key in item),
                None,
            )
            if "action" in item and observation_key:
                events.extend(
                    _event_from_mapping(
                        {"actor": "agent", "kind": "action", "content": item["action"]},
                        episode_id,
                        index * 2,
                    )
                )
                events.extend(
                    _event_from_mapping(
                        {"actor": "environment", "kind": "observation", "content": item[observation_key]},
                        episode_id,
                        index * 2 + 1,
                    )
                )
            else:
                events.extend(_event_from_mapping(item, episode_id, index))
        else:
            events.append(
                RawEvent(
                    id=stable_id("evt", episode_id, index, item),
                    actor=Actor.UNKNOWN,
                    kind=EventKind.MESSAGE,
                    content=item,
                )
            )
    return Episode(id=episode_id, source_id=source_id, events=events, metadata=metadata)


def load_raw_traces(path: str | Path) -> list[Episode]:
    trace_path = Path(path)
    raw = trace_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    episodes = _load_raw_traces(trace_path, raw.decode("utf-8"), f"src_{digest}")
    for episode in episodes:
        episode.metadata.update(source_sha256=digest, source_locator=str(trace_path.resolve()))
    return episodes


def _load_raw_traces(trace_path: Path, text: str, source_id: str) -> list[Episode]:
    """Load JSON, JSONL, or role-prefixed plain text without discarding raw content."""
    suffix = trace_path.suffix.lower()

    if suffix == ".json":
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("episodes"), list):
            objects = value["episodes"]
        elif isinstance(value, list) and value and all(
            isinstance(item, dict) and any(key in item for key in ("events", "messages", "trajectory", "trace"))
            for item in value
        ):
            objects = value
        else:
            objects = [value]
        return [_episode_from_object(obj, source_id, index) for index, obj in enumerate(objects)]

    if suffix == ".jsonl":
        objects = [json.loads(line) for line in text.splitlines() if line.strip()]
        # Event-per-line logs commonly share an episode key; group them.
        if objects and all(isinstance(item, dict) for item in objects) and all(
            not any(key in item for key in ("events", "messages", "trajectory", "trace")) for item in objects
        ):
            groups: dict[str, list[dict[str, Any]]] = {}
            for item in objects:
                key = str(item.get("episode_id", item.get("trajectory_id", "episode-0")))
                groups.setdefault(key, []).append(item)
            return [
                _episode_from_object({"episode_id": key, "events": events}, source_id, index)
                for index, (key, events) in enumerate(groups.items())
            ]
        return [_episode_from_object(obj, source_id, index) for index, obj in enumerate(objects)]

    return [_parse_plain_text(text, source_id)]


LINE_PREFIX = re.compile(
    r"^(?P<label>user|human|assistant|agent|action|observation|environment|tool|system|error)\s*:\s*(?P<body>.*)$",
    re.IGNORECASE,
)


def _parse_plain_text(text: str, source_id: str) -> Episode:
    episode_id = stable_id("ep", source_id, 0)
    events: list[RawEvent] = []
    current_label = "unknown"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if not body:
            buffer = []
            return
        label = current_label.lower()
        actor = ROLE_MAP.get(label, Actor.AGENT if label == "action" else Actor.UNKNOWN)
        if label == "action":
            kind = EventKind.ACTION
        elif label in {"observation", "environment", "tool"}:
            kind = EventKind.OBSERVATION
        elif label == "error":
            kind = EventKind.ERROR
        else:
            kind = EventKind.MESSAGE
        events.append(
            RawEvent(
                id=stable_id("evt", episode_id, len(events), label, body),
                actor=actor,
                kind=kind,
                content=body,
            )
        )
        buffer = []

    for line in text.splitlines():
        match = LINE_PREFIX.match(line)
        if match:
            flush()
            current_label = match.group("label")
            buffer = [match.group("body")]
        else:
            buffer.append(line)
    flush()
    if not events:
        events.append(
            RawEvent(id=stable_id("evt", episode_id, 0), content=text, kind=EventKind.MESSAGE)
        )
    return Episode(id=episode_id, source_id=source_id, events=events)


def segment_transitions(episode: Episode, max_history_events: int = 20) -> list[TransitionSlice]:
    """Prefer explicit correlation; never attribute another call's reply by proximity."""
    transitions: list[TransitionSlice] = []
    events = episode.events
    action_positions = [index for index, event in enumerate(events) if event.kind == EventKind.ACTION]
    call_positions: dict[str, list[int]] = {}
    for index in action_positions:
        call_id = events[index].metadata.get("call_id")
        if call_id:
            call_positions.setdefault(str(call_id), []).append(index)
    for ordinal, start in enumerate(action_positions):
        end = action_positions[ordinal + 1] if ordinal + 1 < len(action_positions) else len(events)
        action = events[start]
        call_id = action.metadata.get("call_id")
        ambiguities = []
        concurrent = []
        alignment = "sequential"
        def is_observation(event):
            return (event.kind in {EventKind.OBSERVATION, EventKind.ERROR, EventKind.STATE}
                    or event.actor in {Actor.ENVIRONMENT, Actor.TOOL})
        if call_id:
            alignment = "correlated"
            matches = [i for i in range(start + 1, len(events))
                       if is_observation(events[i])
                       and str(events[i].metadata.get("tool_call_id", events[i].metadata.get("call_id", ""))) == str(call_id)]
            if len(call_positions[str(call_id)]) != 1:
                matches = []
                ambiguities.append("Duplicate call ID prevents unambiguous result matching.")
            concurrent = [events[i].id for i in action_positions if start < i < max(matches, default=start)]
            # Earlier calls may still be pending even when issued in another message.
            for previous in action_positions:
                previous_id = events[previous].metadata.get("call_id")
                if previous >= start or not previous_id:
                    continue
                replies = [i for i in range(previous + 1, len(events)) if is_observation(events[i])
                           and str(events[i].metadata.get("tool_call_id", events[i].metadata.get("call_id", ""))) == str(previous_id)]
                if not replies or max(replies) > start:
                    concurrent.append(events[previous].id)
            parent = action.metadata.get("parent_event_id")
            concurrent += [events[i].id for i in action_positions if i != start and parent
                           and events[i].metadata.get("parent_event_id") == parent]
            if concurrent:
                ambiguities.append("Overlapping calls: result attribution does not establish mutation order.")
        elif action.metadata.get("batch_size", 1) > 1:
            matches = []
            ambiguities.append("Concurrent calls have no correlation IDs; outputs remain unassigned.")
        else:
            matches = [i for i in range(start + 1, end) if is_observation(events[i])
                       and not events[i].metadata.get("tool_call_id") and not events[i].metadata.get("call_id")]
        observation_ids = [events[i].id for i in matches]
        if not observation_ids:
            alignment = "ambiguous"
            ambiguities.append("No unambiguous observation is available for this action.")
        action_ids = [events[start].id]
        history_start = max(0, start - max_history_events)
        transition_id = stable_id("tr", episode.id, action_ids, observation_ids)
        transitions.append(
            TransitionSlice(
                id=transition_id,
                episode_id=episode.id,
                history_event_ids=[event.id for event in events[history_start:start]],
                action_event_ids=action_ids,
                observation_event_ids=observation_ids,
                delayed_observation_event_ids=[events[i].id for i in matches if i >= end],
                concurrent_action_event_ids=list(dict.fromkeys(concurrent)),
                alignment=alignment,
                ambiguities=ambiguities,
            )
        )
    return transitions


def annotate_episode(episode: Episode, result: TraceAnnotations) -> Episode:
    """Apply type/role annotations to exact source spans; preserve all unclassified text."""
    by_id = {event.id: event for event in episode.events}
    groups = {}
    for annotation in result.annotations:
        if annotation.source_event_id not in by_id:
            raise ValueError("Normalizer referenced an unknown source event")
        groups.setdefault(annotation.source_event_id, []).append(annotation)
    output = []
    for event in episode.events:
        annotations = groups.get(event.id, [])
        if not annotations:
            output.append(event)
            continue
        if not isinstance(event.content, str):
            if len(annotations) != 1 or annotations[0].start is not None or annotations[0].end is not None:
                raise ValueError("Non-text events require a single whole-event annotation")
            a = annotations[0]
            output.append(event.model_copy(update={"actor": a.actor, "kind": a.kind}))
            continue
        spans = []
        for a in annotations:
            start = a.start if a.start is not None else 0
            end = a.end if a.end is not None else len(event.content)
            if not 0 <= start < end <= len(event.content):
                raise ValueError("Normalizer span is outside the source text")
            spans.append((start, end, a))
        cursor = 0
        for start, end, a in sorted(spans, key=lambda value: value[0]):
            if start < cursor:
                raise ValueError("Normalizer spans overlap")
            if start > cursor:
                output.append(event.model_copy(update={"id": stable_id("evt", event.id, cursor, start),
                    "content": event.content[cursor:start], "kind": EventKind.MESSAGE}))
            output.append(event.model_copy(update={"id": stable_id("evt", event.id, start, end),
                "actor": a.actor, "kind": a.kind, "content": event.content[start:end],
                "metadata": {**event.metadata, "source_event_id": event.id, "source_span": [start, end]}}))
            cursor = end
        if cursor < len(event.content):
            output.append(event.model_copy(update={"id": stable_id("evt", event.id, cursor, len(event.content)),
                "content": event.content[cursor:], "kind": EventKind.MESSAGE}))
    return episode.model_copy(update={"events": output})


def events_for_slice(episode: Episode, transition: TransitionSlice) -> dict[str, list[RawEvent]]:
    by_id = {event.id: event for event in episode.events}
    return {
        "history": [by_id[item] for item in transition.history_event_ids],
        "actions": [by_id[item] for item in transition.action_event_ids],
        "observations": [by_id[item] for item in transition.observation_event_ids],
        "delayed_observations": [by_id[item] for item in transition.delayed_observation_event_ids],
    }
