"""Small, auditable storage helpers used by packages and session workspaces."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(to_jsonable(value), ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    lines = [json.dumps(to_jsonable(value), ensure_ascii=False) for value in values]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, value: Any) -> None:
    """Append one durable audit entry. Session state itself is written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(to_jsonable(value), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_models(path: Path, model: type[T]) -> list[T]:
    return [model.model_validate(item) for item in read_jsonl(path)]
