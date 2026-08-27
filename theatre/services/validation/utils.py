"""JSON-safe normalization helpers for validation and experiment evidence."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Recursively convert arbitrary values into JSON-compatible primitives."""
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, Enum):
        return make_json_safe(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        normalized: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = make_json_safe(key)
            if not isinstance(safe_key, (str, int, float, bool)) and safe_key is not None:
                safe_key = str(safe_key)
            normalized[safe_key] = make_json_safe(item)
        return normalized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [make_json_safe(item) for item in value]
    return str(value)
