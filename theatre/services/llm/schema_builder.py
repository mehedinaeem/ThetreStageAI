"""Build request-specific structured-output schemas without mutating Pydantic state."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DurationMinimums:
    scenes: int
    dialogue_per_scene: int
    total_dialogue: int


def duration_minimums(duration_minutes: int) -> DurationMinimums:
    """Return generation minimums aligned with final semantic validation."""
    if duration_minutes < 1:
        raise ValueError("Duration must be at least one minute")
    if duration_minutes <= 5:
        return DurationMinimums(1, 8, 8)
    if duration_minutes <= 10:
        return DurationMinimums(2, 9, 18)
    if duration_minutes <= 20:
        return DurationMinimums(3, 10, 30)
    return DurationMinimums(4, 10, 40)


def build_generation_schema(
    base_schema: dict[str, Any],
    *,
    actor_count: int,
    duration_minutes: int,
    available_lights: list[str],
) -> dict[str, Any]:
    """Apply user constraints to an isolated copy of the Production schema."""
    if actor_count < 1:
        raise ValueError("Actor count must be at least one")
    schema = deepcopy(base_schema)
    minimums = duration_minimums(duration_minutes)

    characters = schema["properties"]["characters"]
    characters["minItems"] = actor_count
    characters["maxItems"] = actor_count
    schema["properties"]["scenes"]["minItems"] = minimums.scenes
    schema["$defs"]["Scene"]["properties"]["dialogue"]["minItems"] = (
        minimums.dialogue_per_scene
    )

    fixtures = [str(item).strip() for item in available_lights if str(item).strip()]
    if fixtures:
        schema["$defs"]["LightingCue"]["properties"]["fixture"]["enum"] = fixtures
    return schema
