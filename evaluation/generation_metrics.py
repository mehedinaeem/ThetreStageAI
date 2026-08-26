"""Objective generation measurements; subjective quality remains human-rated."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationMeasurements:
    validation_success: bool
    repair_attempts: int
    scene_count: int
    dialogue_count: int
    blocking_cue_count: int
    lighting_cue_count: int


def measure_generation(
    production: dict[str, Any] | None,
    *,
    validation_success: bool,
    repair_attempts: int,
) -> GenerationMeasurements:
    """Count observable structure without assigning subjective quality scores."""
    if repair_attempts < 0:
        raise ValueError("Repair attempts cannot be negative")
    scenes = production.get("scenes", []) if isinstance(production, dict) else []
    valid_scenes = [scene for scene in scenes if isinstance(scene, dict)]
    return GenerationMeasurements(
        validation_success=validation_success,
        repair_attempts=repair_attempts,
        scene_count=len(valid_scenes),
        dialogue_count=sum(len(_list(scene.get("dialogue"))) for scene in valid_scenes),
        blocking_cue_count=sum(len(_list(scene.get("blocking"))) for scene in valid_scenes),
        lighting_cue_count=sum(len(_list(scene.get("lighting"))) for scene in valid_scenes),
    )


def extract_generation_components(
    production: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate script, blocking, and lighting artifacts for experiment records."""
    if not isinstance(production, dict):
        return [], [], []
    script: list[dict[str, Any]] = []
    blocking: list[dict[str, Any]] = []
    lighting: list[dict[str, Any]] = []
    for scene in _list(production.get("scenes")):
        if not isinstance(scene, dict):
            continue
        scene_id = str(scene.get("id", ""))
        script.append(
            {
                "scene_id": scene_id,
                "title": scene.get("title", ""),
                "dialogue": _list(scene.get("dialogue")),
                "stage_directions": _list(scene.get("stage_directions")),
            }
        )
        blocking.extend(
            {"scene_id": scene_id, **cue}
            for cue in _list(scene.get("blocking"))
            if isinstance(cue, dict)
        )
        lighting.extend(
            {"scene_id": scene_id, **cue}
            for cue in _list(scene.get("lighting"))
            if isinstance(cue, dict)
        )
    return script, blocking, lighting


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
