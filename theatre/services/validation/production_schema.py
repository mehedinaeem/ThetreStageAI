"""Strict, cross-validated schema for generated theatre productions."""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

ALLOWED_TRIGGERS = frozenset({"scene_start", "scene_end"})


class StageZone(StrEnum):
    USL = "USL"
    USC = "USC"
    USR = "USR"
    CSL = "CSL"
    CSC = "CSC"
    CSR = "CSR"
    DSL = "DSL"
    DSC = "DSC"
    DSR = "DSR"


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Character(StrictSchema):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Dialogue(StrictSchema):
    id: str = Field(min_length=1)
    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class BlockingCue(StrictSchema):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    actor: str = Field(min_length=1)
    from_zone: StageZone = Field(alias="from")
    to: StageZone
    action: str = Field(min_length=1)
    trigger: str = Field(min_length=1)


RGBValue = Annotated[int, Field(ge=0, le=255)]


class LightingCue(StrictSchema):
    cue_id: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    fixture: str = Field(min_length=1)
    focus_zone: StageZone
    rgb: tuple[RGBValue, RGBValue, RGBValue]
    intensity: int = Field(ge=0, le=100)
    fade_seconds: float = Field(ge=0)


class SoundCue(StrictSchema):
    cue_id: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    sound: str = Field(min_length=1)
    volume: float = Field(ge=0, le=1)


class Scene(StrictSchema):
    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    location: str = Field(min_length=1)
    time: str = Field(min_length=1)
    dialogue: list[Dialogue] = Field(min_length=1)
    stage_directions: list[str] = Field(min_length=1)
    blocking: list[BlockingCue] = Field(min_length=1)
    lighting: list[LightingCue] = Field(min_length=1)
    sound: list[SoundCue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scene_references(self) -> Self:
        dialogue_ids = [line.id for line in self.dialogue]
        self._require_unique(dialogue_ids, "Dialogue IDs")
        self._require_unique([cue.cue_id for cue in self.lighting], "Lighting cue IDs")
        self._require_unique([cue.cue_id for cue in self.sound], "Sound cue IDs")
        valid_triggers = ALLOWED_TRIGGERS | set(dialogue_ids)
        for cue_type, cues in (
            ("blocking", self.blocking),
            ("lighting", self.lighting),
            ("sound", self.sound),
        ):
            for cue in cues:
                if cue.trigger not in valid_triggers:
                    raise ValueError(
                        f"Invalid {cue_type} trigger '{cue.trigger}'; expected scene_start, "
                        "scene_end, or a dialogue ID in this scene"
                    )
        return self

    @staticmethod
    def _require_unique(values: list[str], label: str) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for value in values:
            if value in seen:
                duplicates.add(value)
            seen.add(value)
        if duplicates:
            raise ValueError(f"{label} must be unique inside a scene: {sorted(duplicates)}")


class Production(StrictSchema):
    title: str = Field(min_length=1)
    theme: str = Field(min_length=1)
    genre: str = Field(min_length=1)
    characters: list[Character] = Field(min_length=1)
    scenes: list[Scene] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_character_references(self) -> Self:
        character_names = [character.name for character in self.characters]
        Scene._require_unique(character_names, "Character names")
        known_characters = set(character_names)
        for scene in self.scenes:
            for line in scene.dialogue:
                if line.speaker not in known_characters:
                    raise ValueError(
                        f"Dialogue speaker '{line.speaker}' is not in the character list "
                        f"(scene '{scene.id}')"
                    )
            for cue in scene.blocking:
                if cue.actor not in known_characters:
                    raise ValueError(
                        f"Blocking actor '{cue.actor}' is not in the character list "
                        f"(scene '{scene.id}')"
                    )
        return self

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        return cls.model_json_schema(by_alias=True)
