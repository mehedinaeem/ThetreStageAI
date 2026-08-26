"""Backward-compatible imports for the canonical production schema."""

from .production_schema import (
    BlockingCue,
    Character,
    Dialogue as DialogueLine,
    LightingCue,
    Production as GeneratedTheatreProduction,
    Scene as TheatreScene,
    SoundCue,
    StageZone,
)

__all__ = [
    "BlockingCue",
    "Character",
    "DialogueLine",
    "GeneratedTheatreProduction",
    "LightingCue",
    "SoundCue",
    "StageZone",
    "TheatreScene",
]
