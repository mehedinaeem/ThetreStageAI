"""Pydantic schemas and generated-output validation boundary."""

from .output_validator import OutputValidationResult, OutputValidator, ProductionValidationError
from .production_schema import (
    BlockingCue,
    Character,
    Dialogue,
    LightingCue,
    Production,
    Scene,
    SoundCue,
    StageZone,
)

DialogueLine = Dialogue
GeneratedTheatreProduction = Production
TheatreScene = Scene

__all__ = [
    "BlockingCue",
    "Character",
    "Dialogue",
    "DialogueLine",
    "GeneratedTheatreProduction",
    "LightingCue",
    "OutputValidator",
    "OutputValidationResult",
    "Production",
    "ProductionValidationError",
    "Scene",
    "SoundCue",
    "StageZone",
    "TheatreScene",
]
