"""Pydantic schemas and generated-output validation boundary."""

from .constraint_validator import ConstraintValidator
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
from .utils import make_json_safe

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
    "ConstraintValidator",
    "OutputValidationResult",
    "Production",
    "ProductionValidationError",
    "Scene",
    "SoundCue",
    "StageZone",
    "TheatreScene",
    "make_json_safe",
]
