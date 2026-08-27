"""Validate a structurally valid production against the user's requirements."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .production_schema import Production

FORBIDDEN_MODEL_TOKENS = (
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<analysis>",
    "</analysis>",
    "<assistant>",
    "<system>",
    "<user>",
)
DATASET_TITLE_PATTERN = re.compile(r"ছোট\s*নাটক\s*\d+", re.IGNORECASE)
WORD_PATTERN = re.compile(r"[\w\u0980-\u09ff]+", re.UNICODE)
THEME_STOPWORDS = frozenset(
    {"and", "the", "use", "of", "ও", "এবং", "ব্যবহার", "সম্পর্কিত"}
)
THEME_CONCEPTS = (
    frozenset({"fake", "news", "misinformation", "ভুয়া", "ভুয়া", "খবর", "তথ্য"}),
    frozenset({"social", "media", "online", "সামাজিক", "মিডিয়া", "মিডিয়া", "অনলাইন"}),
)


class ConstraintValidator:
    """Return semantic violations without mutating the production."""

    def validate(
        self,
        production: Production,
        requirements: Mapping[str, Any],
    ) -> list[str]:
        errors: list[str] = []
        self._validate_actor_count(production, requirements, errors)
        self._validate_genre(production, requirements, errors)
        self._validate_theme(production, requirements, errors)
        self._validate_duration(production, requirements, errors)
        self._validate_fixtures(production, requirements, errors)
        self._validate_generated_text(production, errors)
        return errors

    @staticmethod
    def _validate_actor_count(
        production: Production,
        requirements: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        requested = requirements.get("actor_count")
        if requested in (None, ""):
            return
        try:
            expected = int(requested)
        except (TypeError, ValueError):
            return
        actual = len(production.characters)
        if actual != expected:
            errors.append(
                f"Actor count mismatch: requested {expected}, generated {actual}."
            )

    @classmethod
    def _validate_genre(
        cls,
        production: Production,
        requirements: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        requested = str(requirements.get("genre") or "").strip()
        if requested and cls._canonical(requested) != cls._canonical(production.genre):
            errors.append(
                f"Genre mismatch: requested '{requested}', generated '{production.genre}'."
            )

    @classmethod
    def _validate_theme(
        cls,
        production: Production,
        requirements: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        requested = str(requirements.get("theme") or "").strip()
        if not requested:
            return
        requested_tokens = cls._theme_tokens(requested)
        generated_tokens = cls._theme_tokens(production.theme)
        direct_overlap = requested_tokens & generated_tokens
        concept_overlap = any(
            requested_tokens & concept and generated_tokens & concept
            for concept in THEME_CONCEPTS
        )
        if not direct_overlap and not concept_overlap:
            errors.append(
                f"Theme mismatch: generated theme '{production.theme}' is not aligned "
                f"with requested theme '{requested}'."
            )

    @staticmethod
    def _validate_duration(
        production: Production,
        requirements: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        requested = requirements.get("duration_minutes")
        if requested in (None, ""):
            return
        try:
            duration = int(requested)
        except (TypeError, ValueError):
            return
        if duration <= 5:
            minimum_scenes, minimum_dialogue = 1, 8
        elif duration <= 10:
            minimum_scenes, minimum_dialogue = 2, 18
        elif duration <= 20:
            minimum_scenes, minimum_dialogue = 3, 30
        else:
            minimum_scenes, minimum_dialogue = 4, 40
        scene_count = len(production.scenes)
        dialogue_count = sum(len(scene.dialogue) for scene in production.scenes)
        if scene_count < minimum_scenes:
            errors.append(
                f"Insufficient scenes for {duration} minutes: minimum "
                f"{minimum_scenes}, generated {scene_count}."
            )
        if dialogue_count < minimum_dialogue:
            errors.append(
                f"Insufficient dialogue for {duration} minutes: minimum "
                f"{minimum_dialogue} lines, generated {dialogue_count}."
            )

    @staticmethod
    def _validate_fixtures(
        production: Production,
        requirements: Mapping[str, Any],
        errors: list[str],
    ) -> None:
        raw_fixtures = requirements.get("available_lights")
        if not raw_fixtures:
            return
        if isinstance(raw_fixtures, str):
            items = re.split(r"[,\r\n]+", raw_fixtures)
        else:
            try:
                items = list(raw_fixtures)
            except TypeError:
                items = [raw_fixtures]
        allowed = {str(item).strip() for item in items if str(item).strip()}
        if not allowed:
            return
        for scene in production.scenes:
            for cue in scene.lighting:
                if cue.fixture not in allowed:
                    errors.append(
                        f"Invalid fixture '{cue.fixture}' in scene '{scene.id}': "
                        f"use only {sorted(allowed)}."
                    )

    @staticmethod
    def _validate_generated_text(production: Production, errors: list[str]) -> None:
        values = ConstraintValidator._text_values(
            production.model_dump(mode="json", by_alias=True)
        )
        for value in values:
            lowered = value.lower()
            token = next(
                (candidate for candidate in FORBIDDEN_MODEL_TOKENS if candidate in lowered),
                None,
            )
            if token is not None:
                errors.append(f"Forbidden model-control token found: {token}")
                break
        if DATASET_TITLE_PATTERN.search(production.title):
            errors.append(
                "Retrieved-title copying detected: generated title contains a synthetic "
                "dataset title and record number."
            )

    @staticmethod
    def _text_values(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Mapping):
            return [
                text
                for child in value.values()
                for text in ConstraintValidator._text_values(child)
            ]
        if isinstance(value, (list, tuple)):
            return [
                text
                for child in value
                for text in ConstraintValidator._text_values(child)
            ]
        return []

    @staticmethod
    def _canonical(value: str) -> str:
        return "_".join(WORD_PATTERN.findall(value.casefold()))

    @staticmethod
    def _theme_tokens(value: str) -> set[str]:
        return {
            token
            for token in WORD_PATTERN.findall(value.casefold())
            if len(token) > 1 and token not in THEME_STOPWORDS
        }
