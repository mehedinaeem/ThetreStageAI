"""Validated theatre generation orchestration."""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from theatre.services.rag.context_builder import ContextBuilder, RetrievalTrace
from theatre.services.retrieval.base import RetrievalResult
from theatre.services.validation import OutputValidator, Production

from .client import LLMProvider
from .prompts import SYSTEM_PROMPT, build_generation_prompt
from .schema_builder import build_generation_schema, duration_minimums

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class GenerationResult:
    production: Production
    retrieval_trace: list[RetrievalTrace]
    raw_output: str
    accepted_output: str
    validation_errors: list[dict[str, object]]
    validation_history: dict[str, list[dict[str, object]]]
    repaired: bool


class TheatreGenerator:
    def __init__(
        self,
        provider: LLMProvider,
        context_builder: ContextBuilder,
        output_validator: OutputValidator | None = None,
    ) -> None:
        self.provider = provider
        self.context_builder = context_builder
        self.output_validator = output_validator or OutputValidator(provider)

    def generate(
        self,
        user_requirements: str,
        scene_results: Sequence[RetrievalResult],
        blocking_results: Sequence[RetrievalResult],
        lighting_results: Sequence[RetrievalResult],
        *,
        constraints: Mapping[str, Any] | None = None,
    ) -> GenerationResult:
        context, trace = self.context_builder.build(
            user_requirements,
            scene_results,
            blocking_results,
            lighting_results,
        )
        base_schema = Production.json_schema()
        minimums = None
        fixtures: list[str] = []
        if constraints:
            actor_count = int(constraints["actor_count"])
            duration_minutes = int(constraints["duration_minutes"])
            fixtures = [
                str(item).strip()
                for item in constraints.get("available_lights", [])
                if str(item).strip()
            ]
            schema = build_generation_schema(
                base_schema,
                actor_count=actor_count,
                duration_minutes=duration_minutes,
                available_lights=fixtures,
            )
            minimums = duration_minimums(duration_minutes)
        else:
            schema = base_schema
        prompt = build_generation_prompt(
            context,
            schema,
            minimums=minimums,
            available_lights=fixtures,
        )
        if settings.DEBUG:
            logger.debug(
                "Final LLM prompt requirement summary: %s",
                self._safe_requirement_summary(user_requirements),
            )
        raw_response = self.provider.generate(
            prompt,
            response_schema=schema,
            system_prompt=SYSTEM_PROMPT,
        )
        validation = self.output_validator.validate_with_details(
            raw_response,
            user_requirements=user_requirements,
            constraints=constraints,
            response_schema=schema,
        )
        return GenerationResult(
            production=validation.production,
            retrieval_trace=trace,
            raw_output=raw_response,
            accepted_output=validation.accepted_output,
            validation_errors=validation.final_errors,
            validation_history={
                "initial": validation.initial_errors,
                "final": validation.final_errors,
            },
            repaired=validation.repaired,
        )

    @staticmethod
    def _safe_requirement_summary(user_requirements: str) -> str:
        """Log bounded requirement values without including retrieved RAG content."""
        allowed_labels = {
            "Story idea",
            "Theme",
            "Genre",
            "Language",
            "Number of actors",
            "Target duration",
            "Stage size",
            "Available lighting fixtures",
            "Scene time",
            "Desired emotion",
        }
        summary: list[str] = []
        for line in user_requirements.splitlines():
            label, separator, value = line.partition(":")
            if separator and label.strip() in allowed_labels:
                clean_value = " ".join(value.split())[:300]
                summary.append(f"{label.strip()}={clean_value}")
        return "; ".join(summary)[:2_000]
