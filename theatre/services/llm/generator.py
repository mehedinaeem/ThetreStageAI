"""Validated theatre generation orchestration."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from theatre.services.rag.context_builder import ContextBuilder, RetrievalTrace
from theatre.services.retrieval.base import RetrievalResult
from theatre.services.validation import OutputValidator, Production

from .client import LLMProvider
from .prompts import SYSTEM_PROMPT, build_generation_prompt

@dataclass(frozen=True, slots=True)
class GenerationResult:
    production: Production
    retrieval_trace: list[RetrievalTrace]


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
    ) -> GenerationResult:
        context, trace = self.context_builder.build(
            user_requirements,
            scene_results,
            blocking_results,
            lighting_results,
        )
        schema = Production.json_schema()
        prompt = build_generation_prompt(context, schema)
        raw_response = self.provider.generate(
            prompt,
            response_schema=schema,
            system_prompt=SYSTEM_PROMPT,
        )
        production = self.output_validator.validate(raw_response)
        return GenerationResult(production=production, retrieval_trace=trace)
