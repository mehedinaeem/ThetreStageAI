"""Validate model output and perform at most one structured correction."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from .production_schema import Production

logger = logging.getLogger(__name__)

CORRECTION_SYSTEM_PROMPT = """আপনি JSON সংশোধনকারী। শুধু schema-সম্মত JSON object ফেরত দিন।
Markdown fence, ব্যাখ্যা বা JSON-এর বাইরের কোনো লেখা দেবেন না। নতুন বিষয়বস্তু যোগ না করে validation error ঠিক করুন।"""


class CorrectionProvider(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str: ...


class ProductionValidationError(RuntimeError):
    """Controlled failure after generated output remains unsafe."""

    def __init__(
        self,
        message: str,
        *,
        initial_errors: list[dict[str, Any]],
        final_errors: list[dict[str, Any]] | None = None,
        correction_error: str | None = None,
        initial_output: str = "",
        corrected_output: str = "",
    ) -> None:
        super().__init__(message)
        self.initial_errors = initial_errors
        self.final_errors = final_errors or []
        self.correction_error = correction_error
        self.initial_output = initial_output
        self.corrected_output = corrected_output


@dataclass(frozen=True, slots=True)
class OutputValidationResult:
    production: Production
    accepted_output: str
    initial_errors: list[dict[str, Any]]
    repaired: bool


class OutputValidator:
    def __init__(self, provider: CorrectionProvider, *, max_response_chars: int = 50_000) -> None:
        if max_response_chars < 1_000:
            raise ValueError("Validation response limit must be at least 1000 characters")
        self.provider = provider
        self.max_response_chars = max_response_chars

    def validate(self, raw_output: str) -> Production:
        """Return only safe output, with exactly one correction attempt when invalid."""
        return self.validate_with_details(raw_output).production

    def validate_with_details(self, raw_output: str) -> OutputValidationResult:
        """Validate while retaining correction evidence for experiment records."""
        try:
            production = self._validate_once(raw_output)
            return OutputValidationResult(production, raw_output, [], False)
        except ValidationError as initial_exception:
            initial_errors = self._errors(initial_exception)
            logger.warning(
                "Generated production failed validation with %d error(s); requesting one correction",
                len(initial_errors),
            )

        schema = Production.json_schema()
        correction_prompt = self._correction_prompt(raw_output, initial_errors, schema)
        try:
            corrected_output = self.provider.generate(
                correction_prompt,
                response_schema=schema,
                system_prompt=CORRECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:
            logger.exception("The single production correction request failed")
            raise ProductionValidationError(
                "Generated production was invalid and its correction request failed",
                initial_errors=initial_errors,
                correction_error=str(exc),
                initial_output=raw_output,
            ) from exc

        try:
            production = self._validate_once(corrected_output)
            return OutputValidationResult(production, corrected_output, initial_errors, True)
        except ValidationError as final_exception:
            final_errors = self._errors(final_exception)
            logger.error(
                "Rejecting generated production after correction; %d validation error(s) remain",
                len(final_errors),
            )
            raise ProductionValidationError(
                "Generated production remained invalid after one correction attempt",
                initial_errors=initial_errors,
                final_errors=final_errors,
                initial_output=raw_output,
                corrected_output=corrected_output,
            ) from final_exception

    def _validate_once(self, raw_output: str) -> Production:
        if len(raw_output) > self.max_response_chars:
            raise ValidationError.from_exception_data(
                "Production",
                [{
                    "type": "value_error",
                    "loc": (),
                    "input": "<oversized response>",
                    "ctx": {"error": ValueError(
                        f"Generated JSON exceeds {self.max_response_chars} characters"
                    )},
                }],
            )
        return Production.model_validate_json(raw_output)

    @staticmethod
    def _errors(exception: ValidationError) -> list[dict[str, Any]]:
        return exception.errors(include_url=False, include_input=False)

    @staticmethod
    def _correction_prompt(
        raw_output: str,
        errors: list[dict[str, Any]],
        schema: dict[str, Any],
    ) -> str:
        return """নিচের invalid JSON-টি একবার সংশোধন করুন।
শুধু সম্পূর্ণ corrected JSON object ফেরত দিন। সংলাপ ও বাংলা বিষয়বস্তু যথাসম্ভব অক্ষুণ্ণ রাখুন।

VALIDATION ERRORS
{errors}

INVALID OUTPUT
{output}

REQUIRED JSON SCHEMA
{schema}""".format(
            errors=json.dumps(errors, ensure_ascii=False, default=str),
            output=raw_output,
            schema=json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        )
