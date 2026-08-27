"""Test the configured LLM provider with one minimal structured request."""
from __future__ import annotations

from time import perf_counter
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from theatre.services.llm import LLMError, create_provider
from theatre.services.llm.gemini_client import GeminiProvider


class Command(BaseCommand):
    help = "Test Gemini or Ollama connectivity with a minimal structured request."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--provider", choices=("gemini", "ollama"))

    def handle(self, *args: Any, **options: Any) -> None:
        provider = create_provider(options.get("provider"))
        schema = {
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": False,
        }
        provider_name = provider.reproducibility_settings().get("provider")
        self.stdout.write("[1] Configuration")
        self.stdout.write(
            "Key configured: "
            + ("YES" if not isinstance(provider, GeminiProvider) or bool(provider.api_key) else "NO")
        )
        self.stdout.write(f"Provider: {provider_name}")
        self.stdout.write(f"Model: {provider.model}")

        if isinstance(provider, GeminiProvider):
            self.stdout.write("\n[2] Plain Gemini call")
            started = perf_counter()
            try:
                plain = provider.generate_plain("Reply only with OK")
            except LLMError as exc:
                self._fail("Plain generation", started, exc)
            if plain.strip().upper() != "OK":
                self.stdout.write("Status: FAILED")
                self.stdout.write(f"Latency: {perf_counter() - started:.2f} sec")
                raise CommandError("Plain generation returned an unexpected response")
            self.stdout.write(self.style.SUCCESS("Status: OK"))
            self.stdout.write(f"Latency: {perf_counter() - started:.2f} sec")

        self.stdout.write("\n[3] Minimal structured-output call")
        started = perf_counter()
        try:
            provider.generate(
                'Return status = "OK".',
                response_schema=schema,
                system_prompt="Return only schema-conforming JSON.",
            )
        except LLMError as exc:
            self._fail("Structured generation", started, exc)
        self.stdout.write(self.style.SUCCESS("Status: OK"))
        self.stdout.write(f"Latency: {perf_counter() - started:.2f} sec")

    def _fail(self, stage: str, started: float, exception: LLMError) -> None:
        self.stdout.write("Status: FAILED")
        self.stdout.write(f"Latency: {perf_counter() - started:.2f} sec")
        self.stdout.write(f"Cause: {type(exception).__name__}: {exception}")
        raise CommandError(f"{stage} failed") from exception
