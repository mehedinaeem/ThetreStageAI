"""Configuration-driven LLM provider selection."""
from __future__ import annotations

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from .base import LLMProvider
from .gemini_client import GeminiProvider
from .ollama_client import OllamaClient


def create_provider(provider_name: str | None = None) -> LLMProvider:
    selected = (provider_name or settings.THETRESTAGEAI_LLM_PROVIDER).strip().lower()
    if selected == "gemini":
        return GeminiProvider(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_MODEL,
            timeout_seconds=settings.GEMINI_TIMEOUT_SECONDS,
            max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
            temperature=settings.GEMINI_TEMPERATURE,
        )
    if selected == "ollama":
        return OllamaClient(
            settings.THETRESTAGEAI_OLLAMA_URL,
            settings.THETRESTAGEAI_LLM_MODEL,
            timeout_seconds=settings.THETRESTAGEAI_LLM_TIMEOUT_SECONDS,
            num_predict=settings.THETRESTAGEAI_LLM_NUM_PREDICT,
        )
    raise ImproperlyConfigured(
        f"Unsupported THETRESTAGEAI_LLM_PROVIDER '{selected}'. "
        "Expected 'gemini' or 'ollama'."
    )
