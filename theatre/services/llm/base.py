"""Provider-neutral LLM contract and typed service errors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMError(RuntimeError):
    """Base error for language-model operations."""


class LLMConnectionError(LLMError):
    """A provider could not be reached."""


class LLMTimeoutError(LLMError):
    """A provider did not respond before its configured timeout."""


class ModelUnavailableError(LLMError):
    """The selected model is unavailable."""


class InvalidLLMResponseError(LLMError):
    """A provider returned an invalid transport envelope or generation."""


class GeminiConfigurationError(LLMError):
    """Gemini configuration is missing or invalid."""


class GeminiAuthenticationError(LLMError):
    """Gemini rejected the configured API credentials."""


class GeminiRateLimitError(LLMError):
    """Gemini quota or request-rate limits were exceeded."""


class GeminiBadRequestError(LLMError):
    """Gemini rejected an invalid request or response schema."""


class GeminiNetworkError(LLMConnectionError):
    """Gemini could not be reached because of a transport failure."""


class GeminiUnavailableError(LLMError):
    """Gemini returned a service-unavailable response."""


class GeminiAPIError(LLMError):
    """Gemini returned a server, network, or unsupported-request error."""


class GeminiInvalidResponseError(InvalidLLMResponseError):
    """Gemini returned empty, malformed, or non-object JSON."""


class LLMProvider(ABC):
    """Logical interface shared by remote and local generation providers."""

    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str:
        """Return generated JSON text constrained by ``response_schema``."""

    def reproducibility_settings(self) -> dict[str, Any]:
        """Return non-secret settings that affect generation."""
        return {}
