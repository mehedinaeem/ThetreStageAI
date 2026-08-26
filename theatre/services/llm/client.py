"""Provider interface and dependency-free Ollama HTTP client."""
from __future__ import annotations

import json
import logging
import socket
from abc import ABC, abstractmethod
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base error for local language-model operations."""


class LLMConnectionError(LLMError):
    """The configured local LLM server could not be reached."""


class LLMTimeoutError(LLMError):
    """The local model did not respond before the configured timeout."""


class ModelUnavailableError(LLMError):
    """The requested local model is not installed or available."""


class InvalidLLMResponseError(LLMError):
    """The provider returned an invalid transport envelope or generation."""


class LLMProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str:
        """Return the provider's generated JSON text."""


class OllamaClient(LLMProvider):
    """Non-streaming client for Ollama's local `/api/generate` endpoint."""

    def __init__(self, base_url: str, model: str, *, timeout_seconds: int = 180) -> None:
        if not base_url.strip():
            raise ValueError("Ollama URL cannot be empty")
        if not model.strip():
            raise ValueError("Ollama model cannot be empty")
        if timeout_seconds < 1:
            raise ValueError("Ollama timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": response_schema,
            "options": {"temperature": 0.2},
        }
        if system_prompt:
            body["system"] = system_prompt
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        logger.info("Requesting structured generation from Ollama model %s", self.model)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = self._http_error_detail(exc)
            if exc.code == 404 or ("model" in detail.lower() and "not found" in detail.lower()):
                raise ModelUnavailableError(
                    f"Ollama model '{self.model}' is unavailable. Run: ollama pull {self.model}"
                ) from exc
            raise LLMConnectionError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise LLMTimeoutError(
                f"Ollama generation exceeded {self.timeout_seconds} seconds"
            ) from exc
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMTimeoutError(
                    f"Ollama generation exceeded {self.timeout_seconds} seconds"
                ) from exc
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is 'ollama serve' running?"
            ) from exc
        except UnicodeError as exc:
            raise InvalidLLMResponseError("Ollama returned non-UTF-8 data") from exc

        try:
            envelope = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise InvalidLLMResponseError("Ollama returned a non-JSON API response") from exc
        if not isinstance(envelope, dict):
            raise InvalidLLMResponseError("Ollama response envelope must be a JSON object")
        if envelope.get("error"):
            detail = str(envelope["error"])
            if "model" in detail.lower() and (
                "not found" in detail.lower() or "pull" in detail.lower()
            ):
                raise ModelUnavailableError(detail)
            raise InvalidLLMResponseError(f"Ollama generation failed: {detail}")
        generated = envelope.get("response")
        if not isinstance(generated, str) or not generated.strip():
            raise InvalidLLMResponseError("Ollama response contains no generated text")
        return generated.strip()

    @staticmethod
    def _http_error_detail(error: HTTPError) -> str:
        try:
            raw = error.read().decode("utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("error"):
                return str(parsed["error"])
            return raw or error.reason
        except (UnicodeError, json.JSONDecodeError, OSError):
            return str(error.reason)
