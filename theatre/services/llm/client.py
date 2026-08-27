"""Provider interface and dependency-free Ollama HTTP client."""
from __future__ import annotations

import json
import logging
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

from .base import (
    InvalidLLMResponseError,
    LLMConnectionError,
    LLMProvider,
    LLMTimeoutError,
    ModelUnavailableError,
)


class OllamaClient(LLMProvider):
    """Non-streaming client for Ollama's local `/api/generate` endpoint."""

    max_http_response_bytes = 2_000_000

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: int = 180,
        num_predict: int = 4096,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Ollama URL cannot be empty")
        if not model.strip():
            raise ValueError("Ollama model cannot be empty")
        if timeout_seconds < 1:
            raise ValueError("Ollama timeout must be positive")
        if num_predict < 1:
            raise ValueError("Ollama num_predict must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.num_predict = num_predict

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
            "options": {"temperature": 0.2, "num_predict": self.num_predict},
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
                raw_bytes = response.read(self.max_http_response_bytes + 1)
                if len(raw_bytes) > self.max_http_response_bytes:
                    raise InvalidLLMResponseError(
                        "Ollama response exceeded the configured size limit"
                    )
                raw_body = raw_bytes.decode("utf-8")
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

    def reproducibility_settings(self) -> dict[str, Any]:
        """Return only non-secret settings that affect model generation."""
        return {
            "provider": "ollama",
            "temperature": 0.2,
            "num_predict": self.num_predict,
            "think": False,
            "timeout_seconds": self.timeout_seconds,
        }

    @staticmethod
    def _http_error_detail(error: HTTPError) -> str:
        try:
            raw = error.read(8_193)
            if len(raw) > 8_192:
                return "Oversized error response"
            raw = raw.decode("utf-8")
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed.get("error"):
                return str(parsed["error"])
            return raw or error.reason
        except (UnicodeError, json.JSONDecodeError, OSError):
            return str(error.reason)
