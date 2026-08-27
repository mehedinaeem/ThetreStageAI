"""Official google-genai structured-output provider."""
from __future__ import annotations

import json
import logging
import re
from copy import deepcopy
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from .base import (
    GeminiAPIError,
    GeminiAuthenticationError,
    GeminiBadRequestError,
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiNetworkError,
    GeminiRateLimitError,
    GeminiUnavailableError,
    LLMProvider,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)
_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*([^\s,;]+)"
)


def _gemini_compatible_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy containing only portable validation annotations.

    Pydantic emits ``default`` annotations for optional/defaulted fields, but
    Gemini structured output does not accept that JSON Schema keyword. Keep
    the caller's request-specific schema unchanged while removing it from the
    wire representation.
    """
    compatible = deepcopy(schema)
    definitions = compatible.get("$defs", {})

    ignored_keywords = {
        "default",
        "$defs",
        "title",
        "minLength",
        "maxLength",
        "additionalProperties",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
    }

    def clean_schema(value: Any) -> Any:
        if isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                target = definitions.get(name)
                if isinstance(target, dict):
                    return clean_schema(deepcopy(target))
            cleaned: dict[str, Any] = {}
            for key, item in value.items():
                if key in ignored_keywords:
                    continue
                if key == "properties" and isinstance(item, dict):
                    # Property names are user data, not schema keywords. A
                    # production property named "title" must never be removed.
                    cleaned[key] = {
                        name: clean_schema(property_schema)
                        for name, property_schema in item.items()
                    }
                else:
                    cleaned[key] = clean_schema(item)
            properties = cleaned.get("properties")
            required = cleaned.get("required")
            if isinstance(properties, dict) and isinstance(required, list):
                cleaned["properties"] = {
                    name: property_schema
                    for name, property_schema in properties.items()
                    if name in required
                }
            # Pydantic represents fixed-length tuples with ``prefixItems``.
            # Gemini rejects this nested tuple form in the full Production
            # schema. RGB uses three identical integer schemas, so express it
            # as homogeneous ``items`` while retaining minItems/maxItems=3.
            prefix_items = cleaned.get("prefixItems")
            if (
                isinstance(prefix_items, list)
                and prefix_items
                and all(item == prefix_items[0] for item in prefix_items)
            ):
                cleaned["items"] = prefix_items[0]
                cleaned.pop("prefixItems", None)
            return cleaned
        if isinstance(value, list):
            return [clean_schema(item) for item in value]
        return value

    return clean_schema(compatible)


class GeminiProvider(LLMProvider):
    """Gemini Developer API provider with no tools or grounding enabled."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.6-flash",
        timeout_seconds: int = 180,
        max_output_tokens: int = 8192,
        temperature: float = 0.2,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if not self.model:
            raise GeminiConfigurationError("GEMINI_MODEL cannot be empty")
        if timeout_seconds < 1 or max_output_tokens < 1:
            raise GeminiConfigurationError(
                "Gemini timeout and maximum output tokens must be positive"
            )
        self._client = client

    @property
    def client(self) -> Any:
        if not self.api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is missing. Add it to the local .env file."
            )
        if self._client is None:
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=self.timeout_seconds * 1000),
            )
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=_gemini_compatible_schema(response_schema),
        )
        logger.info("Requesting structured generation from Gemini model %s", self.model)
        generated = self._generate_content(prompt, config)

        if not isinstance(generated, str) or not generated.strip():
            raise GeminiInvalidResponseError("Gemini returned an empty response")
        try:
            parsed = json.loads(generated)
        except json.JSONDecodeError as exc:
            raise GeminiInvalidResponseError("Gemini returned malformed JSON") from exc
        if not isinstance(parsed, dict):
            raise GeminiInvalidResponseError("Gemini JSON response must be an object")
        return generated.strip()

    def generate_plain(self, prompt: str) -> str:
        """Run a minimal unstructured request for connectivity diagnostics."""
        logger.info("Requesting plain generation from Gemini model %s", self.model)
        generated = self._generate_content(prompt, None)
        if not isinstance(generated, str) or not generated.strip():
            raise GeminiInvalidResponseError("Gemini returned an empty response")
        return generated.strip()

    def _generate_content(
        self, prompt: str, config: types.GenerateContentConfig | None
    ) -> str:
        try:
            arguments: dict[str, Any] = {"model": self.model, "contents": prompt}
            if config is not None:
                arguments["config"] = config
            response = self.client.models.generate_content(**arguments)
            generated = response.text
        except errors.APIError as exc:
            self._log_failure(exc)
            self._raise_api_error(exc)
        except (TimeoutError, httpx.TimeoutException) as exc:
            self._log_failure(exc)
            raise LLMTimeoutError(
                f"Gemini generation exceeded {self.timeout_seconds} seconds"
            ) from exc
        except httpx.NetworkError as exc:
            self._log_failure(exc)
            raise GeminiNetworkError(
                "Gemini could not be reached because of a network transport error"
            ) from exc
        except (AttributeError, ValueError, TypeError) as exc:
            self._log_failure(exc)
            raise GeminiInvalidResponseError(
                "Gemini returned an unreadable response"
            ) from exc
        return response.text

    def reproducibility_settings(self) -> dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
        }

    @staticmethod
    def _raise_api_error(exception: errors.APIError) -> None:
        code = int(getattr(exception, "code", 0) or 0)
        detail = str(exception).lower()
        if code == 429:
            raise GeminiRateLimitError("Gemini quota or rate limit exceeded") from exception
        if code in {401, 403} or "api key not valid" in detail or "unauthenticated" in detail:
            raise GeminiAuthenticationError("Gemini rejected the API key") from exception
        if code == 400:
            raise GeminiBadRequestError(
                "Gemini rejected the request or structured-output schema"
            ) from exception
        if code == 503:
            raise GeminiUnavailableError("Gemini is temporarily unavailable") from exception
        raise GeminiAPIError(f"Gemini API request failed with status {code or 'unknown'}") from exception

    def _log_failure(self, exception: BaseException) -> None:
        status = getattr(exception, "code", None)
        api_status = getattr(exception, "status", None)
        message = getattr(exception, "message", None) or str(exception)
        root = exception
        seen: set[int] = set()
        while (root.__cause__ or root.__context__) is not None and id(root) not in seen:
            seen.add(id(root))
            root = root.__cause__ or root.__context__  # type: ignore[assignment]
        logger.error(
            "Gemini generation failed exception=%s http_status=%s api_status=%s "
            "message=%s root_exception=%s root_message=%s",
            type(exception).__name__,
            status,
            api_status,
            self._sanitize(message),
            type(root).__name__,
            self._sanitize(str(root)),
        )

    def _sanitize(self, value: object) -> str:
        text = str(value).replace(self.api_key, "[REDACTED]") if self.api_key else str(value)
        return _SECRET_PATTERN.sub(r"\1=[REDACTED]", text)[:2_000]
