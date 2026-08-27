from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
from django.test import SimpleTestCase, override_settings
from django.core.exceptions import ImproperlyConfigured
from google.genai import errors

from theatre.services.llm import (
    GeminiAuthenticationError,
    GeminiBadRequestError,
    GeminiConfigurationError,
    GeminiInvalidResponseError,
    GeminiNetworkError,
    GeminiProvider,
    GeminiRateLimitError,
    GeminiUnavailableError,
    LLMTimeoutError,
    OllamaClient,
    create_provider,
)
from theatre.services.llm.generator import TheatreGenerator
from theatre.services.rag.context_builder import ContextBuilder
from theatre.tests.test_constraint_validator import complete_production, requirements


def fake_client(*responses: Any) -> Mock:
    client = Mock()
    client.models.generate_content.side_effect = list(responses)
    return client


class GeminiProviderTests(SimpleTestCase):
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "optional_note": {"type": "string", "default": ""},
        },
        "required": ["title"],
    }

    def test_success_uses_official_structured_output_and_dynamic_schema(self) -> None:
        client = fake_client(SimpleNamespace(text='{"title":"নাটক"}'))
        provider = GeminiProvider(
            api_key="test-key",
            model="gemini-3.6-flash",
            timeout_seconds=12,
            max_output_tokens=8192,
            client=client,
        )

        result = provider.generate(
            "prompt", response_schema=self.schema, system_prompt="system"
        )

        self.assertEqual(json.loads(result), {"title": "নাটক"})
        call = client.models.generate_content.call_args
        self.assertEqual(call.kwargs["model"], "gemini-3.6-flash")
        config = call.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertNotIn("optional_note", config.response_json_schema["properties"])
        self.assertEqual(
            self.schema["properties"]["optional_note"]["default"], ""
        )
        self.assertEqual(config.max_output_tokens, 8192)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.system_instruction, "system")
        self.assertIsNone(config.tools)

    def test_gemini_schema_converts_homogeneous_tuple_to_items(self) -> None:
        schema = {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "prefixItems": [
                {"type": "integer", "minimum": 0, "maximum": 255},
                {"type": "integer", "minimum": 0, "maximum": 255},
                {"type": "integer", "minimum": 0, "maximum": 255},
            ],
        }
        client = fake_client(SimpleNamespace(text='{"value":1}'))
        provider = GeminiProvider(api_key="test-key", client=client)

        provider.generate("prompt", response_schema=schema)

        sent = client.models.generate_content.call_args.kwargs["config"].response_json_schema
        self.assertNotIn("prefixItems", sent)
        self.assertEqual(sent["items"]["type"], "integer")
        self.assertNotIn("maximum", sent["items"])
        self.assertIn("prefixItems", schema)

    def test_gemini_schema_inlines_local_definitions(self) -> None:
        schema = {
            "$defs": {"Item": {"type": "string", "enum": ["A", "B"]}},
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/Item"}},
            "required": ["item"],
        }
        client = fake_client(SimpleNamespace(text='{"item":"A"}'))
        provider = GeminiProvider(api_key="test-key", client=client)

        provider.generate("prompt", response_schema=schema)

        sent = client.models.generate_content.call_args.kwargs["config"].response_json_schema
        self.assertNotIn("$defs", sent)
        self.assertEqual(sent["properties"]["item"]["enum"], ["A", "B"])
        self.assertIn("$defs", schema)

    def test_missing_key_is_configuration_error(self) -> None:
        provider = GeminiProvider(api_key="")
        with self.assertRaises(GeminiConfigurationError):
            provider.generate("prompt", response_schema=self.schema)

    def test_authentication_error_is_mapped(self) -> None:
        client = fake_client(errors.ClientError(401, {"error": {"message": "bad key"}}))
        provider = GeminiProvider(api_key="test-key", client=client)
        with self.assertRaises(GeminiAuthenticationError):
            provider.generate("prompt", response_schema=self.schema)

    def test_rate_limit_is_mapped(self) -> None:
        client = fake_client(errors.ClientError(429, {"error": {"message": "quota"}}))
        provider = GeminiProvider(api_key="test-key", client=client)
        with self.assertRaises(GeminiRateLimitError):
            provider.generate("prompt", response_schema=self.schema)

    def test_bad_request_and_unavailable_are_not_mapped_as_network(self) -> None:
        cases = (
            (400, GeminiBadRequestError),
            (503, GeminiUnavailableError),
        )
        for status, expected in cases:
            with self.subTest(status=status):
                client = fake_client(
                    errors.ClientError(status, {"error": {"message": "request failed"}})
                    if status < 500
                    else errors.ServerError(status, {"error": {"message": "unavailable"}})
                )
                provider = GeminiProvider(api_key="test-key", client=client)
                with self.assertRaises(expected):
                    provider.generate("prompt", response_schema=self.schema)

    def test_real_network_transport_error_is_mapped_as_network(self) -> None:
        request = httpx.Request("POST", "https://example.test")
        provider = GeminiProvider(
            api_key="test-key",
            client=fake_client(httpx.ConnectError("connection failed", request=request)),
        )
        with self.assertRaises(GeminiNetworkError):
            provider.generate("prompt", response_schema=self.schema)

    def test_diagnostic_log_has_sanitized_exception_details(self) -> None:
        client = fake_client(
            errors.ClientError(
                400,
                {"error": {"status": "INVALID_ARGUMENT", "message": "bad schema"}},
            )
        )
        provider = GeminiProvider(api_key="test-key", client=client)
        with self.assertLogs("theatre.services.llm.gemini_client", level="ERROR") as logs:
            with self.assertRaises(GeminiBadRequestError):
                provider.generate("prompt", response_schema=self.schema)
        diagnostic = "\n".join(logs.output)
        self.assertIn("exception=ClientError", diagnostic)
        self.assertIn("http_status=400", diagnostic)
        self.assertIn("api_status=INVALID_ARGUMENT", diagnostic)
        self.assertNotIn("test-key", diagnostic)

    def test_timeout_is_mapped(self) -> None:
        request = httpx.Request("POST", "https://example.test")
        client = fake_client(httpx.ReadTimeout("slow", request=request))
        provider = GeminiProvider(api_key="test-key", client=client)
        with self.assertRaises(LLMTimeoutError):
            provider.generate("prompt", response_schema=self.schema)

    def test_empty_and_malformed_responses_are_rejected(self) -> None:
        for response in (SimpleNamespace(text=""), SimpleNamespace(text="not-json")):
            with self.subTest(response=response.text):
                provider = GeminiProvider(
                    api_key="test-key", client=fake_client(response)
                )
                with self.assertRaises(GeminiInvalidResponseError):
                    provider.generate("prompt", response_schema=self.schema)

    def test_api_key_is_absent_from_settings_and_logs(self) -> None:
        provider = GeminiProvider(api_key="super-secret-key")
        serialized = json.dumps(provider.reproducibility_settings())
        self.assertNotIn("super-secret-key", serialized)
        with self.assertLogs("theatre.services.llm.gemini_client", level="INFO") as logs:
            with self.assertRaises(GeminiConfigurationError):
                GeminiProvider(api_key="").generate("prompt", response_schema=self.schema)
        self.assertNotIn("super-secret-key", "\n".join(logs.output))

    def test_semantic_failure_repairs_with_same_dynamic_gemini_schema(self) -> None:
        invalid = deepcopy(complete_production())
        invalid["scenes"] = [invalid["scenes"][0]]
        invalid["scenes"][0]["dialogue"] = [invalid["scenes"][0]["dialogue"][0]]
        invalid["scenes"][0]["blocking"][0]["trigger"] = (
            invalid["scenes"][0]["dialogue"][0]["id"]
        )
        client = fake_client(
            SimpleNamespace(text=json.dumps(invalid, ensure_ascii=False)),
            SimpleNamespace(text=json.dumps(complete_production(), ensure_ascii=False)),
        )
        provider = GeminiProvider(api_key="test-key", client=client)
        generator = TheatreGenerator(provider, ContextBuilder(max_chars=4_000))
        request_constraints = requirements()

        result = generator.generate(
            "Story idea: ভুয়া পোস্ট নিয়ে নাটক",
            [],
            [],
            [],
            constraints=request_constraints,
        )

        self.assertTrue(result.repaired)
        self.assertEqual(client.models.generate_content.call_count, 2)
        configs = [call.kwargs["config"] for call in client.models.generate_content.call_args_list]
        self.assertEqual(configs[0].response_json_schema, configs[1].response_json_schema)
        def fixture_enum(value: Any) -> list[str] | None:
            if isinstance(value, dict):
                properties = value.get("properties")
                if isinstance(properties, dict) and "fixture" in properties:
                    return properties["fixture"].get("enum")
                for item in value.values():
                    found = fixture_enum(item)
                    if found is not None:
                        return found
            elif isinstance(value, list):
                for item in value:
                    found = fixture_enum(item)
                    if found is not None:
                        return found
            return None

        self.assertEqual(
            fixture_enum(configs[0].response_json_schema),
            request_constraints["available_lights"],
        )


class ProviderFactoryTests(SimpleTestCase):
    @override_settings(
        THETRESTAGEAI_LLM_PROVIDER="gemini",
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="gemini-3.6-flash",
        GEMINI_TIMEOUT_SECONDS=180,
        GEMINI_MAX_OUTPUT_TOKENS=8192,
        GEMINI_TEMPERATURE=0.2,
    )
    def test_gemini_selected_from_settings(self) -> None:
        self.assertIsInstance(create_provider(), GeminiProvider)

    @override_settings(
        THETRESTAGEAI_LLM_PROVIDER="ollama",
        THETRESTAGEAI_OLLAMA_URL="http://localhost:11434",
        THETRESTAGEAI_LLM_MODEL="qwen3:4b",
        THETRESTAGEAI_LLM_TIMEOUT_SECONDS=900,
        THETRESTAGEAI_LLM_NUM_PREDICT=3072,
    )
    def test_ollama_selected_from_settings(self) -> None:
        self.assertIsInstance(create_provider(), OllamaClient)

    @override_settings(THETRESTAGEAI_LLM_PROVIDER="unsupported")
    def test_unsupported_provider_is_not_silently_replaced(self) -> None:
        with self.assertRaises(ImproperlyConfigured):
            create_provider()
