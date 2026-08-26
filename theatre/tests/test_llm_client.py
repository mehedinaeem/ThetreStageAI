from __future__ import annotations

import io
import json
import socket
from typing import Any
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from django.test import SimpleTestCase

from theatre.services.llm.client import (
    InvalidLLMResponseError,
    LLMConnectionError,
    LLMTimeoutError,
    ModelUnavailableError,
    OllamaClient,
)


class FakeResponse:
    def __init__(self, body: dict[str, Any] | str) -> None:
        self.body = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class OllamaClientTests(SimpleTestCase):
    def setUp(self) -> None:
        self.client = OllamaClient("http://localhost:11434", "qwen3:4b", timeout_seconds=5)

    @patch("theatre.services.llm.client.urlopen")
    def test_generate_uses_structured_non_streaming_request(self, mocked_urlopen: Any) -> None:
        mocked_urlopen.return_value = FakeResponse({"response": '{"title":"নাটক"}'})

        generated = self.client.generate(
            "বাংলা prompt",
            response_schema={"type": "object"},
            system_prompt="system",
        )

        self.assertEqual(generated, '{"title":"নাটক"}')
        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "qwen3:4b")
        self.assertFalse(body["stream"])
        self.assertFalse(body["think"])
        self.assertEqual(body["format"], {"type": "object"})
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 5)

    @patch("theatre.services.llm.client.urlopen", side_effect=URLError("refused"))
    def test_connection_error_is_mapped(self, _: Any) -> None:
        with self.assertRaises(LLMConnectionError):
            self.client.generate("prompt", response_schema={})

    @patch(
        "theatre.services.llm.client.urlopen",
        side_effect=URLError(socket.timeout("timed out")),
    )
    def test_timeout_is_mapped(self, _: Any) -> None:
        with self.assertRaises(LLMTimeoutError):
            self.client.generate("prompt", response_schema={})

    @patch("theatre.services.llm.client.urlopen")
    def test_missing_model_is_mapped(self, mocked_urlopen: Any) -> None:
        mocked_urlopen.side_effect = HTTPError(
            "http://localhost:11434/api/generate",
            404,
            "Not Found",
            {},
            io.BytesIO(b'{"error":"model qwen3:4b not found"}'),
        )
        with self.assertRaises(ModelUnavailableError):
            self.client.generate("prompt", response_schema={})

    @patch("theatre.services.llm.client.urlopen")
    def test_invalid_provider_envelope_is_rejected(self, mocked_urlopen: Any) -> None:
        mocked_urlopen.return_value = FakeResponse("not-json")
        with self.assertRaises(InvalidLLMResponseError):
            self.client.generate("prompt", response_schema={})
