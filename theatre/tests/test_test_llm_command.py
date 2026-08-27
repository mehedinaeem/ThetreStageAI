from __future__ import annotations

from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.core.management import call_command
from django.test import SimpleTestCase

from theatre.services.llm import GeminiProvider


class TestLLMCommandTests(SimpleTestCase):
    @patch("theatre.management.commands.test_llm.create_provider")
    def test_command_uses_selected_provider_and_structured_schema(
        self, create_provider: Mock
    ) -> None:
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(text="OK"),
            SimpleNamespace(text='{"status":"OK"}'),
        ]
        provider = GeminiProvider(api_key="configured", client=client)
        create_provider.return_value = provider
        output = StringIO()

        call_command("test_llm", provider="gemini", stdout=output)

        create_provider.assert_called_once_with("gemini")
        calls = client.models.generate_content.call_args_list
        self.assertEqual(calls[0].kwargs["contents"], "Reply only with OK")
        self.assertEqual(
            calls[1].kwargs["config"].response_json_schema["properties"]["status"]["type"],
            "string",
        )
        self.assertIn("Key configured: YES", output.getvalue())
        self.assertIn("[2] Plain Gemini call", output.getvalue())
        self.assertIn("[3] Minimal structured-output call", output.getvalue())
        self.assertIn("Provider: gemini", output.getvalue())
        self.assertIn("Model: gemini-3.6-flash", output.getvalue())
        self.assertIn("Status: OK", output.getvalue())
        self.assertNotIn("API_KEY", output.getvalue())
