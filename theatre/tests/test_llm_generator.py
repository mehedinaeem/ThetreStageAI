from __future__ import annotations

import json
from typing import Any

from django.test import SimpleTestCase

from theatre.services.llm.client import LLMProvider
from theatre.services.llm.generator import TheatreGenerator
from theatre.services.rag.context_builder import ContextBuilder
from theatre.services.validation import ProductionValidationError


def valid_production() -> dict[str, Any]:
    return {
        "title": "নতুন সকাল",
        "theme": "পারিবারিক পুনর্মিলন",
        "genre": "social_drama",
        "characters": [{"name": "মায়া", "description": "বড় বোন"}],
        "scenes": [
            {
                "id": "S01",
                "title": "ফিরে আসা",
                "location": "বসার ঘর",
                "time": "সন্ধ্যা",
                "dialogue": [{"id": "D01", "speaker": "মায়া", "text": "আমি ফিরেছি।"}],
                "stage_directions": ["মায়া ধীরে প্রবেশ করে।"],
                "blocking": [
                    {
                        "actor": "মায়া",
                        "from": "USL",
                        "to": "CSC",
                        "action": "walk_slowly",
                        "trigger": "scene_start",
                    }
                ],
                "lighting": [
                    {
                        "cue_id": "LQ01",
                        "trigger": "scene_start",
                        "fixture": "PAR01",
                        "focus_zone": "CSC",
                        "rgb": [255, 180, 120],
                        "intensity": 55,
                        "fade_seconds": 2.0,
                    }
                ],
            }
        ],
    }


class FakeProvider(LLMProvider):
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt = ""
        self.schema: dict[str, Any] = {}

    def generate(
        self,
        prompt: str,
        *,
        response_schema: dict[str, Any],
        system_prompt: str | None = None,
    ) -> str:
        self.prompt = prompt
        self.schema = response_schema
        return self.response


class TheatreGeneratorTests(SimpleTestCase):
    def test_valid_json_is_returned_as_typed_production(self) -> None:
        provider = FakeProvider(json.dumps(valid_production(), ensure_ascii=False))
        generator = TheatreGenerator(provider, ContextBuilder(max_chars=4_000))

        result = generator.generate("একটি নতুন পারিবারিক নাটক", [], [], [])

        self.assertEqual(result.production.title, "নতুন সকাল")
        self.assertEqual(result.production.scenes[0].blocking[0].from_zone.value, "USL")
        self.assertFalse(result.retrieval_trace)
        self.assertIn("REQUIRED JSON SCHEMA", provider.prompt)
        self.assertIn("from", json.dumps(provider.schema))

    def test_markdown_wrapped_json_is_rejected(self) -> None:
        response = "```json\n" + json.dumps(valid_production(), ensure_ascii=False) + "\n```"
        generator = TheatreGenerator(FakeProvider(response), ContextBuilder())

        with self.assertRaises(ProductionValidationError):
            generator.generate("নাটক", [], [], [])

    def test_invalid_stage_zone_is_rejected(self) -> None:
        production = valid_production()
        production["scenes"][0]["blocking"][0]["from"] = "LEFT"
        generator = TheatreGenerator(
            FakeProvider(json.dumps(production, ensure_ascii=False)), ContextBuilder()
        )

        with self.assertRaises(ProductionValidationError):
            generator.generate("নাটক", [], [], [])
