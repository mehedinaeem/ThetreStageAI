from __future__ import annotations

from copy import deepcopy

from django.test import SimpleTestCase

from theatre.services.llm.prompts import build_generation_prompt
from theatre.services.llm.schema_builder import (
    build_generation_schema,
    duration_minimums,
)
from theatre.services.validation import Production


class DynamicGenerationSchemaTests(SimpleTestCase):
    fixtures = ["RGB_PAR_01", "RGB_PAR_02", "RGB_PAR_03", "RGB_PAR_04"]

    def test_one_minute_schema_constrains_count_dialogue_and_fixtures(self) -> None:
        base = Production.json_schema()
        original = deepcopy(base)

        schema = build_generation_schema(
            base,
            actor_count=2,
            duration_minutes=1,
            available_lights=self.fixtures,
        )

        self.assertEqual(schema["properties"]["characters"]["minItems"], 2)
        self.assertEqual(schema["properties"]["characters"]["maxItems"], 2)
        self.assertEqual(schema["properties"]["scenes"]["minItems"], 1)
        self.assertEqual(
            schema["$defs"]["Scene"]["properties"]["dialogue"]["minItems"], 8
        )
        self.assertEqual(
            schema["$defs"]["LightingCue"]["properties"]["fixture"]["enum"],
            self.fixtures,
        )
        self.assertEqual(
            schema["$defs"]["StageZone"]["enum"],
            ["USL", "USC", "USR", "CSL", "CSC", "CSR", "DSL", "DSC", "DSR"],
        )
        self.assertEqual(base, original, "The shared Pydantic schema must not be mutated")

    def test_ten_minute_schema_requires_two_scenes_and_nine_lines_each(self) -> None:
        schema = build_generation_schema(
            Production.json_schema(),
            actor_count=2,
            duration_minutes=10,
            available_lights=self.fixtures,
        )

        self.assertEqual(schema["properties"]["scenes"]["minItems"], 2)
        self.assertEqual(
            schema["$defs"]["Scene"]["properties"]["dialogue"]["minItems"], 9
        )

    def test_empty_fixture_list_preserves_generic_string_schema(self) -> None:
        schema = build_generation_schema(
            Production.json_schema(),
            actor_count=1,
            duration_minutes=5,
            available_lights=[],
        )

        fixture = schema["$defs"]["LightingCue"]["properties"]["fixture"]
        self.assertNotIn("enum", fixture)
        self.assertEqual(fixture["type"], "string")

    def test_prompt_repeats_ten_minute_numeric_and_fixture_requirements(self) -> None:
        prompt = build_generation_prompt(
            "USER REQUIREMENTS\nbrief",
            {"type": "object"},
            minimums=duration_minimums(10),
            available_lights=self.fixtures,
        )

        self.assertIn("Create at least 2 scenes", prompt)
        self.assertIn("at least 9 dialogue entries", prompt)
        self.assertIn("at least 18 dialogue entries", prompt)
        self.assertIn("AVAILABLE LIGHTING FIXTURES:\nRGB_PAR_01", prompt)
        self.assertIn("Do not use PAR01/PAR02 aliases", prompt)
