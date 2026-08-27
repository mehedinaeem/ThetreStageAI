from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from django.test import SimpleTestCase

from theatre.services.validation import ConstraintValidator, OutputValidator, Production


class RepairProvider:
    def __init__(self, corrected: dict[str, Any]) -> None:
        self.corrected = corrected
        self.prompts: list[str] = []

    def generate(self, prompt: str, **_: Any) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.corrected, ensure_ascii=False)


def requirements() -> dict[str, Any]:
    return {
        "story_idea": "অনলাইনে ছড়িয়ে পড়া একটি ভুয়া পোস্ট নিয়ে দুই শিক্ষার্থীর সংঘাত",
        "theme": "Fake News and Responsible Social Media Use",
        "genre": "social_drama",
        "language": "bn",
        "actor_count": 2,
        "duration_minutes": 10,
        "stage_size": "small",
        "available_lights": [
            "RGB_PAR_01",
            "RGB_PAR_02",
            "RGB_PAR_03",
            "RGB_PAR_04",
        ],
        "scene_time": "সন্ধ্যা",
        "desired_emotion": "রাগ, দ্বিধা, উত্তেজনা, উপলব্ধি",
    }


def complete_production() -> dict[str, Any]:
    characters = [
        {"name": "নীরা", "description": "বিশ্ববিদ্যালয়ের শিক্ষার্থী"},
        {"name": "সজল", "description": "নীরার সহপাঠী"},
    ]
    scenes: list[dict[str, Any]] = []
    for scene_number in (1, 2):
        dialogue = [
            {
                "id": f"D{scene_number}{line_number:02d}",
                "speaker": characters[(line_number - 1) % 2]["name"],
                "text": f"ভুয়া পোস্ট নিয়ে সত্য যাচাইয়ের আলোচনা {line_number}।",
            }
            for line_number in range(1, 10)
        ]
        scenes.append(
            {
                "id": f"S{scene_number:02d}",
                "title": "মুখোমুখি" if scene_number == 1 else "সত্য প্রকাশ",
                "location": "বিশ্ববিদ্যালয়ের ছোট মিলনায়তন",
                "time": "সন্ধ্যা",
                "dialogue": dialogue,
                "stage_directions": ["দুই শিক্ষার্থী ছোট মঞ্চে মুখোমুখি দাঁড়ায়।"],
                "blocking": [
                    {
                        "actor": "নীরা",
                        "from": "USL",
                        "to": "CSC",
                        "action": "ধীরে সামনে এসে ফোনটি দেখায়",
                        "trigger": dialogue[0]["id"],
                    }
                ],
                "lighting": [
                    {
                        "cue_id": f"LQ{scene_number:02d}",
                        "trigger": "scene_start",
                        "fixture": f"RGB_PAR_0{scene_number}",
                        "focus_zone": "CSC",
                        "rgb": [220, 90, 50],
                        "intensity": 65,
                        "fade_seconds": 2,
                    }
                ],
            }
        )
    return {
        "title": "পোস্টের ওপারে সত্য",
        "theme": "Fake News and Responsible Social Media Use",
        "genre": "social_drama",
        "characters": characters,
        "scenes": scenes,
    }


class ConstraintValidatorRegressionTests(SimpleTestCase):
    def test_observed_structurally_valid_failure_is_semantically_rejected(self) -> None:
        invalid = complete_production()
        invalid["title"] = "আু<tool_call>া খবর - ছোট নাটক 497"
        invalid["theme"] = "সাংস্কৃতিক ও রাজনীতির সংমিশ্রণ"
        invalid["genre"] = "প্রেম ও রাজনীতি"
        invalid["scenes"] = [deepcopy(invalid["scenes"][0])]
        invalid["scenes"][0]["dialogue"] = [invalid["scenes"][0]["dialogue"][0]]
        invalid["scenes"][0]["blocking"][0]["trigger"] = invalid["scenes"][0]["dialogue"][0]["id"]
        invalid["scenes"][0]["lighting"][0]["fixture"] = "PAR01"

        errors = ConstraintValidator().validate(
            Production.model_validate(invalid), requirements()
        )
        combined = "\n".join(errors).lower()

        self.assertIn("forbidden model-control token", combined)
        self.assertIn("insufficient scenes", combined)
        self.assertIn("insufficient dialogue", combined)
        self.assertIn("genre mismatch", combined)
        self.assertIn("invalid fixture", combined)
        self.assertIn("retrieved-title copying", combined)

    def test_complete_ten_minute_production_satisfies_constraints(self) -> None:
        production = Production.model_validate(complete_production())

        errors = ConstraintValidator().validate(production, requirements())

        self.assertEqual(errors, [])

    def test_semantic_failure_uses_one_expanding_repair_with_requirements(self) -> None:
        invalid = complete_production()
        invalid["scenes"] = [invalid["scenes"][0]]
        invalid["scenes"][0]["dialogue"] = [invalid["scenes"][0]["dialogue"][0]]
        invalid["scenes"][0]["blocking"][0]["trigger"] = invalid["scenes"][0]["dialogue"][0]["id"]
        full_requirements = "\n".join(
            f"{key}: {value}" for key, value in requirements().items()
        )
        provider = RepairProvider(complete_production())

        result = OutputValidator(provider).validate_with_details(
            json.dumps(invalid, ensure_ascii=False),
            user_requirements=full_requirements,
            constraints=requirements(),
        )

        self.assertTrue(result.repaired)
        self.assertEqual(len(provider.prompts), 1)
        self.assertIn(full_requirements, provider.prompts[0])
        self.assertIn("Insufficient scenes", provider.prompts[0])
        self.assertIn("Correct and EXPAND", provider.prompts[0])
