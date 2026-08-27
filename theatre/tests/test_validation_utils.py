from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from django.test import SimpleTestCase

from theatre.services.validation import make_json_safe


class ExampleState(Enum):
    FAILED = "failed"


class ValidationUtilsTests(SimpleTestCase):
    def test_nested_exceptions_and_tuples_become_json_safe(self) -> None:
        value = {
            "type": "value_error",
            "loc": ("scenes", 0),
            "ctx": {
                "error": ValueError("Invalid blocking trigger D03"),
                "nested": ({"path": Path("scene/one")}, RuntimeError("repair failed")),
            },
            "states": {ExampleState.FAILED},
        }

        normalized = make_json_safe(value)

        self.assertEqual(normalized["loc"], ["scenes", 0])
        self.assertEqual(normalized["ctx"]["error"], "Invalid blocking trigger D03")
        self.assertEqual(normalized["ctx"]["nested"][0]["path"], "scene/one")
        self.assertEqual(normalized["ctx"]["nested"][1], "repair failed")
        self.assertEqual(normalized["states"], ["failed"])
        json.dumps(normalized)
