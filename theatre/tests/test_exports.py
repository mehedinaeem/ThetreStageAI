from __future__ import annotations

import csv
import io
import json

from django.test import TestCase
from django.urls import reverse

from theatre.models import TheatreProject
from theatre.services.export_service import (
    BLOCKING_COLUMNS,
    LIGHTING_COLUMNS,
    export_blocking_csv,
    export_json,
    export_lighting_csv,
    export_script_txt,
)


def production_data() -> dict[str, object]:
    return {
        "title": "শেষ আলো",
        "theme": "সম্পর্ক",
        "genre": "family_drama",
        "characters": [{"name": "মায়া", "description": "বড় বোন"}],
        "scenes": [{
            "id": "S01", "title": "মুখোমুখি", "location": "ঘর", "time": "রাত",
            "dialogue": [{"id": "D01", "speaker": "মায়া", "text": "আজ কথা হবে।"}],
            "stage_directions": ["মায়া ধীরে সামনে আসে।"],
            "blocking": [{
                "actor": "মায়া", "from": "USL", "to": "CSC",
                "action": "ধীরে হাঁটে", "trigger": "D01",
            }],
            "lighting": [{
                "cue_id": "LQ01", "trigger": "scene_start", "fixture": "PAR01",
                "focus_zone": "CSC", "rgb": [220, 120, 60],
                "intensity": 45, "fade_seconds": 2,
            }],
        }],
    }


class ExportServiceTests(TestCase):
    def test_json_export_preserves_complete_bengali_structure(self) -> None:
        exported = json.loads(export_json(production_data()).decode("utf-8"))
        self.assertEqual(exported["title"], "শেষ আলো")
        self.assertEqual(exported["characters"][0]["name"], "মায়া")
        self.assertEqual(exported["scenes"][0]["dialogue"][0]["text"], "আজ কথা হবে।")
        self.assertEqual(exported["scenes"][0]["blocking"][0]["from"], "USL")
        self.assertEqual(exported["scenes"][0]["lighting"][0]["rgb"], [220, 120, 60])

    def test_txt_export_is_human_readable_bengali_script(self) -> None:
        exported = export_script_txt(production_data()).decode("utf-8")
        self.assertIn("চরিত্রসমূহ", exported)
        self.assertIn("দৃশ্য S01: মুখোমুখি", exported)
        self.assertIn("মায়া (D01): আজ কথা হবে।", exported)
        self.assertIn("[মায়া ধীরে সামনে আসে।]", exported)

    def test_blocking_csv_has_exact_columns_and_utf8(self) -> None:
        content = export_blocking_csv(production_data())
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(tuple(rows[0]), BLOCKING_COLUMNS)
        self.assertEqual(rows[0]["scene_id"], "S01")
        self.assertEqual(rows[0]["action"], "ধীরে হাঁটে")
        self.assertEqual(rows[0]["from"], "USL")

    def test_lighting_csv_has_exact_columns_and_rgb_channels(self) -> None:
        content = export_lighting_csv(production_data())
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
        self.assertEqual(tuple(rows[0]), LIGHTING_COLUMNS)
        self.assertEqual(
            (rows[0]["red"], rows[0]["green"], rows[0]["blue"]),
            ("220", "120", "60"),
        )

    def test_csv_text_cells_are_safe_from_spreadsheet_formulas(self) -> None:
        hostile = production_data()
        hostile["scenes"][0]["blocking"][0]["action"] = "=HYPERLINK(\"bad\")"
        content = export_blocking_csv(hostile).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(content)))
        self.assertEqual(rows[0]["action"], "'=HYPERLINK(\"bad\")")


class ExportViewTests(TestCase):
    def setUp(self) -> None:
        self.project = TheatreProject.objects.create(
            title="শেষ আলো", user_prompt="brief", actor_count=1,
            duration_minutes=10, generated_json=production_data(),
        )

    def test_all_export_endpoints_download_expected_formats(self) -> None:
        cases = (
            ("json", "application/json", "production.json"),
            ("txt", "text/plain", "script.txt"),
            ("blocking.csv", "text/csv", "blocking.csv"),
            ("lighting.csv", "text/csv", "lighting.csv"),
        )
        for export_format, content_type, filename in cases:
            with self.subTest(export_format=export_format):
                response = self.client.get(
                    reverse(
                        "theatre:export_production",
                        args=[self.project.pk, export_format],
                    )
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response["Content-Type"].startswith(content_type))
                self.assertIn(filename, response["Content-Disposition"])
                self.assertEqual(response["X-Content-Type-Options"], "nosniff")

    def test_project_without_validated_output_cannot_be_exported(self) -> None:
        empty = TheatreProject.objects.create(
            title="Empty", user_prompt="brief", actor_count=1, duration_minutes=10
        )
        response = self.client.get(
            reverse("theatre:export_production", args=[empty.pk, "json"])
        )
        self.assertEqual(response.status_code, 404)

    def test_generated_page_lists_four_exports_and_no_pdf(self) -> None:
        response = self.client.get(
            reverse("theatre:production_detail", args=[self.project.pk])
        )
        self.assertContains(response, "Complete JSON")
        self.assertContains(response, "Bengali script TXT")
        self.assertContains(response, "Blocking CSV")
        self.assertContains(response, "Lighting CSV")
        self.assertNotContains(response, "PDF")
