from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from django.core.management import call_command
from django.test import SimpleTestCase

from theatre.services.data.validator import inspect_dataset


def write_jsonl(path: Path, records: list[dict[str, Any] | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        record if isinstance(record, str) else json.dumps(record, ensure_ascii=False)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def retrieval_record(view_type: str) -> dict[str, Any]:
    return {
        "id": f"natok_1__{view_type}",
        "view_type": view_type,
        "source_id": "natok_1",
        "search_text": "বাংলা লেখা",
        "metadata": {"language": "bn"},
        "payload": {"summary": "সারাংশ"},
    }


class DatasetValidatorTests(SimpleTestCase):
    def make_dataset(self, root: Path) -> None:
        write_jsonl(
            root / "bangla_natok_500.jsonl",
            [{"id": "natok_1", "title": "পরীক্ষার নাটক"}, "bad-json"],
        )
        write_jsonl(
            root / "retrieval_views" / "scene_view.jsonl",
            [retrieval_record("scene")],
        )
        malformed_blocking = retrieval_record("blocking")
        malformed_blocking.pop("search_text")
        write_jsonl(
            root / "retrieval_views" / "blocking_view.jsonl",
            [retrieval_record("blocking"), malformed_blocking],
        )
        write_jsonl(
            root / "retrieval_views" / "lighting_view.jsonl",
            [retrieval_record("lighting")],
        )

    def test_inspection_aggregates_valid_and_malformed_records(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_dataset(root)

            result = inspect_dataset(root)

        stats = result.statistics
        self.assertEqual(stats.original_records, 1)
        self.assertEqual(stats.scene_records, 1)
        self.assertEqual(stats.blocking_records, 1)
        self.assertEqual(stats.lighting_records, 1)
        self.assertEqual(stats.malformed_records, 2)
        self.assertEqual(stats.missing_search_text, 1)
        self.assertEqual(stats.missing_metadata, 0)
        self.assertEqual(stats.missing_payload, 0)

    def test_management_command_prints_expected_sections(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_dataset(root)
            output = StringIO()

            call_command("inspect_theatre_dataset", dataset_path=root, stdout=output)

        rendered = output.getvalue()
        self.assertIn("ORIGINAL:       1", rendered)
        self.assertIn("SCENE VIEW:     1", rendered)
        self.assertIn("BLOCKING VIEW:  1", rendered)
        self.assertIn("LIGHTING VIEW:  1", rendered)
        self.assertIn("ERRORS:         2", rendered)
