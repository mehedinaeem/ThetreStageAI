from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from theatre.services.data.loader import (
    detect_view_type,
    load_original_records,
    load_retrieval_records,
)
from theatre.services.data.schemas import ViewType


class DataLoaderTests(SimpleTestCase):
    def test_original_loader_preserves_bengali_and_skips_bad_json(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "original.jsonl"
            path.write_text(
                '{"id":"natok_1","title":"বাংলা নাটক"}\nnot-json\n',
                encoding="utf-8",
            )

            records, errors = load_original_records(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].title, "বাংলা নাটক")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].line_number, 2)

    def test_retrieval_loader_returns_typed_document_and_detects_view(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scene_view.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "natok_1__scene",
                        "source_id": "natok_1",
                        "search_text": "বাংলা অনুসন্ধান লেখা",
                        "metadata": {},
                        "payload": {},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            records, errors = load_retrieval_records(path)

        self.assertFalse(errors)
        self.assertEqual(records[0].view_type, ViewType.SCENE)
        self.assertEqual(records[0].search_text, "বাংলা অনুসন্ধান লেখা")

    def test_detect_view_type_rejects_unknown_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unable to detect"):
            detect_view_type({"id": "unknown"})
