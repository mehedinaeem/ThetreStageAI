"""Inspect the configured Bengali theatre JSONL dataset."""
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from theatre.services.data import inspect_dataset


class Command(BaseCommand):
    help = "Validate the theatre dataset and display record statistics."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--dataset-path",
            type=Path,
            default=None,
            help="Dataset root override (defaults to THEATRE_DATASET_PATH).",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        dataset_path = options["dataset_path"] or settings.THEATRE_DATASET_PATH
        result = inspect_dataset(dataset_path)
        stats = result.statistics

        self.stdout.write(self.style.MIGRATE_HEADING("ThetreStageAI Dataset Inspection"))
        self.stdout.write(f"Dataset: {Path(dataset_path).expanduser().resolve()}")
        self.stdout.write(f"ORIGINAL:       {stats.original_records}")
        self.stdout.write(f"SCENE VIEW:     {stats.scene_records}")
        self.stdout.write(f"BLOCKING VIEW:  {stats.blocking_records}")
        self.stdout.write(f"LIGHTING VIEW:  {stats.lighting_records}")
        self.stdout.write(f"ERRORS:         {stats.malformed_records}")
        self.stdout.write(f"  Missing search_text: {stats.missing_search_text}")
        self.stdout.write(f"  Missing metadata:    {stats.missing_metadata}")
        self.stdout.write(f"  Missing payload:     {stats.missing_payload}")
