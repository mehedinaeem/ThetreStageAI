"""Build the three persistent local Qdrant retrieval collections."""
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from theatre.services.data.schemas import ViewType
from theatre.services.rag import (
    COLLECTION_BY_VIEW,
    EmbeddingService,
    MultiViewIndexer,
    QdrantStore,
)


class Command(BaseCommand):
    help = "Embed and index the scene, blocking, and lighting retrieval views."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--dataset-path",
            type=Path,
            default=None,
            help="Dataset root override (defaults to THEATRE_DATASET_PATH).",
        )
        parser.add_argument(
            "--rebuild",
            action="store_true",
            help="Delete and recreate only the three ThetreStageAI collections.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        dataset_path = options["dataset_path"] or settings.THEATRE_DATASET_PATH
        embedder = EmbeddingService(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        store = QdrantStore(settings.QDRANT_PATH)
        indexer = MultiViewIndexer(
            embedder,
            store,
            upsert_batch_size=settings.QDRANT_UPSERT_BATCH_SIZE,
        )

        self.stdout.write(self.style.MIGRATE_HEADING("Building ThetreStageAI RAG index"))
        self.stdout.write(f"Dataset: {Path(dataset_path).expanduser().resolve()}")
        self.stdout.write(f"Qdrant:  {Path(settings.QDRANT_PATH).resolve()}")
        self.stdout.write(f"Model:   {settings.EMBEDDING_MODEL_NAME}")

        def show_progress(view_type: ViewType, completed: int, total: int) -> None:
            self.stdout.write(f"{view_type.value.upper():8} {completed}/{total}")

        try:
            report = indexer.build(
                dataset_path,
                rebuild=options["rebuild"],
                progress=show_progress,
            )
        finally:
            store.close()

        if report.malformed_records:
            self.stdout.write(
                self.style.WARNING(
                    f"Skipped malformed records: {len(report.malformed_records)}"
                )
            )
        self.stdout.write(self.style.SUCCESS("RAG index build complete"))
        for view_type in ViewType:
            self.stdout.write(
                f"{view_type.value.upper():8}: "
                f"{report.counts.get(view_type, 0)} points "
                f"({COLLECTION_BY_VIEW[view_type]})"
            )
