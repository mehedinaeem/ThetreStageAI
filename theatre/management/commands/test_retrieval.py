"""Exercise all three view-specific retrievers from the command line."""
from __future__ import annotations

import json
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from theatre.services.rag import EmbeddingService, QdrantStore
from theatre.services.retrieval import BlockingRetriever, LightingRetriever, SceneRetriever


class Command(BaseCommand):
    help = "Run one request through the scene, blocking, and lighting retrievers."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("request", type=str, help="Bengali theatre-production request")

    def handle(self, *args: Any, **options: Any) -> None:
        embedder = EmbeddingService(
            settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
        )
        store = QdrantStore(settings.QDRANT_PATH)
        retrievers = (
            ("SCENE", SceneRetriever(embedder, store)),
            ("BLOCKING", BlockingRetriever(embedder, store)),
            ("LIGHTING", LightingRetriever(embedder, store)),
        )
        try:
            for label, retriever in retrievers:
                self.stdout.write(f"\n=== {label} RESULTS ===")
                results = retriever.retrieve(options["request"])
                if not results:
                    self.stdout.write("No results.")
                for result in results:
                    self.stdout.write(
                        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2)
                    )
        finally:
            store.close()
