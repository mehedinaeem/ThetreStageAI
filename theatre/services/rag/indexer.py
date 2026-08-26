"""Orchestration for indexing the three independent retrieval views."""
from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from theatre.services.data.loader import load_retrieval_records
from theatre.services.data.schemas import MalformedRecord, RetrievalViewDocument, ViewType

from .embeddings import EmbeddingService
from .qdrant_store import COLLECTION_BY_VIEW, QdrantStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ViewType, int, int], None]

VIEW_FILE_BY_TYPE: dict[ViewType, str] = {
    ViewType.SCENE: "scene_view.jsonl",
    ViewType.BLOCKING: "blocking_view.jsonl",
    ViewType.LIGHTING: "lighting_view.jsonl",
}


@dataclass(slots=True)
class IndexReport:
    counts: dict[ViewType, int] = field(default_factory=dict)
    malformed_records: list[MalformedRecord] = field(default_factory=list)


class MultiViewIndexer:
    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantStore,
        *,
        upsert_batch_size: int = 64,
    ) -> None:
        if upsert_batch_size < 1:
            raise ValueError("Qdrant upsert batch size must be positive")
        self.embedder = embedder
        self.store = store
        self.upsert_batch_size = upsert_batch_size

    def load_documents(
        self, dataset_path: str | Path
    ) -> tuple[dict[ViewType, list[RetrievalViewDocument]], list[MalformedRecord]]:
        root = Path(dataset_path).expanduser().resolve() / "retrieval_views"
        documents: dict[ViewType, list[RetrievalViewDocument]] = {}
        errors: list[MalformedRecord] = []
        for view_type, filename in VIEW_FILE_BY_TYPE.items():
            records, file_errors = load_retrieval_records(root / filename)
            mismatches = [record for record in records if record.view_type is not view_type]
            if mismatches:
                raise ValueError(
                    f"{filename} contains {len(mismatches)} record(s) with a different view_type"
                )
            documents[view_type] = records
            errors.extend(file_errors)
        return documents, errors

    def build(
        self,
        dataset_path: str | Path,
        *,
        rebuild: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexReport:
        documents_by_view, errors = self.load_documents(dataset_path)
        dimension = self.embedder.dimension
        report = IndexReport(malformed_records=errors)

        for view_type, documents in documents_by_view.items():
            collection_name = COLLECTION_BY_VIEW[view_type]
            self.store.ensure_collection(collection_name, dimension, rebuild=rebuild)
            total = len(documents)
            for start in range(0, total, self.upsert_batch_size):
                batch = documents[start : start + self.upsert_batch_size]
                vectors = self.embedder.embed([record.search_text for record in batch])
                self.store.upsert(collection_name, batch, vectors)
                completed = min(start + len(batch), total)
                logger.info("Indexed %s: %d/%d", view_type.value, completed, total)
                if progress is not None:
                    progress(view_type, completed, total)
            report.counts[view_type] = self.store.count(collection_name)

        return report
