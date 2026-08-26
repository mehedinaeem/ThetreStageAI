"""Shared Qdrant retrieval behavior and typed results."""
from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client.http import models

from theatre.services.data.schemas import ViewType
from theatre.services.rag.embeddings import EmbeddingService
from theatre.services.rag.qdrant_store import COLLECTION_BY_VIEW, QdrantStore

from .query_builder import MetadataValue, MultiViewQueryBuilder

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    source_id: str
    view_type: ViewType
    search_text: str
    metadata: dict[str, Any]
    payload: dict[str, Any]


class BaseRetriever:
    view_type: ClassVar[ViewType]
    default_limit: ClassVar[int]
    allowed_metadata_filters: ClassVar[frozenset[str]] = frozenset(
        {
            "language",
            "theme",
            "genre",
            "scene_type",
            "location",
            "time",
            "actors_count",
            "emotion",
            "document_type",
        }
    )

    def __init__(
        self,
        embedder: EmbeddingService,
        store: QdrantStore,
        *,
        query_builder: MultiViewQueryBuilder | None = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.query_builder = query_builder or MultiViewQueryBuilder()

    def retrieve(
        self,
        user_request: str,
        *,
        limit: int | None = None,
        metadata_filters: dict[str, MetadataValue] | None = None,
        score_threshold: float | None = None,
        query_text: str | None = None,
    ) -> list[RetrievalResult]:
        result_limit = self.default_limit if limit is None else limit
        if result_limit < 1:
            raise ValueError("Retrieval limit must be positive")
        query = self.query_builder.build_for_view(
            user_request if query_text is None else query_text,
            self.view_type,
            metadata_filters=metadata_filters,
        )
        embedding_text = query.text if query_text is None else query_text.strip()
        if not embedding_text:
            raise ValueError("Retrieval query cannot be empty")
        vector = self.embedder.embed([embedding_text])[0]
        query_filter = self._build_filter(query.metadata_filters)
        collection = COLLECTION_BY_VIEW[self.view_type]
        logger.info(
            "Retrieving top %d from %s with %d metadata filter(s)",
            result_limit,
            collection,
            len(query.metadata_filters),
        )
        try:
            response = self.store.client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=query_filter,
                limit=result_limit,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
            )
        except Exception:
            logger.exception("Retrieval failed for collection %s", collection)
            raise

        results: list[RetrievalResult] = []
        for rank, point in enumerate(response.points, start=1):
            stored = point.payload or {}
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(point.score),
                    source_id=str(stored.get("source_id", "")),
                    view_type=self.view_type,
                    search_text=str(stored.get("search_text", "")),
                    metadata=self._mapping(stored.get("metadata")),
                    payload=self._mapping(stored.get("payload")),
                )
            )
        return results

    def _build_filter(
        self, metadata_filters: dict[str, MetadataValue]
    ) -> models.Filter | None:
        unknown = set(metadata_filters) - self.allowed_metadata_filters
        if unknown:
            raise ValueError(f"Unsupported metadata filter(s): {', '.join(sorted(unknown))}")
        if not metadata_filters:
            return None
        conditions = [
            models.FieldCondition(
                key=f"metadata.{key}",
                match=models.MatchValue(value=value),
            )
            for key, value in metadata_filters.items()
        ]
        return models.Filter(must=conditions)

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}
