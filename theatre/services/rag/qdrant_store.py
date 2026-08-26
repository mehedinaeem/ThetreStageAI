"""Persistent local Qdrant storage for independent retrieval views."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from theatre.services.data.schemas import RetrievalViewDocument, ViewType

logger = logging.getLogger(__name__)

COLLECTION_BY_VIEW: dict[ViewType, str] = {
    ViewType.SCENE: "thetrestageai_scene",
    ViewType.BLOCKING: "thetrestageai_blocking",
    ViewType.LIGHTING: "thetrestageai_lighting",
}


class QdrantStore:
    """Small adapter around Qdrant local persistent mode."""

    def __init__(self, path: str | Path, *, client: QdrantClient | None = None) -> None:
        self.path = Path(path).expanduser().resolve()
        self._client = client

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self.path.mkdir(parents=True, exist_ok=True)
            logger.info("Opening persistent local Qdrant storage at %s", self.path)
            self._client = QdrantClient(path=str(self.path))
        return self._client

    def ensure_collection(self, name: str, vector_size: int, *, rebuild: bool = False) -> None:
        exists = self.client.collection_exists(name)
        if exists and rebuild:
            logger.warning("Rebuilding Qdrant collection %s", name)
            self.client.delete_collection(name)
            exists = False
        if not exists:
            logger.info("Creating Qdrant collection %s with %d dimensions", name, vector_size)
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(
        self,
        collection_name: str,
        documents: Sequence[RetrievalViewDocument],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        if len(documents) != len(vectors):
            raise ValueError("Document and vector counts must match")
        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, document.id)),
                vector=list(vector),
                payload={
                    "id": document.id,
                    "source_id": document.source_id,
                    "view_type": document.view_type.value,
                    "search_text": document.search_text,
                    "metadata": document.metadata,
                    "payload": document.payload,
                },
            )
            for document, vector in zip(documents, vectors, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=collection_name, points=points, wait=True)

    def count(self, collection_name: str) -> int:
        return int(self.client.count(collection_name=collection_name, exact=True).count)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
