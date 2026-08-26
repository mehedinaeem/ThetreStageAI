"""Lazy sentence-transformers embedding adapter."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SentenceTransformerLike(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...

    def get_sentence_embedding_dimension(self) -> int | None: ...


class EmbeddingService:
    """Generate normalized dense embeddings without loading the model at import time."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        model: SentenceTransformerLike | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> SentenceTransformerLike:
        if self._model is None:
            logger.info("Loading sentence-transformers model %s on %s", self.model_name, self.device)
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required to build the RAG index. "
                    "Install requirements.txt first."
                ) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None or dimension < 1:
            raise RuntimeError("Embedding model did not report a valid vector dimension")
        return dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode a batch of non-empty texts as unit-normalized float vectors."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty search text")

        encoded = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        values = encoded.tolist() if hasattr(encoded, "tolist") else encoded
        vectors = [[float(component) for component in vector] for vector in values]
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding model returned an unexpected number of vectors")
        return vectors
