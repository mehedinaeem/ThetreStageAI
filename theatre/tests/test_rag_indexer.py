from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from django.test import SimpleTestCase
from qdrant_client.http import models

from theatre.services.data.schemas import ViewType
from theatre.services.rag.indexer import MultiViewIndexer
from theatre.services.rag.qdrant_store import COLLECTION_BY_VIEW, QdrantStore


class FakeEmbedder:
    dimension = 3

    def __init__(self) -> None:
        self.batch_lengths: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_lengths.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def retrieval_record(view_type: ViewType, number: int) -> dict[str, Any]:
    return {
        "id": f"natok_{number}__{view_type.value}",
        "view_type": view_type.value,
        "source_id": f"natok_{number}",
        "search_text": f"বাংলা {view_type.value} {number}",
        "metadata": {"language": "bn"},
        "payload": {"number": number},
    }


def make_dataset(root: Path, count: int = 3) -> None:
    views = root / "retrieval_views"
    views.mkdir(parents=True, exist_ok=True)
    for view_type in ViewType:
        path = views / f"{view_type.value}_view.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(retrieval_record(view_type, number), ensure_ascii=False)
                for number in range(count)
            )
            + "\n",
            encoding="utf-8",
        )


class MultiViewIndexerTests(SimpleTestCase):
    def test_build_creates_cosine_collections_and_preserves_payload(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_dataset(root)
            embedder = FakeEmbedder()
            store = QdrantStore(root / "qdrant")
            indexer = MultiViewIndexer(embedder, store, upsert_batch_size=2)  # type: ignore[arg-type]

            report = indexer.build(root)

            self.assertEqual(report.counts[ViewType.SCENE], 3)
            self.assertEqual(embedder.batch_lengths, [2, 1, 2, 1, 2, 1])
            collection = store.client.get_collection(COLLECTION_BY_VIEW[ViewType.SCENE])
            self.assertEqual(collection.config.params.vectors.distance, models.Distance.COSINE)
            points, _ = store.client.scroll(
                COLLECTION_BY_VIEW[ViewType.SCENE], limit=1, with_payload=True
            )
            payload = points[0].payload or {}
            self.assertEqual(
                set(payload),
                {"id", "source_id", "view_type", "search_text", "metadata", "payload"},
            )
            self.assertIn("বাংলা", payload["search_text"])
            store.close()

    def test_rebuild_removes_points_no_longer_in_source(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_dataset(root, count=2)
            store = QdrantStore(root / "qdrant")
            indexer = MultiViewIndexer(FakeEmbedder(), store)  # type: ignore[arg-type]
            indexer.build(root)
            make_dataset(root, count=1)

            report = indexer.build(root, rebuild=True)

            self.assertTrue(all(count == 1 for count in report.counts.values()))
            store.close()
