from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from django.test import SimpleTestCase

from theatre.services.data.schemas import RetrievalViewDocument, ViewType
from theatre.services.rag.qdrant_store import COLLECTION_BY_VIEW, QdrantStore
from theatre.services.retrieval import (
    BlockingRetriever,
    LightingRetriever,
    MultiViewQueryBuilder,
    SceneRetriever,
)


class RecordingEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def document(view_type: ViewType, number: int, theme: str = "পরিবার") -> RetrievalViewDocument:
    return RetrievalViewDocument(
        id=f"natok_{number}__{view_type.value}",
        view_type=view_type,
        source_id=f"natok_{number}",
        search_text=f"বাংলা {view_type.value} অনুসন্ধান {number}",
        metadata={"theme": theme, "actors_count": 2},
        payload={"record": number},
    )


class QueryBuilderTests(SimpleTestCase):
    def test_queries_are_distinct_and_view_focused(self) -> None:
        queries = MultiViewQueryBuilder().build("দুই চরিত্রের পারিবারিক সংঘাত")

        self.assertNotEqual(queries.scene.text, queries.blocking.text)
        self.assertNotEqual(queries.blocking.text, queries.lighting.text)
        self.assertIn("সংলাপের ভঙ্গি", queries.scene.text)
        self.assertIn("মঞ্চে চলাচল", queries.blocking.text)
        self.assertIn("RGB আলো", queries.lighting.text)

    def test_empty_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            MultiViewQueryBuilder().build(" ")


class RetrieverTests(SimpleTestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.store = QdrantStore(Path(self.temporary_directory.name) / "qdrant")
        for view_type in ViewType:
            collection = COLLECTION_BY_VIEW[view_type]
            self.store.ensure_collection(collection, 3)
            documents = [
                document(view_type, number, "পরিবার" if number < 4 else "পরিবেশ")
                for number in range(6)
            ]
            vectors = [[1.0, float(number) / 20, 0.0] for number in range(6)]
            self.store.upsert(collection, documents, vectors)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary_directory.cleanup()

    def test_retrievers_use_independent_queries_and_default_limits(self) -> None:
        embedder = RecordingEmbedder()

        scene = SceneRetriever(embedder, self.store).retrieve("পারিবারিক সংঘাত")
        blocking = BlockingRetriever(embedder, self.store).retrieve("পারিবারিক সংঘাত")
        lighting = LightingRetriever(embedder, self.store).retrieve("পারিবারিক সংঘাত")

        self.assertEqual(len(scene), 5)
        self.assertEqual(len(blocking), 3)
        self.assertEqual(len(lighting), 3)
        self.assertEqual(len(set(embedder.queries)), 3)
        self.assertEqual(scene[0].rank, 1)
        self.assertEqual(scene[0].view_type, ViewType.SCENE)
        self.assertTrue(scene[0].search_text)
        self.assertIn("record", scene[0].payload)

    def test_metadata_filter_is_applied(self) -> None:
        results = SceneRetriever(RecordingEmbedder(), self.store).retrieve(
            "পারিবারিক সংঘাত",
            metadata_filters={"theme": "পরিবেশ"},
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.metadata["theme"] == "পরিবেশ" for result in results))

    def test_unknown_metadata_filter_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported metadata"):
            SceneRetriever(RecordingEmbedder(), self.store).retrieve(
                "সংঘাত", metadata_filters={"unknown": "value"}
            )

    def test_malformed_qdrant_payloads_are_skipped_without_breaking_results(self) -> None:
        response = SimpleNamespace(points=[
            SimpleNamespace(payload=["not", "an", "object"], score=0.99),
            SimpleNamespace(payload={"search_text": "missing source"}, score=0.98),
            SimpleNamespace(payload={
                "source_id": "safe-source", "search_text": "নিরাপদ রেফারেন্স",
                "metadata": {}, "payload": {},
            }, score=0.9),
        ])
        fake_store = SimpleNamespace(
            client=SimpleNamespace(query_points=lambda **_: response)
        )

        results = SceneRetriever(RecordingEmbedder(), fake_store).retrieve("সংঘাত")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_id, "safe-source")
        self.assertEqual(results[0].rank, 1)
