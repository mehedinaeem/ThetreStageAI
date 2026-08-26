from django.test import SimpleTestCase

from theatre.services.rag.embeddings import EmbeddingService


class FakeArray:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, sentences: list[str], **kwargs: object) -> FakeArray:
        self.options = kwargs
        return FakeArray([[1.0, 0.0, 0.0] for _ in sentences])


class EmbeddingServiceTests(SimpleTestCase):
    def test_embedding_requests_normalized_batched_vectors(self) -> None:
        model = FakeSentenceTransformer()
        service = EmbeddingService("test-model", batch_size=7, model=model)

        vectors = service.embed(["বাংলা দৃশ্য", "আলো"])

        self.assertEqual(len(vectors), 2)
        self.assertEqual(service.dimension, 3)
        self.assertEqual(model.options["batch_size"], 7)
        self.assertIs(model.options["normalize_embeddings"], True)
        self.assertIs(model.options["convert_to_numpy"], True)

    def test_empty_search_text_is_rejected(self) -> None:
        service = EmbeddingService("test-model", model=FakeSentenceTransformer())
        with self.assertRaisesRegex(ValueError, "empty search text"):
            service.embed(["  "])
