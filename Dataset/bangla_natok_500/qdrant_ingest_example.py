from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import json

MODEL_NAME = "BAAI/bge-m3"
VIEW_FILE = "retrieval_views/scene_view.jsonl"

model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(path="./qdrant_db")

docs = [json.loads(line) for line in open(VIEW_FILE, encoding="utf-8") if line.strip()]
vectors = model.encode([d["search_text"] for d in docs], normalize_embeddings=True)

collection = "natya_scene"
client.recreate_collection(
    collection_name=collection,
    vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
)

points = []
for i, (doc, vec) in enumerate(zip(docs, vectors)):
    points.append(PointStruct(
        id=i,
        vector=vec.tolist(),
        payload={
            "id": doc["id"],
            "source_id": doc["source_id"],
            "search_text": doc["search_text"],
            "metadata": doc["metadata"],
            "payload": doc["payload"]
        }
    ))

client.upsert(collection_name=collection, points=points)
print(f"Indexed {len(points)} documents into {collection}")
