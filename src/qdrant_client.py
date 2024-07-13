from qdrant_client import QdrantClient, models
from config import QDRANT_URL, COLLECTION_NAME

class QdrantWrapper:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)

    def create_collection(self, vector_size):
        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def insert_points(self, points):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def multi_stage_query(self, query_vector, limit=10):
        return self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )