from qdrant_client import QdrantClient, models
from config import QDRANT_URL, COLLECTION_NAME

class QdrantWrapper:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL)

    def create_collection(self, vector_size):
        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "default": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
                "colbert": models.VectorParams(size=384, distance=models.Distance.COSINE),  # Adjust size based on your model
                "mrl_byte": models.VectorParams(size=4, distance=models.Distance.COSINE)
            }
        )

    def insert_points(self, points):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def multi_stage_query(self, colbert_embedding, byte_vector, full_embedding):
        return self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=models.Prefetch(
                prefetch=models.Prefetch(
                    query=byte_vector,
                    using="mrl_byte",
                    limit=1000,
                ),
                query=full_embedding,
                using="default",
                limit=100,
            ),
            query=colbert_embedding,
            using="colbert",
            limit=10,
        )