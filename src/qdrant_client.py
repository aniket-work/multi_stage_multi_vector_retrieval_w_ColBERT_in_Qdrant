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

    def multi_stage_query(self, query_vectors, byte_vector):
        return self.client.search(
            collection_name=COLLECTION_NAME,
            query_filter=None,
            search_params=models.SearchParams(
                hnsw_ef=128,
                exact=False
            ),
            query_vector=query_vectors[0],  # Use the first vector for the main query
            limit=10,
            with_payload=True,
            with_vectors=True,
        )