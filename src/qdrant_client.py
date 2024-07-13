from qdrant_client import QdrantClient, models

class QdrantWrapper:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(url=f"http://{host}:{port}")

    def create_collection(self, vector_sizes):
        self.client.recreate_collection(
            collection_name="multi_stage_collection",
            vectors_config={
                name: models.VectorParams(size=size, distance=models.Distance.COSINE)
                for name, size in vector_sizes.items()
            },
        )

    def insert_points(self, points):
        self.client.upsert(
            collection_name="multi_stage_collection",
            points=points,
        )

    def multi_stage_query(self, query_vectors):
        response = self.client.query(
            collection_name="multi_stage_collection",
            query_vector=query_vectors["default"],
            search_params=models.SearchParams(
                using="default",
                limit=10,
                prefetch=[
                    models.Prefetch(
                        query=query_vectors["mrl_byte"],
                        using="mrl_byte",
                        limit=1000,
                    ),
                    models.Prefetch(
                        query=query_vectors["colbert"],
                        using="colbert",
                        limit=100,
                    ),
                ]
            )
        )
        return response
