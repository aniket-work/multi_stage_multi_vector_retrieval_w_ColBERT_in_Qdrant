from qdrant_client import models
from src.embeddings import EmbeddingGenerator
from src.qdrant_client import QdrantWrapper

class MultiStageQuery:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.qdrant_wrapper = QdrantWrapper()

    def prepare_data(self, documents):
        sample_embedding = self.embedding_generator.generate_embedding(documents[0])
        self.qdrant_wrapper.create_collection(vector_size=len(sample_embedding))

        points = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_generator.generate_embedding(doc)
            points.append(models.PointStruct(
                id=i,
                vector=embedding,
                payload={"text": doc}
            ))

        self.qdrant_wrapper.insert_points(points)

    def query(self, query_text):
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        results = self.qdrant_wrapper.multi_stage_query(query_embedding)
        return results