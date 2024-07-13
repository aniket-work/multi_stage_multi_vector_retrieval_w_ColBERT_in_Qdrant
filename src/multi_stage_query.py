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
            colbert_embedding = self.embedding_generator.generate_colbert_embedding(doc)
            byte_vector = self.embedding_generator.generate_byte_vector(embedding)

            points.append(models.PointStruct(
                id=i,
                vector={
                    "default": embedding,
                    "colbert": colbert_embedding,
                    "mrl_byte": byte_vector
                },
                payload={"text": doc}
            ))

        self.qdrant_wrapper.insert_points(points)

    def query(self, query_text):
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        query_colbert = self.embedding_generator.generate_colbert_embedding(query_text)
        query_byte = self.embedding_generator.generate_byte_vector(query_embedding)

        results = self.qdrant_wrapper.multi_stage_query(query_colbert, query_byte, query_embedding)
        return results