from src.embeddings import EmbeddingGenerator
from src.qdrant_client import QdrantWrapper


class MultiStageQuery:
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.qdrant_wrapper = QdrantWrapper()

    def prepare_data(self, documents):
        points = []
        for i, doc in enumerate(documents):
            embedding = self.embedding_generator.generate_embedding(doc)
            colbert_embedding = self.embedding_generator.generate_colbert_embedding(doc)
            byte_vector = self.embedding_generator.generate_byte_vector(embedding)

            points.append(models.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": doc,
                    "colbert": colbert_embedding,
                    "mrl_byte": byte_vector
                }
            ))

        self.qdrant_wrapper.create_collection(len(embedding))
        self.qdrant_wrapper.insert_points(points)

    def query(self, query_text):
        query_embedding = self.embedding_generator.generate_embedding(query_text)
        query_colbert = self.embedding_generator.generate_colbert_embedding(query_text)
        query_byte = self.embedding_generator.generate_byte_vector(query_embedding)

        results = self.qdrant_wrapper.multi_stage_query(query_colbert, query_byte)
        return results