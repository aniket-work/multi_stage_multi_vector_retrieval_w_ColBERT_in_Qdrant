from qdrant_client import models
from src.embeddings import EmbeddingGenerator
from src.qdrant_client import QdrantWrapper

class MultiStageQuery:
    def __init__(self, colbert_embedding_dim=12):
        self.embedding_generator = EmbeddingGenerator(colbert_embedding_dim=colbert_embedding_dim)
        self.qdrant_wrapper = QdrantWrapper()

    def prepare_data(self, documents):
        sample_embedding = self.embedding_generator.generate_embedding(documents[0])
        sample_colbert = self.embedding_generator.generate_colbert_embedding(documents[0])
        sample_byte = self.embedding_generator.generate_byte_vector(sample_embedding)

        print(f"Default embedding dimension: {len(sample_embedding)}")
        print(f"ColBERT embedding dimension: {len(sample_colbert)}")
        print(f"Byte vector dimension: {len(sample_byte)}")

        self.qdrant_wrapper.create_collection({
            "default": len(sample_embedding),
            "colbert": len(sample_colbert),
            "mrl_byte": len(sample_byte)
        })

        points = []
        for i, doc in enumerate(documents):
            default_embedding = self.embedding_generator.generate_embedding(doc)
            colbert_embedding = self.embedding_generator.generate_colbert_embedding(doc)
            byte_vector = self.embedding_generator.generate_byte_vector(default_embedding)

            points.append(models.PointStruct(
                id=i,
                vector={
                    "default": default_embedding,
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

        print(f"Query default embedding dimension: {len(query_embedding)}")
        print(f"Query ColBERT embedding dimension: {len(query_colbert)}")
        print(f"Query byte vector dimension: {len(query_byte)}")

        results = self.qdrant_wrapper.multi_stage_query({
            "default": query_embedding,
            "colbert": query_colbert,
            "mrl_byte": query_byte
        })
        return results
