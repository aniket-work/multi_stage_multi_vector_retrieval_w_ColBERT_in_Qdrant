from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, text):
        return self.model.encode(text)

    def generate_colbert_embedding(self, text):
        tokens = self.model.tokenize([text])
        embeddings = self.model.encode(tokens, output_value='token_embeddings')[0]
        return embeddings.tolist()

    def generate_byte_vector(self, vector, size=4):
        return (np.array(vector) * 255).astype(np.uint8)[:size].tolist()