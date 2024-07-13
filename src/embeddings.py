from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, text):
        return self.model.encode(text).tolist()

    def generate_colbert_embedding(self, text):
        embeddings = self.model.encode(text, output_value='token_embeddings')
        if len(embeddings.shape) == 2:
            embeddings = [embeddings]
        return embeddings[0].tolist()

    def generate_byte_vector(self, vector, size=4):
        return (np.array(vector) * 255).astype(np.uint8)[:size].tolist()

def load_documents(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]
