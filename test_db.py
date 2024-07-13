from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

try:
    # This should return a list of collection names
    collections = client.get_collections()
    print("Connected successfully. Collections:", collections)
except Exception as e:
    print("Connection failed:", str(e))