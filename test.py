from qdrant_client import QdrantClient, models
import numpy as np

# Initialize the client
client = QdrantClient(url="http://localhost:6333")

# Define the collection name
collection_name = "multi_vector_collection"

# Define vector configurations
vector_config = {
    "mrl_byte": models.VectorParams(size=4, distance=models.Distance.COSINE),
    "full": models.VectorParams(size=5, distance=models.Distance.COSINE),
    "colbert": models.VectorParams(size=5, distance=models.Distance.COSINE),
}

# Create the collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=vector_config
)
print(f"Created collection: {collection_name}")

# Insert points with multiple vector types
num_points = 1000
points_to_insert = [
    models.PointStruct(
        id=i,
        vector={
            "mrl_byte": np.random.uniform(0, 1, size=4).tolist(),  # Changed to float values
            "full": np.random.rand(5).tolist(),
            "colbert": np.random.rand(5).tolist(),  # Single vector for colbert
        },
        payload={"metadata": f"Point {i}"}
    ) for i in range(num_points)
]

client.upsert(collection_name=collection_name, points=points_to_insert)
print(f"Inserted {num_points} points into the collection.")

# Perform the multi-stage, multi-vector query
results = client.query_points(
    collection_name=collection_name,
    prefetch=models.Prefetch(
        prefetch=models.Prefetch(
            query=[0.1, 0.23, 0.45, 0.67],  # Changed to float values
            using="mrl_byte",
            limit=1000,
        ),
        query=[0.01, 0.45, 0.67, 0.89, 0.23],
        using="full",
        limit=100,
    ),
    query=[0.1, 0.2, 0.3, 0.4, 0.5],  # Single vector for colbert
    using="colbert",
    limit=10,
)

# Process and print results
print(f"\nFound {results} results:")
