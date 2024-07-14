import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, Prefetch

client = QdrantClient(url="http://localhost:6333")

# Define vector configurations
vector_config = {
    "mrl_byte": models.VectorParams(size=4, distance=models.Distance.COSINE),
    "full": models.VectorParams(size=5, distance=models.Distance.COSINE),
    "colbert": models.VectorParams(
        size=5, distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    ),
}

# Check if collection exists and create if it doesn't
if not client.get_collection("multi_vector_collection"):
    client.create_collection(
        collection_name="multi_vector_collection",
        vectors_config=vector_config
    )
    print("Created collection: multi_vector_collection")
else:
    print("Collection already exists: multi_vector_collection")

# Generate realistic vectors
num_points = 1000
points_to_insert = []
for i in range(num_points):
    mrl_byte = np.random.uniform(0, 1, size=4)
    full = np.concatenate([mrl_byte, [np.random.uniform(0, 1)]])
    colbert = np.random.uniform(0, 1, size=5)

    points_to_insert.append(
        PointStruct(
            id=i,
            vector={
                "mrl_byte": mrl_byte.tolist(),
                "full": full.tolist(),
                "colbert": colbert.tolist(),
            },
            payload={"metadata": f"Point {i}"}
        )
    )

# Insert points into the collection
client.upsert(
    collection_name="multi_vector_collection",
    points=points_to_insert
)
print(f"Inserted {num_points} points into the collection.")

# Perform the multi-stage, multi-vector query
query_mrl = [0.2, 0.4, 0.6, 0.8]
query_full = [0.2, 0.4, 0.6, 0.8, 0.5]
query_colbert = [
    [0.1, 0.3, 0.5, 0.7, 0.9],
    [0.2, 0.4, 0.6, 0.8, 0.1],
    [0.9, 0.7, 0.5, 0.3, 0.1],
]

results = client.query_points(
    collection_name="multi_vector_collection",
    prefetch=Prefetch(
        prefetch=Prefetch(
            query=query_mrl,
            using="mrl_byte",
            limit=100,
        ),
        query=query_full,
        using="full",
        limit=20,
    ),
    query=query_colbert,
    using="colbert",
    limit=5,
)

# Process and print results
print(f"Found {len(results.points)} results:")
for point in results.points:
    print(point)

# Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot all points
all_vectors = np.array([p.vector["full"] for p in points_to_insert])
ax.scatter(all_vectors[:, 0], all_vectors[:, 1], all_vectors[:, 2], c='gray', alpha=0.1, label='All points')

# Plot MRL stage (first 100 points for demonstration)
mrl_vectors = np.array([p.vector["mrl_byte"] for p in points_to_insert[:100]])
ax.scatter(mrl_vectors[:, 0], mrl_vectors[:, 1], mrl_vectors[:, 2], c='blue', alpha=0.5, label='MRL stage')

# Plot Full vector stage (first 20 points for demonstration)
full_vectors = np.array([p.vector["full"] for p in points_to_insert[:20]])
ax.scatter(full_vectors[:, 0], full_vectors[:, 1], full_vectors[:, 2], c='green', alpha=0.7, label='Full vector stage')

# Plot ColBERT stage (using the IDs from the results)
colbert_vectors = np.array([points_to_insert[point.id].vector["colbert"] for point in results.points])
ax.scatter(colbert_vectors[:, 0], colbert_vectors[:, 1], colbert_vectors[:, 2], c='red', s=100, label='ColBERT stage')

# Plot query points
ax.scatter(query_mrl[0], query_mrl[1], query_mrl[2], c='purple', s=200, marker='*', label='Query MRL')
ax.scatter(query_full[0], query_full[1], query_full[2], c='orange', s=200, marker='*', label='Query Full')
ax.scatter(query_colbert[0][0], query_colbert[0][1], query_colbert[0][2], c='pink', s=200, marker='*',
           label='Query ColBERT')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Multi-stage, Multi-vector Retrieval with ColBERT in Qdrant')
ax.legend()

plt.tight_layout()
plt.show()