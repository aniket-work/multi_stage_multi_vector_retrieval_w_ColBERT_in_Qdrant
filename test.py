
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qdrant_client import QdrantClient, models

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

# Create the collection
client.recreate_collection(
    collection_name="multi_vector_collection",
    vectors_config=vector_config
)
print("Created collection: multi_vector_collection")

import numpy as np
from qdrant_client.http.models import PointStruct

# Generate random vectors for demonstration
num_points = 1000
points_to_insert = [
    PointStruct(
        id=i,
        vector={
            "mrl_byte": np.random.uniform(0, 1, size=4).tolist(),
            "full": np.random.rand(5).tolist(),
            "colbert": np.random.rand(5).tolist(),
        },
        payload={"metadata": f"Point {i}"}
    ) for i in range(num_points)
]

# Insert points into the collection
client.upsert(
    collection_name="multi_vector_collection",
    points=points_to_insert
)
print(f"Inserted {num_points} points into the collection.")

from qdrant_client.http.models import Prefetch

# Perform the multi-stage, multi-vector query
results = client.query_points(
    collection_name="multi_vector_collection",
    prefetch=Prefetch(
        prefetch=Prefetch(
            query=[0.01, 0.45, 0.67, 0.89],  # Small byte vector
            using="mrl_byte",
            limit=1000,
        ),
        query=[0.01, 0.45, 0.67, 0.89, 0.23],  # Full dense vector
        using="full",
        limit=100,
    ),
    query=[
        [0.1, 0.2, 0.3, 0.4, 0.5],  # Multi-vector for ColBERT
        [0.2, 0.1, 0.4, 0.3, 0.5],
        [0.8, 0.9, 0.7, 0.6, 0.5],
    ],
    using="colbert",
    limit=10,
)

# Process and print results
print(f"Found {len(results.points)} results:")
for point in results.points:
    print(point)



