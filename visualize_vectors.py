from qdrant_client import QdrantClient, models
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ... (Your existing code for initializing client, creating collection, and inserting points)

# Define query vectors
query_vectors = {
    "mrl_byte": [0.1, 0.23, 0.45, 0.67],
    "full": [0.01, 0.45, 0.67, 0.89, 0.23],
    "colbert": [0.1, 0.2, 0.3, 0.4, 0.5]
}

# Perform queries for each vector type
results = {}
for vector_type, query_vector in query_vectors.items():
    results[vector_type] = client.search(
        collection_name=collection_name,
        query_vector=(vector_type, query_vector),
        limit=5
    )

# Visualization
fig = plt.figure(figsize=(18, 6))

for i, (vector_type, query_result) in enumerate(results.items()):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')

    # Plot query vector
    query_vector = query_vectors[vector_type]
    ax.scatter(query_vector[0], query_vector[1], query_vector[2], c='r', s=100, label='Query')

    # Plot top 5 results
    for result in query_result:
        vector = result.vector[vector_type]
        ax.scatter(vector[0], vector[1], vector[2], c='b', alpha=0.5)
        ax.text(vector[0], vector[1], vector[2], f"ID: {result.id}", fontsize=8)

    ax.set_title(f"{vector_type.capitalize()} Vectors")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

plt.tight_layout()
plt.show()

# Print results
for vector_type, query_result in results.items():
    print(f"\n{vector_type.capitalize()} Results:")
    for result in query_result:
        print(f"ID: {result.id}, Score: {result.score}")