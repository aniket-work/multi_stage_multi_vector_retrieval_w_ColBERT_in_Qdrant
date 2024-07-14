# visualization.py
"""
Module for visualizing multi-vector retrieval results.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def visualize_results(points, results):
    """
    Visualize the results of the multi-stage, multi-vector query in 3D.

    Args:
        points (list): List of all PointStruct objects.
        results (QueryResult): Results from the multi-stage query.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    all_vectors = np.array([p.vector["full"] for p in points])
    ax.scatter(all_vectors[:, 0], all_vectors[:, 1], all_vectors[:, 2], c='gray', alpha=0.1, label='All points')

    # Plot MRL stage (first 100 points for demonstration)
    mrl_vectors = np.array([p.vector["mrl_byte"] for p in points[:100]])
    ax.scatter(mrl_vectors[:, 0], mrl_vectors[:, 1], mrl_vectors[:, 2], c='blue', alpha=0.5, label='MRL stage')

    # Plot Full vector stage (first 20 points for demonstration)
    full_vectors = np.array([p.vector["full"] for p in points[:20]])
    ax.scatter(full_vectors[:, 0], full_vectors[:, 1], full_vectors[:, 2], c='green', alpha=0.7,
               label='Full vector stage')

    # Plot ColBERT stage (using the IDs from the results)
    colbert_vectors = np.array([points[point.id].vector["colbert"] for point in results.points])
    ax.scatter(colbert_vectors[:, 0], colbert_vectors[:, 1], colbert_vectors[:, 2], c='red', s=100,
               label='ColBERT stage')

    # Plot query points
    query_mrl = [0.2, 0.4, 0.6, 0.8]
    query_full = [0.2, 0.4, 0.6, 0.8, 0.5]
    query_colbert = [0.1, 0.3, 0.5, 0.7, 0.9]
    ax.scatter(query_mrl[0], query_mrl[1], query_mrl[2], c='purple', s=200, marker='*', label='Query MRL')
    ax.scatter(query_full[0], query_full[1], query_full[2], c='orange', s=200, marker='*', label='Query Full')
    ax.scatter(query_colbert[0], query_colbert[1], query_colbert[2], c='pink', s=200, marker='*', label='Query ColBERT')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multi-stage, Multi-vector Retrieval with ColBERT in Qdrant')
    ax.legend()

    plt.tight_layout()
    plt.show()