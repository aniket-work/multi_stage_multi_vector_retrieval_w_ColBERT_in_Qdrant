# data_generation.py
"""
Module for generating and inserting sample data into Qdrant.
"""

import numpy as np
from qdrant_client.http.models import PointStruct

import config


def generate_and_insert_points(client, num_points):
    """
    Generate and insert sample points into the Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.
        num_points (int): Number of points to generate and insert.

    Returns:
        list: List of generated PointStruct objects.
    """
    points_to_insert = []
    for i in range(num_points):
        mrl_byte = np.random.uniform(0, 1, size=4)
        full = np.concatenate([mrl_byte, [np.random.uniform(0, 1)]])
        colbert = np.random.uniform(0, 1, size=5)

        points_to_insert.append(
            PointStruct(
                id=i,
                vector={
                    config.MRL_BYTE: mrl_byte.tolist(),
                    config.FULL: full.tolist(),
                    config.COLBERT: colbert.tolist(),
                },
                payload={"metadata": f"Point {i}"}
            )
        )

    client.upsert(
        collection_name=config.COLLECTION_NAME,
        points=points_to_insert
    )
    print(f"Inserted {num_points} points into the collection.")

    return points_to_insert