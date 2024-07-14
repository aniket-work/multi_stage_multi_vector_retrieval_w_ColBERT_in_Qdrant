# qdrant_operations.py
"""
Module for Qdrant client operations.
"""

from qdrant_client import models

import config


def setup_collection(client):
    """
    Set up the Qdrant collection with the specified vector configurations.

    Args:
        client (QdrantClient): The Qdrant client instance.
    """
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


    client.recreate_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=vector_config
    )
    print("Created collection: multi_vector_collection")
