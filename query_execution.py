# query_execution.py
"""
Module for executing multi-stage, multi-vector queries in Qdrant.
"""

from qdrant_client.http.models import Prefetch

import config

def perform_multistage_query(client):
    """
    Perform a multi-stage, multi-vector query on the Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client instance.

    Returns:
        QueryResult: The results of the multi-stage query.
    """
    query_mrl = [0.2, 0.4, 0.6, 0.8]
    query_full = [0.2, 0.4, 0.6, 0.8, 0.5]
    query_colbert = [
        [0.1, 0.3, 0.5, 0.7, 0.9],
        [0.2, 0.4, 0.6, 0.8, 0.1],
        [0.9, 0.7, 0.5, 0.3, 0.1],
    ]

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        prefetch=Prefetch(
            prefetch=Prefetch(
                query=query_mrl,
                using=config.MRL_BYTE,
                limit=100,
            ),
            query=query_full,
            using=config.FULL,
            limit=20,
        ),
        query=query_colbert,
        using=config.COLBERT,
        limit=5,
    )

    print(f"Found {len(results.points)} results:")
    for point in results.points:
        print(point)

    return results