# main.py
"""
Main entry point for the multi-vector retrieval demonstration using Qdrant.
"""

from qdrant_client import QdrantClient

import config
from qdrant_operations import setup_collection
from data_generation import generate_and_insert_points
from query_execution import perform_multistage_query
from visualization import visualize_results


def main():
    """
    Main function to demonstrate multi-vector retrieval with Qdrant.
    """
    client = QdrantClient(url=config.QDRANT_URL)

    # Setup collection
    setup_collection(client)

    # Generate and insert points
    num_points = config.DATASET_SIZE
    points = generate_and_insert_points(client, num_points)

    # Perform multi-stage query
    results = perform_multistage_query(client)

    # Visualize results
    visualize_results(points, results)


if __name__ == "__main__":
    main()