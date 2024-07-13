from src.embeddings import load_documents
from src.multi_stage_query import MultiStageQuery
from config import DOCUMENTS_FILE_PATH

def main():
    """
    Main function to demonstrate multi-stage querying with ColBERT approach in Qdrant.

    This function loads documents, prepares the data for querying,
    performs a sample query, and prints the results.
    """
    # Load documents from the specified file path
    documents = load_documents(DOCUMENTS_FILE_PATH)

    # Initialize the MultiStageQuery object
    multi_stage_query = MultiStageQuery()

    # Prepare the data by processing and indexing the documents
    multi_stage_query.prepare_data(documents)

    # Define a sample query
    query = "Are dogs aggressive?"

    # Perform the query and get results
    results = multi_stage_query.query(query)

    # Print the query and results
    print("Query:", query)
    print("Top results:")
    for result in results:
        print(f"- {result.payload['text']} (Score: {result.score})")

if __name__ == "__main__":
    main()