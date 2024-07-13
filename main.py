from src.embeddings import load_documents
from src.multi_stage_query import MultiStageQuery
from config import DOCUMENTS_FILE_PATH

def main():
    documents = load_documents(DOCUMENTS_FILE_PATH)
    multi_stage_query = MultiStageQuery()
    multi_stage_query.prepare_data(documents)

    query = "Are dogs aggressive?"
    results = multi_stage_query.query(query)

    print("Query:", query)
    print("Top results:")
    for result in results:
        print(f"- {result.payload['text']} (Score: {result.score})")

if __name__ == "__main__":
    main()