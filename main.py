from src.data_loader import load_documents
from src.multi_stage_query import MultiStageQuery

def main():
    documents = load_documents('data/documents.txt')
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