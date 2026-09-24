from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "colleges"

def build_hybrid_retriever():
    """
    Build a hybrid retriever combining BM25 (keyword) and vector (semantic) search.
    """
    # --- 1. Load vector store and get all documents ---
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    
    # Vector retriever
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # --- 2. Get all documents from the vector store to build BM25 index ---
    # Chroma allows us to fetch all stored documents
    all_data = vectorstore.get()
    
    docs = []
    for i, text in enumerate(all_data["documents"]):
        metadata = all_data["metadatas"][i] if all_data["metadatas"] else {}
        docs.append(Document(page_content=text, metadata=metadata))
    
    print(f"Loaded {len(docs)} documents for BM25 index")
    
    # --- 3. Build BM25 retriever ---
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5
    
    # --- 4. Combine both retrievers with ensemble ---
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.1, 0.8]  # equal weight to keyword and semantic
    )
    
    return ensemble


if __name__ == "__main__":
    retriever = build_hybrid_retriever()
    
    # Test with the queries that failed before
    test_queries = [
        "What is the fee range for private engineering colleges in Bihar?",  # q21
        "What is the NIRF management rank of IIM Bodh Gaya?",              # q16
        "Tell me about IIT Bombay",
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        results = retriever.invoke(query)
        for i, doc in enumerate(results[:3], 1):
            name = doc.metadata.get("college_name") or doc.metadata.get("source", "unknown")
            print(f"\n  Result {i}: [{doc.metadata.get('type', '?')}] {name}")
            print(f"  Preview: {doc.page_content[:200]}...")