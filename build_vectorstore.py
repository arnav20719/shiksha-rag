import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from build_documents import build_all_documents

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "iit_colleges"

def build_vectorstore():
    """Load documents, embed them, and persist to ChromaDB."""
    print("Loading documents...")
    docs = build_all_documents("data/iit_colleges.json")
    print(f"Loaded {len(docs)} documents")

    print("\nInitializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    print(f"Vector store persisted to ./{CHROMA_DIR}")
    return vectorstore

def load_vectorstore():
    """Load an existing vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

def test_retrieval(vectorstore):
    """Test semantic search with sample queries."""
    queries = [
        "Which IIT has the highest placement salary?",
        "Tell me about IIT Bombay",
        "Colleges in Mumbai",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        results = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}: {doc.metadata['college_name']}")
            print(f"  Location: {doc.metadata['location']}")
            print(f"  Rating: {doc.metadata['rating']}")
            print(f"  Content preview: {doc.page_content[:120]}...")

if __name__ == "__main__":
    if os.path.exists(CHROMA_DIR):
        print(f"Loading existing vector store from ./{CHROMA_DIR}")
        vectorstore = load_vectorstore()
    else:
        vectorstore = build_vectorstore()

    test_retrieval(vectorstore)