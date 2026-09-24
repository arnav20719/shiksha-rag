import os
import shutil
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from build_documents import build_all_documents
from ingest_bihar import load_bihar_document

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "colleges"

def rebuild():
    # Step 1: Delete old ChromaDB so we start clean
    if os.path.exists(CHROMA_DIR):
        print(f"Deleting existing {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)
    
    # Step 2: Load all documents from both sources
    print("\nLoading IIT colleges from JSON...")
    iit_docs = build_all_documents("data/iit_colleges.json")
    print(f"  -> {len(iit_docs)} IIT documents")
    
    print("\nLoading Bihar colleges from text...")
    bihar_docs = load_bihar_document("data/bihar_colleges.txt")
    print(f"  -> {len(bihar_docs)} Bihar chunks")
    
    # Step 3: Combine
    all_docs = iit_docs + bihar_docs
    print(f"\nTotal documents to embed: {len(all_docs)}")
    
    # Step 4: Embed and store
    print("\nInitializing embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Embedding and storing in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )
    
    print(f"\nRebuilt vector store with {len(all_docs)} documents")
    print(f"Persisted to ./{CHROMA_DIR}")
    return vectorstore

def test_retrieval(vectorstore):
    """Test retrieval across both data sources."""
    queries = [
        "Tell me about IIT Bombay",                       # IIT JSON
        "What is the fee for MIT Muzaffarpur?",           # Bihar text
        "Which IIT has the highest placement salary?",    # IIT JSON
        "What is the NAAC grade of Patna University?",    # Bihar text
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        results = vectorstore.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            source_type = doc.metadata.get("type", "unknown")
            name = doc.metadata.get("college_name", doc.metadata.get("source", "Bihar doc"))
            print(f"\n  Result {i} [{source_type}] {name}")
            print(f"  Preview: {doc.page_content[:150]}...")

if __name__ == "__main__":
    vectorstore = rebuild()
    test_retrieval(vectorstore)