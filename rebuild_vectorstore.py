import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from build_documents import build_all_documents
from ingest_bihar import load_bihar_document

load_dotenv()

COLLECTION_NAME = "colleges"

# Module-level in-memory vectorstore (survives across function calls)
_global_vectorstore = None

def rebuild(force=True):
    """Build vector store in memory (no disk persistence)."""
    global _global_vectorstore
    
    if _global_vectorstore is not None and not force:
        print("Vector store already in memory, skipping.")
        return _global_vectorstore

    print("Loading IIT colleges...")
    iit_docs = build_all_documents("data/iit_colleges.json")
    print(f"  -> {len(iit_docs)} IIT documents")

    print("Loading Bihar colleges...")
    bihar_docs = load_bihar_document("data/bihar_colleges.txt")
    print(f"  -> {len(bihar_docs)} Bihar chunks")

    all_docs = iit_docs + bihar_docs
    print(f"Total: {len(all_docs)} documents")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding and storing in MEMORY (no disk)...")
    
    # In-memory ChromaDB — no persist_directory
    _global_vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )
    
    print(f"Done: {len(all_docs)} documents embedded in memory")
    return _global_vectorstore


def get_vectorstore():
    """Get the in-memory vectorstore, building if needed."""
    global _global_vectorstore
    if _global_vectorstore is None:
        rebuild(force=True)
    return _global_vectorstore


if __name__ == "__main__":
    rebuild(force=True)