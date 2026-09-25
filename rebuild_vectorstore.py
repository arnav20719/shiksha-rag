import os
import shutil
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from build_documents import build_all_documents
from ingest_bihar import load_bihar_document

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "colleges"


def rebuild(force=False):
    """Rebuild the vector store. Skip if exists and force=False."""
    if os.path.exists(CHROMA_DIR) and not force:
        print(f"Vector store exists at ./{CHROMA_DIR}, skipping.")
        return

    if os.path.exists(CHROMA_DIR):
        print(f"Deleting {CHROMA_DIR}...")
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError:
            time.sleep(2)
            shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    print("\nLoading IIT colleges...")
    iit_docs = build_all_documents("data/iit_colleges.json")
    print(f"  -> {len(iit_docs)} IIT documents")

    print("\nLoading Bihar colleges...")
    bihar_docs = load_bihar_document("data/bihar_colleges.txt")
    print(f"  -> {len(bihar_docs)} Bihar chunks")

    all_docs = iit_docs + bihar_docs
    print(f"\nTotal: {len(all_docs)} documents")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embedding and storing...")
    Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"Done: {len(all_docs)} documents persisted to ./{CHROMA_DIR}")


if __name__ == "__main__":
    rebuild(force=True)