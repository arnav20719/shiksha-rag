from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_bihar_document(path="data/bihar_colleges.txt"):
    """Load the Bihar college document and chunk it."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Loaded {len(text)} characters from {path}")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    
    chunks = splitter.split_text(text)
    
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(Document(
            page_content=chunk,
            metadata={
                "source": "Bihar College Data 2026",
                "source_url": "https://github.com/arnav20719/shiksha-rag",
                "type": "bihar_colleges",
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
        ))
    
    return docs

if __name__ == "__main__":
    docs = load_bihar_document()
    print(f"\nCreated {len(docs)} chunks from Bihar document")
    print(f"\n--- First chunk preview (first 300 chars) ---")
    print(docs[0].page_content[:300])
    print(f"\n--- Metadata ---")
    print(docs[0].metadata)
    print(f"\n--- Last chunk preview (first 300 chars) ---")
    print(docs[-1].page_content[:300])