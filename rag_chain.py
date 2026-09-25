import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "colleges"

# ----- 1. Load vector store -----
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# Vector-only retriever (hybrid removed — didn't improve this dataset)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# ----- 2. Initialize LLM -----
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ----- 3. Prompt template -----
prompt = ChatPromptTemplate.from_template("""
You are a helpful college information assistant. Answer the user's question
using ONLY the context provided below. If the answer is not in the context,
say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:
""")

# ----- 4. Helper to format retrieved documents -----
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# ----- 5. Helper to extract unique source URLs from retrieved docs -----
def extract_sources(docs):
    sources = []
    seen = set()
    for doc in docs:
        url = doc.metadata.get("source_url")
        name = doc.metadata.get("college_name") or doc.metadata.get("source")
        if url and url not in seen:
            sources.append({"college": name, "url": url})
            seen.add(url)
    return sources

# ----- 6. Build chain that returns answer + sources -----
def rag_with_sources(question: str) -> dict:
    # Step 1: retrieve
    docs = retriever.invoke(question)

    # Step 2: build context and generate answer
    context = format_docs(docs)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    # Step 3: only include sources if the LLM gave a real answer
    if "don't have enough information" in answer.lower() or "i don't know" in answer.lower():
        sources = []
    else:
        sources = extract_sources(docs)

    return {
        "answer": answer,
        "sources": sources
    }

# ----- 7. Test -----
if __name__ == "__main__":
    questions = [
        "Which IIT has the highest placement salary?",
        "Tell me about IIT Bombay",
        "What is the fee for MIT Muzaffarpur?",
        "What is the NAAC grade of Patna University?",
        "What is the fee for B.Tech at IIT Kanpur?",
    ]

    for q in questions:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print('='*70)
        result = rag_with_sources(q)
        print(f"A: {result['answer']}")
        print(f"\nSources:")
        if not result["sources"]:
            print("  (no sources — answer not grounded in retrieved context)")
        for src in result["sources"]:
            print(f"  - {src['college']}")
            print(f"    {src['url']}")