import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rebuild_vectorstore import get_vectorstore

load_dotenv()

# Get the in-memory vectorstore
vectorstore = get_vectorstore()

# Vector-only retriever
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
    docs = retriever.invoke(question)
    context = format_docs(docs)
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })

    if "don't have enough information" in answer.lower() or "i don't know" in answer.lower():
        sources = []
    else:
        sources = extract_sources(docs)

    return {
        "answer": answer,
        "sources": sources
    }