from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import rag_with_sources

app = FastAPI(
    title="CareersAthhi RAG API",
    description="College information Q&A powered by RAG",
    version="1.0.0"
)

# Allow your website to call this API from a browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["https://careersathhi.com"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Request / Response models -----
class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    college: str
    url: str

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# ----- Endpoints -----
@app.get("/")
def root():
    return {"status": "ok", "message": "CareersAthhi RAG API is running"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    result = rag_with_sources(request.question)
    return AskResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]]
    )

@app.get("/health")
def health():
    return {"status": "healthy"}