"""
FastAPI HTTP wrapper around the LangGraph Q&A agent.

POST /ask
    Request:  { "question": str, "country": str, "language": str }
    Response: { "answer": str, "language_used": str, "citations": [...], "trace": {...} }

GET /health
    Returns { "status": "ok", "collection_size": int }
"""
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Multi-Country Content Q&A",
    description="RAG-based Q&A system with country/language scoping and citations.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    country: str = Field(..., min_length=1, max_length=10)
    language: str = Field(..., min_length=2, max_length=10)


class CitationResponse(BaseModel):
    content_id: str
    type: str
    excerpt: str
    match_score: float


class TraceResponse(BaseModel):
    retrieval_count: int
    latency_ms: int
    model: str


class AskResponse(BaseModel):
    answer: str
    language_used: str
    citations: list[CitationResponse]
    trace: TraceResponse


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    logger.info(
        "Received question | country=%s lang=%s question=%r",
        req.country,
        req.language,
        req.question[:80],
    )

    try:
        from src.agent.graph import ask
        result = ask(
            question=req.question,
            country=req.country,
            language=req.language,
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Response | citations=%d latency_ms=%d",
        len(result.get("citations", [])),
        result.get("trace", {}).get("latency_ms", 0),
    )

    return AskResponse(
        answer=result["answer"],
        language_used=result["language_used"],
        citations=[CitationResponse(**c) for c in result.get("citations", [])],
        trace=TraceResponse(**result["trace"]),
    )


@app.get("/health")
def health():
    from src.db.vector_store import count_documents
    try:
        n = count_documents()
    except Exception:
        n = -1
    return {"status": "ok", "collection_size": n}
