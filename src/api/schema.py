"""
Pydantic schemas for the HTTP API (request/response shapes).
"""
from pydantic import BaseModel, Field


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
