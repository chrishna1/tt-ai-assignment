"""
Pydantic schemas for the HTTP API (request/response shapes).
"""

import re

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    country: str = Field(..., min_length=1, max_length=10)
    language: str = Field(..., min_length=2, max_length=10)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v

    @field_validator("country")
    @classmethod
    def country_format(cls, v: str) -> str:
        v = v.strip().upper()
        if not v.isalpha():
            raise ValueError("country must contain only letters (e.g. 'A', 'US')")
        return v

    @field_validator("language")
    @classmethod
    def language_format(cls, v: str) -> str:
        v = v.strip()
        if not re.fullmatch(r"[a-zA-Z]{2,8}(_[a-zA-Z]{2,4})?", v):
            raise ValueError(
                "language must be a valid language code (e.g. 'en', 'hi', 'fr_CA')"
            )
        return v


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
