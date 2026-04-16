"""
Internal data models used by the LangGraph agent.

Citation and Trace are the core value objects produced by the agent.
InputState is the public API boundary (caller-facing fields).
AgentState is the full LangGraph state passed between nodes.
"""
from typing import Optional
from typing_extensions import TypedDict


class Citation(TypedDict):
    content_id: str
    type: str
    excerpt: str
    match_score: float


class Trace(TypedDict):
    retrieval_count: int
    latency_ms: int
    model: str


class InputState(TypedDict):
    """Public API boundary — the fields callers must supply."""
    question: str
    country: str
    language: str


class AgentState(TypedDict):
    # Inputs (same as InputState — duplicated for Python 3.9 compatibility)
    question: str
    country: str
    language: str

    # Intermediate
    search_query: str
    retrieved_chunks: list
    fallback_triggered: bool
    fallback_reason: Optional[str]

    # Outputs
    answer: str
    language_used: str
    citations: list
    trace: Trace

    # Timing (set at entry, used to compute latency_ms)
    start_time: float
