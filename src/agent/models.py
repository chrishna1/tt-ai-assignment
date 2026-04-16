"""
Internal data models used by the LangGraph agent.

Citation and Trace are the core value objects produced by the agent.
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


class AgentState(TypedDict):
    # Inputs
    question: str
    country: str
    language: str

    # Intermediate
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
