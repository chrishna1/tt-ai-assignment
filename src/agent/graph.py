"""
LangGraph agent graph definition.

Graph structure:
                      ┌─────────────────┐
                      │ validate_request │
                      └────────┬────────┘
                               │
              ┌────────────────┼────────────────┐
              │ fallback?      │ no fallback     │
              ▼                ▼                 │
    ┌─────────────────┐  ┌──────────┐           │
    │ handle_fallback │  │ retrieve │           │
    └────────┬────────┘  └────┬─────┘           │
             │                │                  │
             │           ┌────▼──────┐           │
             │           │ synthesize│           │
             │           └────┬──────┘           │
             │                │                  │
             │      ┌─────────▼──────────┐       │
             │      │ extract_citations  │       │
             │      └─────────┬──────────┘       │
             │                │                  │
             └────────────────▼──────────────────┘
                            END
"""
import time

from langgraph.graph import StateGraph, END

from src.agent.models import AgentState, InputState
from src.agent.nodes import (
    validate_request,
    retrieve,
    synthesize,
    extract_citations,
    handle_fallback,
    route_after_validation,
)


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("validate_request", validate_request)
    builder.add_node("retrieve", retrieve)
    builder.add_node("synthesize", synthesize)
    builder.add_node("extract_citations", extract_citations)
    builder.add_node("handle_fallback", handle_fallback)

    builder.set_entry_point("validate_request")

    builder.add_conditional_edges(
        "validate_request",
        route_after_validation,
        {
            "retrieve": "retrieve",
            "handle_fallback": "handle_fallback",
        },
    )

    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", "extract_citations")
    builder.add_edge("extract_citations", END)
    builder.add_edge("handle_fallback", END)

    return builder.compile()


# Module-level compiled graph (singleton)
graph = build_graph()


def ask(question: str, country: str, language: str) -> dict:
    """
    Run the agent for a single question.

    InputState is the public interface — callers supply question, country, language.
    Returns the final AgentState as a plain dict.
    """
    initial_state: AgentState = {
        "question": question,
        "country": country,
        "language": language,
        "search_query": "",
        "retrieved_chunks": [],
        "fallback_triggered": False,
        "fallback_reason": None,
        "answer": "",
        "language_used": language,
        "citations": [],
        "trace": {"retrieval_count": 0, "latency_ms": 0, "model": ""},
        "start_time": time.time(),
    }

    result = graph.invoke(initial_state)
    return result
