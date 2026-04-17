import time

from langgraph.graph import END, StateGraph

from src.agent.configuration import Configuration
from src.agent.models import AgentState, InputState
from src.agent.nodes import (
    extract_citations,
    generate_query,
    handle_fallback,
    retrieve,
    synthesize,
    topic_guard,
    validate_request,
)


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState, input=InputState, config_schema=Configuration)

    builder.add_node("validate_request", validate_request)
    builder.add_node("topic_guard", topic_guard)
    builder.add_node("generate_query", generate_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("synthesize", synthesize)
    builder.add_node("extract_citations", extract_citations)
    builder.add_node("handle_fallback", handle_fallback)

    builder.set_entry_point("validate_request")

    # validate_request, topic_guard, and retrieve always return Command.
    # Only static edges for nodes that return plain state dicts.
    builder.add_edge("generate_query", "retrieve")
    builder.add_edge("synthesize", "extract_citations")
    builder.add_edge("extract_citations", END)
    builder.add_edge("handle_fallback", END)

    return builder.compile()


# Module-level compiled graph (singleton)
graph = build_graph()
graph.name = "MultiCountryQAGraph"


async def ask_async(question: str, country: str, language: str) -> dict:
    """Run the agent for a single question. Returns the final AgentState as a plain dict."""
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
    result = await graph.ainvoke(initial_state)
    return result
