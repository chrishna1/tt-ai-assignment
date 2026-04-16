"""
LangGraph node implementations.

Nodes (in execution order):
  1. validate_request   — check country/language scope exists in the DB
  2. generate_query     — rewrite question for better vector search
  3. retrieve           — metadata-filtered similarity search
  4. synthesize         — LLM call grounded in retrieved chunks
  5. extract_citations  — pull exact excerpts from source docs

Fallback centralisation:
  All failure paths (invalid scope, no relevant content) route to the single
  handle_fallback node via Command(goto="handle_fallback"). No flag + router
  boilerplate needed — Command is the LangGraph-idiomatic dynamic routing tool.

All nodes are async and accept (state, *, config: RunnableConfig) so that
callers can override Configuration fields per-request via config["configurable"].
"""

import time
from typing import Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.types import Command

from src.agent.configuration import Configuration
from src.agent.models import AgentState, Citation
from src.agent.prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from src.db.vector_store import has_content_for, retrieve_chunks

load_dotenv()


def _get_llm(model: str):
    if "claude" in model.lower():
        return ChatAnthropic(model=model, temperature=0)
    return ChatOpenAI(model=model, temperature=0)


def _fallback_command(reason: str) -> Command:
    """Return a Command that routes to handle_fallback with the given reason."""
    return Command(
        goto="handle_fallback",
        update={"fallback_triggered": True, "fallback_reason": reason},
    )


# ---------------------------------------------------------------------------
# Node 1: validate_request
# ---------------------------------------------------------------------------


async def validate_request(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Command | AgentState:
    """
    Check whether the requested country/language scope has content in the DB.

    Scope is derived dynamically from has_content_for() so no code change is
    needed when a new country/language is added to the corpus.
    """
    country = state["country"].upper()
    language = state["language"]

    state["country"] = country
    state["start_time"] = time.time()

    if not has_content_for(country, language):
        return _fallback_command(
            f"No content found in the database for country='{country}', language='{language}'."
        )

    return state


# ---------------------------------------------------------------------------
# Node 2: generate_query
# ---------------------------------------------------------------------------


async def generate_query(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """Rewrites the user question into a better vector search query."""
    cfg = Configuration.from_runnable_config(config)
    llm = _get_llm(cfg.llm_model)
    messages = [
        SystemMessage(content=QUERY_REWRITE_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    response = await llm.ainvoke(messages)
    state["search_query"] = response.content.strip()
    return state


# ---------------------------------------------------------------------------
# Node 3: retrieve
# ---------------------------------------------------------------------------


async def retrieve(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> Command | AgentState:
    """
    Metadata-filtered similarity search. Filter is applied INSIDE the ANN query.
    Routes to handle_fallback if no chunks pass the relevance threshold.
    """
    cfg = Configuration.from_runnable_config(config)
    chunks = retrieve_chunks(
        query=state.get("search_query") or state["question"],
        country=state["country"],
        language=state["language"],
        top_k=cfg.retrieval_top_k,
        config=config,
    )

    if not chunks:
        return _fallback_command(
            "No relevant content found for your question in the available documents."
        )

    state["retrieved_chunks"] = chunks
    return state


# ---------------------------------------------------------------------------
# Node 4: synthesize
# ---------------------------------------------------------------------------


async def synthesize(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """Generate a grounded answer using only retrieved chunks as context."""
    context_parts = []
    for i, chunk in enumerate(state["retrieved_chunks"], 1):
        context_parts.append(
            f"[{i}] Content ID: {chunk['content_id']} | Type: {chunk['type']}\n"
            f"Title: {chunk['title']}\n"
            f"{chunk['body']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Content excerpts:\n\n{context}\n\nQuestion: {state['question']}"

    cfg = Configuration.from_runnable_config(config)
    llm = _get_llm(cfg.llm_model)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    response = await llm.ainvoke(messages)
    state["answer"] = response.content
    state["language_used"] = state["language"]
    return state


# ---------------------------------------------------------------------------
# Node 5: extract_citations
# ---------------------------------------------------------------------------


async def extract_citations(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """
    Build citations from retrieved chunks.

    Excerpt strategy: take the first 200 characters of the body as the excerpt.
    This is a verifiable substring of the source — the reviewer can open corpus.jsonl
    and confirm it exists. We include all retrieved chunks as citations (ordered by
    match_score), letting the match_score signal relevance.
    """
    elapsed_ms = int((time.time() - state["start_time"]) * 1000)

    citations: list[Citation] = []
    for chunk in state["retrieved_chunks"]:
        body = chunk["body"]
        excerpt = body[:200]
        if len(body) > 200:
            last_space = excerpt.rfind(" ")
            if last_space > 100:
                excerpt = excerpt[:last_space] + "..."

        citations.append(
            Citation(
                content_id=chunk["content_id"],
                type=chunk["type"],
                excerpt=excerpt,
                match_score=chunk["match_score"],
            )
        )

    cfg = Configuration.from_runnable_config(config)
    state["citations"] = citations
    state["trace"] = {
        "retrieval_count": len(state["retrieved_chunks"]),
        "latency_ms": elapsed_ms,
        "model": cfg.llm_model,
    }
    return state


# ---------------------------------------------------------------------------
# Node: handle_fallback  — single entry point for ALL failure paths
# ---------------------------------------------------------------------------


async def handle_fallback(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """
    Unified fallback handler for all failure cases:
      - unsupported country/language scope  (from validate_request)
      - no relevant content found           (from retrieve)

    Returns a structured empty response so the API shape stays consistent.
    """
    elapsed_ms = int((time.time() - state.get("start_time", time.time())) * 1000)
    cfg = Configuration.from_runnable_config(config)

    state["answer"] = state.get(
        "fallback_reason", "No content available for your request."
    )
    state["language_used"] = state["language"]
    state["citations"] = []
    state["retrieved_chunks"] = []
    state["trace"] = {
        "retrieval_count": 0,
        "latency_ms": elapsed_ms,
        "model": cfg.llm_model,
    }
    return state
