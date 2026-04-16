"""
LangGraph node implementations.

Nodes (in execution order):
  1. validate_request   — check country/language scope exists in the DB
  2. generate_query     — rewrite question for better vector search
  3. retrieve           — metadata-filtered similarity search
  4. synthesize         — LLM call grounded in retrieved chunks
  5. extract_citations  — pull exact excerpts from source docs

All nodes are async and accept (state, *, config: RunnableConfig) so that
callers can override Configuration fields per-request via config["configurable"].
"""

import time
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.agent.configuration import Configuration
from src.agent.models import AgentState, Citation
from src.agent.prompts import QUERY_REWRITE_PROMPT, SYSTEM_PROMPT
from src.db.vector_store import has_content_for, retrieve_chunks

load_dotenv()


def _get_llm(model: str):
    if "claude" in model.lower():
        return ChatAnthropic(model=model, temperature=0)
    return ChatOpenAI(model=model, temperature=0)


# ---------------------------------------------------------------------------
# Node 1: validate_request
# ---------------------------------------------------------------------------


async def validate_request(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """
    Check whether the requested country/language scope has content in the DB.

    Scope is derived dynamically from has_content_for() so no code change is
    needed when a new country/language is added to the corpus.

    Fallback strategy (deliberate choice):
      - If the country+language combo has no content → fallback with clear message.
      - We do NOT silently switch languages — the caller explicitly requested one.
    """
    country = state["country"].upper()
    language = state["language"]

    state["country"] = country
    state["fallback_triggered"] = False
    state["fallback_reason"] = None
    state["start_time"] = time.time()

    if not has_content_for(country, language):
        state["fallback_triggered"] = True
        state["fallback_reason"] = (
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
) -> AgentState:
    """Metadata-filtered similarity search. Filter is applied INSIDE the ANN query."""
    cfg = Configuration.from_runnable_config(config)
    chunks = retrieve_chunks(
        query=state.get("search_query") or state["question"],
        country=state["country"],
        language=state["language"],
        top_k=cfg.retrieval_top_k,
        config=config,
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
    chunks = state["retrieved_chunks"]

    if not chunks:
        state["answer"] = (
            "I'm sorry, I could not find relevant information to answer your question "
            "in the content available for your region."
        )
        state["language_used"] = state["language"]
        state["citations"] = []
        return state

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Content ID: {chunk['content_id']} | Type: {chunk['type']}\n"
            f"Title: {chunk['title']}\n"
            f"{chunk['body']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Content excerpts:\n\n{context}\n\n Question: {state['question']}"

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
# Node: handle_fallback
# ---------------------------------------------------------------------------


async def handle_fallback(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    """Return a structured empty response for unsupported scopes."""
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


# ---------------------------------------------------------------------------
# Routing function (sync — routers don't do I/O)
# ---------------------------------------------------------------------------


def route_after_validation(
    state: AgentState,
) -> Literal["generate_query", "handle_fallback"]:
    if state.get("fallback_triggered"):
        return "handle_fallback"
    return "generate_query"
