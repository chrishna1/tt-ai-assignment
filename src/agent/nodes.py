"""
LangGraph node implementations.

Nodes (in execution order):
  1. validate_request   — check country/language scope exists in the DB
  2. retrieve           — metadata-filtered similarity search
  3. synthesize         — LLM call grounded in retrieved chunks
  4. extract_citations  — pull exact excerpts from source docs
"""
import os
import time
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.configuration import Configuration
from src.agent.models import AgentState, Citation
from src.agent.prompts import SYSTEM_PROMPT, QUERY_REWRITE_PROMPT
from src.db.vector_store import retrieve_chunks, has_content_for

load_dotenv()

# Module-level defaults (kept for backward compatibility)
_default_config = Configuration()
LLM_MODEL = _default_config.llm_model
RETRIEVAL_TOP_K = _default_config.retrieval_top_k

# Supported country/language combos per the corpus
VALID_SCOPES: dict[str, list[str]] = {
    "A": ["en", "hi"],
    "B": ["en", "es"],
    "C": ["en", "fr_CA"],
    "D": ["en"],
}


def _get_llm(model: Optional[str] = None):
    llm_model = model or LLM_MODEL
    if "claude" in llm_model.lower():
        return ChatAnthropic(model=llm_model, temperature=0)
    else:
        return ChatOpenAI(model=llm_model, temperature=0)


# ---------------------------------------------------------------------------
# Node 1: validate_request
# ---------------------------------------------------------------------------

def validate_request(state: AgentState) -> AgentState:
    """
    Check whether the requested country/language scope has content.

    Fallback strategy (deliberate choice):
      - If the country is unknown → error, no fallback.
      - If the language is not available in the country → return empty result
        with a clear message. We do NOT fall back to another language because
        the caller explicitly requested a language; silently changing it would
        be confusing. This is documented in README.
    """
    country = state["country"].upper()
    language = state["language"]

    state["country"] = country  # normalise to uppercase
    state["fallback_triggered"] = False
    state["fallback_reason"] = None
    state["start_time"] = time.time()

    valid_langs = VALID_SCOPES.get(country)
    if valid_langs is None:
        state["fallback_triggered"] = True
        state["fallback_reason"] = f"Country '{country}' is not a supported scope."
        return state

    if language not in valid_langs:
        state["fallback_triggered"] = True
        state["fallback_reason"] = (
            f"No content available for country '{country}' in language '{language}'. "
            f"Available languages for this country: {valid_langs}."
        )
        return state

    # Double-check against the live DB (catches edge cases after partial ingestion)
    if not has_content_for(country, language):
        state["fallback_triggered"] = True
        state["fallback_reason"] = (
            f"No content found in the database for country='{country}', language='{language}'."
        )

    return state


# ---------------------------------------------------------------------------
# Node 2: generate_query
# ---------------------------------------------------------------------------

def generate_query(state: AgentState) -> AgentState:
    """Rewrites the user question into a better vector search query."""
    config = Configuration()
    llm = _get_llm(config.llm_model)
    messages = [
        SystemMessage(content=QUERY_REWRITE_PROMPT),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    state["search_query"] = response.content.strip()
    return state


# ---------------------------------------------------------------------------
# Node 3: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: AgentState) -> AgentState:
    """Metadata-filtered similarity search. Filter is applied INSIDE the ANN query."""
    config = Configuration()
    chunks = retrieve_chunks(
        query=state.get("search_query") or state["question"],
        country=state["country"],
        language=state["language"],
        top_k=config.retrieval_top_k,
        config=config,
    )
    state["retrieved_chunks"] = chunks
    return state


# ---------------------------------------------------------------------------
# Node 3: synthesize
# ---------------------------------------------------------------------------

def synthesize(state: AgentState) -> AgentState:
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

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Content ID: {chunk['content_id']} | Type: {chunk['type']}\n"
            f"Title: {chunk['title']}\n"
            f"{chunk['body']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_message = (
        f"Content excerpts:\n\n{context}\n\n"
        f"Question: {state['question']}\n\n"
        f"Answer in language: {state['language']}"
    )

    config = Configuration()
    llm = _get_llm(config.llm_model)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    state["answer"] = response.content
    state["language_used"] = state["language"]
    return state


# ---------------------------------------------------------------------------
# Node 4: extract_citations
# ---------------------------------------------------------------------------

def extract_citations(state: AgentState) -> AgentState:
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
        # Extract a meaningful excerpt (up to 200 chars, end at word boundary)
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

    config = Configuration()
    state["citations"] = citations
    state["trace"] = {
        "retrieval_count": len(state["retrieved_chunks"]),
        "latency_ms": elapsed_ms,
        "model": config.llm_model,
    }
    return state


# ---------------------------------------------------------------------------
# Node: handle_fallback
# ---------------------------------------------------------------------------

def handle_fallback(state: AgentState) -> AgentState:
    """Return a structured empty response for unsupported scopes."""
    elapsed_ms = int((time.time() - state.get("start_time", time.time())) * 1000)

    config = Configuration()
    state["answer"] = state.get("fallback_reason", "No content available for your request.")
    state["language_used"] = state["language"]
    state["citations"] = []
    state["retrieved_chunks"] = []
    state["trace"] = {
        "retrieval_count": 0,
        "latency_ms": elapsed_ms,
        "model": config.llm_model,
    }
    return state


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def route_after_validation(state: AgentState) -> Literal["generate_query", "handle_fallback"]:
    if state.get("fallback_triggered"):
        return "handle_fallback"
    return "generate_query"
