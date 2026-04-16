"""
Vector store retrieval layer.

Public API:
    make_retriever(config)  — @contextmanager, yields a VectorStoreRetriever
    retrieve_chunks(...)    — convenience wrapper used by nodes
    count_documents()       — admin / health check
    has_content_for(...)    — used by validate_request node

Metadata stored per chunk: content_id, country, language, type, version, title.
Filtering is applied AT query time (metadata filter passed to ChromaDB query),
not post-retrieval — this is the multi-tenant isolation guarantee.

ChromaDB singleton:
    ChromaDB 1.5+ raises ValueError if two PersistentClient instances are
    created for the same path. We keep one module-level client and reuse it
    inside every make_retriever() call.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

from src.agent.configuration import Configuration

COLLECTION_NAME = "content_corpus"
# Cosine distance gives relevance = cosine_similarity ∈ [-1, 1].
# L2 (the ChromaDB default) produces negative relevance for typical short docs.
COLLECTION_METADATA = {"hnsw:space": "cosine"}
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# ── ChromaDB singleton ────────────────────────────────────────────────────────
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_client(persist_dir: Optional[str] = None) -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=persist_dir or CHROMA_PERSIST_DIR
        )
    return _chroma_client


def _make_embeddings(model: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model)


# ── Backend factories ─────────────────────────────────────────────────────────


@contextmanager
def _make_chroma_retriever(
    configuration: Configuration,
) -> Generator[VectorStoreRetriever, None, None]:
    vstore = Chroma(
        client=_get_client(configuration.chroma_persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=_make_embeddings(configuration.embedding_model),
        collection_metadata=COLLECTION_METADATA,
    )
    yield vstore.as_retriever(search_kwargs={"k": configuration.retrieval_top_k})


# ── Public API ────────────────────────────────────────────────────────────────


@contextmanager
def make_retriever(
    config: Optional[RunnableConfig] = None,
) -> Generator[VectorStoreRetriever, None, None]:
    """Yield a VectorStoreRetriever configured from the LangGraph RunnableConfig.

    Usage::

        with make_retriever(config) as retriever:
            docs = await retriever.ainvoke(query)
    """
    configuration = Configuration.from_runnable_config(config)
    match configuration.retriever_provider:
        case "chroma":
            with _make_chroma_retriever(configuration) as retriever:
                yield retriever
        case _:
            raise ValueError(
                f"Unsupported retriever_provider: {configuration.retriever_provider!r}. "
                "Expected: 'chroma'"
            )


def retrieve_chunks(
    query: str,
    country: str,
    language: str,
    top_k: int = 5,
    config: Optional[RunnableConfig] = None,
) -> list[dict]:
    """
    Retrieve top-K chunks filtered by country AND language BEFORE similarity ranking.

    The where clause is passed directly to ChromaDB so the filter is applied
    inside the ANN search, not after fetching top-K globally. This prevents
    cross-country content from ever entering the result set.
    """
    configuration = Configuration.from_runnable_config(config)
    vstore = Chroma(
        client=_get_client(configuration.chroma_persist_dir),
        collection_name=COLLECTION_NAME,
        embedding_function=_make_embeddings(configuration.embedding_model),
        collection_metadata=COLLECTION_METADATA,
    )

    where_filter = {
        "$and": [
            {"country": {"$eq": country}},
            {"language": {"$eq": language}},
        ]
    }

    results = vstore.similarity_search_with_relevance_scores(
        query=query,
        k=top_k,
        filter=where_filter,
    )

    chunks = []
    for doc, score in results:
        chunks.append(
            {
                "content_id": doc.metadata["content_id"],
                "type": doc.metadata["type"],
                "country": doc.metadata["country"],
                "language": doc.metadata["language"],
                "title": doc.metadata.get("title", ""),
                "version": doc.metadata.get("version", ""),
                "body": doc.page_content,
                "match_score": round(float(score), 4),
            }
        )

    chunks.sort(key=lambda x: x["match_score"], reverse=True)

    min_score = configuration.min_relevance_score
    chunks = [c for c in chunks if c["match_score"] >= min_score]

    return chunks


def count_documents() -> int:
    """Return total document count in the collection."""
    collection = _get_client().get_collection(COLLECTION_NAME)
    return collection.count()
