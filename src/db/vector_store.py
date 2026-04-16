"""
ChromaDB vector store wrapper.

Metadata stored per chunk: content_id, country, language, type, version, title.
Filtering is applied AT query time (metadata filter passed to ChromaDB query),
not post-retrieval — this is the multi-tenant isolation guarantee.

Client singleton: ChromaDB 1.5+ raises ValueError if two PersistentClient instances
are created for the same path with different settings. We solve this by keeping a
single module-level client and passing it to Chroma(client=...) everywhere.
"""
import os
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "content_corpus"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ── Singletons ────────────────────────────────────────────────────────────────
# One client for the entire process lifetime. Passed to every Chroma() call so
# there is never more than one PersistentClient for this path.
_chroma_client: Optional[chromadb.PersistentClient] = None


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> Chroma:
    """Return a Chroma vector store using the shared client."""
    return Chroma(
        client=_get_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve_chunks(
    query: str,
    country: str,
    language: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-K chunks filtered by country AND language BEFORE similarity ranking.

    The where clause is passed directly to ChromaDB so the filter is applied
    inside the ANN search, not after fetching top-K globally. This prevents
    cross-country content from ever entering the result set.
    """
    store = get_vector_store()

    where_filter = {
        "$and": [
            {"country": {"$eq": country}},
            {"language": {"$eq": language}},
        ]
    }

    results = store.similarity_search_with_relevance_scores(
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
    return chunks


def count_documents() -> int:
    """Return total document count in the collection."""
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def has_content_for(country: str, language: str) -> bool:
    """Check whether any content exists for the given country+language scope."""
    try:
        collection = _get_client().get_collection(COLLECTION_NAME)
    except Exception:
        return False

    results = collection.get(
        where={
            "$and": [
                {"country": {"$eq": country}},
                {"language": {"$eq": language}},
            ]
        },
        limit=1,
        include=[],
    )
    return len(results["ids"]) > 0
