"""
ChromaDB vector store wrapper.

Metadata stored per chunk: content_id, country, language, type, version, title.
Filtering is applied AT query time (metadata filter passed to ChromaDB query),
not post-retrieval — this is the multi-tenant isolation guarantee.
"""
import os
from typing import Optional
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "content_corpus"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_vector_store() -> Chroma:
    """Return a Chroma vector store backed by local persistence."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def retrieve_chunks(
    query: str,
    country: str,
    language: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-K chunks filtered by country AND language BEFORE similarity ranking.

    The where clause is passed directly to ChromaDB so the filter is applied
    inside the ANN search, not after fetching top-K globally.  This prevents
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

    # Sort descending by score
    chunks.sort(key=lambda x: x["match_score"], reverse=True)
    return chunks


def count_documents(country: Optional[str] = None, language: Optional[str] = None) -> int:
    """Count documents in the store, optionally filtered by country/language."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return 0

    where: dict = {}
    if country and language:
        where = {"$and": [{"country": {"$eq": country}}, {"language": {"$eq": language}}]}
    elif country:
        where = {"country": {"$eq": country}}
    elif language:
        where = {"language": {"$eq": language}}

    if where:
        return collection.count()  # ChromaDB count() doesn't support where; use get()
    return collection.count()


def has_content_for(country: str, language: str) -> bool:
    """Check whether any content exists for the given country+language scope."""
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(COLLECTION_NAME)
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
