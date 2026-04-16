"""
Ingestion logic — shared by the CLI script and the /ingest HTTP endpoint.
"""

import json
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.agent.configuration import Configuration
from src.db.vector_store import (
    COLLECTION_METADATA,
    COLLECTION_NAME,
    _get_client,
    _make_embeddings,
)


def parse_jsonl(raw: str | bytes) -> list[dict]:
    """Parse JSONL text (str or bytes) into a list of dicts."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def build_documents(items: list[dict]) -> list[Document]:
    docs = []
    for item in items:
        doc = Document(
            page_content=item["body"],
            metadata={
                "content_id": item["content_id"],
                "country": item["country"],
                "language": item["language"],
                "type": item["type"],
                "version": str(item.get("version", "")),
                "title": item.get("title", ""),
                "updated_at": item.get("updated_at", ""),
            },
        )
        docs.append(doc)
    return docs


def ingest_items(items: list[dict], reset: bool = False) -> dict:
    """
    Embed and store items in ChromaDB.

    Args:
        items: Parsed corpus records.
        reset: If True, wipe the collection before ingesting.

    Returns:
        Summary dict: { "ingested": int, "breakdown": {country/language: count} }
    """
    if reset:
        try:
            _get_client().delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    docs = build_documents(items)
    ids = [doc.metadata["content_id"] for doc in docs]

    cfg = Configuration()
    store = Chroma(
        client=_get_client(),
        collection_name=COLLECTION_NAME,
        embedding_function=_make_embeddings(cfg.embedding_model),
        collection_metadata=COLLECTION_METADATA,
    )
    store.add_documents(documents=docs, ids=ids)

    counts = Counter(f"{d.metadata['country']}/{d.metadata['language']}" for d in docs)
    return {"ingested": len(docs), "breakdown": dict(sorted(counts.items()))}


def ingest_file(path: str, reset: bool = False) -> dict:
    """Convenience wrapper that reads a file path."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    items = parse_jsonl(raw)
    return ingest_items(items, reset=reset)
