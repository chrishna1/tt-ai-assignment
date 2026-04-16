"""
Ingestion script — loads corpus.jsonl into ChromaDB.

Usage:
    python ingest.py
    python ingest.py --corpus path/to/corpus.jsonl --reset
"""
import argparse
import json
import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CORPUS_FILE = "corpus.jsonl"


def load_corpus(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_documents(items: list[dict]) -> list[Document]:
    """
    Convert corpus items to LangChain Documents.
    Each item becomes one document (body as page_content, all fields as metadata).
    The body is the full text — no sub-chunking needed given the small corpus size.
    """
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


def ingest(corpus_path: str = CORPUS_FILE, reset: bool = False) -> int:
    """
    Embed and store all corpus items.

    Args:
        corpus_path: Path to the .jsonl corpus file.
        reset:       If True, wipe the existing collection before ingesting.

    Returns:
        Number of documents ingested.
    """
    # Import here so env vars are loaded first
    from src.db.vector_store import get_embeddings, COLLECTION_NAME
    import chromadb
    from chromadb.config import Settings
    from langchain_chroma import Chroma

    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Loading corpus from {corpus_path} ...")
    items = load_corpus(corpus_path)
    print(f"[ingest] {len(items)} items loaded.")

    docs = build_documents(items)

    if reset:
        print("[ingest] Resetting existing collection ...")
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[ingest] Existing collection deleted.")
        except Exception:
            pass

    print(f"[ingest] Embedding {len(docs)} documents into ChromaDB ...")
    embeddings = get_embeddings()

    # Use ids = content_id so re-ingestion is idempotent
    ids = [doc.metadata["content_id"] for doc in docs]

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    store.add_documents(documents=docs, ids=ids)

    print(f"[ingest] Done. {len(docs)} documents stored in {CHROMA_PERSIST_DIR}.")

    # Summary breakdown
    from collections import Counter
    counts = Counter((d.metadata["country"], d.metadata["language"]) for d in docs)
    for (country, lang), n in sorted(counts.items()):
        print(f"         Country {country} / {lang}: {n} items")

    return len(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into ChromaDB")
    parser.add_argument("--corpus", default=CORPUS_FILE, help="Path to corpus.jsonl")
    parser.add_argument("--reset", action="store_true", help="Wipe collection before ingesting")
    args = parser.parse_args()
    ingest(corpus_path=args.corpus, reset=args.reset)
