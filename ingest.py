"""
CLI ingestion script — thin wrapper around src.db.ingest.

Usage:
    python ingest.py
    python ingest.py --corpus path/to/corpus.jsonl --reset
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

CORPUS_FILE = "corpus.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into ChromaDB")
    parser.add_argument("--corpus", default=CORPUS_FILE, help="Path to corpus.jsonl")
    parser.add_argument(
        "--reset", action="store_true", help="Wipe collection before ingesting"
    )
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        print(f"[ERROR] Corpus file not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    from src.db.ingest import ingest_file

    print(f"[ingest] Loading {args.corpus} ...")
    summary = ingest_file(args.corpus, reset=args.reset)
    print(f"[ingest] Done. {summary['ingested']} documents stored.")
