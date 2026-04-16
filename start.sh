#!/usr/bin/env bash
# start.sh — single command to ingest corpus and start the API server
# Usage: bash start.sh [--reset]
set -e

RESET_FLAG=""
if [[ "$1" == "--reset" ]]; then
  RESET_FLAG="--reset"
  echo "[start] Resetting vector DB..."
fi

echo "[start] Step 1: Ingesting corpus into ChromaDB..."
python ingest.py $RESET_FLAG

echo "[start] Step 2: Starting FastAPI server..."
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
