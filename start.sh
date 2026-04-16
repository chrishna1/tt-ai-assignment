#!/usr/bin/env bash
set -e

python ingest.py "$@"
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
