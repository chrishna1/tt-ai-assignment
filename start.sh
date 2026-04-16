#!/usr/bin/env bash
set -e

uv run python ingest.py "$@"
uv run uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload
