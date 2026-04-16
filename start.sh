#!/usr/bin/env bash
# start.sh — start the API server, then ingest corpus.jsonl via the /ingest endpoint
#
# Usage:
#   bash start.sh           # ingest corpus.jsonl, then serve
#   bash start.sh --reset   # wipe DB, re-ingest, then serve
set -e

RESET_PARAM="false"
if [[ "$1" == "--reset" ]]; then
  RESET_PARAM="true"
  echo "[start] Reset flag set — existing collection will be wiped."
fi

CORPUS_FILE="${CORPUS:-corpus.jsonl}"
API_PORT="${API_PORT:-8000}"

if [[ ! -f "$CORPUS_FILE" ]]; then
  echo "[ERROR] Corpus file not found: $CORPUS_FILE" >&2
  exit 1
fi

# Start server in background
echo "[start] Starting FastAPI server on port $API_PORT ..."
uvicorn src.api.server:app --host 0.0.0.0 --port "$API_PORT" &
SERVER_PID=$!

# Wait for server to be ready
echo "[start] Waiting for server to be ready ..."
for i in $(seq 1 20); do
  if curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
    echo "[start] Server is up."
    break
  fi
  sleep 1
done

# Ingest via the /ingest endpoint
echo "[start] Ingesting $CORPUS_FILE via POST /ingest ..."
RESPONSE=$(curl -sf -X POST "http://localhost:$API_PORT/ingest" \
  -F "file=@$CORPUS_FILE;type=application/octet-stream" \
  -F "reset=$RESET_PARAM")

echo "[start] Ingest response: $RESPONSE"

echo "[start] Ready. Server PID=$SERVER_PID"
echo "[start] Try: curl -X POST http://localhost:$API_PORT/ask -H 'Content-Type: application/json' \\"
echo "              -d '{\"question\":\"What is your return policy?\",\"country\":\"B\",\"language\":\"en\"}'"

# Bring server to foreground
wait $SERVER_PID
