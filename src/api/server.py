"""
FastAPI HTTP wrapper around the LangGraph Q&A agent.

POST /ingest
    Upload a corpus.jsonl file to embed and store in ChromaDB.
    Form params: file (UploadFile), reset (bool, default False)

POST /ask
    Request:  { "question": str, "country": str, "language": str }
    Response: { "answer": str, "language_used": str, "citations": [...], "trace": {...} }

GET /health
    Returns { "status": "ok", "collection_size": int }
"""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.api.schema import AskRequest, AskResponse, CitationResponse, TraceResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="Multi-Country Content Q&A",
    description="RAG-based Q&A system with country/language scoping and citations.",
    version="1.0.0",
)


@app.post("/ingest")
async def ingest_endpoint(
    file: UploadFile = File(..., description="corpus.jsonl file to ingest"),
    reset: bool = Form(False, description="Wipe existing collection before ingesting"),
):
    """
    Upload a .jsonl corpus file and embed it into ChromaDB.
    Each line must be a JSON object with fields: content_id, country, language,
    type, version, title, body, updated_at.
    """
    if not file.filename or not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="File must be a .jsonl file")

    raw = await file.read()
    if not raw.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    logger.info(
        "Ingest request | file=%s reset=%s size=%d bytes",
        file.filename,
        reset,
        len(raw),
    )

    try:
        from src.db.ingest import ingest_items, parse_jsonl

        items = parse_jsonl(raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse JSONL: {e}")

    if not items:
        raise HTTPException(status_code=422, detail="No valid JSON lines found in file")

    try:
        summary = ingest_items(items, reset=reset)
    except Exception as e:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Ingest complete | ingested=%d", summary["ingested"])
    return {"status": "ok", **summary}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    logger.info(
        "Received question | country=%s lang=%s question=%r",
        req.country,
        req.language,
        req.question[:80],
    )

    try:
        from src.agent.graph import ask_async

        result = await ask_async(
            question=req.question,
            country=req.country,
            language=req.language,
        )
    except Exception as e:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Response | citations=%d latency_ms=%d",
        len(result.get("citations", [])),
        result.get("trace", {}).get("latency_ms", 0),
    )

    return AskResponse(
        answer=result["answer"],
        language_used=result["language_used"],
        citations=[CitationResponse(**c) for c in result.get("citations", [])],
        trace=TraceResponse(**result["trace"]),
    )


@app.get("/health")
def health():
    from src.db.vector_store import count_documents

    try:
        n = count_documents()
    except Exception:
        n = -1
    return {"status": "ok", "collection_size": n}
