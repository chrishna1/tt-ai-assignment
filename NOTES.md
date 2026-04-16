# Submission Notes

## What I Built

A working end-to-end RAG system that answers natural-language questions grounded in country- and language-scoped content. The core components:

- **Ingestion pipeline** (`ingest.py`): embeds 44 corpus items into ChromaDB with full metadata
- **LangGraph agent** (`src/agent/`): 4-node graph — validate → retrieve → synthesize → cite, with conditional fallback edge
- **FastAPI server** (`src/api/server.py`): `POST /ask` endpoint returning answer + citations + trace
- **Evaluation harness** (`eval/run_eval.py`): 10 test questions including 2 isolation tests
- **Unit tests** (`tests/test_core.py`): 4 tests covering filter correctness, citation fidelity, score ranking

## What I Chose to Skip

- **Authentication/rate limiting**: out of scope per requirements
- **Sub-document chunking**: corpus items are short; single-doc-per-item is sufficient for the prototype
- **Streaming**: not required; would add complexity without improving the inspection target
- **Distributed cache**: an in-memory cache would be trivial but the latency is acceptable
- **Custom embedding models**: used `text-embedding-3-small` — fast, cheap, good multilingual support

## What I'm Most Proud Of

The multi-tenant isolation. The `where` filter is passed to ChromaDB's ANN query — not applied as a post-processing step. This means cross-country content is excluded before any similarity computation, making the isolation a structural guarantee rather than a best-effort filter. The isolation tests in the eval harness verify this end-to-end.

## What I Would Do Differently

- Start with sub-document chunking from day one — even for a small corpus it improves citation granularity
- Add a reranker step in the graph to reduce noisy citations
- Use `langsmith` tracing from the start rather than hand-rolling the `trace` dict

## Gotchas for the Reviewer

- Requires Python 3.10+
- Must run `python ingest.py` (or `bash start.sh`) before the server will return results
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` must be set in `.env`
- If using OpenAI for the LLM, also set `LLM_MODEL=gpt-4o-mini` in `.env`
- ChromaDB data persists in `./chroma_db/` — delete this directory to start fresh
- `pytest-mock` is required for unit tests: `pip install pytest-mock`
