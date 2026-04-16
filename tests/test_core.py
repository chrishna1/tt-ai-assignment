"""
Unit tests covering the three most important functions:
  1. Metadata filtering (multi-tenant isolation)
  2. Citation excerpt verification (excerpt is substring of source body)
  3. Score ranking (chunks sorted descending by match_score)
"""

# ---------------------------------------------------------------------------
# Test 1: Metadata filter correctness
# ---------------------------------------------------------------------------


class TestMetadataFilter:
    """
    The retrieve_chunks function must apply country+language filter at query time.
    We test the filter logic by mocking the ChromaDB call and verifying that
    the where clause passed to it contains the correct country and language.
    """

    def test_filter_contains_correct_country_and_language(self, mocker):
        mock_store = mocker.MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mocker.patch(
            "src.db.vector_store._make_embeddings", return_value=mocker.MagicMock()
        )
        mocker.patch("src.db.vector_store.Chroma", return_value=mock_store)

        from src.db.vector_store import retrieve_chunks

        retrieve_chunks(query="test question", country="B", language="es", top_k=3)

        call_kwargs = mock_store.similarity_search_with_relevance_scores.call_args
        where = call_kwargs.kwargs.get("filter") or call_kwargs[1].get("filter")
        assert where is not None, "No filter passed to similarity search"

        # The filter must contain both country and language conditions
        filter_str = str(where)
        assert "B" in filter_str, f"Country 'B' not found in filter: {where}"
        assert "es" in filter_str, f"Language 'es' not found in filter: {where}"

    def test_different_country_produces_different_filter(self, mocker):
        mock_store = mocker.MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mocker.patch(
            "src.db.vector_store._make_embeddings", return_value=mocker.MagicMock()
        )
        mocker.patch("src.db.vector_store.Chroma", return_value=mock_store)

        from src.db.vector_store import retrieve_chunks

        retrieve_chunks(query="test", country="A", language="en", top_k=3)

        call_kwargs = mock_store.similarity_search_with_relevance_scores.call_args
        where = call_kwargs.kwargs.get("filter") or call_kwargs[1].get("filter")
        filter_str = str(where)
        assert "A" in filter_str
        assert "en" in filter_str
        # Country B must NOT appear in the filter for a Country A query
        assert "'B'" not in filter_str and '"B"' not in filter_str


# ---------------------------------------------------------------------------
# Test 2: Citation excerpt verification
# ---------------------------------------------------------------------------


class TestCitationExcerpt:
    """
    The excerpt in every citation must be a substring of the source body.
    This is the core citation fidelity check.
    """

    async def _run_extract_citations(self, chunks: list[dict]) -> list[dict]:
        import time

        from src.agent.nodes import extract_citations
        from src.agent.state import AgentState

        state: AgentState = {
            "question": "test",
            "country": "A",
            "language": "en",
            "retrieved_chunks": chunks,
            "fallback_triggered": False,
            "fallback_reason": None,
            "answer": "test answer",
            "language_used": "en",
            "citations": [],
            "trace": {"retrieval_count": 0, "latency_ms": 0, "model": ""},
            "start_time": time.time(),
        }
        result = await extract_citations(state)
        return result["citations"]

    async def test_excerpt_is_substring_of_body(self):
        body = (
            "Returns are accepted within 48 hours of delivery for defective "
            "or incorrect items. Perishable goods cannot be returned once accepted."
        )
        chunks = [
            {
                "content_id": "a_faq_returns_en",
                "type": "FAQ",
                "country": "A",
                "language": "en",
                "title": "Return Policy",
                "version": "1.0",
                "body": body,
                "match_score": 0.92,
            }
        ]
        citations = await self._run_extract_citations(chunks)
        assert len(citations) == 1
        excerpt = citations[0]["excerpt"]
        # Strip trailing ellipsis before checking substring
        clean_excerpt = excerpt.rstrip(".").rstrip()
        assert clean_excerpt in body, (
            f"Excerpt not found in source body.\nExcerpt: {excerpt!r}\nBody: {body!r}"
        )

    async def test_long_body_excerpt_truncated(self):
        body = "A" * 300  # 300 chars
        chunks = [
            {
                "content_id": "test_long",
                "type": "FAQ",
                "country": "A",
                "language": "en",
                "title": "Test",
                "version": "1.0",
                "body": body,
                "match_score": 0.8,
            }
        ]
        citations = await self._run_extract_citations(chunks)
        assert len(citations) == 1
        # Excerpt must be at most 200 chars (plus possible "...")
        assert len(citations[0]["excerpt"]) <= 203  # 200 + "..."


# ---------------------------------------------------------------------------
# Test 3: Score ranking
# ---------------------------------------------------------------------------


class TestScoreRanking:
    """Retrieved chunks must be sorted descending by match_score."""

    def test_chunks_sorted_by_score_descending(self, mocker):
        # Simulate ChromaDB returning chunks in arbitrary order
        from langchain_core.documents import Document

        raw_results = [
            (
                Document(
                    page_content="low score doc",
                    metadata={
                        "content_id": "low",
                        "type": "FAQ",
                        "country": "A",
                        "language": "en",
                        "title": "T",
                        "version": "1",
                    },
                ),
                0.55,
            ),
            (
                Document(
                    page_content="high score doc",
                    metadata={
                        "content_id": "high",
                        "type": "FAQ",
                        "country": "A",
                        "language": "en",
                        "title": "T",
                        "version": "1",
                    },
                ),
                0.95,
            ),
            (
                Document(
                    page_content="mid score doc",
                    metadata={
                        "content_id": "mid",
                        "type": "FAQ",
                        "country": "A",
                        "language": "en",
                        "title": "T",
                        "version": "1",
                    },
                ),
                0.73,
            ),
        ]

        mock_store = mocker.MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = raw_results
        mocker.patch(
            "src.db.vector_store._make_embeddings", return_value=mocker.MagicMock()
        )
        mocker.patch("src.db.vector_store.Chroma", return_value=mock_store)

        from src.db.vector_store import retrieve_chunks

        chunks = retrieve_chunks("test", "A", "en", top_k=3)

        scores = [c["match_score"] for c in chunks]
        assert scores == sorted(scores, reverse=True), (
            f"Chunks not sorted by score: {scores}"
        )
        assert chunks[0]["content_id"] == "high"
        assert chunks[-1]["content_id"] == "low"
