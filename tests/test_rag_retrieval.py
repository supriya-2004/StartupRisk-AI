# tests/test_rag_retrieval.py
"""
Unit tests for Node 1 — RAG Retrieval (Section 13.1 + additional tests).

Test matrix
-----------
From Section 13.1 (spec):
  test_vectorstore_populated           — at least 100 chunks indexed
  test_market_retrieval_relevance      — market query hits market_data/ folder

Additional tests:
  test_financial_retrieval_relevance   — financial query hits financial_models/
  test_viability_retrieval_relevance   — viability query hits startup_patterns/
  test_chunking_count                  — total chunk count 150 ≤ n ≤ 350
  test_retrieval_returns_strings       — helper returns List[str]
  test_rag_node_populates_state        — full async node integration
  test_rag_node_appends_live_headlines — headlines appear in chunk lists
  test_rag_node_records_timing         — node_timings["rag_retrieval"] present
  test_rag_node_default_fallbacks      — handles missing sector / funding_stage
"""

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared fixture — build or load the ChromaDB vectorstore once per test
# session so we don't pay the embedding cost on every test.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vectorstore():
    """Load (or build) the ChromaDB vectorstore from the knowledge base."""
    from rag.indexer import build_or_load_vectorstore
    return build_or_load_vectorstore()


# ---------------------------------------------------------------------------
# Helper used by several tests — retrieve k chunks for an arbitrary query.
# ---------------------------------------------------------------------------

def retrieve_chunks(vectorstore, query: str, k: int = 3) -> List[str]:
    """Synchronous wrapper around the async retriever for test convenience."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    # LangChain retrievers expose a synchronous .invoke() in addition to
    # the async .ainvoke() used by the production code.
    results = retriever.invoke(query)
    return [doc.page_content for doc in results]


# ===========================================================================
# Vectorstore population tests
# ===========================================================================

class TestVectorstorePopulation:

    def test_vectorstore_populated(self, vectorstore):
        """Vectorstore must contain at least 100 indexed chunks."""
        count = vectorstore._collection.count()
        assert count >= 100, (
            f"Expected >= 100 chunks but found {count}. "
            "Run the indexer against the full knowledge base."
        )

    def test_chunking_count(self, vectorstore):
        """Total chunk count should sit between 150 and 350 for the 20-doc KB."""
        count = vectorstore._collection.count()
        assert 150 <= count <= 350, (
            f"Chunk count {count} is outside the expected 150–350 range. "
            "Check chunk_size / chunk_overlap settings in rag/indexer.py."
        )


# ===========================================================================
# Retrieval relevance tests
# ===========================================================================

class TestRetrievalRelevance:

    def test_market_retrieval_relevance(self, vectorstore):
        """
        A market-themed query should surface at least one chunk sourced from
        the market_data/ folder.
        """
        results_raw = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ).invoke("fintech market competition barriers")

        assert len(results_raw) == 3, "Expected exactly 3 results"

        sources = [r.metadata.get("source", "") for r in results_raw]
        assert any("market_data" in s for s in sources), (
            f"No result from market_data/ — sources returned: {sources}"
        )

    def test_financial_retrieval_relevance(self, vectorstore):
        """
        A financial-themed query should surface at least one chunk sourced
        from the financial_models/ folder.
        """
        results_raw = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ).invoke("burn rate runway seed stage")

        sources = [r.metadata.get("source", "") for r in results_raw]
        assert any("financial_models" in s for s in sources), (
            f"No result from financial_models/ — sources returned: {sources}"
        )

    def test_viability_retrieval_relevance(self, vectorstore):
        """
        A viability-themed query should surface at least one chunk sourced
        from the startup_patterns/ folder.
        """
        results_raw = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ).invoke("startup failure product market fit")

        sources = [r.metadata.get("source", "") for r in results_raw]
        assert any("startup_patterns" in s for s in sources), (
            f"No result from startup_patterns/ — sources returned: {sources}"
        )


# ===========================================================================
# Return-type tests
# ===========================================================================

class TestReturnTypes:

    def test_retrieval_returns_strings(self, vectorstore):
        """retrieve_chunks() helper must return a non-empty List[str]."""
        results = retrieve_chunks(vectorstore, "SaaS unit economics CAC LTV", k=3)
        assert isinstance(results, list), "Expected a list"
        assert len(results) > 0,          "Expected at least one result"
        assert all(isinstance(r, str) for r in results), (
            "All items in the returned list must be plain strings"
        )


# ===========================================================================
# Full async node integration tests
# ===========================================================================

class TestRagRetrievalNode:
    """Tests that exercise rag_retrieval_node() end-to-end."""

    def _make_state(self, sector="SaaS", funding_stage="Seed", live_headlines=None):
        """Build a minimal AnalysisState-compatible dict for testing."""
        query = MagicMock()
        query.sector        = sector
        query.funding_stage = funding_stage

        return {
            "query":          query,
            "live_headlines": live_headlines or [],
            "node_timings":   {},
            # Remaining AnalysisState fields — set to None / [] as defaults
            "rag_market_chunks":    [],
            "rag_financial_chunks": [],
            "rag_viability_chunks": [],
            "market_output":        None,
            "financial_output":     None,
            "viability_output":     None,
            "sentiment_result":     None,
            "weighted_risk_score":  None,
            "risk_level":           None,
            "final_report":         None,
            "error":                None,
        }

    @pytest.mark.asyncio
    async def test_rag_node_populates_state(self, vectorstore):
        """Node must populate all three chunk-list keys."""
        from graph.nodes.rag_retrieval import rag_retrieval_node

        state  = self._make_state()
        result = await rag_retrieval_node(state, vectorstore)

        assert "rag_market_chunks"    in result
        assert "rag_financial_chunks" in result
        assert "rag_viability_chunks" in result

        assert len(result["rag_market_chunks"])    > 0
        assert len(result["rag_financial_chunks"]) > 0
        assert len(result["rag_viability_chunks"]) > 0

    @pytest.mark.asyncio
    async def test_rag_node_appends_live_headlines(self, vectorstore):
        """
        Live headlines from Node 0 must appear at the tail of every chunk list.
        """
        from graph.nodes.rag_retrieval import rag_retrieval_node

        headlines = ["Fintech funding slows in Q2.", "AI startups see record valuation."]
        state  = self._make_state(live_headlines=headlines)
        result = await rag_retrieval_node(state, vectorstore)

        for key in ("rag_market_chunks", "rag_financial_chunks", "rag_viability_chunks"):
            chunk_list = result[key]
            # The last N items should be the live headlines (order preserved)
            tail = chunk_list[-len(headlines):]
            assert tail == headlines, (
                f"{key} tail {tail!r} does not match live_headlines {headlines!r}"
            )

    @pytest.mark.asyncio
    async def test_rag_node_records_timing(self, vectorstore):
        """node_timings must contain a 'rag_retrieval' key after the node runs."""
        from graph.nodes.rag_retrieval import rag_retrieval_node

        state  = self._make_state()
        result = await rag_retrieval_node(state, vectorstore)

        assert "rag_retrieval" in result["node_timings"], (
            "Expected 'rag_retrieval' key in node_timings"
        )
        elapsed = result["node_timings"]["rag_retrieval"]
        assert isinstance(elapsed, float) and elapsed >= 0, (
            f"Timing value should be a non-negative float, got {elapsed!r}"
        )

    @pytest.mark.asyncio
    async def test_rag_node_default_fallbacks(self, vectorstore):
        """
        Node must handle missing sector and funding_stage gracefully by
        substituting 'startup' and 'seed' respectively.
        """
        from graph.nodes.rag_retrieval import rag_retrieval_node

        state  = self._make_state(sector=None, funding_stage=None)
        # Should not raise
        result = await rag_retrieval_node(state, vectorstore)

        assert len(result["rag_market_chunks"])    > 0
        assert len(result["rag_financial_chunks"]) > 0
        assert len(result["rag_viability_chunks"]) > 0

    @pytest.mark.asyncio
    async def test_rag_node_preserves_existing_state_keys(self, vectorstore):
        """
        The node must not clobber unrelated state keys such as error or
        final_report.
        """
        from graph.nodes.rag_retrieval import rag_retrieval_node

        state = self._make_state()
        state["error"] = "some_prior_error"

        result = await rag_retrieval_node(state, vectorstore)

        assert result["error"] == "some_prior_error", (
            "rag_retrieval_node must not overwrite pre-existing state keys"
        )
