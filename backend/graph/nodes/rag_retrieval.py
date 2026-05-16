# graph/nodes/rag_retrieval.py
"""
Node 1 — RAG Retrieval
Queries ChromaDB with three semantically-tuned search queries (one per downstream
analysis agent) and appends the live headlines collected by Node 0 to each
chunk list before storing everything in AnalysisState.
"""

import time
from typing import TYPE_CHECKING

# Avoid a hard import of AnalysisState at module level so this file can be
# imported in test environments that have not wired up the full graph yet.
if TYPE_CHECKING:
    from graph.state import AnalysisState
    from langchain_chroma import Chroma


async def rag_retrieval_node(
    state: "AnalysisState",
    vectorstore: "Chroma",
) -> "AnalysisState":
    """
    Retrieves the top-4 most relevant knowledge-base chunks for each of the
    three downstream analysis agents (Market, Financial, Viability).

    Live headlines stored by Node 0 (live_fetch) are appended to every
    chunk list so that all three agents receive current market context
    alongside the static knowledge-base content.

    Parameters
    ----------
    state       : AnalysisState   — current pipeline state (read-only fields used here)
    vectorstore : Chroma          — pre-built ChromaDB instance injected by the graph builder

    Returns
    -------
    AnalysisState with the following keys populated:
        rag_market_chunks    — List[str]
        rag_financial_chunks — List[str]
        rag_viability_chunks — List[str]
        node_timings         — updated with "rag_retrieval" → elapsed seconds
    """
    t_start = time.perf_counter()

    query = state["query"]

    # ------------------------------------------------------------------
    # Build semantically-tuned retrieval queries for each agent.
    # Defaults handle the case where the user left sector / funding_stage
    # blank.
    # ------------------------------------------------------------------
    sector        = (query.sector        or "startup").strip()
    funding_stage = (query.funding_stage or "seed").strip()

    market_query    = f"market size competition trends {sector} industry"
    financial_query = f"financial risk revenue model unit economics {funding_stage}"
    viability_query = "startup success failure patterns product market fit execution risk"

    # ------------------------------------------------------------------
    # Create a single retriever configured to return top-4 chunks.
    # Each ainvoke() call is independent, so results are non-overlapping
    # in terms of intent even if ChromaDB may return some shared chunks.
    # ------------------------------------------------------------------
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    market_docs    = await retriever.ainvoke(market_query)
    financial_docs = await retriever.ainvoke(financial_query)
    viability_docs = await retriever.ainvoke(viability_query)

    # ------------------------------------------------------------------
    # Convert Document objects → plain strings and append live headlines.
    # live_headlines is guaranteed to exist (Node 0 always writes it,
    # defaulting to [] on failure).
    # ------------------------------------------------------------------
    live_headlines: list[str] = state.get("live_headlines", [])

    rag_market_chunks    = [d.page_content for d in market_docs]    + live_headlines
    rag_financial_chunks = [d.page_content for d in financial_docs] + live_headlines
    rag_viability_chunks = [d.page_content for d in viability_docs] + live_headlines

    # ------------------------------------------------------------------
    # Timing bookkeeping — merge into existing timings dict.
    # ------------------------------------------------------------------
    elapsed = round(time.perf_counter() - t_start, 3)
    node_timings = {**state.get("node_timings", {}), "rag_retrieval": elapsed}

    print(
        f"[rag_retrieval] Retrieved {len(market_docs)} market, "
        f"{len(financial_docs)} financial, {len(viability_docs)} viability chunks "
        f"+ {len(live_headlines)} live headlines in {elapsed}s"
    )

    return {
        **state,
        "rag_market_chunks":    rag_market_chunks,
        "rag_financial_chunks": rag_financial_chunks,
        "rag_viability_chunks": rag_viability_chunks,
        "node_timings":         node_timings,
    }
