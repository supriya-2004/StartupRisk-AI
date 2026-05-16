# graph/state.py

"""
AnalysisState — the single shared state object that flows through every node
of the LangGraph pipeline.

Every node receives the full state dict, makes its updates, and returns the
updated dict.  LangGraph merges the return value back into the live state.

All type annotations match the field definitions in Section 2.4 of the
technical documentation exactly.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from typing_extensions import TypedDict

from models.schemas import (
    StartupQuery,
    MarketAgentOutput,
    FinancialAgentOutput,
    ViabilityAgentOutput,
    SentimentResult,
    RiskReport,
)


class AnalysisState(TypedDict):
    #  Input 
    query: StartupQuery

    #  Live Fetch Output (Node 0) 
    live_headlines: List[str]           # Top-10 live NewsAPI headlines for sector

    #  RAG Retrieved Context (Node 1) 
    rag_market_chunks: List[str]        # Top-4 market knowledge chunks
    rag_financial_chunks: List[str]     # Top-4 financial framework chunks
    rag_viability_chunks: List[str]     # Top-4 startup pattern chunks

    #  Analysis Agent Outputs (Node 2) 
    market_output: Optional[MarketAgentOutput]
    financial_output: Optional[FinancialAgentOutput]
    viability_output: Optional[ViabilityAgentOutput]

    #  Local Model Outputs (Node 3) 
    sentiment_result: Optional[SentimentResult]

    #  Computed Score (Node 4) 
    weighted_risk_score: Optional[float]
    risk_level: Optional[str]

    #  Final Report (Node 5) 
    final_report: Optional[RiskReport]

    #  Diagnostics 
    error: Optional[str]
    node_timings: Dict[str, float]      # Wall-clock seconds per node, e.g. {"live_fetch": 0.82}


#  Factory 

def get_empty_state(query: StartupQuery) -> AnalysisState:
    """
    Return a fully-initialised AnalysisState with *query* set and every other
    field at its zero value.

    Usage
    -----
    Pass the returned dict as the ``initial_state`` argument to
    ``graph.ainvoke(initial_state, config=config)`` in the FastAPI router.

    Parameters
    ----------
    query : StartupQuery
        The validated user query (startup description + optional metadata).

    Returns
    -------
    AnalysisState
        A dict whose keys satisfy every field declared in AnalysisState.
    """
    return AnalysisState(
        query=query,

        # Live fetch
        live_headlines=[],

        # RAG chunks — populated by Node 1
        rag_market_chunks=[],
        rag_financial_chunks=[],
        rag_viability_chunks=[],

        # Agent outputs — populated by Node 2 (parallel)
        market_output=None,
        financial_output=None,
        viability_output=None,

        # Sentiment — populated by Node 3
        sentiment_result=None,

        # Score — populated by Node 4
        weighted_risk_score=None,
        risk_level=None,

        # Final report — populated by Node 5
        final_report=None,

        # Diagnostics
        error=None,
        node_timings={},
    )