"""
LangGraph Pipeline Builder — Production Version
================================================
Defines, wires, and compiles the full 6-node AnalysisState directed graph:

  Node 0: live_fetch          — NewsAPI live sector headlines
  Node 1: rag_retrieval       — ChromaDB semantic retrieval (3 targeted queries)
  Node 2: parallel_agents     — Market / Financial / Viability agents (asyncio.gather)
  Node 3: finbert_sentiment   — Local FinBERT financial tone scoring
  Node 4: score_calculation   — Deterministic weighted risk formula (pure Python)
  Node 5: claude_synthesis    — Single Claude Sonnet external API call

Execution order:
  live_fetch → rag_retrieval → parallel_agents → finbert_sentiment
             → score_calculation → claude_synthesis → END

The vectorstore (ChromaDB) is injected into rag_retrieval_node via a closure
so the node signature stays compatible with LangGraph's async node protocol.

State is checkpointed using MemorySaver, which supports both sync and async
operations. Pipeline results are persisted to SQLite via SQLAlchemy.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AnalysisState
from models.schemas import StartupQuery

# ── Node imports (all real implementations — no stubs) ───────────────────────
from graph.nodes.live_fetch        import live_fetch_node
from graph.nodes.rag_retrieval     import rag_retrieval_node
from graph.nodes.parallel_agents   import parallel_agents_node
from graph.nodes.finbert_sentiment import finbert_sentiment_node
from graph.nodes.score_calculator  import score_calculation_node
from graph.nodes.claude_synthesis  import claude_synthesis_node


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_analysis_graph(vectorstore):
    """
    Build and compile the full StartupRisk AI v2 LangGraph pipeline.

    Parameters
    ----------
    vectorstore : Chroma
        A pre-built / pre-loaded ChromaDB vectorstore instance.  It is
        injected into rag_retrieval_node via closure so the node function
        receives it without breaking LangGraph's (state) -> state contract.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to be called with .ainvoke().
    """

    # ── Closure wrappers ──────────────────────────────────────────────────────
    # LangGraph nodes must accept a single positional argument (state).
    # We use closures here to capture `vectorstore` for the retrieval node;
    # all other nodes are wrapped uniformly for consistency.

    async def fetch_node(state: AnalysisState) -> AnalysisState:
        return await live_fetch_node(state)

    async def rag_node(state: AnalysisState) -> AnalysisState:
        # vectorstore captured from outer scope
        return await rag_retrieval_node(state, vectorstore)

    async def agents_node(state: AnalysisState) -> AnalysisState:
        return await parallel_agents_node(state)

    async def finbert_node(state: AnalysisState) -> AnalysisState:
        return await finbert_sentiment_node(state)

    async def score_node(state: AnalysisState) -> AnalysisState:
        return await score_calculation_node(state)

    async def synth_node(state: AnalysisState) -> AnalysisState:
        return await claude_synthesis_node(state)

    # ── Graph definition ──────────────────────────────────────────────────────
    graph = StateGraph(AnalysisState)

    # Register nodes
    graph.add_node("live_fetch",        fetch_node)
    graph.add_node("rag_retrieval",     rag_node)
    graph.add_node("parallel_agents",   agents_node)
    graph.add_node("finbert_sentiment", finbert_node)
    graph.add_node("score_calculation", score_node)
    graph.add_node("claude_synthesis",  synth_node)

    # Entry point
    graph.set_entry_point("live_fetch")

    # Execution edges (linear pipeline — parallel execution happens inside
    # parallel_agents_node via asyncio.gather, not at the graph level)
    graph.add_edge("live_fetch",        "rag_retrieval")
    graph.add_edge("rag_retrieval",     "parallel_agents")
    graph.add_edge("parallel_agents",   "finbert_sentiment")
    graph.add_edge("finbert_sentiment", "score_calculation")
    graph.add_edge("score_calculation", "claude_synthesis")
    graph.add_edge("claude_synthesis",  END)

    # ── Checkpointer (MemorySaver) ─────────────────────────────────────────
    # MemorySaver supports both sync (.invoke) and async (.ainvoke) operations.
    # Pipeline results are persisted to SQLite via SQLAlchemy in the router,
    # so in-memory checkpointing is sufficient for graph execution.
    checkpointer = MemorySaver()
    compiled_graph = graph.compile(checkpointer=checkpointer)
    return compiled_graph


# ── Initial state factory ─────────────────────────────────────────────────────

def get_initial_state(query: StartupQuery) -> dict:
    """
    Returns a fully-populated initial AnalysisState dict with:
      - query set to the provided StartupQuery
      - all list fields initialised to empty lists
      - all Optional fields initialised to None
      - node_timings initialised to an empty dict

    Usage
    -----
    >>> state = get_initial_state(query)
    >>> result = await graph.ainvoke(state, config=config)
    """
    return {
        # ── Input ──────────────────────────────────────────────────────────
        "query": query,

        # ── Node 0 output ──────────────────────────────────────────────────
        "live_headlines": [],

        # ── Node 1 outputs ─────────────────────────────────────────────────
        "rag_market_chunks":    [],
        "rag_financial_chunks": [],
        "rag_viability_chunks": [],

        # ── Node 2 outputs ─────────────────────────────────────────────────
        "market_output":    None,
        "financial_output": None,
        "viability_output": None,

        # ── Node 3 output ──────────────────────────────────────────────────
        "sentiment_result": None,

        # ── Node 4 outputs ─────────────────────────────────────────────────
        "weighted_risk_score": None,
        "risk_level":          None,

        # ── Node 5 output ──────────────────────────────────────────────────
        "final_report": None,

        # ── Error tracking & diagnostics ───────────────────────────────────
        "error":        None,
        "node_timings": {},
    }