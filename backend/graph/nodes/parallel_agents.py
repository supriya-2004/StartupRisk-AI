# graph/nodes/parallel_agents.py
"""
StartupRisk AI v2 — Node 2: Parallel Analysis Agents
Runs MarketAgent, FinancialAgent, and ViabilityAgent concurrently via
asyncio.gather().  All three agents call Ollama locally — no external API.

Execution model

  Node 1 (rag_retrieval) populates three separate chunk lists in state.
  This node fans out to all three agents in parallel, then fans back in,
  merging the outputs into AnalysisState before handing off to Node 3
  (finbert_sentiment).

Error handling

  If a single agent raises an exception its output is set to None and the
  state["error"] key is populated with a descriptive message.  The pipeline
  continues so that remaining nodes can still produce a (degraded) report.
  If multiple agents fail, the last failure message is retained.
"""

import asyncio
import logging
import time
from typing import Optional

from agents.market_agent import MarketAgent
from agents.financial_agent import FinancialAgent
from agents.viability_agent import ViabilityAgent
from graph.state import AnalysisState
from models.schemas import (
    MarketAgentOutput,
    FinancialAgentOutput,
    ViabilityAgentOutput,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons — instantiated once, reused across every request.
# Ollama / LangChain clients are stateless between calls so this is safe.
# ---------------------------------------------------------------------------
market_agent: MarketAgent = MarketAgent()
financial_agent: FinancialAgent = FinancialAgent()
viability_agent: ViabilityAgent = ViabilityAgent()


# ---------------------------------------------------------------------------
# Helper — run a single agent with isolated exception handling
# ---------------------------------------------------------------------------
async def _run_agent(name: str, coro):
    """
    Await *coro* and return (name, result, error_message).
    On any exception the result is None and error_message is set.
    """
    try:
        result = await coro
        return name, result, None
    except Exception as exc:
        msg = f"Agent {name} failed: {exc}"
        logger.exception(msg)
        return name, None, msg


# ---------------------------------------------------------------------------
# Node 2 — parallel_agents_node
# ---------------------------------------------------------------------------
async def parallel_agents_node(state: AnalysisState) -> AnalysisState:
    """
    Fan out to all three analysis agents concurrently and merge results.

    Reads from state
    ----------------
    query.startup_description   — forwarded to every agent as startup_desc
    rag_market_chunks           — RAG context for the Market agent
    rag_financial_chunks        — RAG context for the Financial agent
    rag_viability_chunks        — RAG context for the Viability agent

    Writes to state
    ---------------
    market_output               — MarketAgentOutput | None
    financial_output            — FinancialAgentOutput | None
    viability_output            — ViabilityAgentOutput | None
    node_timings["parallel_agents"] — wall-clock seconds for this node
    error                       — populated if any agent raised an exception
    """
    desc: str = state["query"].startup_description
    t0: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Build coroutines — each agent receives its own RAG chunk list
    # ------------------------------------------------------------------
    market_coro = market_agent.analyze(
        startup_desc=desc,
        rag_chunks=state.get("rag_market_chunks", []),
    )
    financial_coro = financial_agent.analyze(
        startup_desc=desc,
        rag_chunks=state.get("rag_financial_chunks", []),
    )
    viability_coro = viability_agent.analyze(
        startup_desc=desc,
        rag_chunks=state.get("rag_viability_chunks", []),
    )

    # ------------------------------------------------------------------
    # Run all three concurrently — individual errors are isolated
    # ------------------------------------------------------------------
    results = await asyncio.gather(
        _run_agent("MarketAgent", market_coro),
        _run_agent("FinancialAgent", financial_coro),
        _run_agent("ViabilityAgent", viability_coro),
    )

    # ------------------------------------------------------------------
    # Unpack results
    # ------------------------------------------------------------------
    market_output: Optional[MarketAgentOutput] = None
    financial_output: Optional[FinancialAgentOutput] = None
    viability_output: Optional[ViabilityAgentOutput] = None
    error_message: Optional[str] = state.get("error")  # preserve upstream errors

    for agent_name, output, error in results:
        if error:
            # Last failing agent's message wins (edge-case: multiple failures)
            error_message = error
            logger.warning("Partial pipeline failure — %s", error)

        if agent_name == "MarketAgent":
            market_output = output
        elif agent_name == "FinancialAgent":
            financial_output = output
        elif agent_name == "ViabilityAgent":
            viability_output = output

    elapsed: float = round(time.perf_counter() - t0, 3)
    logger.info(
        "parallel_agents_node completed in %.3fs  "
        "(market=%s, financial=%s, viability=%s)",
        elapsed,
        "OK" if market_output else "FAILED",
        "OK" if financial_output else "FAILED",
        "OK" if viability_output else "FAILED",
    )

    # ------------------------------------------------------------------
    # Merge into state — spread operator preserves all existing keys
    # ------------------------------------------------------------------
    node_timings: dict = {**state.get("node_timings", {}), "parallel_agents": elapsed}

    return {
        **state,
        "market_output": market_output,
        "financial_output": financial_output,
        "viability_output": viability_output,
        "node_timings": node_timings,
        "error": error_message,
    }
