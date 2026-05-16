# tests/test_local_agents.py
"""
StartupRisk AI v2 — Unit Tests: Local Analysis Agents & Parallel Node

Coverage

  test_market_agent_valid_json            — MarketAgent returns correct output schema
  test_financial_agent_valid_json         — FinancialAgent returns correct output schema
  test_viability_agent_valid_json         — ViabilityAgent returns correct output schema
  test_agent_retry_on_prefix_bleed        — _parse_output strips preamble & extracts JSON
  test_parallel_agents_all_run            — asyncio.gather fires all three agents
  test_parallel_agents_partial_failure    — one agent failure → graceful degradation

Strategy

  ChatOllama.ainvoke is mocked at the class level using unittest.mock.AsyncMock
  so that no real Ollama server or model is required.  All assertions operate on
  the Pydantic output models to verify schema compliance.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Shared fixture data — minimal but schema-valid JSON payloads
# ---------------------------------------------------------------------------

MARKET_JSON: dict = {
    "market_risk_score": 0.42,
    "market_size_assessment": "Large",
    "competition_level": "High",
    "timing_assessment": "Good",
    "market_tailwinds": ["Growing SaaS adoption", "Remote work trends"],
    "market_headwinds": ["Incumbent competition", "Price sensitivity", "Slow enterprise sales cycles"],
    "key_competitors_mentioned": ["Salesforce", "HubSpot"],
    "confidence": 0.75,
    "reasoning": (
        "The B2B SaaS logistics market is growing at 15 % CAGR with strong tailwinds "
        "from digital transformation initiatives. However, established players with "
        "deep integration ecosystems create significant barriers to entry for a "
        "bootstrapped team at this stage."
    ),
}

FINANCIAL_JSON: dict = {
    "financial_risk_score": 0.58,
    "revenue_model_type": "SaaS",
    "revenue_model_viability": "Moderate",
    "burn_rate_risk": "Medium",
    "funding_stage_risk": "High",
    "path_to_profitability": "Possible",
    "key_financial_risks": [
        "Long enterprise sales cycles inflate CAC",
        "Bootstrapped — no runway buffer",
        "Churn risk without dedicated customer success",
    ],
    "financial_strengths": ["Recurring revenue model"],
    "confidence": 0.70,
    "reasoning": (
        "Monthly subscription at Rs.4,999 provides predictable revenue but CAC for SMB "
        "logistics clients tends to be high relative to LTV at this price point. "
        "Bootstrapped status limits marketing spend and extends the runway risk."
    ),
}

VIABILITY_JSON: dict = {
    "viability_risk_score": 0.35,
    "pmf_signal": "Moderate",
    "moat_type": "Switching Costs",
    "moat_strength": "Moderate",
    "traction_level": "Pilots",
    "team_signal": "Strong",
    "execution_complexity": "Medium",
    "key_execution_risks": [
        "Scaling beyond 10 pilots without additional engineering headcount",
        "Integration complexity with legacy fleet management systems",
        "Customer onboarding friction for non-technical logistics SMBs",
    ],
    "positive_signals": [
        "Founders have 5 years domain experience — reduces product risk",
        "10 paying pilot customers confirm early demand signal",
    ],
    "confidence": 0.80,
    "reasoning": (
        "Strong domain expertise from founders combined with 10 active pilots suggests "
        "genuine PMF exploration. Switching costs from data lock-in provide a moderate "
        "moat. Execution risk is manageable given team background."
    ),
}

# Minimal startup description used across all agent tests
STARTUP_DESC = (
    "A B2B SaaS platform helping small logistics companies track their fleet in "
    "real-time. Monthly subscription Rs.4999. 10 pilot customers. Founded by two "
    "engineers with 5 years in logistics software. Currently bootstrapped."
)

RAG_CHUNKS = ["Chunk about SaaS benchmarks.", "Chunk about logistics market TAM."]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(payload: dict) -> MagicMock:
    """Return an object that mimics langchain_core.messages.AIMessage."""
    mock = MagicMock()
    mock.content = json.dumps(payload)
    return mock


def _make_mock_response_with_preamble(payload: dict) -> MagicMock:
    """Simulate a model that bleeds preamble text before the JSON object."""
    mock = MagicMock()
    mock.content = f"Sure! Here is the output: {json.dumps(payload)}"
    return mock


# ===========================================================================
# DELIVERABLE 3 — TESTS
# ===========================================================================


# ---------------------------------------------------------------------------
# test_market_agent_valid_json
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_market_agent_valid_json():
    """
    MarketAgent.analyze() should return a MarketAgentOutput instance with
    field values that match the mocked JSON payload exactly.
    """
    from agents.market_agent import MarketAgent
    from models.schemas import MarketAgentOutput

    agent = MarketAgent()

    with patch.object(
        agent.llm, "ainvoke", new=AsyncMock(return_value=_make_mock_response(MARKET_JSON))
    ):
        output = await agent.analyze(
            startup_desc=STARTUP_DESC,
            rag_chunks=RAG_CHUNKS,
        )

    assert isinstance(output, MarketAgentOutput), (
        f"Expected MarketAgentOutput, got {type(output)}"
    )
    assert output.market_risk_score == pytest.approx(MARKET_JSON["market_risk_score"])
    assert output.market_size_assessment == MARKET_JSON["market_size_assessment"]
    assert output.competition_level == MARKET_JSON["competition_level"]
    assert output.timing_assessment == MARKET_JSON["timing_assessment"]
    assert output.market_tailwinds == MARKET_JSON["market_tailwinds"]
    assert output.market_headwinds == MARKET_JSON["market_headwinds"]
    assert output.key_competitors_mentioned == MARKET_JSON["key_competitors_mentioned"]
    assert output.confidence == pytest.approx(MARKET_JSON["confidence"])
    assert isinstance(output.reasoning, str)
    assert len(output.reasoning) > 0


# ---------------------------------------------------------------------------
# test_financial_agent_valid_json
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_financial_agent_valid_json():
    """
    FinancialAgent.analyze() should return a FinancialAgentOutput instance
    with field values matching the mocked JSON payload.
    """
    from agents.financial_agent import FinancialAgent
    from models.schemas import FinancialAgentOutput

    agent = FinancialAgent()

    with patch.object(
        agent.llm, "ainvoke", new=AsyncMock(return_value=_make_mock_response(FINANCIAL_JSON))
    ):
        output = await agent.analyze(
            startup_desc=STARTUP_DESC,
            rag_chunks=RAG_CHUNKS,
        )

    assert isinstance(output, FinancialAgentOutput), (
        f"Expected FinancialAgentOutput, got {type(output)}"
    )
    assert output.financial_risk_score == pytest.approx(FINANCIAL_JSON["financial_risk_score"])
    assert output.revenue_model_type == FINANCIAL_JSON["revenue_model_type"]
    assert output.revenue_model_viability == FINANCIAL_JSON["revenue_model_viability"]
    assert output.burn_rate_risk == FINANCIAL_JSON["burn_rate_risk"]
    assert output.funding_stage_risk == FINANCIAL_JSON["funding_stage_risk"]
    assert output.path_to_profitability == FINANCIAL_JSON["path_to_profitability"]
    assert output.key_financial_risks == FINANCIAL_JSON["key_financial_risks"]
    assert output.financial_strengths == FINANCIAL_JSON["financial_strengths"]
    assert output.confidence == pytest.approx(FINANCIAL_JSON["confidence"])
    assert isinstance(output.reasoning, str)


# ---------------------------------------------------------------------------
# test_viability_agent_valid_json
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_viability_agent_valid_json():
    """
    ViabilityAgent.analyze() should return a ViabilityAgentOutput instance
    with field values matching the mocked JSON payload.
    """
    from agents.viability_agent import ViabilityAgent
    from models.schemas import ViabilityAgentOutput

    agent = ViabilityAgent()

    with patch.object(
        agent.llm, "ainvoke", new=AsyncMock(return_value=_make_mock_response(VIABILITY_JSON))
    ):
        output = await agent.analyze(
            startup_desc=STARTUP_DESC,
            rag_chunks=RAG_CHUNKS,
        )

    assert isinstance(output, ViabilityAgentOutput), (
        f"Expected ViabilityAgentOutput, got {type(output)}"
    )
    assert output.viability_risk_score == pytest.approx(VIABILITY_JSON["viability_risk_score"])
    assert output.pmf_signal == VIABILITY_JSON["pmf_signal"]
    assert output.moat_type == VIABILITY_JSON["moat_type"]
    assert output.moat_strength == VIABILITY_JSON["moat_strength"]
    assert output.traction_level == VIABILITY_JSON["traction_level"]
    assert output.team_signal == VIABILITY_JSON["team_signal"]
    assert output.execution_complexity == VIABILITY_JSON["execution_complexity"]
    assert output.key_execution_risks == VIABILITY_JSON["key_execution_risks"]
    assert output.positive_signals == VIABILITY_JSON["positive_signals"]
    assert output.confidence == pytest.approx(VIABILITY_JSON["confidence"])
    assert isinstance(output.reasoning, str)


# ---------------------------------------------------------------------------
# test_agent_retry_on_prefix_bleed
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_agent_retry_on_prefix_bleed():
    """
    _parse_output must strip any preamble text that precedes the JSON object.
    Models sometimes emit "Sure! Here is the output: {...}" instead of bare JSON.
    The ViabilityAgent is used as the representative agent for this test.
    """
    from agents.viability_agent import ViabilityAgent
    from models.schemas import ViabilityAgentOutput

    agent = ViabilityAgent()
    bleed_response = _make_mock_response_with_preamble(VIABILITY_JSON)

    with patch.object(
        agent.llm, "ainvoke", new=AsyncMock(return_value=bleed_response)
    ):
        output = await agent.analyze(
            startup_desc=STARTUP_DESC,
            rag_chunks=RAG_CHUNKS,
        )

    assert isinstance(output, ViabilityAgentOutput), (
        "_parse_output should strip preamble and still return a valid ViabilityAgentOutput"
    )
    assert output.viability_risk_score == pytest.approx(VIABILITY_JSON["viability_risk_score"])
    assert output.pmf_signal == VIABILITY_JSON["pmf_signal"]


# ---------------------------------------------------------------------------
# test_parallel_agents_all_run
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_agents_all_run():
    """
    parallel_agents_node() must invoke all three agents concurrently and
    populate market_output, financial_output, and viability_output in state.

    We verify that asyncio.gather is called exactly once (one gather call
    that fans out to three coroutines) and all three output fields are set.
    """
    from models.schemas import (
        MarketAgentOutput,
        FinancialAgentOutput,
        ViabilityAgentOutput,
        StartupQuery,
    )

    # Build minimal state
    query = StartupQuery(
        startup_description=STARTUP_DESC,
        sector="SaaS",
        funding_stage="Bootstrapped",
    )
    state = {
        "query": query,
        "rag_market_chunks": RAG_CHUNKS,
        "rag_financial_chunks": RAG_CHUNKS,
        "rag_viability_chunks": RAG_CHUNKS,
        "node_timings": {},
        "error": None,
        "live_headlines": [],
    }

    market_out = MarketAgentOutput(**MARKET_JSON)
    financial_out = FinancialAgentOutput(**FINANCIAL_JSON)
    viability_out = ViabilityAgentOutput(**VIABILITY_JSON)

    # Patch the module-level agent singletons used inside parallel_agents.py
    with (
        patch(
            "graph.nodes.parallel_agents.market_agent.analyze",
            new=AsyncMock(return_value=market_out),
        ),
        patch(
            "graph.nodes.parallel_agents.financial_agent.analyze",
            new=AsyncMock(return_value=financial_out),
        ),
        patch(
            "graph.nodes.parallel_agents.viability_agent.analyze",
            new=AsyncMock(return_value=viability_out),
        ),
    ):
        # Wrap asyncio.gather to count calls without disrupting its behaviour
        original_gather = asyncio.gather
        gather_call_count = []

        async def tracking_gather(*coros, **kwargs):
            gather_call_count.append(1)
            return await original_gather(*coros, **kwargs)

        with patch("graph.nodes.parallel_agents.asyncio.gather", side_effect=tracking_gather):
            from graph.nodes.parallel_agents import parallel_agents_node
            result = await parallel_agents_node(state)

    # asyncio.gather called exactly once (inside _run_agent triples)
    assert len(gather_call_count) == 1, (
        f"Expected asyncio.gather to be called once, was called {len(gather_call_count)} times"
    )

    # All three outputs populated
    assert isinstance(result["market_output"], MarketAgentOutput)
    assert isinstance(result["financial_output"], FinancialAgentOutput)
    assert isinstance(result["viability_output"], ViabilityAgentOutput)

    # No error
    assert result["error"] is None

    # Timing recorded
    assert "parallel_agents" in result["node_timings"]
    assert isinstance(result["node_timings"]["parallel_agents"], float)


# ---------------------------------------------------------------------------
# test_parallel_agents_partial_failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parallel_agents_partial_failure():
    """
    If financial_agent raises ValueError, parallel_agents_node must:
      — set financial_output to None
      — leave market_output and viability_output populated
      — set state["error"] to a non-None string describing the failure
      — NOT raise an exception itself (graceful degradation)
    """
    from models.schemas import (
        MarketAgentOutput,
        ViabilityAgentOutput,
        StartupQuery,
    )

    query = StartupQuery(
        startup_description=STARTUP_DESC,
        sector="SaaS",
        funding_stage="Bootstrapped",
    )
    state = {
        "query": query,
        "rag_market_chunks": RAG_CHUNKS,
        "rag_financial_chunks": RAG_CHUNKS,
        "rag_viability_chunks": RAG_CHUNKS,
        "node_timings": {},
        "error": None,
        "live_headlines": [],
    }

    market_out = MarketAgentOutput(**MARKET_JSON)
    viability_out = ViabilityAgentOutput(**VIABILITY_JSON)

    async def _raise_value_error(*args, **kwargs):
        raise ValueError("Simulated Ollama JSON parse failure")

    with (
        patch(
            "graph.nodes.parallel_agents.market_agent.analyze",
            new=AsyncMock(return_value=market_out),
        ),
        patch(
            "graph.nodes.parallel_agents.financial_agent.analyze",
            new=AsyncMock(side_effect=_raise_value_error),
        ),
        patch(
            "graph.nodes.parallel_agents.viability_agent.analyze",
            new=AsyncMock(return_value=viability_out),
        ),
    ):
        from graph.nodes.parallel_agents import parallel_agents_node
        result = await parallel_agents_node(state)

    # Market and Viability succeeded
    assert isinstance(result["market_output"], MarketAgentOutput), (
        "market_output should be populated even when financial_agent fails"
    )
    assert isinstance(result["viability_output"], ViabilityAgentOutput), (
        "viability_output should be populated even when financial_agent fails"
    )

    # Financial failed — must be None
    assert result["financial_output"] is None, (
        "financial_output must be None when FinancialAgent raises an exception"
    )

    # Error must be set and describe the failure
    assert result["error"] is not None, (
        "state['error'] must be populated when any agent fails"
    )
    assert "FinancialAgent" in result["error"], (
        f"error message should identify the failing agent; got: {result['error']!r}"
    )
    assert "failed" in result["error"].lower(), (
        f"error message should mention 'failed'; got: {result['error']!r}"
    )
