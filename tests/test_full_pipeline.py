# # tests/test_full_pipeline.py
# import asyncio
# import json
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest

# # ---------------------------------------------------------------------------
# # Constants
# # ---------------------------------------------------------------------------
# B2B_SAAS_DESCRIPTION = (
#     "We are building a B2B SaaS platform for logistics route optimisation "
#     "targeting mid-market freight companies in India. Our AI engine reduces "
#     "fuel costs by 18% and idle time by 23% compared to manual planning. "
#     "We have 3 paying customers generating Rs.12L ARR, with a pipeline of "
#     "Rs.45L in active trials. The founding team has 12 years of combined "
#     "experience in logistics and ML."
# )

# VALID_QUERY = {
#     "startup_description": B2B_SAAS_DESCRIPTION,
#     "sector": "SaaS",
#     "funding_stage": "Seed",
# }

# # ---------------------------------------------------------------------------
# # Mock payloads — field names MUST match Pydantic schemas exactly
# # ---------------------------------------------------------------------------
# MOCK_MARKET_OUTPUT = {
#     "market_risk_score": 0.42,
#     "market_size_assessment": "Medium",
#     "competition_level": "Medium",
#     "timing_assessment": "Good",
#     "market_tailwinds": ["India logistics SaaS growing", "Government infra push"],
#     "market_headwinds": ["Established players", "Long sales cycles", "Price sensitivity"],
#     "key_competitors_mentioned": ["FarEye", "LogiNext", "Locus"],
#     "confidence": 0.75,
#     "reasoning": "India logistics SaaS market is growing rapidly with strong tailwinds.",
# }

# MOCK_FINANCIAL_OUTPUT = {
#     "financial_risk_score": 0.38,
#     "revenue_model_type": "SaaS",
#     "revenue_model_viability": "Strong",
#     "burn_rate_risk": "Low",
#     "funding_stage_risk": "Low",
#     "path_to_profitability": "Clear",
#     "key_financial_risks": ["Customer concentration", "Slow enterprise sales", "Churn risk"],
#     "financial_strengths": ["Early ARR validated"],
#     "confidence": 0.80,
#     "reasoning": "Strong early ARR with validated unit economics.",
# }

# MOCK_VIABILITY_OUTPUT = {
#     "viability_risk_score": 0.40,
#     "pmf_signal": "Moderate",
#     "moat_type": "IP",
#     "moat_strength": "Moderate",
#     "traction_level": "Revenue",
#     "team_signal": "Strong",
#     "execution_complexity": "Medium",
#     "key_execution_risks": ["Enterprise sales cycle", "Integration complexity", "Scaling ops"],
#     "positive_signals": ["Paying customers", "Domain expertise"],
#     "confidence": 0.78,
#     "reasoning": "Strong founding team with relevant domain expertise.",
# }

# MOCK_SYNTHESIS_OUTPUT = {
#     "short_term_prediction": (
#         "High probability of closing the Rs.45L trial pipeline within 6 months "
#         "given 3 paying references and measurable ROI metrics."
#     ),
#     "long_term_prediction": (
#         "If the team executes on enterprise expansion, Rs.5Cr ARR within 3 years "
#         "is achievable given current market trajectory."
#     ),
#     "investment_thesis": (
#         "Differentiated AI moat in an underserved mid-market logistics segment "
#         "with early revenue validation and an experienced founding team."
#     ),
#     "top_risk_factors": [
#         "Enterprise sales cycle length may constrain early growth velocity",
#         "Dependency on 3 early customers creates concentration risk",
#         "Established players like FarEye have deeper enterprise relationships",
#         "Integration complexity with legacy TMS systems may slow onboarding",
#         "Fundraising timeline pressure if trials do not convert",
#     ],
#     "positive_signals": [
#         "18% fuel cost reduction is quantifiable and replicable across customers",
#         "Rs.12L ARR from 3 paying customers validates willingness to pay",
#         "12-year combined founding team experience reduces execution risk",
#     ],
#     "recommendation": "Consider with Caution",
# }

# FAKE_HEADLINES = [
#     {"title": "India logistics SaaS sees record funding", "description": "Q1 2024 data."},
#     {"title": "FarEye raises $100M Series D", "description": "Expansion into Southeast Asia."},
#     {"title": "Fuel costs surge for freight companies", "description": "Diesel up 12% YoY."},
# ]


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------
# def _make_anthropic_response(json_payload: dict) -> MagicMock:
#     content_block = MagicMock()
#     content_block.text = json.dumps(json_payload)
#     response = MagicMock()
#     response.content = [content_block]
#     return response


# def _make_newsapi_response() -> MagicMock:
#     resp = MagicMock()
#     resp.status_code = 200
#     resp.json.return_value = {"status": "ok", "articles": FAKE_HEADLINES}
#     return resp


# # ---------------------------------------------------------------------------
# # Fixtures
# # ---------------------------------------------------------------------------
# @pytest.fixture
# def event_loop():
#     loop = asyncio.new_event_loop()
#     yield loop
#     loop.close()


# # ---------------------------------------------------------------------------
# # Shared mock context builder — avoids copy-paste across tests
# # ---------------------------------------------------------------------------
# def _build_mock_context(newsapi_side_effect=None, claude_side_effect=None, claude_response=None):
#     """Returns the patches needed by all pipeline tests."""
#     from models.schemas import MarketAgentOutput, FinancialAgentOutput, ViabilityAgentOutput

#     market_out = MarketAgentOutput(**MOCK_MARKET_OUTPUT)
#     financial_out = FinancialAgentOutput(**MOCK_FINANCIAL_OUTPUT)
#     viability_out = ViabilityAgentOutput(**MOCK_VIABILITY_OUTPUT)

#     # Agents are module-level singletons created at import time — patching
#     # ChatOllama after the fact has no effect.  Patch BaseLocalAgent.analyze
#     # directly so all three agents use the mock regardless of instantiation order.
#     analyze_mock = AsyncMock(side_effect=[market_out, financial_out, viability_out])

#     newsapi_kwargs = (
#         {"side_effect": newsapi_side_effect}
#         if newsapi_side_effect
#         else {"return_value": _make_newsapi_response()}
#     )

#     claude_kwargs = (
#         {"side_effect": claude_side_effect}
#         if claude_side_effect
#         else {"return_value": claude_response or _make_anthropic_response(MOCK_SYNTHESIS_OUTPUT)}
#     )

#     return analyze_mock, newsapi_kwargs, claude_kwargs


# # ---------------------------------------------------------------------------
# # Test 1 — full happy-path pipeline
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# async def test_full_pipeline_b2b_saas_logistics():
#     from graph.builder import build_analysis_graph, get_initial_state
#     from models.schemas import StartupQuery

#     query = StartupQuery(**VALID_QUERY)
#     mock_vectorstore = MagicMock()
#     mock_retriever = MagicMock()
#     mock_retriever.ainvoke = AsyncMock(return_value=[])
#     mock_vectorstore.as_retriever.return_value = mock_retriever

#     analyze_mock, newsapi_kwargs, claude_kwargs = _build_mock_context()

#     import torch
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.return_value = {
#         "input_ids": torch.zeros((1, 10), dtype=torch.long),
#         "attention_mask": torch.ones((1, 10), dtype=torch.long),
#     }
#     mock_model = MagicMock()
#     mock_model.eval.return_value = None
#     mock_output = MagicMock()
#     mock_output.logits = torch.tensor([[2.0, -1.0, 0.5]])
#     mock_model.return_value = mock_output

#     with (
#         patch("httpx.AsyncClient.get", new_callable=AsyncMock, **newsapi_kwargs),
#         patch("agents.base_local_agent.BaseLocalAgent.analyze", analyze_mock),
#         patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
#         patch("graph.nodes.finbert_sentiment._get_model", return_value=(mock_tokenizer, mock_model)),
#     ):
#         mock_claude_client.messages = MagicMock()
#         mock_claude_client.messages.create = AsyncMock(**claude_kwargs)

#         graph = build_analysis_graph(mock_vectorstore)
#         initial_state = get_initial_state(query)
#         result = await graph.ainvoke(
#             initial_state, config={"configurable": {"thread_id": "test-thread-full-pipeline"}}
#         )

#     assert result.get("error") is None, f"Pipeline error: {result.get('error')}"
#     report = result["final_report"]
#     assert report is not None
#     assert 0.0 <= report.overall_risk_score <= 1.0
#     assert report.risk_level in ["Low", "Moderate", "Moderate-High", "High"]
#     assert len(report.top_risk_factors) == 5
#     assert len(report.positive_signals) == 3
#     assert report.recommendation in ["Invest", "Consider with Caution", "High Caution", "Avoid"]
#     assert result["sentiment_result"] is not None
#     assert result["market_output"] is not None
#     assert result["financial_output"] is not None
#     assert result["viability_output"] is not None

#     m = result["market_output"].market_risk_score
#     f = result["financial_output"].financial_risk_score
#     v = result["viability_output"].viability_risk_score
#     ns = result["sentiment_result"].financial_tone_risk
#     base = 0.35 * m + 0.30 * f + 0.35 * v
#     modifier = 1.00 if ns < 0.15 else 1.03 if ns < 0.35 else 1.07 if ns < 0.60 else 1.12
#     expected_score = min(round(base * modifier, 3), 1.0)
#     assert abs(report.overall_risk_score - expected_score) < 0.01, (
#         f"Score formula mismatch: got {report.overall_risk_score}, expected ~{expected_score}"
#     )


# # ---------------------------------------------------------------------------
# # Test 2 — Claude API failure surfaces as exception
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# async def test_pipeline_missing_claude_raises_500():
#     import anthropic
#     from graph.builder import build_analysis_graph, get_initial_state
#     from models.schemas import StartupQuery

#     query = StartupQuery(**VALID_QUERY)
#     mock_vectorstore = MagicMock()
#     mock_retriever = MagicMock()
#     mock_retriever.ainvoke = AsyncMock(return_value=[])
#     mock_vectorstore.as_retriever.return_value = mock_retriever

#     analyze_mock, newsapi_kwargs, _ = _build_mock_context()

#     import torch
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.return_value = {
#         "input_ids": torch.zeros((1, 10), dtype=torch.long),
#         "attention_mask": torch.ones((1, 10), dtype=torch.long),
#     }
#     mock_model = MagicMock()
#     mock_output = MagicMock()
#     mock_output.logits = torch.tensor([[1.0, -0.5, 0.2]])
#     mock_model.return_value = mock_output

#     with (
#         patch("httpx.AsyncClient.get", new_callable=AsyncMock, **newsapi_kwargs),
#         patch("agents.base_local_agent.BaseLocalAgent.analyze", analyze_mock),
#         patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
#         patch("graph.nodes.finbert_sentiment._get_model", return_value=(mock_tokenizer, mock_model)),
#     ):
#         mock_claude_client.messages = MagicMock()
#         mock_claude_client.messages.create = AsyncMock(
#             side_effect=anthropic.APIConnectionError(request=MagicMock())
#         )

#         graph = build_analysis_graph(mock_vectorstore)
#         initial_state = get_initial_state(query)

#         with pytest.raises(Exception):
#             await graph.ainvoke(
#                 initial_state,
#                 config={"configurable": {"thread_id": "test-claude-fail"}},
#             )


# # ---------------------------------------------------------------------------
# # Test 3 — NewsAPI failure is graceful
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# async def test_pipeline_newsapi_failure_graceful():
#     from graph.builder import build_analysis_graph, get_initial_state
#     from models.schemas import StartupQuery

#     query = StartupQuery(**VALID_QUERY)
#     mock_vectorstore = MagicMock()
#     mock_retriever = MagicMock()
#     mock_retriever.ainvoke = AsyncMock(return_value=[])
#     mock_vectorstore.as_retriever.return_value = mock_retriever

#     analyze_mock, _, claude_kwargs = _build_mock_context(
#         newsapi_side_effect=Exception("NewsAPI unreachable")
#     )

#     import torch
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.return_value = {
#         "input_ids": torch.zeros((1, 10), dtype=torch.long),
#         "attention_mask": torch.ones((1, 10), dtype=torch.long),
#     }
#     mock_model = MagicMock()
#     mock_output = MagicMock()
#     mock_output.logits = torch.tensor([[1.0, -0.5, 0.2]])
#     mock_model.return_value = mock_output

#     with (
#         patch("httpx.AsyncClient.get", new_callable=AsyncMock, side_effect=Exception("NewsAPI unreachable")),
#         patch("agents.base_local_agent.BaseLocalAgent.analyze", analyze_mock),
#         patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
#         patch("graph.nodes.finbert_sentiment._get_model", return_value=(mock_tokenizer, mock_model)),
#     ):
#         mock_claude_client.messages = MagicMock()
#         mock_claude_client.messages.create = AsyncMock(**claude_kwargs)

#         graph = build_analysis_graph(mock_vectorstore)
#         initial_state = get_initial_state(query)
#         result = await graph.ainvoke(
#             initial_state,
#             config={"configurable": {"thread_id": "test-newsapi-fail"}},
#         )

#     assert result["live_headlines"] == [], "live_headlines should be [] on NewsAPI failure"
#     assert result["final_report"] is not None


# # ---------------------------------------------------------------------------
# # Test 4 — short description rejected at API layer
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# async def test_short_description_rejected():
#     from httpx import AsyncClient, ASGITransport
#     from unittest.mock import patch, MagicMock

#     with patch("backend.main.build_or_load_vectorstore", return_value=MagicMock()), \
#          patch("backend.main.build_analysis_graph", return_value=MagicMock()), \
#          patch("backend.main.database.Base.metadata.create_all"):

#         from backend.main import app

#         async with AsyncClient(
#             transport=ASGITransport(app=app), base_url="http://test"
#         ) as client:
#             response = await client.post(
#                 "/api/v2/analyze",
#                 json={"startup_description": "Too short.", "sector": "SaaS", "funding_stage": "Seed"},
#             )

#     assert response.status_code == 400
#     assert "80" in response.text


# # ---------------------------------------------------------------------------
# # Test 5 — health check
# # ---------------------------------------------------------------------------
# @pytest.mark.asyncio
# async def test_health_check():
#     from httpx import AsyncClient, ASGITransport

#     with patch("backend.main.build_or_load_vectorstore", return_value=MagicMock()), \
#          patch("backend.main.build_analysis_graph", return_value=MagicMock()), \
#          patch("backend.main.database.Base.metadata.create_all"):

#         from backend.main import app

#         async with AsyncClient(
#             transport=ASGITransport(app=app), base_url="http://test"
#         ) as client:
#             response = await client.get("/")

#     assert response.status_code == 200
#     data = response.json()
#     assert data["status"] == "ok"
#     assert data["version"] == "2.0.0"


# tests/test_full_pipeline.py
"""
End-to-end integration test for the StartupRisk AI v2 pipeline.

Mocking strategy
----------------
- NewsAPI  : patch httpx.AsyncClient.get
- Agents   : patch agents.base_local_agent.BaseLocalAgent.analyze
             (NOT ChatOllama — agents are module-level singletons created at
             import time, so patching the class constructor after the fact has
             no effect on already-created instances)
- Claude   : patch graph.nodes.claude_synthesis.client
             (NOT anthropic.AsyncAnthropic — same singleton reason)
- FinBERT  : patch graph.nodes.finbert_sentiment._get_model
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
B2B_SAAS_DESCRIPTION = (
    "We are building a B2B SaaS platform for logistics route optimisation "
    "targeting mid-market freight companies in India. Our AI engine reduces "
    "fuel costs by 18% and idle time by 23% compared to manual planning. "
    "We have 3 paying customers generating Rs.12L ARR, with a pipeline of "
    "Rs.45L in active trials. The founding team has 12 years of combined "
    "experience in logistics and ML."
)

VALID_QUERY = {
    "startup_description": B2B_SAAS_DESCRIPTION,
    "sector": "SaaS",
    "funding_stage": "Seed",
}

# ---------------------------------------------------------------------------
# Mock payloads — field names match Pydantic schemas EXACTLY
# ---------------------------------------------------------------------------
MOCK_MARKET_OUTPUT = {
    "market_risk_score": 0.42,
    "market_size_assessment": "Medium",
    "competition_level": "Medium",
    "timing_assessment": "Good",
    "market_tailwinds": ["India logistics SaaS growing", "Government infra push"],
    "market_headwinds": ["Established players", "Long sales cycles", "Price sensitivity"],
    "key_competitors_mentioned": ["FarEye", "LogiNext", "Locus"],
    "confidence": 0.75,
    "reasoning": "India logistics SaaS market is growing rapidly.",
}

MOCK_FINANCIAL_OUTPUT = {
    "financial_risk_score": 0.38,
    "revenue_model_type": "SaaS",
    "revenue_model_viability": "Strong",
    "burn_rate_risk": "Low",
    "funding_stage_risk": "Low",
    "path_to_profitability": "Clear",
    "key_financial_risks": ["Customer concentration", "Slow enterprise sales", "Churn risk"],
    "financial_strengths": ["Early ARR validated"],
    "confidence": 0.80,
    "reasoning": "Strong early ARR with validated unit economics.",
}

MOCK_VIABILITY_OUTPUT = {
    "viability_risk_score": 0.40,
    "pmf_signal": "Moderate",
    "moat_type": "IP",
    "moat_strength": "Moderate",
    "traction_level": "Revenue",
    "team_signal": "Strong",
    "execution_complexity": "Medium",
    "key_execution_risks": ["Enterprise sales cycle", "Integration complexity", "Scaling ops"],
    "positive_signals": ["Paying customers", "Domain expertise"],
    "confidence": 0.78,
    "reasoning": "Strong founding team with relevant domain expertise.",
}

MOCK_SYNTHESIS_OUTPUT = {
    "short_term_prediction": (
        "High probability of closing the Rs.45L trial pipeline within 6 months "
        "given 3 paying references and measurable ROI metrics. The team should "
        "prioritise converting active trials and building case studies from "
        "existing customers to accelerate the enterprise sales cycle."
    ),
    "long_term_prediction": (
        "If the team executes on enterprise expansion, Rs.5Cr ARR within 3 years "
        "is achievable given current market trajectory. Expansion into adjacent "
        "logistics verticals and Southeast Asian markets could accelerate growth "
        "significantly beyond the initial India mid-market focus."
    ),
    "investment_thesis": (
        "Differentiated AI moat in an underserved mid-market logistics segment "
        "with early revenue validation and an experienced founding team."
    ),
    "top_risk_factors": [
        "Enterprise sales cycle length may constrain early growth velocity",
        "Dependency on 3 early customers creates concentration risk",
        "Established players like FarEye have deeper enterprise relationships",
        "Integration complexity with legacy TMS systems may slow onboarding",
        "Fundraising timeline pressure if trials do not convert",
    ],
    "positive_signals": [
        "18% fuel cost reduction is quantifiable and replicable across customers",
        "Rs.12L ARR from 3 paying customers validates willingness to pay",
        "12-year combined founding team experience reduces execution risk",
    ],
    "recommendation": "Consider with Caution",
}

FAKE_HEADLINES = [
    {"title": "India logistics SaaS sees record funding", "description": "Q1 2024 data."},
    {"title": "FarEye raises $100M Series D", "description": "Expansion into Southeast Asia."},
    {"title": "Fuel costs surge for freight companies", "description": "Diesel up 12% YoY."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_anthropic_response(json_payload: dict) -> MagicMock:
    content_block = MagicMock()
    content_block.text = json.dumps(json_payload)
    response = MagicMock()
    response.content = [content_block]
    return response


def _make_newsapi_response() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "articles": FAKE_HEADLINES}
    return resp


def _make_finbert_mocks():
    """Return (mock_tokenizer, mock_model) with positive-dominant logits."""
    import torch
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.zeros((1, 10), dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
    }
    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_output = MagicMock()
    mock_output.logits = torch.tensor([[2.0, -1.0, 0.5]])   # positive dominant
    mock_model.return_value = mock_output
    return mock_tokenizer, mock_model


def _make_agent_analyze_mock():
    """
    Return an AsyncMock for BaseLocalAgent.analyze that returns the three
    schema-valid output objects in market / financial / viability order.
    """
    from models.schemas import MarketAgentOutput, FinancialAgentOutput, ViabilityAgentOutput
    return AsyncMock(side_effect=[
        MarketAgentOutput(**MOCK_MARKET_OUTPUT),
        FinancialAgentOutput(**MOCK_FINANCIAL_OUTPUT),
        ViabilityAgentOutput(**MOCK_VIABILITY_OUTPUT),
    ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Test 1 — full happy-path pipeline
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_full_pipeline_b2b_saas_logistics():
    """
    Full end-to-end integration test.
    All external I/O is mocked. Asserts every requirement from Section 13.4.
    """
    from graph.builder import build_analysis_graph, get_initial_state
    from models.schemas import StartupQuery

    query = StartupQuery(**VALID_QUERY)

    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    mock_vectorstore.as_retriever.return_value = mock_retriever

    mock_tokenizer, mock_model = _make_finbert_mocks()

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock,
              return_value=_make_newsapi_response()),
        # Patch the method on the class — affects ALL agent instances
        patch("agents.base_local_agent.BaseLocalAgent.analyze",
              _make_agent_analyze_mock()),
        # Patch the module-level singleton — NOT the constructor
        patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
        patch("graph.nodes.finbert_sentiment._get_model",
              return_value=(mock_tokenizer, mock_model)),
    ):
        mock_claude_client.messages = MagicMock()
        mock_claude_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response(MOCK_SYNTHESIS_OUTPUT)
        )

        graph = build_analysis_graph(mock_vectorstore)
        initial_state = get_initial_state(query)
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "test-thread-full-pipeline"}},
        )

    # Core assertions
    assert result.get("error") is None, f"Pipeline error: {result.get('error')}"
    report = result["final_report"]
    assert report is not None
    assert 0.0 <= report.overall_risk_score <= 1.0
    assert report.risk_level in ["Low", "Moderate", "Moderate-High", "High"]
    assert len(report.top_risk_factors) == 5
    assert len(report.positive_signals) == 3
    assert report.recommendation in ["Invest", "Consider with Caution", "High Caution", "Avoid"]

    # Agent + sentiment outputs present
    assert result["sentiment_result"] is not None
    assert result["market_output"] is not None
    assert result["financial_output"] is not None
    assert result["viability_output"] is not None

    # Score formula verification
    m_score = result["market_output"].market_risk_score
    f_score = result["financial_output"].financial_risk_score
    v_score = result["viability_output"].viability_risk_score
    ns = result["sentiment_result"].financial_tone_risk
    base = 0.35 * m_score + 0.30 * f_score + 0.35 * v_score
    modifier = 1.00 if ns < 0.15 else 1.03 if ns < 0.35 else 1.07 if ns < 0.60 else 1.12
    expected = min(round(base * modifier, 3), 1.0)
    assert abs(report.overall_risk_score - expected) < 0.01, (
        f"Score mismatch: got {report.overall_risk_score}, expected ~{expected}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Claude API failure surfaces as exception
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_missing_claude_raises_500():
    """If Claude synthesis raises APIError, the pipeline must raise."""
    import anthropic as anthropic_lib
    from graph.builder import build_analysis_graph, get_initial_state
    from models.schemas import StartupQuery

    query = StartupQuery(**VALID_QUERY)
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    mock_vectorstore.as_retriever.return_value = mock_retriever

    mock_tokenizer, mock_model = _make_finbert_mocks()

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock,
              return_value=_make_newsapi_response()),
        patch("agents.base_local_agent.BaseLocalAgent.analyze",
              _make_agent_analyze_mock()),
        patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
        patch("graph.nodes.finbert_sentiment._get_model",
              return_value=(mock_tokenizer, mock_model)),
    ):
        mock_claude_client.messages = MagicMock()
        mock_claude_client.messages.create = AsyncMock(
            side_effect=anthropic_lib.APIConnectionError(request=MagicMock())
        )

        graph = build_analysis_graph(mock_vectorstore)
        initial_state = get_initial_state(query)

        with pytest.raises(Exception):
            await graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": "test-claude-fail"}},
            )


# ---------------------------------------------------------------------------
# Test 3 — NewsAPI failure is graceful
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pipeline_newsapi_failure_graceful():
    """NewsAPI failure must not crash the pipeline — live_headlines should be []."""
    from graph.builder import build_analysis_graph, get_initial_state
    from models.schemas import StartupQuery

    query = StartupQuery(**VALID_QUERY)
    mock_vectorstore = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    mock_vectorstore.as_retriever.return_value = mock_retriever

    mock_tokenizer, mock_model = _make_finbert_mocks()

    with (
        patch("httpx.AsyncClient.get", new_callable=AsyncMock,
              side_effect=Exception("NewsAPI unreachable")),
        patch("agents.base_local_agent.BaseLocalAgent.analyze",
              _make_agent_analyze_mock()),
        patch("graph.nodes.claude_synthesis.client") as mock_claude_client,
        patch("graph.nodes.finbert_sentiment._get_model",
              return_value=(mock_tokenizer, mock_model)),
    ):
        mock_claude_client.messages = MagicMock()
        mock_claude_client.messages.create = AsyncMock(
            return_value=_make_anthropic_response(MOCK_SYNTHESIS_OUTPUT)
        )

        graph = build_analysis_graph(mock_vectorstore)
        initial_state = get_initial_state(query)
        result = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": "test-newsapi-fail"}},
        )

    assert result["live_headlines"] == [], "live_headlines must be [] on NewsAPI failure"
    assert result["final_report"] is not None


# ---------------------------------------------------------------------------
# Test 4 — short description rejected at API layer with HTTP 400
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_short_description_rejected():
    from httpx import AsyncClient, ASGITransport

    with patch("backend.main.build_or_load_vectorstore", return_value=MagicMock()), \
         patch("backend.main.build_analysis_graph", return_value=MagicMock()), \
         patch("backend.main.database.Base.metadata.create_all"):

        from backend.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/v2/analyze",
                json={
                    "startup_description": "Too short.",
                    "sector": "SaaS",
                    "funding_stage": "Seed",
                },
            )

    assert response.status_code == 400
    assert "80" in response.text


# ---------------------------------------------------------------------------
# Test 5 — health check
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_health_check():
    from httpx import AsyncClient, ASGITransport

    with patch("backend.main.build_or_load_vectorstore", return_value=MagicMock()), \
         patch("backend.main.build_analysis_graph", return_value=MagicMock()), \
         patch("backend.main.database.Base.metadata.create_all"):

        from backend.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"