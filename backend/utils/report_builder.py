# utils/report_builder.py
"""
StartupRisk AI v2 — Report Builder
Assembles the final RiskReport Pydantic model from the completed AnalysisState
and the SynthesisOutput returned by the Claude synthesis node (Node 5).

This module is a pure data-assembly step — no LLM calls, no IO.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

# Pydantic schemas — imported at runtime; TYPE_CHECKING guard avoids circulars
# in unit-test environments where the full app stack may not be installed.
if TYPE_CHECKING:
    from graph.state import AnalysisState

from models.schemas import (
    RiskReport,
    SynthesisOutput,
    MarketAgentOutput,
    FinancialAgentOutput,
    ViabilityAgentOutput,
    SentimentResult,
    StartupQuery,
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(tz=timezone.utc).isoformat()


def build_final_report(
    state: "AnalysisState",
    synthesis: SynthesisOutput,
) -> RiskReport:
    """
    Assemble a complete RiskReport from the finished pipeline state and the
    Claude synthesis output.

    Parameters
    ----------
    state : AnalysisState
        The fully-populated LangGraph state after all nodes have executed.
        Expected to contain:
          - query                 : StartupQuery
          - weighted_risk_score   : float
          - risk_level            : str
          - market_output         : MarketAgentOutput | None
          - financial_output      : FinancialAgentOutput | None
          - viability_output      : ViabilityAgentOutput | None
          - sentiment_result      : SentimentResult | None
          - node_timings          : dict[str, float]
          - live_headlines        : list[str]

    synthesis : SynthesisOutput
        Structured output from the Claude synthesis node containing:
          - short_term_prediction
          - long_term_prediction
          - investment_thesis
          - top_risk_factors
          - positive_signals
          - recommendation

    Returns
    -------
    RiskReport
        Fully populated report ready for API serialisation and DB persistence.
    """
    # ── Core identity / provenance ────────────────────────────────────────────
    query: StartupQuery = state["query"]
    created_at: str = _utc_now_iso()

    # ── Risk score & level (computed by Node 4) ───────────────────────────────
    overall_risk_score: float = float(state["weighted_risk_score"])
    risk_level: str = state["risk_level"]

    # ── Agent outputs (may be None if an agent failed — guard defensively) ────
    market_output: MarketAgentOutput | None = state.get("market_output")
    financial_output: FinancialAgentOutput | None = state.get("financial_output")
    viability_output: ViabilityAgentOutput | None = state.get("viability_output")

    # ── FinBERT sentiment result ──────────────────────────────────────────────
    sentiment_result: SentimentResult | None = state.get("sentiment_result")

    # ── Pipeline performance metadata ─────────────────────────────────────────
    node_timings: dict = dict(state.get("node_timings") or {})
    total_time_seconds: float = round(sum(node_timings.values()), 3)

    # ── Live context metadata ─────────────────────────────────────────────────
    live_headlines: list[str] = list(state.get("live_headlines") or [])

    # ── Assemble and return the final report ──────────────────────────────────
    return RiskReport(
        # ── Provenance ────────────────────────────────────────────────────────
        created_at=created_at,
        startup_description=query.startup_description,
        sector=query.sector,
        funding_stage=query.funding_stage,

        # ── Risk scoring ──────────────────────────────────────────────────────
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,

        # ── Claude synthesis fields ────────────────────────────────────────────
        short_term_prediction=synthesis.short_term_prediction,
        long_term_prediction=synthesis.long_term_prediction,
        investment_thesis=synthesis.investment_thesis,
        top_risk_factors=synthesis.top_risk_factors,        # exactly 5 strings
        positive_signals=synthesis.positive_signals,        # exactly 3 strings
        recommendation=synthesis.recommendation,

        # ── Agent sub-reports ─────────────────────────────────────────────────
        market_analysis=market_output,
        financial_analysis=financial_output,
        viability_analysis=viability_output,

        # ── Sentiment ─────────────────────────────────────────────────────────
        sentiment=sentiment_result,

        # ── Pipeline metadata ─────────────────────────────────────────────────
        node_timings=node_timings,
        total_time_seconds=total_time_seconds,
        live_headlines_used=live_headlines,
        pipeline_version="v2",
    )
