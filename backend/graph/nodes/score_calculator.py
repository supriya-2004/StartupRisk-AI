# graph/nodes/score_calculator.py
"""
StartupRisk AI v2 — Node 4: Score Calculation
LangGraph node wrapper around utils.score_calculator.compute_risk_score().

Reads structured agent outputs from AnalysisState, extracts the three risk
scores and the FinBERT negative probability, delegates to the pure utility
function, and writes results back into the state.
"""

import time
from typing import TYPE_CHECKING

from utils.score_calculator import compute_risk_score

# AnalysisState is imported for type hints only; we use TYPE_CHECKING to avoid
# circular imports during testing (the state module imports node files).
if TYPE_CHECKING:
    from graph.state import AnalysisState


# ── Default fallback score used when an agent output is missing ──────────────
_MISSING_SCORE_DEFAULT: float = 0.5


def _extract_market_score(state: dict) -> tuple[float, bool]:
    """Return (market_risk_score, is_default)."""
    output = state.get("market_output")
    if output is None:
        return _MISSING_SCORE_DEFAULT, True
    return float(output.market_risk_score), False


def _extract_financial_score(state: dict) -> tuple[float, bool]:
    """Return (financial_risk_score, is_default)."""
    output = state.get("financial_output")
    if output is None:
        return _MISSING_SCORE_DEFAULT, True
    return float(output.financial_risk_score), False


def _extract_viability_score(state: dict) -> tuple[float, bool]:
    """Return (viability_risk_score, is_default)."""
    output = state.get("viability_output")
    if output is None:
        return _MISSING_SCORE_DEFAULT, True
    return float(output.viability_risk_score), False


def _extract_finbert_score(state: dict) -> tuple[float, bool]:
    """Return (financial_tone_risk / negative_score, is_default)."""
    sentiment = state.get("sentiment_result")
    if sentiment is None:
        return _MISSING_SCORE_DEFAULT, True
    # financial_tone_risk == negative_score as set in finbert_sentiment.py
    return float(sentiment.financial_tone_risk), False


async def score_calculation_node(state: "AnalysisState") -> "AnalysisState":
    """
    LangGraph Node 4 — Deterministic Risk Scoring.

    Extracts agent scores from state, calls compute_risk_score(), and stores:
      • state["weighted_risk_score"]  — final float score (0–1)
      • state["risk_level"]           — e.g. "Moderate-High"
      • state["node_timings"]["score_calculation"]  — wall-clock seconds
      • state["error"]                — warning string if any input was missing
                                        (existing errors are preserved)

    Parameters
    ----------
    state : AnalysisState
        The shared LangGraph state dict at the point Node 4 executes.

    Returns
    -------
    AnalysisState
        Updated copy of state with scoring fields populated.
    """
    node_start = time.perf_counter()
    warnings: list[str] = []

    # ── Extract scores, noting any missing agent outputs ─────────────────────
    market_score,    market_missing    = _extract_market_score(state)
    financial_score, financial_missing = _extract_financial_score(state)
    viability_score, viability_missing = _extract_viability_score(state)
    finbert_score,   finbert_missing   = _extract_finbert_score(state)

    if market_missing:
        warnings.append("market_output missing — defaulting market_score to 0.5")
    if financial_missing:
        warnings.append("financial_output missing — defaulting financial_score to 0.5")
    if viability_missing:
        warnings.append("viability_output missing — defaulting viability_score to 0.5")
    if finbert_missing:
        warnings.append("sentiment_result missing — defaulting finbert_score to 0.5")

    # ── Compute score (pure deterministic utility) ────────────────────────────
    final_score, risk_level = compute_risk_score(
        market_score=market_score,
        financial_score=financial_score,
        viability_score=viability_score,
        finbert_negative_score=finbert_score,
    )

    # ── Record timing ─────────────────────────────────────────────────────────
    elapsed = round(time.perf_counter() - node_start, 4)
    timings: dict = dict(state.get("node_timings") or {})
    timings["score_calculation"] = elapsed

    # ── Merge any new warnings with existing errors ───────────────────────────
    existing_error: str | None = state.get("error")
    if warnings:
        warning_text = "; ".join(warnings)
        combined_error = (
            f"{existing_error}; {warning_text}"
            if existing_error
            else warning_text
        )
    else:
        combined_error = existing_error

    return {
        **state,
        "weighted_risk_score": final_score,
        "risk_level": risk_level,
        "node_timings": timings,
        "error": combined_error,
    }
