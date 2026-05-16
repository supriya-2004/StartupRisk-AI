# utils/score_calculator.py
"""
StartupRisk AI v2 — Risk Score Calculator (Node 4 utility)
Pure deterministic arithmetic — no LLM involved.
Section 6.1 of the technical specification.
"""

# ── Sentinel weights (must sum to 1.0) ──────────────────────────────────────
WEIGHT_MARKET     = 0.35
WEIGHT_FINANCIAL  = 0.30
WEIGHT_VIABILITY  = 0.35

# ── FinBERT negative-score → multiplier mapping ──────────────────────────────
# Thresholds are EXCLUSIVE upper bounds (ns < threshold → use that modifier).
_SENTIMENT_MODIFIERS: list[tuple[float, float]] = [
    (0.15, 1.00),   # ns < 0.15  → positive/neutral-leaning
    (0.35, 1.03),   # ns < 0.35  → slight negative tone
    (0.60, 1.07),   # ns < 0.60  → moderate negative tone
    (1.01, 1.12),   # ns >= 0.60 → strongly negative tone  (1.01 acts as "else")
]

# ── Risk-level bands (Section 6.2) ───────────────────────────────────────────
_RISK_BANDS: list[tuple[float, str]] = [
    (0.35, "Low"),
    (0.55, "Moderate"),
    (0.74, "Moderate-High"),
    (1.00, "High"),
]


def _resolve_modifier(finbert_negative_score: float) -> float:
    """Return the FinBERT sentiment multiplier for a given negative-score."""
    for threshold, modifier in _SENTIMENT_MODIFIERS:
        if finbert_negative_score < threshold:
            return modifier
    return 1.12  # fallback — should never be reached


def _resolve_risk_level(score: float) -> str:
    """Map a [0, 1] risk score to its human-readable risk level."""
    for ceiling, label in _RISK_BANDS:
        if score <= ceiling:
            return label
    return "High"  # fallback for score == 1.0


def compute_risk_score(
    market_score: float,
    financial_score: float,
    viability_score: float,
    finbert_negative_score: float,
) -> tuple[float, str]:
    """
    Compute the overall investment risk score and risk level.

    Parameters
    ----------
    market_score : float
        Market analysis risk score in [0.0, 1.0] from the Market Agent.
    financial_score : float
        Financial risk score in [0.0, 1.0] from the Financial Agent.
    viability_score : float
        Startup viability risk score in [0.0, 1.0] from the Viability Agent.
    finbert_negative_score : float
        FinBERT ``negative`` class probability in [0.0, 1.0].

    Returns
    -------
    tuple[float, str]
        (final_score rounded to 3 d.p., risk_level string)

    Formula (Section 6.1)
    ----------------------
    base_score  = 0.35 * market + 0.30 * financial + 0.35 * viability
    modifier    = f(finbert_negative_score)   # one of 1.00 / 1.03 / 1.07 / 1.12
    final_score = min(base_score * modifier, 1.0)
    """
    # ── Weighted base score ──────────────────────────────────────────────────
    base_score: float = (
        WEIGHT_MARKET    * market_score
        + WEIGHT_FINANCIAL * financial_score
        + WEIGHT_VIABILITY * viability_score
    )

    # ── Apply FinBERT modifier ───────────────────────────────────────────────
    modifier: float = _resolve_modifier(finbert_negative_score)
    final_score: float = min(base_score * modifier, 1.0)

    # ── Round and classify ───────────────────────────────────────────────────
    final_score = round(final_score, 3)
    risk_level: str = _resolve_risk_level(final_score)

    return final_score, risk_level
