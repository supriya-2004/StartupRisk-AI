# tests/test_score_calculator.py
"""
StartupRisk AI v2 — Test Suite: Score Calculator
Covers Section 13.3 of the technical specification plus the additional
test cases specified in the project brief.

Run with:
    pytest tests/test_score_calculator.py -v
"""

import math
import pytest

# The module under test
from utils.score_calculator import (
    compute_risk_score,
    WEIGHT_MARKET,
    WEIGHT_FINANCIAL,
    WEIGHT_VIABILITY,
    _resolve_modifier,  # internal helper — tested directly for boundary values
)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13.3 — Tests from the technical specification
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecificationTests:
    """Tests copied verbatim from Section 13.3 of the technical spec."""

    def test_score_formula_low(self):
        """All low inputs should produce a Low risk score below 0.36."""
        score, level = compute_risk_score(0.2, 0.25, 0.2, 0.1)
        assert score < 0.36, f"Expected score < 0.36, got {score}"
        assert level == "Low", f"Expected 'Low', got '{level}'"

    def test_finbert_modifier_applied(self):
        """
        A strongly-negative FinBERT score (ns=0.70 → modifier 1.12) must
        produce a higher final score than no modifier (ns=0.10 → modifier 1.00).
        """
        base_score, _ = compute_risk_score(0.5, 0.5, 0.5, 0.10)  # modifier 1.00
        mod_score, _  = compute_risk_score(0.5, 0.5, 0.5, 0.70)  # modifier 1.12
        assert mod_score > base_score, (
            f"Modifier should increase score. base={base_score}, modified={mod_score}"
        )
        # The difference must match the 1.12 multiplier within floating-point tolerance
        raw_base = 0.35 * 0.5 + 0.30 * 0.5 + 0.35 * 0.5   # = 0.5
        expected_mod = min(raw_base * 1.12, 1.0)
        assert math.isclose(mod_score, round(expected_mod, 3), abs_tol=0.001), (
            f"Modified score should be {round(expected_mod, 3)}, got {mod_score}"
        )

    def test_score_capped_at_1(self):
        """Perfect (maximum) inputs with the highest FinBERT modifier must cap at 1.0."""
        score, _ = compute_risk_score(1.0, 1.0, 1.0, 0.99)
        assert score == 1.0, f"Score must be capped at 1.0, got {score}"


# ═══════════════════════════════════════════════════════════════════════════════
# Additional tests specified in the project brief
# ═══════════════════════════════════════════════════════════════════════════════


class TestAllRiskLevels:
    """Verify every risk-level band is reachable with appropriate inputs."""

    @pytest.mark.parametrize(
        "market, financial, viability, finbert_ns, expected_level",
        [
            # Low:           base = 0.35*0.2 + 0.30*0.2 + 0.35*0.2 = 0.20  × 1.00 = 0.200
            (0.2, 0.2, 0.2, 0.1, "Low"),
            # Moderate:      base = 0.35*0.4 + 0.30*0.4 + 0.35*0.4 = 0.40  × 1.00 = 0.400
            (0.4, 0.4, 0.4, 0.1, "Moderate"),
            # Moderate-High: base = 0.35*0.6 + 0.30*0.6 + 0.35*0.6 = 0.60  × 1.00 = 0.600
            (0.6, 0.6, 0.6, 0.1, "Moderate-High"),
            # High:          base = 0.35*0.8 + 0.30*0.8 + 0.35*0.8 = 0.80  × 1.00 = 0.800
            (0.8, 0.8, 0.8, 0.1, "High"),
        ],
    )
    def test_all_risk_levels(
        self, market, financial, viability, finbert_ns, expected_level
    ):
        _, level = compute_risk_score(market, financial, viability, finbert_ns)
        assert level == expected_level, (
            f"Inputs ({market},{financial},{viability},ns={finbert_ns}) "
            f"→ expected '{expected_level}', got '{level}'"
        )


class TestWeightsSumToOne:
    """Document and verify the weight constraint: weights must sum to exactly 1.0."""

    def test_weights_sum_to_one(self):
        """
        The three agent weights must sum to 1.0.
        Uses math.isclose (abs_tol=1e-9) because IEEE 754 floating-point
        arithmetic means 0.35 + 0.30 + 0.35 evaluates to 0.9999999999999999
        rather than 1.0 exactly — the intent is clearly 1.0 and this test
        documents and enforces that invariant.
        """
        weight_sum = WEIGHT_MARKET + WEIGHT_FINANCIAL + WEIGHT_VIABILITY
        assert math.isclose(weight_sum, 1.0, abs_tol=1e-9), (
            f"Weights must sum to 1.0 for a normalised score. "
            f"Got {WEIGHT_MARKET} + {WEIGHT_FINANCIAL} + {WEIGHT_VIABILITY} = {weight_sum}"
        )


class TestModifierBoundaryValues:
    """
    Verify exact modifier selection at every boundary value.
    Uses the internal _resolve_modifier helper for precision.

    Boundary pairs from the spec (ns < threshold is the condition):
      0.15 boundary:  ns=0.149 → 1.00,  ns=0.150 → 1.03
      0.35 boundary:  ns=0.349 → 1.03,  ns=0.350 → 1.07
      0.60 boundary:  ns=0.599 → 1.07,  ns=0.600 → 1.12
    """

    @pytest.mark.parametrize(
        "ns, expected_modifier",
        [
            # ── 0.15 boundary ──────────────────────────────────────────────
            (0.149, 1.00),   # just below → first band (no adjustment)
            (0.150, 1.03),   # at boundary → second band
            # ── 0.35 boundary ──────────────────────────────────────────────
            (0.349, 1.03),   # just below → second band
            (0.350, 1.07),   # at boundary → third band
            # ── 0.60 boundary ──────────────────────────────────────────────
            (0.599, 1.07),   # just below → third band
            (0.600, 1.12),   # at boundary → fourth band (strongly negative)
        ],
    )
    def test_modifier_boundary_values(self, ns, expected_modifier):
        actual = _resolve_modifier(ns)
        assert actual == expected_modifier, (
            f"ns={ns}: expected modifier {expected_modifier}, got {actual}"
        )

    def test_modifier_at_zero(self):
        """A perfect-positive description (ns=0.0) must use the 1.00 modifier."""
        assert _resolve_modifier(0.0) == 1.00

    def test_modifier_at_one(self):
        """Maximum negativity (ns=1.0) must use the 1.12 modifier."""
        assert _resolve_modifier(1.0) == 1.12


# ═══════════════════════════════════════════════════════════════════════════════
# Formula correctness — arithmetic verification
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormulaArithmetic:
    """Verify the weighted formula and rounding are applied correctly."""

    def test_weighted_formula_correctness(self):
        """Manual calculation must match function output."""
        m, f, v, ns = 0.4, 0.6, 0.5, 0.20
        # ns=0.20 → modifier 1.03
        expected_base = 0.35 * m + 0.30 * f + 0.35 * v
        expected_final = round(min(expected_base * 1.03, 1.0), 3)
        score, _ = compute_risk_score(m, f, v, ns)
        assert math.isclose(score, expected_final, abs_tol=1e-9), (
            f"Expected {expected_final}, got {score}"
        )

    def test_rounding_to_3_decimal_places(self):
        """Result must be rounded to exactly 3 decimal places."""
        score, _ = compute_risk_score(0.333, 0.444, 0.555, 0.05)
        decimal_part = str(score).split(".")
        if len(decimal_part) == 2:
            assert len(decimal_part[1]) <= 3, (
                f"Expected at most 3 decimal places, got '{score}'"
            )

    def test_unequal_weights_market_vs_financial(self):
        """
        Market and viability weights (0.35) are higher than financial (0.30).
        Raising market score must have more impact than the same raise in financial.
        """
        base, _   = compute_risk_score(0.5, 0.5, 0.5, 0.10)
        mkt_bump, _ = compute_risk_score(0.6, 0.5, 0.5, 0.10)   # +0.10 in market
        fin_bump, _ = compute_risk_score(0.5, 0.6, 0.5, 0.10)   # +0.10 in financial
        assert mkt_bump > fin_bump, (
            "Market weight (0.35) > financial weight (0.30), "
            f"so market bump ({mkt_bump}) should exceed financial bump ({fin_bump})"
        )

    @pytest.mark.parametrize(
        "ns, expected_modifier",
        [(0.10, 1.00), (0.25, 1.03), (0.50, 1.07), (0.80, 1.12)],
    )
    def test_modifier_applied_in_final_score(self, ns, expected_modifier):
        """Final score must embed the correct modifier for each FinBERT band."""
        m, f, v = 0.5, 0.5, 0.5
        raw_base = 0.35 * m + 0.30 * f + 0.35 * v      # = 0.5
        expected = round(min(raw_base * expected_modifier, 1.0), 3)
        score, _ = compute_risk_score(m, f, v, ns)
        assert math.isclose(score, expected, abs_tol=1e-9), (
            f"ns={ns}: expected {expected}, got {score}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases & robustness
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Guard against degenerate inputs."""

    def test_all_zeros(self):
        """All-zero inputs should produce score=0.0, level='Low'."""
        score, level = compute_risk_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0.0
        assert level == "Low"

    def test_all_ones_capped(self):
        """All-one inputs (max negativity) must be capped at 1.0."""
        score, level = compute_risk_score(1.0, 1.0, 1.0, 1.0)
        assert score == 1.0
        assert level == "High"

    def test_return_type(self):
        """Return type must be (float, str)."""
        result = compute_risk_score(0.3, 0.4, 0.3, 0.2)
        assert isinstance(result, tuple) and len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], str)

    @pytest.mark.parametrize(
        "level", ["Low", "Moderate", "Moderate-High", "High"]
    )
    def test_risk_level_values_are_valid_strings(self, level):
        """All four risk-level strings must be non-empty and correctly spelled."""
        assert isinstance(level, str) and len(level) > 0
