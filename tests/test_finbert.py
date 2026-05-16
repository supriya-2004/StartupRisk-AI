# tests/test_finbert.py
"""
Unit tests for Node 3 — FinBERT Sentiment Analysis
StartupRisk AI v2  ·  Final Year Engineering Project 2025-26

Covers:
  - Section 13.2 tests (negative / positive descriptions)
  - Additional spec tests:
      test_neutral_description
      test_financial_tone_risk_equals_negative_score
      test_scores_sum_to_one
      test_truncation_no_error

Run with:
    pytest tests/test_finbert.py -v

Note: First run downloads ProsusAI/finbert (~440 MB). Subsequent runs use
the HuggingFace local cache and are fast.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or tests/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub out models.schemas so the test file is importable without the full
# FastAPI project installed.  We define a minimal SentimentResult dataclass
# that mirrors the Pydantic model used in production.
# ---------------------------------------------------------------------------
import types

_schemas_stub = types.ModuleType("models.schemas")


@dataclass
class _SentimentResult:
    dominant_sentiment: str
    positive_score: float
    negative_score: float
    neutral_score: float
    financial_tone_risk: float


_schemas_stub.SentimentResult = _SentimentResult  # type: ignore[attr-defined]

# Inject stub before importing the node module
sys.modules.setdefault("models", types.ModuleType("models"))
sys.modules["models.schemas"] = _schemas_stub

# Also stub graph.state so the TYPE_CHECKING import resolves cleanly
_state_stub = types.ModuleType("graph.state")
sys.modules.setdefault("graph", types.ModuleType("graph"))
sys.modules["graph.state"] = _state_stub

# Now import the implementation under test
from backend.graph.nodes.finbert_sentiment import run_finbert  # noqa: E402


# ===========================================================================
# Section 13.2 — Core tests from the project spec
# ===========================================================================


class TestNegativeDescription:
    """FinBERT should score strongly negative financial language as negative."""

    def test_negative_description(self):
        text = "The company faces severe losses, declining revenue, and credit risk."
        result = run_finbert(text)
        assert result.dominant_sentiment == "negative", (
            f"Expected 'negative', got '{result.dominant_sentiment}'"
        )
        assert result.negative_score > 0.5, (
            f"Expected negative_score > 0.5, got {result.negative_score:.4f}"
        )


class TestPositiveDescription:
    """FinBERT should score strong financial performance language as positive."""

    def test_positive_description(self):
        text = "Strong revenue growth, profitable unit economics, expanding market share."
        result = run_finbert(text)
        assert result.dominant_sentiment == "positive", (
            f"Expected 'positive', got '{result.dominant_sentiment}'"
        )
        assert result.positive_score > 0.5, (
            f"Expected positive_score > 0.5, got {result.positive_score:.4f}"
        )


# ===========================================================================
# Additional spec tests
# ===========================================================================


class TestNeutralDescription:
    """Generic, emotionally-flat business language should not be negative-dominant."""

    def test_neutral_description(self):
        text = "The company operates in the software industry with a monthly subscription."
        result = run_finbert(text)
        assert result.dominant_sentiment in ("neutral", "positive"), (
            f"Expected 'neutral' or 'positive', got '{result.dominant_sentiment}'"
        )
        assert result.neutral_score + result.positive_score > result.negative_score, (
            f"neutral + positive ({result.neutral_score + result.positive_score:.4f}) "
            f"should exceed negative ({result.negative_score:.4f})"
        )


class TestFinancialToneRiskEqualsNegativeScore:
    """financial_tone_risk must be exactly equal to negative_score for all inputs."""

    @pytest.mark.parametrize("text", [
        "Strong revenue growth, profitable unit economics, expanding market share.",
        "The company faces severe losses, declining revenue, and credit risk.",
        "The company operates in the software industry with a monthly subscription.",
        "Early-stage startup targeting underserved SMB segment with freemium model.",
    ])
    def test_financial_tone_risk_equals_negative_score(self, text: str):
        result = run_finbert(text)
        assert result.financial_tone_risk == result.negative_score, (
            f"financial_tone_risk ({result.financial_tone_risk}) "
            f"!= negative_score ({result.negative_score})"
        )


class TestScoresSumToOne:
    """positive + negative + neutral must sum to 1.0 (within floating-point tolerance)."""

    @pytest.mark.parametrize("text", [
        "Strong revenue growth, profitable unit economics, expanding market share.",
        "The company faces severe losses, declining revenue, and credit risk.",
        "The company operates in the software industry with a monthly subscription.",
    ])
    def test_scores_sum_to_one(self, text: str):
        result = run_finbert(text)
        total = result.positive_score + result.negative_score + result.neutral_score
        assert abs(total - 1.0) < 0.01, (
            f"Scores sum to {total:.6f}, expected ~1.0 "
            f"(pos={result.positive_score:.4f}, neg={result.negative_score:.4f}, "
            f"neu={result.neutral_score:.4f})"
        )


class TestTruncationNoError:
    """Inputs longer than 512 characters must be handled without raising an exception."""

    def test_truncation_no_error(self):
        long_text = "a " * 1000  # 2000 characters, well beyond BERT's 512-token limit
        try:
            result = run_finbert(long_text)
        except Exception as exc:
            pytest.fail(
                f"run_finbert raised {type(exc).__name__} on 2000-char input: {exc}"
            )
        # Basic sanity: result is still a valid SentimentResult
        assert result.dominant_sentiment in ("positive", "negative", "neutral")
        total = result.positive_score + result.negative_score + result.neutral_score
        assert abs(total - 1.0) < 0.01


# ===========================================================================
# Extra edge-case tests (good practice for a production codebase)
# ===========================================================================


class TestOutputShape:
    """All score fields must be floats in [0, 1]."""

    def test_all_scores_in_range(self):
        text = "Innovative AI platform for financial risk detection with SaaS model."
        result = run_finbert(text)
        for field, value in [
            ("positive_score", result.positive_score),
            ("negative_score", result.negative_score),
            ("neutral_score", result.neutral_score),
            ("financial_tone_risk", result.financial_tone_risk),
        ]:
            assert 0.0 <= value <= 1.0, (
                f"{field} = {value:.4f} is outside [0, 1]"
            )

    def test_dominant_is_argmax(self):
        """dominant_sentiment must correspond to the highest individual score."""
        text = "Severe cash burn, imminent default, no revenue in sight."
        result = run_finbert(text)
        scores = {
            "positive": result.positive_score,
            "negative": result.negative_score,
            "neutral": result.neutral_score,
        }
        expected_dominant = max(scores, key=scores.get)
        assert result.dominant_sentiment == expected_dominant, (
            f"dominant='{result.dominant_sentiment}' but argmax='{expected_dominant}' "
            f"(scores={scores})"
        )


class TestModelCaching:
    """_get_model() must return the same objects on repeated calls (lazy singleton)."""

    def test_model_is_cached(self):
        from backend.graph.nodes.finbert_sentiment import _get_model
        tok1, mod1 = _get_model()
        tok2, mod2 = _get_model()
        assert tok1 is tok2, "Tokenizer should be the same cached instance"
        assert mod1 is mod2, "Model should be the same cached instance"

    def test_model_in_eval_mode(self):
        from backend.graph.nodes.finbert_sentiment import _get_model
        _, model = _get_model()
        assert not model.training, (
            "Model must be in eval() mode to disable dropout during inference"
        )
