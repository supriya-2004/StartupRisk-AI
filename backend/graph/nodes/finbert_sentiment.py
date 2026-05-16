# graph/nodes/finbert_sentiment.py
"""
Node 3 — FinBERT Sentiment Analysis
StartupRisk AI v2  ·  Final Year Engineering Project 2025-26

Scores the startup pitch description (and live NewsAPI headlines if present)
using ProsusAI/FinBERT — a BERT model pre-trained on financial text.

Label order from FinBERT logits: ["positive", "negative", "neutral"]

Dependencies:
    pip install transformers torch
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Type stubs — imported only during type checking to avoid circular imports
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    from graph.state import AnalysisState
    from models.schemas import SentimentResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME: str = "ProsusAI/finbert"

# FinBERT's logit order (fixed by the model's label2id mapping)
LABELS: list[str] = ["positive", "negative", "neutral"]

# ---------------------------------------------------------------------------
# Module-level cached singletons (lazy-loaded on first inference call)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


def _get_model():
    """
    Lazy-load FinBERT tokenizer and model on the first call.
    Subsequent calls return the already-loaded cached instances.

    Returns:
        tuple[PreTrainedTokenizer, PreTrainedModel]
    """
    global _tokenizer, _model

    if _tokenizer is None:
        print("Loading FinBERT model...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()  # Disable dropout / BatchNorm training behaviour
        print("FinBERT ready.")

    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Internal helper — score a single piece of text
# ---------------------------------------------------------------------------
def _score_text(text: str) -> dict[str, float]:
    """
    Run FinBERT inference on *text* and return a probability dict.

    Args:
        text: Raw input string; will be truncated to 512 tokens by the tokenizer.

    Returns:
        {"positive": float, "negative": float, "neutral": float}
    """
    tokenizer, model = _get_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).squeeze()  # shape: (3,)
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


# ---------------------------------------------------------------------------
# Public synchronous helper — for direct / test use without a full state
# ---------------------------------------------------------------------------
def run_finbert(text: str) -> "SentimentResult":
    """
    Synchronous, state-free entry point for FinBERT scoring.
    Useful for unit tests and ad-hoc usage.

    Args:
        text: Startup description or any financial text.

    Returns:
        SentimentResult Pydantic model.
    """
    # Inline import so this module stays importable even when models.schemas
    # is not on the path (e.g. isolated unit-test environments mock it).
    try:
        from models.schemas import SentimentResult
    except ModuleNotFoundError:
        # Fallback: return a plain dataclass-like object for test environments
        # that haven't wired up the full project structure.
        from dataclasses import dataclass

        @dataclass
        class SentimentResult:  # type: ignore[no-redef]
            dominant_sentiment: str
            positive_score: float
            negative_score: float
            neutral_score: float
            financial_tone_risk: float

    scores = _score_text(text[:512])
    dominant = max(scores, key=scores.get)

    return SentimentResult(
        dominant_sentiment=dominant,
        positive_score=scores["positive"],
        negative_score=scores["negative"],
        neutral_score=scores["neutral"],
        financial_tone_risk=scores["negative"],  # used directly in Node 4 formula
    )


# ---------------------------------------------------------------------------
# LangGraph async node
# ---------------------------------------------------------------------------
async def finbert_sentiment_node(state: "AnalysisState") -> "AnalysisState":
    """
    LangGraph Node 3 — FinBERT Sentiment Analysis.

    Pipeline:
      Step 1 — Score the startup pitch description (char-truncated to 512).
      Step 2 — Score live headlines if present (joined + char-truncated to 512).
      Step 3 — If headlines exist, average scores element-wise with pitch scores.
      Step 4 — Determine dominant label (argmax over averaged scores).
      Step 5 — Build SentimentResult.
      Step 6 — Store in state["sentiment_result"].
      Step 7 — Record wall-clock timing in state["node_timings"]["finbert_sentiment"].

    Args:
        state: Shared AnalysisState TypedDict carrying all pipeline data.

    Returns:
        Updated AnalysisState with sentiment_result and node_timings populated.
    """
    from models.schemas import SentimentResult

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Score the startup pitch description
    # ------------------------------------------------------------------
    pitch_text: str = state["query"].startup_description[:512]
    pitch_scores: dict[str, float] = _score_text(pitch_text)

    # ------------------------------------------------------------------
    # Step 2 — Score live headlines (if any were fetched by Node 0)
    # ------------------------------------------------------------------
    live_headlines: list[str] = state.get("live_headlines", []) or []
    headlines_text: str = " ".join(live_headlines)[:512]

    # ------------------------------------------------------------------
    # Step 3 — Element-wise average of pitch + headline scores
    # ------------------------------------------------------------------
    if headlines_text:
        headline_scores: dict[str, float] = _score_text(headlines_text)
        averaged_scores: dict[str, float] = {
            label: (pitch_scores[label] + headline_scores[label]) / 2.0
            for label in LABELS
        }
    else:
        averaged_scores = pitch_scores

    # ------------------------------------------------------------------
    # Step 4 — Dominant label
    # ------------------------------------------------------------------
    dominant: str = max(averaged_scores, key=averaged_scores.get)

    # ------------------------------------------------------------------
    # Step 5 — Build SentimentResult
    # ------------------------------------------------------------------
    sentiment_result = SentimentResult(
        dominant_sentiment=dominant,
        positive_score=averaged_scores["positive"],
        negative_score=averaged_scores["negative"],
        neutral_score=averaged_scores["neutral"],
        financial_tone_risk=averaged_scores["negative"],  # Node 4 uses this directly
    )

    # ------------------------------------------------------------------
    # Step 7 — Record timing
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    node_timings: dict = dict(state.get("node_timings") or {})
    node_timings["finbert_sentiment"] = round(elapsed, 4)

    # ------------------------------------------------------------------
    # Step 6 — Return updated state
    # ------------------------------------------------------------------
    return {
        **state,
        "sentiment_result": sentiment_result,
        "node_timings": node_timings,
    }
