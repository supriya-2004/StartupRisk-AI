# graph/nodes/live_fetch.py

"""
Node 0 — Live Fetch

Fetches the 10 most recent news headlines for the startup's sector from
NewsAPI before any RAG retrieval or agent analysis runs.

The headlines are stored in state['live_headlines'] and later:
  - Appended to every agent's RAG context chunks (Node 1)
  - Averaged with the pitch sentiment score by FinBERT (Node 3)

On any network or API error the node fails gracefully: live_headlines is
set to [] and the pipeline continues without live data.

Environment variable required:
  NEWSAPI_KEY  — free tier at newsapi.org (100 requests/day)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING
from dotenv import load_dotenv

load_dotenv()

import httpx

if TYPE_CHECKING:
    from graph.state import AnalysisState

NEWSAPI_URL: str = "https://newsapi.org/v2/everything"
REQUEST_TIMEOUT: float = 5.0
PAGE_SIZE: int = 10


async def live_fetch_node(state: "AnalysisState") -> "AnalysisState":
    """
    LangGraph node that retrieves live sector headlines from NewsAPI.

    Parameters
    ----------
    state : AnalysisState
        The current pipeline state.  Reads ``state['query'].sector``.

    Returns
    -------
    AnalysisState
        Updated state with:
        - ``live_headlines`` — list of up to 10 headline strings
          (format: ``"<title>. <description>"``)
        - ``node_timings["live_fetch"]`` — wall-clock seconds taken
    """
    t_start = time.perf_counter()

    sector: str = (state["query"].sector or "startup").strip()
    api_key: str | None = os.getenv("NEWSAPI_KEY")

    headlines: list[str] = []

    try:
        if not api_key:
            raise ValueError(
                "NEWSAPI_KEY environment variable is not set. "
                "Sign up for a free key at https://newsapi.org."
            )

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                NEWSAPI_URL,
                params={
                    "q": sector,
                    "sortBy": "publishedAt",
                    "pageSize": PAGE_SIZE,
                    "apiKey": api_key,
                },
            )
            response.raise_for_status()
            data = response.json()

        articles: list[dict] = data.get("articles", [])

        headlines = [
            article["title"] + ". " + (article.get("description") or "")
            for article in articles
            if article.get("title")          # skip entries with no title
        ]

    except Exception as exc:
        # Fail gracefully — the pipeline is fully functional without live data.
        # A missing API key, network timeout, or API error will land here.
        print(
            f"[live_fetch] WARNING: Could not fetch live headlines "
            f"for sector='{sector}'. Reason: {exc!r}. "
            "Continuing with empty live_headlines."
        )
        headlines = []

    elapsed = round(time.perf_counter() - t_start, 3)

    # Merge updated fields back into state (LangGraph pattern)
    updated_timings: dict = {**state.get("node_timings", {}), "live_fetch": elapsed}

    return {
        **state,
        "live_headlines": headlines,
        "node_timings": updated_timings,
    }
