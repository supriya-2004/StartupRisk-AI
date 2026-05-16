# tests/test_live_fetch.py
"""
Unit tests for Node 0 — live_fetch_node() (Section 3.5).

All tests mock httpx so they run offline with no real NewsAPI key required.

Test matrix
-----------
  test_live_fetch_returns_list        — successful response → List[str]
  test_live_fetch_graceful_failure    — httpx raises Exception → live_headlines == []
  test_live_fetch_headline_format     — each headline contains the article title
  test_live_fetch_timing_recorded     — node_timings["live_fetch"] is present
  test_live_fetch_empty_articles      — API returns empty articles list → []
  test_live_fetch_missing_description — article with no description still included
  test_live_fetch_sector_used_in_query— sector from query is forwarded to NewsAPI
  test_live_fetch_default_sector      — missing sector falls back to "startup"
"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(sector: str = "FinTech", live_headlines=None) -> dict:
    """Return a minimal AnalysisState-compatible dict."""
    query         = MagicMock()
    query.sector  = sector

    return {
        "query":          query,
        "live_headlines": live_headlines or [],
        "node_timings":   {},
        # Remaining fields — not touched by live_fetch but must exist
        "rag_market_chunks":    [],
        "rag_financial_chunks": [],
        "rag_viability_chunks": [],
        "market_output":        None,
        "financial_output":     None,
        "viability_output":     None,
        "sentiment_result":     None,
        "weighted_risk_score":  None,
        "risk_level":           None,
        "final_report":         None,
        "error":                None,
    }


def _make_mock_response(articles: list) -> MagicMock:
    """
    Build a mock httpx response whose .json() returns the standard
    NewsAPI /v2/everything payload shape.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {"articles": articles}
    return mock_response


def _fake_articles(n: int = 3) -> list:
    """Produce *n* realistic-looking NewsAPI article dicts."""
    return [
        {
            "title":       f"FinTech Headline {i}: Market Update",
            "description": f"Description for article {i} with useful context.",
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Fixture — patch the NEWSAPI_KEY env var so the node does not abort early
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "test-key-abc123")


# ===========================================================================
# Core behaviour tests
# ===========================================================================

class TestLiveFetchCore:

    @pytest.mark.asyncio
    async def test_live_fetch_returns_list(self):
        """Successful API response must yield a non-empty List[str]."""
        from graph.nodes.live_fetch import live_fetch_node

        articles      = _fake_articles(5)
        mock_response = _make_mock_response(articles)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx    = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        headlines = result["live_headlines"]
        assert isinstance(headlines, list),            "live_headlines must be a list"
        assert len(headlines) == 5,                    "Expected one headline per article"
        assert all(isinstance(h, str) for h in headlines), (
            "Every element of live_headlines must be a str"
        )

    @pytest.mark.asyncio
    async def test_live_fetch_graceful_failure(self):
        """
        When httpx raises any exception the node must catch it, log nothing
        harmful, and return live_headlines == [].
        """
        from graph.nodes.live_fetch import live_fetch_node

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(side_effect=Exception("Network unreachable"))
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        assert result["live_headlines"] == [], (
            "live_headlines must be [] when the HTTP call fails"
        )

    @pytest.mark.asyncio
    async def test_live_fetch_headline_format(self):
        """
        Each headline string must contain the article's title text.
        (The node concatenates title + '. ' + description.)
        """
        from graph.nodes.live_fetch import live_fetch_node

        articles      = _fake_articles(3)
        mock_response = _make_mock_response(articles)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx    = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        headlines = result["live_headlines"]
        for i, article in enumerate(articles):
            assert article["title"] in headlines[i], (
                f"Headline at index {i} does not contain the article title "
                f"'{article['title']}'. Got: '{headlines[i]}'"
            )

    @pytest.mark.asyncio
    async def test_live_fetch_timing_recorded(self):
        """node_timings must contain 'live_fetch' after the node runs."""
        from graph.nodes.live_fetch import live_fetch_node

        mock_response = _make_mock_response(_fake_articles(2))
        mock_client   = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx      = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        assert "live_fetch" in result["node_timings"], (
            "Expected 'live_fetch' key in node_timings"
        )
        elapsed = result["node_timings"]["live_fetch"]
        assert isinstance(elapsed, float) and elapsed >= 0, (
            f"Timing should be a non-negative float, got {elapsed!r}"
        )


# ===========================================================================
# Edge-case and contract tests
# ===========================================================================

class TestLiveFetchEdgeCases:

    @pytest.mark.asyncio
    async def test_live_fetch_empty_articles(self):
        """
        API returning an empty 'articles' list must yield live_headlines == [].
        """
        from graph.nodes.live_fetch import live_fetch_node

        mock_response = _make_mock_response([])
        mock_client   = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx      = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        assert result["live_headlines"] == []

    @pytest.mark.asyncio
    async def test_live_fetch_missing_description(self):
        """
        Articles with a None or absent 'description' must still produce a
        valid headline string (no KeyError / TypeError).
        """
        from graph.nodes.live_fetch import live_fetch_node

        articles = [
            {"title": "No Description Article", "description": None},
            {"title": "Missing Key Article"},        # 'description' key absent
        ]
        mock_response = _make_mock_response(articles)
        mock_client   = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx      = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(_make_state())

        headlines = result["live_headlines"]
        assert len(headlines) == 2
        assert all(isinstance(h, str) for h in headlines)
        assert "No Description Article" in headlines[0]
        assert "Missing Key Article"    in headlines[1]

    @pytest.mark.asyncio
    async def test_live_fetch_sector_used_in_query(self):
        """
        The sector from state['query'].sector must be forwarded as the 'q'
        parameter in the NewsAPI HTTP request.
        """
        from graph.nodes.live_fetch import live_fetch_node

        mock_response = _make_mock_response(_fake_articles(1))
        mock_client   = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx      = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            await live_fetch_node(_make_state(sector="HealthTech"))

        _call_kwargs = mock_client.get.call_args
        # params is passed as a keyword argument to httpx AsyncClient.get()
        params = _call_kwargs.kwargs.get("params") or _call_kwargs.args[1] if len(_call_kwargs.args) > 1 else {}
        assert params.get("q") == "HealthTech", (
            f"Expected query param 'q'='HealthTech', got {params!r}"
        )

    @pytest.mark.asyncio
    async def test_live_fetch_default_sector(self):
        """
        When query.sector is None the node must substitute 'startup' as the
        query term — it must not raise and must return a valid list.
        """
        from graph.nodes.live_fetch import live_fetch_node

        mock_response = _make_mock_response(_fake_articles(2))
        mock_client   = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx      = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__  = AsyncMock(return_value=False)

        state = _make_state(sector=None)

        with patch("httpx.AsyncClient", return_value=mock_ctx):
            result = await live_fetch_node(state)

        assert isinstance(result["live_headlines"], list)
