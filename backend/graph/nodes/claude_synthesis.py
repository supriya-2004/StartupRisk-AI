# # graph/nodes/claude_synthesis.py
# import time
# import anthropic

# from graph.state import AnalysisState
# from models.schemas import SynthesisOutput
# from utils.report_builder import build_final_report

# client = anthropic.AsyncAnthropic()

# SYNTHESIS_SYSTEM_PROMPT = """You are a senior investment risk analyst synthesizing the outputs of three specialized AI analysis agents into a final investment risk report. You do NOT re-analyze the startup — you synthesize what the agents found. Be precise, honest about uncertainty, and produce investment-grade reasoning. Return ONLY a valid JSON object matching the RiskReport schema exactly."""


# def build_synthesis_prompt(state: AnalysisState) -> str:
#     q = state["query"]
#     m = state["market_output"]
#     f = state["financial_output"]
#     v = state["viability_output"]
#     s = state["sentiment_result"]
#     score = state["weighted_risk_score"]
#     level = state["risk_level"]

#     if m:
#         market_section = (
#             f"   Risk Score: {m.market_risk_score:.2f} | Market Size: {m.market_size_assessment}\n"
#             f"   Competition: {m.competition_level} | Timing: {m.timing_assessment}\n"
#             f"   Tailwinds: {m.market_tailwinds}\n"
#             f"   Headwinds: {m.market_headwinds}\n"
#             f"   Reasoning: {m.reasoning}"
#         )
#     else:
#         market_section = "   [Market agent failed — no data available]"

#     if f:
#         financial_section = (
#             f"   Risk Score: {f.financial_risk_score:.2f} | Revenue Model: {f.revenue_model_type}\n"
#             f"   Viability: {f.revenue_model_viability} | Burn Risk: {f.burn_rate_risk}\n"
#             f"   Path to Profit: {f.path_to_profitability}\n"
#             f"   Key Risks: {f.key_financial_risks}\n"
#             f"   Reasoning: {f.reasoning}"
#         )
#     else:
#         financial_section = "   [Financial agent failed — no data available]"

#     if v:
#         viability_section = (
#             f"   Risk Score: {v.viability_risk_score:.2f} | PMF Signal: {v.pmf_signal}\n"
#             f"   Moat: {v.moat_type} ({v.moat_strength}) | Traction: {v.traction_level}\n"
#             f"   Team Signal: {v.team_signal}\n"
#             f"   Execution Risks: {v.key_execution_risks}\n"
#             f"   Positive Signals: {v.positive_signals}\n"
#             f"   Reasoning: {v.reasoning}"
#         )
#     else:
#         viability_section = "   [Viability agent failed — no data available]"

#     sentiment_line = (
#         f"FinBERT Sentiment: {s.dominant_sentiment} "
#         f"(positive: {s.positive_score:.2f}, negative: {s.negative_score:.2f})"
#         if s else "FinBERT Sentiment: [unavailable]"
#     )

#     return f"""STARTUP QUERY SUMMARY:
# Description: {q.startup_description[:300]}
# Sector: {q.sector or 'Not specified'}
# Funding Stage: {q.funding_stage or 'Not specified'}

# AGENT OUTPUTS:
# 1. Market Analysis Agent:
# {market_section}

# 2. Financial Risk Agent:
# {financial_section}

# 3. Startup Viability Agent:
# {viability_section}

# {sentiment_line}

# COMPUTED RISK SCORE: {score} -> Risk Level: {level}

# Generate the final RiskReport JSON with these fields:
# short_term_prediction (6-month outlook, 80-100 words),
# long_term_prediction (3-year investment outlook, 80-100 words),
# investment_thesis (2-3 sentence recommendation, max 60 words),
# top_risk_factors (list of exactly 5 strings),
# positive_signals (list of exactly 3 strings),
# recommendation (one of: Invest | Consider with Caution | High Caution | Avoid)"""


# async def claude_synthesis_node(state: AnalysisState) -> AnalysisState:
#     t_start = time.perf_counter()

#     try:
#         user_message = build_synthesis_prompt(state)

#         response = await client.messages.create(
#             model="claude-sonnet-4-20250514",
#             max_tokens=1200,
#             temperature=0.0,
#             system=SYNTHESIS_SYSTEM_PROMPT,
#             messages=[{"role": "user", "content": user_message}],
#         )

#         raw_text = response.content[0].text
#         synthesis = SynthesisOutput.model_validate_json(raw_text)
#         report = build_final_report(state, synthesis)

#         elapsed = round(time.perf_counter() - t_start, 3)
#         updated_timings = {**state.get("node_timings", {}), "claude_synthesis": elapsed}

#         return {**state, "final_report": report, "node_timings": updated_timings}

#     except anthropic.APIError as exc:
#         elapsed = round(time.perf_counter() - t_start, 3)
#         updated_timings = {**state.get("node_timings", {}), "claude_synthesis": elapsed}
#         error_msg = f"Claude API error in claude_synthesis_node: {exc}"
#         raise RuntimeError(error_msg) from exc

# graph/nodes/claude_synthesis.py
"""
Node 5 — Claude Synthesis (Single External API Call)
"""

import time
import anthropic

from graph.state import AnalysisState
from models.schemas import SynthesisOutput
from utils.report_builder import build_final_report

client = anthropic.AsyncAnthropic()

SYNTHESIS_SYSTEM_PROMPT = """You are a senior investment risk analyst synthesizing the outputs of three specialized AI analysis agents into a final investment risk report. You do NOT re-analyze the startup — you synthesize what the agents found. Be precise, honest about uncertainty, and produce investment-grade reasoning. Return ONLY a valid JSON object matching the RiskReport schema exactly."""


def build_synthesis_prompt(state: AnalysisState) -> str:
    """
    Build the Claude user message from pipeline state.
    Handles None agent outputs gracefully — if an agent failed, its section
    shows a placeholder so synthesis can still proceed with partial data.
    """
    q = state["query"]
    m = state["market_output"]
    f = state["financial_output"]
    v = state["viability_output"]
    s = state["sentiment_result"]
    score = state["weighted_risk_score"]
    level = state["risk_level"]

    # Per-agent sections — degrade gracefully if an agent returned None
    if m:
        market_section = (
            f"   Risk Score: {m.market_risk_score:.2f} | Market Size: {m.market_size_assessment}\n"
            f"   Competition: {m.competition_level} | Timing: {m.timing_assessment}\n"
            f"   Tailwinds: {m.market_tailwinds}\n"
            f"   Headwinds: {m.market_headwinds}\n"
            f"   Reasoning: {m.reasoning}"
        )
    else:
        market_section = "   [Market agent failed — no data available]"

    if f:
        financial_section = (
            f"   Risk Score: {f.financial_risk_score:.2f} | Revenue Model: {f.revenue_model_type}\n"
            f"   Viability: {f.revenue_model_viability} | Burn Risk: {f.burn_rate_risk}\n"
            f"   Path to Profit: {f.path_to_profitability}\n"
            f"   Key Risks: {f.key_financial_risks}\n"
            f"   Reasoning: {f.reasoning}"
        )
    else:
        financial_section = "   [Financial agent failed — no data available]"

    if v:
        viability_section = (
            f"   Risk Score: {v.viability_risk_score:.2f} | PMF Signal: {v.pmf_signal}\n"
            f"   Moat: {v.moat_type} ({v.moat_strength}) | Traction: {v.traction_level}\n"
            f"   Team Signal: {v.team_signal}\n"
            f"   Execution Risks: {v.key_execution_risks}\n"
            f"   Positive Signals: {v.positive_signals}\n"
            f"   Reasoning: {v.reasoning}"
        )
    else:
        viability_section = "   [Viability agent failed — no data available]"

    sentiment_line = (
        f"FinBERT Sentiment: {s.dominant_sentiment} "
        f"(positive: {s.positive_score:.2f}, negative: {s.negative_score:.2f})"
        if s else "FinBERT Sentiment: [unavailable]"
    )

    return f"""STARTUP QUERY SUMMARY:
Description: {q.startup_description[:300]}
Sector: {q.sector or 'Not specified'}
Funding Stage: {q.funding_stage or 'Not specified'}

AGENT OUTPUTS:
1. Market Analysis Agent:
{market_section}

2. Financial Risk Agent:
{financial_section}

3. Startup Viability Agent:
{viability_section}

{sentiment_line}

COMPUTED RISK SCORE: {score} -> Risk Level: {level}

Generate the final RiskReport JSON with these fields:
short_term_prediction (6-month outlook, 80-100 words),
long_term_prediction (3-year investment outlook, 80-100 words),
investment_thesis (2-3 sentence recommendation, max 60 words),
top_risk_factors (list of exactly 5 strings),
positive_signals (list of exactly 3 strings),
recommendation (one of: Invest | Consider with Caution | High Caution | Avoid)"""


async def claude_synthesis_node(state: AnalysisState) -> AnalysisState:
    """LangGraph Node 5 — single Claude Sonnet API call for final synthesis."""
    t_start = time.perf_counter()

    try:
        user_message = build_synthesis_prompt(state)

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            temperature=0.0,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        import re

        raw_text = response.content[0].text

        # Step 1: remove markdown fences
        clean = re.sub(r"```json|```", "", raw_text).strip()

        # Step 2: extract ONLY JSON object
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in Claude response")

        json_str = match.group(0)

        # Step 3: validate safely
        synthesis = SynthesisOutput.model_validate_json(json_str)
        report = build_final_report(state, synthesis)

        elapsed = round(time.perf_counter() - t_start, 3)
        updated_timings = {**state.get("node_timings", {}), "claude_synthesis": elapsed}
        return {**state, "final_report": report, "node_timings": updated_timings}

    except anthropic.APIError as exc:
        elapsed = round(time.perf_counter() - t_start, 3)
        updated_timings = {**state.get("node_timings", {}), "claude_synthesis": elapsed}
        error_msg = f"Claude API error in claude_synthesis_node: {exc}"
        # Re-raise so LangGraph surfaces the error — do NOT return state here
        raise RuntimeError(error_msg) from exc