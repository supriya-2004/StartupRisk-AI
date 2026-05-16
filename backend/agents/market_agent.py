
from langchain_community.chat_models import ChatOllama
from agents.base_local_agent import BaseLocalAgent
from models.schemas import MarketAgentOutput

# ---------------------------------------------------------------------------
# System prompt (verbatim from Section 4.3 of the technical documentation)
# ---------------------------------------------------------------------------

MARKET_SYSTEM_PROMPT = """You are a market analysis specialist for startup investments. You receive a KNOWLEDGE BASE CONTEXT containing real market data, and a startup description. Use the knowledge base to inform your analysis.

Analyze the MARKET-LEVEL risk only. Assess:
- Market size and growth potential relevant to this startup
- Competitive landscape and barriers to entry
- Market timing (is the window open or closing?)
- Geographic market dynamics

Return ONLY this JSON structure (no other text):
{
  "market_risk_score": <float 0.0-1.0>,
  "market_size_assessment": "Large|Medium|Small|Niche",
  "competition_level": "Very High|High|Medium|Low",
  "timing_assessment": "Excellent|Good|Fair|Poor",
  "market_tailwinds": ["<string>", "<string>"],
  "market_headwinds": ["<string>", "<string>", "<string>"],
  "key_competitors_mentioned": ["<string>"],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<string max 120 words>"
}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MarketAgent(BaseLocalAgent):
    """Market Analysis Agent — powered by Llama 3.2 3B via Ollama.

    Assesses market-level risk: market size, competition, timing, and
    geographic dynamics. All logic lives in ``BaseLocalAgent``; this class
    only supplies the domain-specific system prompt and output schema.
    """

    def __init__(self) -> None:
        super().__init__(
            system_prompt=MARKET_SYSTEM_PROMPT,
            output_schema=MarketAgentOutput,
        )
