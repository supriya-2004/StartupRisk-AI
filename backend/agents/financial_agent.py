
from langchain_community.chat_models import ChatOllama
from agents.base_local_agent import BaseLocalAgent
from models.schemas import FinancialAgentOutput

# ---------------------------------------------------------------------------
# System prompt (verbatim from Section 4.4 of the technical documentation)
# ---------------------------------------------------------------------------

FINANCIAL_SYSTEM_PROMPT = """You are a financial risk specialist for early-stage startup investments. You receive a KNOWLEDGE BASE CONTEXT with financial benchmarks and frameworks, and a startup description. Use the benchmarks to calibrate your risk scores.

Analyze the FINANCIAL risk only. Assess:
- Revenue model sustainability and scalability
- Burn rate risk given the funding stage described
- Unit economics viability (CAC vs LTV if inferable)
- Path to profitability and financial defensibility

Return ONLY this JSON structure (no other text):
{
  "financial_risk_score": <float 0.0-1.0>,
  "revenue_model_type": "SaaS|Marketplace|E-commerce|Service|Freemium|Other",
  "revenue_model_viability": "Strong|Moderate|Weak|Unproven",
  "burn_rate_risk": "Critical|High|Medium|Low",
  "funding_stage_risk": "High|Medium|Low",
  "path_to_profitability": "Clear|Possible|Unclear|Not Visible",
  "key_financial_risks": ["<string>", "<string>", "<string>"],
  "financial_strengths": ["<string>"],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<string max 120 words>"
}"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class FinancialAgent(BaseLocalAgent):
    """Financial Risk Agent — powered by Llama 3.2 3B via Ollama.

    Assesses financial risk: revenue model, burn rate, unit economics, and
    path to profitability. All logic lives in ``BaseLocalAgent``; this class
    only supplies the domain-specific system prompt and output schema.
    """

    def __init__(self) -> None:
        super().__init__(
            system_prompt=FINANCIAL_SYSTEM_PROMPT,
            output_schema=FinancialAgentOutput,
        )
