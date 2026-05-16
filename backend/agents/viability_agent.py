
"""
StartupRisk AI v2 — Startup Viability Agent
Powered by Llama 3.2 3B via Ollama (local inference, zero API cost).
Analyses execution risk, PMF signals, moat strength, team quality, and traction.
"""
from langchain_community.chat_models import ChatOllama
from agents.base_local_agent import BaseLocalAgent
from models.schemas import ViabilityAgentOutput

# ---------------------------------------------------------------------------
# System prompt — reproduced verbatim from Section 4.5 of the technical spec
# ---------------------------------------------------------------------------
_VIABILITY_SYSTEM_PROMPT = """You are a startup execution and viability specialist. You receive a KNOWLEDGE BASE CONTEXT with startup success/failure patterns, and a startup description. Use the patterns to calibrate your assessments.

Analyze the EXECUTION and VIABILITY risk only. Assess:
- Product-market fit signals present in the description
- Business model defensibility and competitive moat
- Team and founder signals (domain expertise, relevant background)
- Traction indicators (users, customers, revenue, pilots mentioned)
- Operational complexity and execution risk

Return ONLY this JSON structure (no other text):
{
  "viability_risk_score": <float 0.0-1.0>,
  "pmf_signal": "Strong|Moderate|Weak|None",
  "moat_type": "Network Effects|Switching Costs|IP|Brand|None",
  "moat_strength": "Strong|Moderate|Weak|None",
  "traction_level": "Revenue|Pilots|Users|Pre-Launch",
  "team_signal": "Strong|Moderate|Weak|Not Mentioned",
  "execution_complexity": "Very High|High|Medium|Low",
  "key_execution_risks": ["<string>", "<string>", "<string>"],
  "positive_signals": ["<string>", "<string>"],
  "confidence": <float 0.0-1.0>,
  "reasoning": "<string max 120 words>"
}"""


class ViabilityAgent(BaseLocalAgent):
    """
    Agent 3 — Startup Viability Agent.

    Assesses execution risk, product-market fit, moat defensibility,
    traction indicators, and team quality using Llama 3.2 3B (local)
    augmented with RAG context from the startup_patterns and
    investment_frameworks knowledge collections.
    """

    def __init__(self) -> None:
        super().__init__(
            system_prompt=_VIABILITY_SYSTEM_PROMPT,
            output_schema=ViabilityAgentOutput,
        )
