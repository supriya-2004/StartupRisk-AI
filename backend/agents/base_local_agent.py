# agents/base_local_agent.py

import json
import os
import re
from typing import Type, TypeVar

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# OLLAMA_MODEL = "llama3.2:3b-instruct-q4_K_M"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Base URL can be overridden via environment variable (e.g. in Docker / CI)
_OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

T = TypeVar("T", bound=BaseModel)

class AsyncLLMWrapper:
    def __init__(self, llm):
        self._llm = llm

    async def ainvoke(self, messages):
        import asyncio
        return await asyncio.to_thread(self._llm.invoke, messages)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
  
    
async def check_ollama_health() -> bool:
    """Return True if Ollama is reachable and the target model is available.

    Sends a GET request to <OLLAMA_BASE_URL>/api/tags and checks that the
    model name defined by ``OLLAMA_MODEL`` appears in the response payload.
    Returns False on *any* exception so callers can gate startup gracefully.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            return False
        payload = response.json()
        models = payload.get("models", [])
        # Each entry has a "name" key, e.g. "llama3.2:3b-instruct-q4_K_M"
        return any(OLLAMA_MODEL in m.get("name", "") for m in models)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------


class BaseLocalAgent:
    """Shared foundation for all local Ollama/Llama analysis agents.

    Subclasses only need to supply a system prompt and an output Pydantic
    schema; all LLM communication, JSON extraction, and error recovery live
    here.
    """

    def __init__(self, system_prompt: str, output_schema: Type[T]) -> None:
        base_llm = ChatOllama(
            model=OLLAMA_MODEL,
            temperature=0.1,
            format="json",      # Forces JSON output mode in Ollama
            num_predict=800,    # Max tokens — enough for structured output
        )

        self.llm = AsyncLLMWrapper(base_llm)
        
        self.system_prompt = system_prompt
        self.output_schema = output_schema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, startup_desc: str, rag_chunks: list[str]) -> T:
        """Run the agent and return a validated Pydantic output object.

        Parameters
        ----------
        startup_desc:
            Raw startup description text provided by the user.
        rag_chunks:
            List of text chunks retrieved from ChromaDB (plus live headlines)
            for this agent's focus area.
        """
        rag_context = "\n---\n".join(rag_chunks)

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=(
                    f"KNOWLEDGE BASE CONTEXT:\n{rag_context}\n\n"
                    "---\n\n"
                    f"STARTUP DESCRIPTION TO ANALYZE:\n{startup_desc}\n\n"
                    "Return ONLY a valid JSON object. "
                    "No explanation, no markdown fences."
                )
            ),
        ]

        response = await self.llm.ainvoke(messages)
        return self._parse_output(response.content)
    
    # ------------------------------------------------------------------
    # Sanitize the agent outputs
    # ------------------------------------------------------------------
    
    def _sanitize_data(self, data: dict) -> dict:
        """
        Normalize fields that the local LLM sometimes returns in non-standard formats.
        Handles pipe-separated values like 'Network Effects|IP' by taking the first value.
        """
        LITERAL_FIELDS = {
            "revenue_model_type": ["SaaS","Marketplace","E-commerce","Service","Freemium","Other",],
            "moat_type": ["Network Effects", "Switching Costs", "IP", "Brand", "None"],
            "moat_strength": ["Strong", "Moderate", "Weak", "None"],
            "pmf_signal": ["Strong", "Moderate", "Weak", "None"],
            "traction_level": ["Revenue", "Pilots", "Users", "Pre-Launch"],
            "team_signal": ["Strong", "Moderate", "Weak", "Not Mentioned"],
            "execution_complexity": ["Very High", "High", "Medium", "Low"],
            "market_size_assessment": ["Large", "Medium", "Small", "Niche"],
            "competition_level": ["Very High", "High", "Medium", "Low"],
            "timing_assessment": ["Excellent", "Good", "Fair", "Poor"],
            "revenue_model_viability": ["Strong", "Moderate", "Weak", "Unproven"],
            "burn_rate_risk": ["Critical", "High", "Medium", "Low"],
            "funding_stage_risk": ["High", "Medium", "Low"],
            "path_to_profitability": ["Clear", "Possible", "Unclear", "Not Visible"],
        }
        sanitized = dict(data)

        MIN_LIST_FIELDS = {
            "market_headwinds": "Market competition or adoption challenges",
            "market_tailwinds": "Favorable industry growth trends",
            "key_risks": "Execution or scaling risks",
        }

        for field, default_value in MIN_LIST_FIELDS.items():
            if field in sanitized:
                if not isinstance(sanitized[field], list) or len(sanitized[field]) == 0:
                    sanitized[field] = [default_value]
        
        for field, allowed in LITERAL_FIELDS.items():
            if field in sanitized and isinstance(sanitized[field], str):
                val = sanitized[field]

                # Already valid → keep
                if val in allowed:
                    continue

                # Special handling for revenue_model_type
                if field == "revenue_model_type":
                    val_lower = val.lower()

                    if "saas" in val_lower or "subscription" in val_lower:
                        sanitized[field] = "SaaS"
                        continue
                    elif "marketplace" in val_lower:
                        sanitized[field] = "Marketplace"
                        continue
                    elif "ecommerce" in val_lower or "e-commerce" in val_lower:
                        sanitized[field] = "E-commerce"
                        continue
                    elif "service" in val_lower:
                        sanitized[field] = "Service"
                        continue
                    elif "freemium" in val_lower:
                        sanitized[field] = "Freemium"
                        continue
                    else:
                        sanitized[field] = "Other"
                        continue

                # Generic fallback (your existing logic)
                for token in re.split(r"[|/,]", val):
                    token = token.strip()
                    if token in allowed:
                        sanitized[field] = token
                        break
                else:
                    sanitized[field] = allowed[-1]
        return sanitized

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_output(self, raw: str) -> T:
        """Parse Ollama's raw text into the output schema.

        Attempt 1 — strip markdown fences and parse directly.
        Attempt 2 — slice from the first ``{`` to the last ``}`` to handle
                    any prefix/suffix bleed from the model.
        Raises ``ValueError`` if both attempts fail.
        """
        # Attempt 1: strip ```json ... ``` fences
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(clean)
            return self.output_schema(**self._sanitize_data(data))
        except json.JSONDecodeError:
            pass  # Fall through to attempt 2

        # Attempt 2: extract the outermost JSON object
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            trimmed = clean[start : end + 1]
            try:
                data = json.loads(trimmed)
                return self.output_schema(**self._sanitize_data(data))
            except json.JSONDecodeError:
                pass  # Fall through to error

        raise ValueError(
            f"[BaseLocalAgent] Failed to parse JSON from model output after "
            f"two attempts.\n\nRaw output was:\n{raw}"
        )

