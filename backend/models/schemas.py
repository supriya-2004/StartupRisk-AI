from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


#  Input Schema 

class StartupQuery(BaseModel):
    startup_description: str = Field(
        ...,
        description="Full description of the startup or business idea"
    )
    sector: Optional[str] = Field(
        default=None,
        description="e.g. FinTech, HealthTech, SaaS, EdTech, E-commerce"
    )
    funding_stage: Optional[str] = Field(
        default=None,
        description="e.g. Bootstrapped, Pre-Seed, Seed, Series A"
    )


#  Agent Output Schemas 

class MarketAgentOutput(BaseModel):
    market_risk_score: float = Field(..., ge=0.0, le=1.0)
    market_size_assessment: Literal["Large", "Medium", "Small", "Niche"]
    competition_level: Literal["Very High", "High", "Medium", "Low"]
    timing_assessment: Literal["Excellent", "Good", "Fair", "Poor"]
    market_tailwinds: List[str] = Field(..., min_length=1)
    market_headwinds: List[str] = Field(..., min_length=1)
    key_competitors_mentioned: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=800)


class FinancialAgentOutput(BaseModel):
    financial_risk_score: float = Field(..., ge=0.0, le=1.0)
    revenue_model_type: Literal[
        "SaaS", "Marketplace", "E-commerce", "Service", "Freemium", "Other"
    ]
    revenue_model_viability: Literal["Strong", "Moderate", "Weak", "Unproven"]
    burn_rate_risk: Literal["Critical", "High", "Medium", "Low"]
    funding_stage_risk: Literal["High", "Medium", "Low"]
    path_to_profitability: Literal["Clear", "Possible", "Unclear", "Not Visible"]
    key_financial_risks: List[str] = Field(..., min_length=1)
    financial_strengths: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=800)


class ViabilityAgentOutput(BaseModel):
    viability_risk_score: float = Field(..., ge=0.0, le=1.0)
    pmf_signal: Literal["Strong", "Moderate", "Weak", "None"]
    moat_type: Literal[
        "Network Effects", "Switching Costs", "IP", "Brand", "None"
    ]
    moat_strength: Literal["Strong", "Moderate", "Weak", "None"]
    traction_level: Literal["Revenue", "Pilots", "Users", "Pre-Launch"]
    team_signal: Literal["Strong", "Moderate", "Weak", "Not Mentioned"]
    execution_complexity: Literal["Very High", "High", "Medium", "Low"]
    key_execution_risks: List[str] = Field(..., min_length=1)
    positive_signals: List[str] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., max_length=800)


#  Sentiment Schema 

class SentimentResult(BaseModel):
    dominant_sentiment: Literal["positive", "negative", "neutral"]
    positive_score: float = Field(..., ge=0.0, le=1.0)
    negative_score: float = Field(..., ge=0.0, le=1.0)
    neutral_score: float = Field(..., ge=0.0, le=1.0)
    financial_tone_risk: float = Field(
        ..., ge=0.0, le=1.0,
        description="Equals negative_score — used in risk score formula"
    )


#  Synthesis & Final Report Schemas 

class SynthesisOutput(BaseModel):
    short_term_prediction: str = Field(
        ..., description="6-month outlook, 80-100 words"
    )
    long_term_prediction: str = Field(
        ..., description="3-year investment outlook, 80-100 words"
    )
    investment_thesis: str = Field(
        ..., max_length=500,
        description="2-3 sentence recommendation, max 60 words"
    )
    top_risk_factors: List[str] = Field(..., min_length=5, max_length=5)
    positive_signals: List[str] = Field(..., min_length=3, max_length=3)
    recommendation: Literal[
        "Invest", "Consider with Caution", "High Caution", "Avoid"
    ]


class RiskReport(BaseModel):
    #  Core scores 
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: Literal["Low", "Moderate", "Moderate-High", "High"]
    recommendation: Literal[
        "Invest", "Consider with Caution", "High Caution", "Avoid"
    ]

    #  Synthesis narrative 
    short_term_prediction: str
    long_term_prediction: str
    investment_thesis: str
    top_risk_factors: List[str] = Field(..., min_length=5, max_length=5)
    positive_signals: List[str] = Field(..., min_length=3, max_length=3)

    #  Agent outputs (nested) 
    market_analysis: Optional[MarketAgentOutput] = None
    financial_analysis: Optional[FinancialAgentOutput] = None
    viability_analysis: Optional[ViabilityAgentOutput] = None

    #  Sentiment 
    sentiment: Optional[SentimentResult] = None

    #  Query context 
    sector: Optional[str] = None
    funding_stage: Optional[str] = None
    startup_description_excerpt: Optional[str] = None

    #  RAG chunk attribution 
    rag_market_chunks: Optional[List[str]] = None
    rag_financial_chunks: Optional[List[str]] = None
    rag_viability_chunks: Optional[List[str]] = None

    #  Metadata 
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    node_timings: Optional[dict] = None
    thread_id: Optional[str] = None