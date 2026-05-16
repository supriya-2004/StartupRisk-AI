// frontend/src/types/index.ts

export interface StartupQuery {
  startup_description: string;
  sector?: string;
  funding_stage?: string;
}

export interface SentimentResult {
  dominant_sentiment: 'positive' | 'negative' | 'neutral';
  positive_score: number;
  negative_score: number;
  neutral_score: number;
  financial_tone_risk: number;
}

export interface MarketAgentOutput {
  market_risk_score: number;
  market_size_assessment: string;
  competition_level: string;
  timing_assessment: string;
  market_tailwinds: string[];
  market_headwinds: string[];
  key_competitors_mentioned: string[];
  confidence: number;
  reasoning: string;
}

export interface FinancialAgentOutput {
  financial_risk_score: number;
  revenue_model_type: string;
  revenue_model_viability: string;
  burn_rate_risk: string;
  funding_stage_risk: string;
  path_to_profitability: string;
  key_financial_risks: string[];
  financial_strengths: string[];
  confidence: number;
  reasoning: string;
}

export interface ViabilityAgentOutput {
  viability_risk_score: number;
  pmf_signal: string;
  moat_type: string;
  moat_strength: string;
  traction_level: string;
  team_signal: string;
  execution_complexity: string;
  key_execution_risks: string[];
  positive_signals: string[];
  confidence: number;
  reasoning: string;
}

export interface RiskReport {
  // scores
  overall_risk_score: number;
  risk_level: 'Low' | 'Moderate' | 'Moderate-High' | 'High';
  recommendation: 'Invest' | 'Consider with Caution' | 'High Caution' | 'Avoid';
  // synthesis
  short_term_prediction: string;
  long_term_prediction: string;
  investment_thesis: string;
  top_risk_factors: string[];
  positive_signals: string[];
  // agent outputs — backend field names
  market_analysis: MarketAgentOutput | null;
  financial_analysis: FinancialAgentOutput | null;
  viability_analysis: ViabilityAgentOutput | null;
  // sentiment — backend field name
  sentiment: SentimentResult | null;
  // context
  sector: string | null;
  funding_stage: string | null;
  startup_description_excerpt: string | null;
  // RAG attribution
  rag_market_chunks: string[] | null;
  rag_financial_chunks: string[] | null;
  rag_viability_chunks: string[] | null;
  // metadata
  created_at: string;
  node_timings: Record<string, number> | null;
  thread_id: string | null;
}