// frontend/src/components/RiskDashboard.tsx

import type { RiskReport } from '../types';
import ScoreCard from './ScoreCard';
import AgentRadarChart from './RadarChart';
import SentimentBadge from './SentimentBadge';
import PredictionPanel from './PredictionPanel';
import AgentBreakdown from './AgentBreakdown';

interface RiskDashboardProps {
  report: RiskReport;
}

export default function RiskDashboard({ report }: RiskDashboardProps) {
  const {
    overall_risk_score,
    risk_level,
    recommendation,
    market_analysis,
    financial_analysis,
    viability_analysis,
    sentiment,
    short_term_prediction,
    long_term_prediction,
    top_risk_factors,
    positive_signals,
    investment_thesis,
    rag_market_chunks,
    rag_financial_chunks,
    rag_viability_chunks,
  } = report;

  return (
    <div className="risk-dashboard">
      {/* Header row */}
      <div className="dashboard-topbar">
        <div className="dashboard-meta">
          <span className="meta-sector">{report.sector ?? 'Startup'}</span>
          <span className="meta-stage">{report.funding_stage ?? 'Unknown Stage'}</span>
          <span className="meta-ts">{new Date(report.created_at).toLocaleString()}</span>
        </div>
        <button className="btn-pdf" onClick={() => window.print()}>
          ⬇ Download PDF
        </button>
      </div>

      {/* Row 1: ScoreCard + RadarChart */}
      <div className="dashboard-row dashboard-row--split">
        <div className="dashboard-col dashboard-col--third">
          <ScoreCard
            score={overall_risk_score}
            riskLevel={risk_level}
            recommendation={recommendation}
          />
        </div>
        <div className="dashboard-col dashboard-col--twothird">
          {market_analysis && financial_analysis && viability_analysis && (
            <AgentRadarChart
              marketScore={market_analysis.market_risk_score}
              financialScore={financial_analysis.financial_risk_score}
              viabilityScore={viability_analysis.viability_risk_score}
            />
          )}
        </div>
      </div>

      {/* Row 2: Sentiment */}
      {sentiment && (
        <div className="dashboard-row">
          <SentimentBadge sentiment={sentiment} />
        </div>
      )}

      {/* Row 3: Predictions */}
      <div className="dashboard-row">
        <PredictionPanel
          shortTermPrediction={short_term_prediction}
          longTermPrediction={long_term_prediction}
        />
      </div>

      {/* Investment thesis */}
      {investment_thesis && (
        <div className="dashboard-row">
          <div className="thesis-card">
            <span className="thesis-label">Investment Thesis</span>
            <p className="thesis-text">{investment_thesis}</p>
          </div>
        </div>
      )}

      {/* Row 4: Agent Breakdown */}
      {market_analysis && financial_analysis && viability_analysis && (
        <div className="dashboard-row">
          <AgentBreakdown
            marketOutput={market_analysis}
            financialOutput={financial_analysis}
            viabilityOutput={viability_analysis}
            ragMarketChunks={rag_market_chunks ?? []}
            ragFinancialChunks={rag_financial_chunks ?? []}
            ragViabilityChunks={rag_viability_chunks ?? []}
          />
        </div>
      )}

      {/* Row 5: Risk factors + Positive signals */}
      <div className="dashboard-row dashboard-row--split">
        <div className="dashboard-col">
          <div className="signals-card signals-card--risk">
            <h4 className="signals-title">⚠ Top Risk Factors</h4>
            <ol className="signals-list">
              {top_risk_factors.map((f, i) => (
                <li key={i} className="signal-item signal-item--risk">
                  <span className="signal-num">{i + 1}</span>
                  {f}
                </li>
              ))}
            </ol>
          </div>
        </div>
        <div className="dashboard-col">
          <div className="signals-card signals-card--pos">
            <h4 className="signals-title">✓ Positive Signals</h4>
            <ul className="signals-list">
              {positive_signals.map((s, i) => (
                <li key={i} className="signal-item signal-item--pos">
                  <span className="signal-check">✓</span>
                  {s}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}