// frontend/src/components/AgentBreakdown.tsx

import { useState } from 'react';
import type { MarketAgentOutput, FinancialAgentOutput, ViabilityAgentOutput } from '../types';

interface AgentBreakdownProps {
  marketOutput: MarketAgentOutput;
  financialOutput: FinancialAgentOutput;
  viabilityOutput: ViabilityAgentOutput;
  ragMarketChunks?: string[];
  ragFinancialChunks?: string[];
  ragViabilityChunks?: string[];
}

interface ChunkCardProps {
  chunk: string;
  index: number;
}

function ChunkCard({ chunk, index }: ChunkCardProps) {
  const [expanded, setExpanded] = useState(false);
  const preview = chunk.slice(0, 120);
  const hasMore = chunk.length > 120;

  // Try to extract a source filename hint from the chunk
  const fileMatch = chunk.match(/\b[\w_]+\.(txt|md)\b/);
  const sourceLabel = fileMatch ? fileMatch[0] : `Source ${index + 1}`;

  return (
    <div
      className="chunk-card"
      onClick={() => hasMore && setExpanded((p) => !p)}
      style={{ cursor: hasMore ? 'pointer' : 'default' }}
    >
      <div className="chunk-source-label">{sourceLabel}</div>
      <p className="chunk-text">
        {expanded ? chunk : preview}
        {!expanded && hasMore && <span className="chunk-more">… <span className="expand-hint">click to expand</span></span>}
      </p>
      {expanded && (
        <button className="chunk-collapse" onClick={(e) => { e.stopPropagation(); setExpanded(false); }}>
          ↑ collapse
        </button>
      )}
    </div>
  );
}

interface RiskBadgeProps { score: number }
function RiskBadge({ score }: RiskBadgeProps) {
  const pct = (score * 100).toFixed(0);
  const color =
    score <= 0.35 ? '#22c55e' :
    score <= 0.55 ? '#eab308' :
    score <= 0.74 ? '#f97316' : '#ef4444';
  return (
    <span className="risk-badge" style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}>
      {pct}% risk
    </span>
  );
}

interface AgentSectionProps {
  title: string;
  icon: string;
  score: number;
  reasoning: string;
  fields: Record<string, unknown>;
  chunks?: string[];
}

function AgentSection({ title, icon, score, reasoning, fields, chunks }: AgentSectionProps) {
  const [open, setOpen] = useState(false);

  const renderValue = (val: unknown): string => {
    if (Array.isArray(val)) return val.join(', ');
    if (typeof val === 'number') return val.toFixed ? val.toFixed(3) : String(val);
    return String(val ?? '—');
  };

  const formatKey = (k: string) =>
    k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  const omitKeys = new Set(['reasoning', 'market_risk_score', 'financial_risk_score', 'viability_risk_score']);
  const displayFields = Object.entries(fields).filter(([k]) => !omitKeys.has(k));

  return (
    <div className={`agent-section ${open ? 'agent-section--open' : ''}`}>
      <button className="agent-header" onClick={() => setOpen((p) => !p)}>
        <span className="agent-icon">{icon}</span>
        <span className="agent-title">{title}</span>
        <RiskBadge score={score} />
        <span className="agent-reasoning-preview">{reasoning.slice(0, 80)}{reasoning.length > 80 ? '…' : ''}</span>
        <span className="agent-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="agent-body">
          <div className="agent-kv-grid">
            {displayFields.map(([k, v]) => (
              <div key={k} className="kv-row">
                <dt className="kv-key">{formatKey(k)}</dt>
                <dd className="kv-val">{renderValue(v)}</dd>
              </div>
            ))}
          </div>

          <div className="agent-reasoning-full">
            <span className="section-label">Agent Reasoning</span>
            <p>{reasoning}</p>
          </div>

          {chunks && chunks.length > 0 && (
            <div className="rag-attribution">
              <span className="section-label">⟁ Knowledge Sources Used</span>
              <div className="chunk-grid">
                {chunks.slice(0, 4).map((c, i) => (
                  <ChunkCard key={i} chunk={c} index={i} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function AgentBreakdown({
  marketOutput,
  financialOutput,
  viabilityOutput,
  ragMarketChunks,
  ragFinancialChunks,
  ragViabilityChunks,
}: AgentBreakdownProps) {
  return (
    <div className="agent-breakdown">
      <h3 className="breakdown-title">Agent Analysis Breakdown</h3>
      <AgentSection
        title="Market Analysis Agent"
        icon="📊"
        score={marketOutput.market_risk_score}
        reasoning={marketOutput.reasoning}
        fields={marketOutput}
        chunks={ragMarketChunks}
      />
      <AgentSection
        title="Financial Risk Agent"
        icon="💰"
        score={financialOutput.financial_risk_score}
        reasoning={financialOutput.reasoning}
        fields={financialOutput}
        chunks={ragFinancialChunks}
      />
      <AgentSection
        title="Viability Agent"
        icon="🚀"
        score={viabilityOutput.viability_risk_score}
        reasoning={viabilityOutput.reasoning}
        fields={viabilityOutput}
        chunks={ragViabilityChunks}
      />
    </div>
  );
}
