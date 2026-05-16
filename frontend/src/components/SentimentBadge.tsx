// frontend/src/components/SentimentBadge.tsx

import type { SentimentResult } from '../types';

interface SentimentBadgeProps {
  sentiment: SentimentResult;
}

function getModifier(negScore: number): { label: string; value: string } {
  if (negScore < 0.15) return { label: 'Very low negative tone — minimal modifier', value: '×1.00' };
  if (negScore < 0.35) return { label: 'Low-moderate negative tone', value: '×1.03' };
  if (negScore < 0.60) return { label: 'Elevated negative tone', value: '×1.07' };
  return { label: 'High negative financial tone', value: '×1.12' };
}

const SENTIMENT_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  positive: { bg: '#16a34a22', text: '#22c55e', border: '#22c55e55' },
  neutral:  { bg: '#6b728022', text: '#9ca3af', border: '#9ca3af55' },
  negative: { bg: '#dc262622', text: '#ef4444', border: '#ef444455' },
};

const BAR_COLORS: Record<string, string> = {
  positive: '#22c55e',
  neutral: '#9ca3af',
  negative: '#ef4444',
};

export default function SentimentBadge({ sentiment }: SentimentBadgeProps) {
  const { dominant_sentiment, positive_score, negative_score, neutral_score, financial_tone_risk } = sentiment;
  const colors = SENTIMENT_COLORS[dominant_sentiment] ?? SENTIMENT_COLORS.neutral;
  const modifier = getModifier(financial_tone_risk);

  const bars = [
    { label: 'Positive', score: positive_score, color: BAR_COLORS.positive },
    { label: 'Neutral',  score: neutral_score,  color: BAR_COLORS.neutral  },
    { label: 'Negative', score: negative_score, color: BAR_COLORS.negative  },
  ];

  return (
    <div className="sentiment-badge-card">
      <div className="sentiment-top-row">
        <span className="sentiment-label">FinBERT Sentiment</span>
        <span
          className="sentiment-pill"
          style={{ background: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}
        >
          {dominant_sentiment.toUpperCase()}
        </span>
      </div>

      <div className="sentiment-bars">
        {bars.map(({ label, score, color }) => (
          <div key={label} className="sentiment-bar-row">
            <span className="bar-label">{label}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(score * 100).toFixed(1)}%`, background: color }}
              />
            </div>
            <span className="bar-value">{(score * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div className="sentiment-modifier">
        <span className="modifier-icon">⚙</span>
        <div className="modifier-text">
          <span className="modifier-desc">{modifier.label}</span>
          <span className="modifier-val">Sentiment modifier applied to risk score: <strong>{modifier.value}</strong></span>
        </div>
      </div>
    </div>
  );
}
