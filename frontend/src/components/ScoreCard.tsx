// frontend/src/components/ScoreCard.tsx

import { useEffect, useState } from 'react';

interface ScoreCardProps {
  score: number;
  riskLevel: string;
  recommendation: string;
}

const LEVEL_COLORS: Record<string, string> = {
  Low:            '#22c55e',
  Moderate:       '#eab308',
  'Moderate-High':'#f97316',
  High:           '#ef4444',
};

const REC_COLORS: Record<string, string> = {
  Invest:                 '#22c55e',
  'Consider with Caution':'#eab308',
  'High Caution':         '#f97316',
  Avoid:                  '#ef4444',
};

export default function ScoreCard({ score, riskLevel, recommendation }: ScoreCardProps) {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    let frame: number;
    let start: number | null = null;
    const target = score * 100;
    const duration = 900;
    const animate = (ts: number) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      setDisplayed(progress * target);
      if (progress < 1) frame = requestAnimationFrame(animate);
    };
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [score]);

  const color = LEVEL_COLORS[riskLevel] ?? '#9ca3af';
  const recColor = REC_COLORS[recommendation] ?? '#9ca3af';

  return (
    <div className="score-card">
      <div className="score-card__label">Overall Risk Score</div>
      <div className="score-card__dial" style={{ '--ring-color': color } as React.CSSProperties}>
        <svg viewBox="0 0 120 120" className="score-dial-svg">
          <circle cx="60" cy="60" r="50" fill="none" stroke="#1e293b" strokeWidth="10" />
          <circle
            cx="60" cy="60" r="50"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={`${2 * Math.PI * 50}`}
            strokeDashoffset={`${2 * Math.PI * 50 * (1 - score)}`}
            strokeLinecap="round"
            transform="rotate(-90 60 60)"
            style={{ transition: 'stroke-dashoffset 0.9s ease-out' }}
          />
        </svg>
        <div className="score-dial-inner">
          <span className="score-pct" style={{ color }}>{displayed.toFixed(1)}<span className="score-unit">%</span></span>
        </div>
      </div>
      <div className="score-card__level" style={{ background: `${color}22`, color, border: `1px solid ${color}55` }}>
        {riskLevel}
      </div>
      <div className="score-card__rec-label">Recommendation</div>
      <div className="score-card__rec" style={{ color: recColor }}>
        {recommendation}
      </div>
    </div>
  );
}
