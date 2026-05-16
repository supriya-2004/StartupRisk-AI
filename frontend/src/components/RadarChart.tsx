// frontend/src/components/RadarChart.tsx

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';

interface RadarChartProps {
  marketScore: number;
  financialScore: number;
  viabilityScore: number;
}

export default function AgentRadarChart({ marketScore, financialScore, viabilityScore }: RadarChartProps) {
  const data = [
    { axis: 'Market Risk',    score: parseFloat((marketScore * 100).toFixed(1)) },
    { axis: 'Financial Risk', score: parseFloat((financialScore * 100).toFixed(1)) },
    { axis: 'Viability Risk', score: parseFloat((viabilityScore * 100).toFixed(1)) },
  ];

  return (
    <div className="radar-card">
      <div className="radar-title">Agent Risk Breakdown</div>
      <ResponsiveContainer width="100%" height={260}>
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#334155" />
          <PolarAngleAxis
            dataKey="axis"
            tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: "'IBM Plex Mono', monospace" }}
          />
          <PolarRadiusAxis
            angle={90}
            domain={[0, 100]}
            tick={{ fill: '#64748b', fontSize: 10 }}
            tickCount={4}
          />
          <Radar
            name="Risk Score"
            dataKey="score"
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.25}
            strokeWidth={2}
          />
          <Tooltip
            contentStyle={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: '6px', color: '#e2e8f0' }}
            formatter={(val: number) => [`${val}%`, 'Risk Score']}
          />
          <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
