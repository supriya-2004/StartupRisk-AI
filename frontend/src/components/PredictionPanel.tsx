// frontend/src/components/PredictionPanel.tsx

interface PredictionPanelProps {
  shortTermPrediction: string;
  longTermPrediction: string;
}

export default function PredictionPanel({ shortTermPrediction, longTermPrediction }: PredictionPanelProps) {
  return (
    <div className="prediction-panel">
      <div className="prediction-card prediction-card--short">
        <div className="prediction-horizon">
          <span className="horizon-icon">📅</span>
          <span className="horizon-label">6-Month Outlook</span>
        </div>
        <p className="prediction-text">{shortTermPrediction}</p>
      </div>
      <div className="prediction-card prediction-card--long">
        <div className="prediction-horizon">
          <span className="horizon-icon">🔭</span>
          <span className="horizon-label">3-Year Outlook</span>
        </div>
        <p className="prediction-text">{longTermPrediction}</p>
      </div>
    </div>
  );
}
