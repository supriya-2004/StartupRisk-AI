// frontend/src/App.tsx

import { useState } from 'react';
import type { RiskReport, StartupQuery } from './types';
import { analyzeStartup } from './services/api';
import InputForm from './components/InputForm';
import RiskDashboard from './components/RiskDashboard';
import PipelineProgress from './components/PipelineProgress';

export default function App() {
  const [report, setReport]               = useState<RiskReport | null>(null);
  const [isLoading, setIsLoading]         = useState(false);
  const [error, setError]                 = useState<string | null>(null);
  const [currentThreadId, setThreadId]    = useState<string | null>(null);

  const handleSubmit = async (query: StartupQuery) => {
    setIsLoading(true);
    setError(null);
    setReport(null);

    // Generate a client-side thread id for pipeline polling
    const tid = crypto.randomUUID();
    setThreadId(tid);

    try {
      const result = await analyzeStartup(query);
      setReport(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-shell">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="header-inner">
          <div className="header-brand">
            <span className="brand-icon">⬡</span>
            <span className="brand-name">StartupRisk<span className="brand-v">AI</span></span>
          </div>
          <nav className="header-nav">
            <span className="nav-tag">Multi-Agent · RAG · FinBERT · LangGraph</span>
          </nav>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">
        <div className="main-inner">

          {/* Intro line shown before first result */}
          {!report && !isLoading && (
            <div className="intro-block">
              <h1 className="intro-heading">Startup Risk Intelligence</h1>
              <p className="intro-sub">
                Powered by multiple LLMs, FinBERT sentiment analysis, RAG and LangGraph orchastration.
                Submit a startup description for a full multi-agent risk assessment.
              </p>
            </div>
          )}

          {/* Pipeline progress — shown while loading */}
          {isLoading && (
            <PipelineProgress
              threadId={currentThreadId}
              isActive={isLoading}
            />
          )}

          {/* Input form — always shown */}
          <div className="form-wrapper">
            <InputForm onSubmit={handleSubmit} isLoading={isLoading} />
          </div>

          {/* Error banner */}
          {error && (
            <div className="error-banner" role="alert">
              <span className="error-icon">✕</span>
              <span>{error}</span>
              <button className="error-dismiss" onClick={() => setError(null)}>Dismiss</button>
            </div>
          )}

          {/* Results */}
          {report && !isLoading && (
            <div className="results-wrapper">
              <RiskDashboard report={report} />
            </div>
          )}
        </div>
      </main>

    </div>
  );
}
