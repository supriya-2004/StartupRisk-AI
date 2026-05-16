// frontend/src/components/InputForm.tsx

import { useState } from 'react';
import type { StartupQuery } from '../types';

const SECTORS = ['FinTech', 'HealthTech', 'EdTech', 'SaaS', 'E-commerce', 'Other'];
const STAGES  = ['Bootstrapped', 'Pre-Seed', 'Seed', 'Series A', 'Other'];
const MIN_LEN = 80;

interface InputFormProps {
  onSubmit: (query: StartupQuery) => void;
  isLoading: boolean;
}

export default function InputForm({ onSubmit, isLoading }: InputFormProps) {
  const [desc, setDesc]     = useState('');
  const [sector, setSector] = useState('');
  const [stage, setStage]   = useState('');
  const [touched, setTouched] = useState(false);

  const tooShort = desc.length < MIN_LEN;
  const showError = touched && tooShort;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (tooShort) return;
    onSubmit({ startup_description: desc, sector: sector || undefined, funding_stage: stage || undefined });
  };

  return (
    <form className="input-form" onSubmit={handleSubmit} noValidate>
      <div className="form-field">
        <label className="form-label" htmlFor="desc">
          Startup Description
          <span className={`char-count ${tooShort ? 'char-count--warn' : 'char-count--ok'}`}>
            {desc.length} / {MIN_LEN}+ chars
          </span>
        </label>
        <textarea
          id="desc"
          className={`form-textarea ${showError ? 'form-textarea--error' : ''}`}
          rows={6}
          placeholder="Describe the startup — its product, target market, revenue model, team, and traction…"
          value={desc}
          onChange={(e) => setDesc(e.target.value)}
          onBlur={() => setTouched(true)}
        />
        {showError && (
          <p className="form-error">Description must be at least {MIN_LEN} characters.</p>
        )}
      </div>

      <div className="form-row">
        <div className="form-field">
          <label className="form-label" htmlFor="sector">Sector</label>
          <select id="sector" className="form-select" value={sector} onChange={(e) => setSector(e.target.value)}>
            <option value="">— Select —</option>
            {SECTORS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="form-field">
          <label className="form-label" htmlFor="stage">Funding Stage</label>
          <select id="stage" className="form-select" value={stage} onChange={(e) => setStage(e.target.value)}>
            <option value="">— Select —</option>
            {STAGES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      <button className="form-submit" type="submit" disabled={isLoading}>
        {isLoading ? (
          <><span className="btn-spinner" /> Analyzing…</>
        ) : (
          <>⚡ Analyze Startup Risk</>
        )}
      </button>
    </form>
  );
}
