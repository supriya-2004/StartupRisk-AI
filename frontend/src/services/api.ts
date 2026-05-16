// frontend/src/services/api.ts

import type { StartupQuery, RiskReport } from '../types';

const BASE_URL = (import.meta as any).env?.VITE_API_URL ?? 'http://localhost:8000';

export type { StartupQuery, RiskReport };

export async function analyzeStartup(query: StartupQuery): Promise<RiskReport> {
  const res = await fetch(`${BASE_URL}/api/v2/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(query),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? 'Analysis failed');
  }
  return res.json();
}

export async function getPipelineStatus(threadId: string): Promise<Record<string, number>> {
  const res = await fetch(`${BASE_URL}/api/v2/status/${threadId}`);
  if (!res.ok) return {};
  return res.json();
}

export async function getHistory(): Promise<RiskReport[]> {
  const res = await fetch(`${BASE_URL}/api/v2/history`);
  if (!res.ok) return [];
  return res.json();
}
