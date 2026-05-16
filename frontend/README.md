# StartupRisk AI v2 — React Frontend

**Final Year Engineering Project · 2025–26 · Academic Use Only**

## Stack

- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS (utility styling)
- Recharts (radar chart)
- IBM Plex Mono + Syne (typography)

## Project Structure

```
src/
├── types/
│   └── index.ts              ← All shared TypeScript interfaces
├── services/
│   └── api.ts                ← API service layer (analyzeStartup, getPipelineStatus, getHistory)
├── components/
│   ├── InputForm.tsx         ← Startup description form with validation
│   ├── ScoreCard.tsx         ← Animated risk score dial
│   ├── RadarChart.tsx        ← Three-axis agent risk radar (Recharts)
│   ├── PredictionPanel.tsx   ← 6-month + 3-year outlook cards
│   ├── SentimentBadge.tsx    ← FinBERT sentiment bar chart + modifier display
│   ├── PipelineProgress.tsx  ← Live polling pipeline stage tracker
│   ├── AgentBreakdown.tsx    ← Accordion per-agent outputs + RAG attribution
│   └── RiskDashboard.tsx     ← Full dashboard assembly
├── App.tsx                   ← Root app with state management
├── main.tsx                  ← React entry point
└── index.css                 ← Global styles (CSS variables, all component styles)
```

## Setup & Run

```bash
# 1. Install dependencies
npm install

# 2. Copy env file and set your backend URL
cp .env.example .env.local
# Edit VITE_API_URL if your backend runs on a different port

# 3. Start development server
npm run dev
# → http://localhost:5173

# 4. Build for production
npm run build
```

## Backend Connection

The frontend expects the FastAPI backend at `VITE_API_URL` (default `http://localhost:8000`).

Endpoints used:
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v2/analyze` | Submit startup for analysis |
| GET | `/api/v2/status/{threadId}` | Poll node timings for pipeline progress |
| GET | `/api/v2/history` | Fetch last 20 reports |

## Component Notes

### PipelineProgress
Polls `/api/v2/status/{threadId}` every 2 seconds while `isActive=true`. Infers stage status from which `node_timings` keys are present in the response. Stops polling when `isActive` becomes false.

### SentimentBadge
Derives the FinBERT sentiment modifier from `financial_tone_risk` (= `negative_score`) using exact thresholds from the project spec:
- `< 0.15` → ×1.00
- `< 0.35` → ×1.03
- `< 0.60` → ×1.07
- `≥ 0.60` → ×1.12

### AgentBreakdown
RAG source attribution shows up to 4 knowledge chunks per agent, truncated to 120 chars with click-to-expand. Source filenames are extracted from chunk content when available.
