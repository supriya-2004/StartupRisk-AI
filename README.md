# StartupRisk AI

> **Final Year Engineering Project · 2025–26**  
> Hybrid Open-Source Multi-Agent System with RAG · Local LLMs · FinBERT · LangGraph

---

## What It Does

A user describes a startup or business idea and asks: *Should I invest in this?*  
StartupRisk AI produces a structured investment risk profile — including a risk score, recommendation, short-term and long-term predictions, and fully explainable agent reasoning — in **20–40 seconds on a standard laptop**.

**95% of the analysis pipeline runs entirely on your machine** using free, open-source models. Only the final synthesis step makes a single external API call to Claude.

| | v1 | v2 |
|---|---|---|
| External API calls | 4 | **1** |
| Cost per query | ~$0.04 | **~$0.003** |
| Local LLM | None | Llama 3.2 3B via Ollama |
| Sentiment Analysis | None | FinBERT (local) |
| Knowledge Base | None | RAG · ChromaDB · 20 documents |
| Orchestration | asyncio | LangGraph stateful graph |

---

## Architecture

```
START
  │
  ▼
Node 0: live_fetch         ← NewsAPI · live sector headlines (NEW)
  │
  ▼
Node 1: rag_retrieval      ← ChromaDB · sentence-transformers embeddings
  │
  ├──────────────── parallel ────────────────┐
  ▼                 ▼                        ▼
Node 2a: market   Node 2b: financial    Node 2c: viability
  (Llama 3.2)       (Llama 3.2)           (Llama 3.2)
  │                 │                        │
  └──────────────────────────────────────────┘
  │
  ▼
Node 3: finbert_sentiment  ← Local FinBERT scoring
  │
  ▼
Node 4: score_calculator   ← Deterministic weighted formula (no LLM)
  │
  ▼
Node 5: claude_synthesis   ← SINGLE external API call
  │
END
```

### Component Map

| Component | Technology | Type |
|---|---|---|
| Orchestrator | LangGraph 0.2 | Open Source |
| Local LLM Runtime | Ollama 0.4 | Open Source |
| Local LLM Model | Llama 3.2 3B Instruct | Open Source (Meta) |
| Sentiment Model | FinBERT (ProsusAI) | Pre-trained (HuggingFace) |
| Embeddings | all-MiniLM-L6-v2 | Pre-trained (HuggingFace) |
| Vector Database | ChromaDB 0.6 | Open Source |
| RAG Framework | LangChain 0.3 | Open Source |
| External LLM | Claude Sonnet (Anthropic) | External API — 1 call only |
| Backend API | FastAPI + Python 3.11 | Open Source |
| Database | SQLite + SQLAlchemy | Open Source |
| Frontend | React 18 + TypeScript | Open Source |

---

## Risk Scoring

The overall investment risk score is computed **deterministically** — no LLM involved:

```
Base Score  = (0.35 × market_risk) + (0.30 × financial_risk) + (0.35 × viability_risk)
Final Score = Base Score × FinBERT sentiment modifier
```

| Score | Risk Level | Recommendation |
|---|---|---|
| 0.00–0.35 | Low | Invest — Favorable |
| 0.36–0.55 | Moderate | Consider with Caution |
| 0.56–0.74 | Moderate-High | High Caution |
| 0.75–1.00 | High | Avoid |

---

## Project Structure

```
startup-risk-ai-v2/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── graph/
│   │   ├── builder.py
│   │   ├── state.py
│   │   └── nodes/
│   │       ├── live_fetch.py         ← Node 0: NewsAPI headlines
│   │       ├── rag_retrieval.py      ← Node 1: ChromaDB retrieval
│   │       ├── parallel_agents.py    ← Node 2: 3 agents in parallel
│   │       ├── finbert_sentiment.py  ← Node 3: local FinBERT
│   │       ├── score_calculator.py   ← Node 4: deterministic scoring
│   │       └── claude_synthesis.py   ← Node 5: single Claude API call
│   ├── agents/
│   │   ├── base_local_agent.py
│   │   ├── market_agent.py
│   │   ├── financial_agent.py
│   │   └── viability_agent.py
│   ├── rag/
│   │   ├── indexer.py
│   │   └── retriever.py
│   └── models/
│       ├── schemas.py
│       └── database.py
├── knowledge_base/
│   ├── market_data/          (6 files)
│   ├── startup_patterns/     (4 files)
│   ├── investment_frameworks/(4 files)
│   └── financial_models/     (4 files)
├── frontend/
│   └── src/
│       └── components/
│           ├── InputForm.tsx
│           ├── RiskDashboard.tsx
│           ├── RadarChart.tsx
│           ├── SentimentBadge.tsx    ← NEW
│           ├── PipelineProgress.tsx  ← NEW
│           └── AgentBreakdown.tsx
└── tests/
    ├── test_rag_retrieval.py
    ├── test_local_agents.py
    ├── test_finbert.py
    ├── test_score_calculator.py
    └── test_full_pipeline.py
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- [Ollama](https://ollama.ai) installed (free, no account needed)
- Anthropic API key — [console.anthropic.com](https://console.anthropic.com)
- NewsAPI key (free tier, 100 req/day) — [newsapi.org](https://newsapi.org)
- 8 GB RAM minimum (16 GB recommended)
- ~6 GB free disk space

### Step 1 — Install and Start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

ollama serve
ollama pull llama3.2:3b-instruct-q4_K_M    # ~2 GB, one-time download

# Verify
curl http://localhost:11434/api/tags
```

### Step 2 — Backend Setup

```bash
git clone https://github.com/your-team/startup-risk-ai-v2.git
cd startup-risk-ai-v2/backend

python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install PyTorch (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OLLAMA_BASE_URL=http://localhost:11434
#   CHROMA_DB_PATH=../chroma_db
#   KNOWLEDGE_BASE_PATH=../knowledge_base
#   NEWSAPI_KEY=your_key_here

# Start the backend (first run builds ChromaDB — ~30 seconds)
uvicorn main:app --reload --port 8000
```

### Step 3 — Frontend Setup

```bash
cd ../frontend
npm install
echo 'VITE_API_URL=http://localhost:8000' > .env
npm run dev
# Running at http://localhost:5173
```

---

## Performance

| Stage | Typical Time | What's Happening |
|---|---|---|
| Node 0: live_fetch | ~0.5–1 s | NewsAPI HTTP call |
| Node 1: RAG retrieval | ~0.5 s | ChromaDB similarity search |
| Node 2: Agents (parallel) | ~8–15 s | 3 Llama 3.2 calls concurrently |
| Node 3: FinBERT | ~0.5–1 s | Local BERT inference |
| Node 4: Score calc | < 0.1 s | Pure Python arithmetic |
| Node 5: Claude synthesis | ~4–6 s | 1 external API call |
| **Total** | **~16–26 s** | GPU cuts agent time to ~4–6 s |

---

## Hardware Requirements

| | Minimum (CPU Only) | Recommended (GPU) |
|---|---|---|
| CPU | Quad-core (i5 / Ryzen 5) | 8-core (i7 / Ryzen 7) |
| RAM | 8 GB | 16 GB |
| GPU | Not required | 4 GB VRAM (NVIDIA) |
| Disk | 6 GB free | 10 GB free |
| OS | Windows 10 / Ubuntu 22 / macOS 12+ | Same |

**Memory breakdown:** Llama 3.2 3B ~2.5 GB · FinBERT ~440 MB · all-MiniLM-L6-v2 ~80 MB · ChromaDB ~50–200 MB

---

## Running Tests

```bash
cd backend
pytest tests/ -v

# Individual test files
pytest tests/test_rag_retrieval.py
pytest tests/test_finbert.py
pytest tests/test_score_calculator.py
pytest tests/test_full_pipeline.py
```

---

## RAG Knowledge Base

20 curated plain-text documents form the system's domain expertise, organized into four collections:

| Collection | Files | Used By |
|---|---|---|
| `market_data/` | 6 files — FinTech, HealthTech, EdTech, SaaS, E-commerce, India ecosystem | Market Agent |
| `startup_patterns/` | 4 files — CB Insights failure reasons, YC success patterns, PMF signals, pivots | Viability Agent |
| `investment_frameworks/` | 4 files — Angel checklist, seed valuation, due diligence, moat analysis | Both agents |
| `financial_models/` | 4 files — SaaS metrics, marketplace economics, burn rate, revenue models | Financial Agent |

Content sourced from CB Insights, Y Combinator, Sequoia Capital, and NASSCOM public reports.

---

## API Cost

Claude Sonnet charges ~$3 per million input tokens. A typical synthesis call uses ~2,000 tokens input and ~600 tokens output — approximately **$0.006–0.009 per query** (under ₹1 per analysis).

---

## Team

| Member | Owns |
|---|---|
| Member 1 | LangGraph orchestration, graph builder, state design, checkpointing |
| Member 2 | Local LLM agents, Ollama setup, prompt engineering |
| Member 3 | RAG pipeline, knowledge base, ChromaDB setup |
| Member 4 | FinBERT, scoring, Claude synthesis node, API router |
| Member 5 | React frontend, Tailwind styling, E2E integration tests |

---

## Known Limitations

- **Llama 3.2 3B quality** — 3B models occasionally produce malformed JSON on ambiguous inputs. BaseLocalAgent retries once on parse failure. Use Llama 3.1 8B if hardware permits.
- **Static knowledge base** — Documents require manual updates for background facts. Live sector news (Node 0) mitigates staleness for current market signals.
- **FinBERT token limit** — Processes only the first 512 tokens. System warns if input exceeds 400 words.
- **Single-user prototype** — FastAPI is single-process. Concurrent users require worker processes (Celery), which is out of scope.
- **No real financial data** — All analysis is text-based; no balance sheets or market price data.

---

## Future Enhancements

1. **Llama 3.1 8B / Mistral 7B** — Higher quality agent reasoning (requires 16 GB RAM)
2. **Knowledge Base Auto-Update** — Background scraping of CB Insights, YC blog, NASSCOM
3. **PDF Pitch Deck Input** — PyMuPDF text extraction injected into the pipeline
4. **Streaming API Response** — Stream node results to frontend as they complete
5. **Zero External API Mode** — Llama 3.1 70B via Ollama for fully offline operation
6. **Sector-Specific Fine-Tuning** — LoRA fine-tuning on startup pitch/outcome datasets
7. **PostgreSQL + Redis** — Replace SQLite; add caching for repeated queries
8. **HITL Review Gate** — Human-review queue via LangGraph `interrupt_before` for High-risk cases

---

## References

- Meta AI (2024). [Llama 3.2](https://meta.ai/llama). Meta Llama 3.2 Community License.
- Araci, D. (2019). [FinBERT: Financial Sentiment Analysis](https://huggingface.co/ProsusAI/finbert). arXiv:1908.10063.
- Reimers & Gurevych (2019). [Sentence-BERT](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). EMNLP 2019.
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- [Anthropic Claude API](https://docs.anthropic.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Ollama Documentation](https://ollama.ai/docs)

---

> **Academic Disclaimer:** This system is a final year engineering prototype for educational purposes. Its outputs do not constitute professional financial advice. Do not use recommendations for actual investment decisions.
