# backend/main.py

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from models import database
from models.database import engine, SessionLocal, RiskReportRecord
from rag.indexer import build_or_load_vectorstore
from graph.builder import build_analysis_graph
from routers import risk_router as risk_router_module

# ---------------------------------------------------------------------------
# Module-level singletons (set during startup)
# ---------------------------------------------------------------------------
_vectorstore = None
_graph = None


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vectorstore, _graph

    # 1. Build / load ChromaDB vectorstore
    _vectorstore = build_or_load_vectorstore()

    # 2. Build LangGraph pipeline
    _graph = build_analysis_graph(_vectorstore)

    # 3. Expose via app.state so routers can use dependency injection
    app.state.graph = _graph

    # 4. Create SQLAlchemy tables (idempotent)
    database.Base.metadata.create_all(bind=engine)

    print("Startup complete. API ready at http://localhost:8000/docs")

    yield  # application runs here

    # Shutdown: nothing special needed


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="StartupRisk AI v2",
    version="2.0.0",
    description=(
        "Hybrid multi-agent startup risk analysis system. "
        "Uses local LLMs (Ollama/Llama 3.2), FinBERT, RAG, and Claude synthesis."
    ),
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(risk_router_module.router, prefix="/api/v2")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
async def health_check():
    """Root health check — returns API status and version."""
    return {"status": "ok", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# History endpoint  (GET /api/v2/history)
# ---------------------------------------------------------------------------
@app.get("/api/v2/history", tags=["history"])
async def get_history() -> List[dict]:
    """Return the last 20 completed risk analyses from SQLite."""
    db: Session = SessionLocal()
    try:
        records = (
            db.query(RiskReportRecord)
            .order_by(RiskReportRecord.created_at.desc())
            .limit(20)
            .all()
        )
        return [record.to_dict() for record in records]
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Pipeline status endpoint  (GET /api/v2/status/{thread_id})
# ---------------------------------------------------------------------------
@app.get("/api/v2/status/{thread_id}", tags=["status"])
async def get_pipeline_status(thread_id: str, request: Request) -> dict:
    """
    Return node_timings from the LangGraph checkpointer for the given thread_id.
    Used by the frontend for live pipeline progress polling.
    """
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # LangGraph compiled graph exposes get_state() for checkpoint access
        state_snapshot = await graph.aget_state(config)
        if state_snapshot and state_snapshot.values:
            timings = state_snapshot.values.get("node_timings", {})
            return {"thread_id": thread_id, "node_timings": timings}
    except Exception as exc:  # noqa: BLE001
        # Thread not found or checkpointer miss — return empty timings
        return {"thread_id": thread_id, "node_timings": {}, "detail": str(exc)}

    return {"thread_id": thread_id, "node_timings": {}}