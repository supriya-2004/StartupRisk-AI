# routers/risk_router.py

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from graph.builder import get_initial_state
from models.database import SessionLocal, RiskReportRecord
from models.schemas import RiskReport, StartupQuery

router = APIRouter(tags=["analysis"])


# ---------------------------------------------------------------------------
# Dependency: SQLAlchemy session
# ---------------------------------------------------------------------------
def get_db():
    """Yield a SQLAlchemy session and close it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Dependency: compiled LangGraph (injected via app.state)
# ---------------------------------------------------------------------------
def get_graph(request: Request) -> Any:
    """
    Retrieve the compiled LangGraph graph from app.state.
    The router never imports _graph directly — always via this dependency.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis pipeline not ready. Server is still initialising.",
        )
    return graph


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------
@router.post("/analyze", response_model=RiskReport, summary="Analyse a startup's risk profile")
async def analyze_startup(
    query: StartupQuery,
    request: Request,
    db: Session = Depends(get_db),
) -> RiskReport:
    """
    Run the full StartupRisk AI v2 multi-agent pipeline.

    Steps
    -----
    1. Validate input (≥ 80 characters).
    2. Generate a UUID4 thread_id for LangGraph checkpointing.
    3. Build initial state.
    4. Run the graph asynchronously.
    5. Persist the result to SQLite.
    6. Return the final RiskReport.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate minimum description length
    # ------------------------------------------------------------------ #
    if not query.startup_description or len(query.startup_description.strip()) < 80:
        raise HTTPException(
            status_code=400,
            detail=(
                "startup_description must be at least 80 characters. "
                f"Received {len(query.startup_description.strip())} characters."
            ),
        )
    
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=503,
            detail="Analysis pipeline not ready. Server is still initialising.",
        )

    # ------------------------------------------------------------------ #
    # 2. Generate thread_id for this request
    # ------------------------------------------------------------------ #
    thread_id = str(uuid.uuid4())

    # ------------------------------------------------------------------ #
    # 3. Build initial LangGraph state
    # ------------------------------------------------------------------ #
    initial_state = get_initial_state(query)

    # ------------------------------------------------------------------ #
    # 4. Run the graph
    # ------------------------------------------------------------------ #
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    # ------------------------------------------------------------------ #
    # 5. Surface pipeline-level errors
    # ------------------------------------------------------------------ #
    if result.get("error"):
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {result['error']}",
        )

    final_report: RiskReport = result.get("final_report")
    if final_report is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline completed but produced no final report.",
        )

    # ------------------------------------------------------------------ #
    # 6. Persist to SQLite
    # ------------------------------------------------------------------ #
    try:
        record = RiskReportRecord(
            thread_id=thread_id,
            startup_description_excerpt=query.startup_description[:300],
            sector=query.sector or "Unknown",
            funding_stage=query.funding_stage or "Unknown",
            overall_risk_score=final_report.overall_risk_score,
            risk_level=final_report.risk_level,
            recommendation=final_report.recommendation,
            full_report_json=final_report.model_dump_json(),
        )
        db.add(record)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        # Persistence failure should not block the API response
        db.rollback()
        print(f"[WARNING] Failed to persist report to SQLite: {exc}")

    # ------------------------------------------------------------------ #
    # 7. Return the report
    # ------------------------------------------------------------------ #
    return final_report
