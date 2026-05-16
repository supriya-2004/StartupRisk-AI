from __future__ import annotations
import os
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Text, DateTime, JSON
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./startup_risk.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite only
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class RiskReportRecord(Base):
    """Persists a completed RiskReport to SQLite."""
    __tablename__ = "risk_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(64), unique=True, index=True, nullable=True)

    # Core output
    overall_risk_score = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False)
    recommendation = Column(String(64), nullable=False)

    # Narrative
    short_term_prediction = Column(Text, nullable=True)
    long_term_prediction = Column(Text, nullable=True)
    investment_thesis = Column(Text, nullable=True)
    top_risk_factors = Column(Text, nullable=True)   # JSON-serialised list
    positive_signals = Column(Text, nullable=True)   # JSON-serialised list

    # Agent scores (summary)
    market_risk_score = Column(Float, nullable=True)
    financial_risk_score = Column(Float, nullable=True)
    viability_risk_score = Column(Float, nullable=True)

    # Sentiment
    dominant_sentiment = Column(String(16), nullable=True)
    sentiment_negative_score = Column(Float, nullable=True)

    # Query context
    sector = Column(String(64), nullable=True)
    funding_stage = Column(String(64), nullable=True)
    startup_description_excerpt = Column(Text, nullable=True)

    # Full report blob (for history retrieval)
    full_report_json = Column(Text, nullable=True)   # full RiskReport as JSON

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<RiskReportRecord id={self.id} "
            f"risk_level='{self.risk_level}' "
            f"score={self.overall_risk_score:.3f} "
            f"created_at='{self.created_at}'>"
        )

    @classmethod
    def from_risk_report(cls, report, thread_id: str | None = None):
        """Factory: build a DB record from a RiskReport Pydantic object."""
        return cls(
            thread_id=thread_id or report.thread_id,
            overall_risk_score=report.overall_risk_score,
            risk_level=report.risk_level,
            recommendation=report.recommendation,
            short_term_prediction=report.short_term_prediction,
            long_term_prediction=report.long_term_prediction,
            investment_thesis=report.investment_thesis,
            top_risk_factors=json.dumps(report.top_risk_factors),
            positive_signals=json.dumps(report.positive_signals),
            market_risk_score=(
                report.market_analysis.market_risk_score
                if report.market_analysis else None
            ),
            financial_risk_score=(
                report.financial_analysis.financial_risk_score
                if report.financial_analysis else None
            ),
            viability_risk_score=(
                report.viability_analysis.viability_risk_score
                if report.viability_analysis else None
            ),
            dominant_sentiment=(
                report.sentiment.dominant_sentiment
                if report.sentiment else None
            ),
            sentiment_negative_score=(
                report.sentiment.negative_score
                if report.sentiment else None
            ),
            sector=report.sector,
            funding_stage=report.funding_stage,
            startup_description_excerpt=report.startup_description_excerpt,
            full_report_json=report.model_dump_json(),
        )


class QueryHistoryRecord(Base):
    """Stores each incoming query for audit and history display."""
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thread_id = Column(String(64), index=True, nullable=True)
    startup_description = Column(Text, nullable=False)
    sector = Column(String(64), nullable=True)
    funding_stage = Column(String(64), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return (
            f"<QueryHistoryRecord id={self.id} "
            f"sector='{self.sector}' "
            f"timestamp='{self.timestamp}'>"
        )


def create_all():
    """Create all tables. Called once at app startup."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created (SQLite).")


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()