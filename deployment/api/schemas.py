"""Pydantic models for API requests and responses, separate from dataset schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class RetrievedEvidenceResponse(BaseModel):
    """One retrieved corpus item, as returned to API callers."""

    pair_id: str
    caption: str
    similarity_score: float


class PredictionResponse(BaseModel):

    answer: str
    question: str
    retrieved_evidence: list[RetrievedEvidenceResponse] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
