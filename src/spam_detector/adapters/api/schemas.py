"""HTTP schema models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4096)


class PredictResponse(BaseModel):
    label: str
    score: float
    is_spam: bool


class HealthResponse(BaseModel):
    status: str
    backend: str
