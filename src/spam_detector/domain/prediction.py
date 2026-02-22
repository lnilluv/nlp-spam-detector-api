"""Domain entities for spam predictions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float
    is_spam: bool
