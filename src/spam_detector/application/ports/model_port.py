"""Model inference port."""

from __future__ import annotations

from typing import Protocol


class ModelPort(Protocol):
    """Port for model adapters that provide spam probabilities."""

    def predict_proba(self, messages: list[str]) -> list[float]:
        """Return probability of spam for each message."""
